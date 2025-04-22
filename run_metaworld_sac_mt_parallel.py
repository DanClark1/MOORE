import metaworld
# mushroomrl
from mushroom_rl.core import Logger
# deeplearning frameworks
import torch.optim as optim
import torch.nn.functional as F
# continual proto-value functions
from moore.core import VecCore
from moore.algorithms.actor_critic import MTSAC
from moore.environments.metaworld_env import make_env
from moore.environments import SubprocVecEnv
from moore.utils.dataset import get_stats
from moore.utils.argparser import argparser
import moore.utils.networks_sac as Network
# data handling
import numpy as np
# visualization
from tqdm import trange
import wandb
# Utils
import pickle
import os
from collections import defaultdict

# Helper to split a flat trajectory into episodes

def split_episodes(traj):
    episodes = []
    current = []
    for reward, done in traj:
        current.append(reward)
        if done:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes

# The function is used to run a single experiment 
def run_experiment(args, save_dir, exp_id = 0, seed = None):
    import matplotlib
    matplotlib.use('Agg')

    np.random.seed(seed)

    single_logger = Logger(f"seed_{exp_id if seed is None else seed}", results_dir=save_dir, log_console=True)
    save_dir = single_logger.path

    n_epochs = args.n_epochs
    n_steps = args.n_steps
    n_episodes_test = args.n_episodes_test

    # MDP
    exp_type = args.exp_type
    horizon = args.horizon
    gamma = args.gamma
    gamma_eval = args.gamma_eval

    benchmark = getattr(metaworld, exp_type)()

    mdp = SubprocVecEnv([
        make_env(env_name=env_name,
                 env_cls=env_cls,
                 train_tasks=benchmark.train_tasks,
                 horizon=horizon,
                 gamma=gamma,
                 normalize_reward=args.normalize_reward,
                 sample_task_per_episode=args.sample_task_per_episode)
        for env_name, env_cls in benchmark.train_classes.items()
    ])
    n_contexts = mdp.num_envs

    # Settings
    initial_replay_size = args.initial_replay_size
    max_replay_size = int(args.max_replay_size)
    batch_size = args.batch_size
    train_frequency = args.train_frequency
    tau = args.tau
    warmup_transitions = args.warmup_transitions
    log_std_min = args.log_std_min
    log_std_max = args.log_std_max

    append_context_actor = "Single" in args.actor_network
    append_context_mu_actor = "Single" in args.actor_mu_network
    append_context_sigma_actor = "Single" in args.actor_sigma_network
    append_context_critic = "Single" in args.critic_network

    # Networks and params
    if args.shared_mu_sigma:
        actor_network = getattr(Network, args.actor_network)
        actor_n_features = args.actor_n_features
    else:
        actor_mu_network = getattr(Network, args.actor_mu_network)
        actor_sigma_network = getattr(Network, args.actor_sigma_network)
        actor_mu_n_features = args.actor_mu_n_features
        actor_sigma_n_features = args.actor_sigma_n_features

    critic_network = getattr(Network, args.critic_network)
    critic_n_features = args.critic_n_features

    lr_alpha = args.lr_alpha
    lr_actor = args.lr_actor
    lr_critic = args.lr_critic
    target_entropy = args.target_entropy

    # Actor / critic params dicts
    actor_params = None
    actor_mu_params = None
    actor_sigma_params = None

    # Shared or separate mu/sigma
    if args.shared_mu_sigma:
        inp = mdp.observation_space.shape
        if append_context_actor:
            inp = (inp[0] + n_contexts,)
        actor_params = dict(
            network=actor_network,
            n_features=actor_n_features,
            input_shape=inp,
            output_shape=mdp.action_space.shape,
            shared_mu_sigma=True,
            use_cuda=args.use_cuda,
            n_contexts=n_contexts,
            activation=args.activation,
            orthogonal=args.orthogonal,
            n_experts=args.n_experts,
            agg_activation=args.agg_activation,
        )
    else:
        mu_inp = mdp.observation_space.shape
        if append_context_mu_actor:
            mu_inp = (mu_inp[0] + n_contexts,)
        sigma_inp = mdp.observation_space.shape
        if append_context_sigma_actor:
            sigma_inp = (sigma_inp[0] + n_contexts,)
        actor_mu_params = dict(
            network=actor_mu_network,
            n_features=actor_mu_n_features,
            input_shape=mu_inp,
            output_shape=mdp.action_space.shape,
            use_cuda=args.use_cuda,
            n_contexts=n_contexts,
            activation=args.activation,
            orthogonal=args.orthogonal,
            n_experts=args.n_experts,
            agg_activation=args.agg_activation,
        )
        actor_sigma_params = dict(
            network=actor_sigma_network,
            n_features=actor_sigma_n_features,
            input_shape=sigma_inp,
            output_shape=mdp.action_space.shape,
            use_cuda=args.use_cuda,
            n_contexts=n_contexts,
            activation=args.activation,
            orthogonal=args.orthogonal,
            n_experts=args.n_experts,
            agg_activation=args.agg_activation,
        )

    actor_optimizer = {'class': optim.Adam, 'params': {'lr': lr_actor, 'betas': (0.9, 0.999)}}

    crit_inp = (mdp.observation_space.shape[0] + mdp.action_space.shape[0],)
    if append_context_critic:
        crit_inp = (crit_inp[0] + n_contexts,)
    critic_params = dict(
        network=critic_network,
        optimizer={'class': optim.Adam, 'params': {'lr': lr_critic, 'betas': (0.9, 0.999)}},
        loss=F.mse_loss,
        n_features=critic_n_features,
        input_shape=crit_inp,
        output_shape=(1,),
        use_cuda=args.use_cuda,
        n_contexts=n_contexts,
        activation=args.activation,
        orthogonal=args.orthogonal,
        n_experts=args.n_experts,
        agg_activation=args.agg_activation,
    )

    # Debug overrides
    if args.debug:
        initial_replay_size = 150
        batch_size = 8
        n_epochs = 2
        n_steps = 150
        n_episodes_test = 1
        args.wandb = False
        warmup_transitions = 150

    # Weights & directories
    if args.wandb:
        wandb.init(
            name=f"seed_{exp_id}", project="MOORE",
            group=f"metaworld_{args.env_name or args.exp_type}",
            job_type=args.exp_name, config=vars(args)
        )
    os.makedirs(save_dir, exist_ok=True)
    for sub in ("actor", "critic", "agent"):
        os.makedirs(os.path.join(save_dir, sub), exist_ok=True)

    # Create agent
    agent = MTSAC(
        mdp_info=mdp.info,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        warmup_transitions=warmup_transitions,
        tau=tau,
        lr_alpha=lr_alpha,
        actor_params=actor_params,
        actor_mu_params=actor_mu_params,
        actor_sigma_params=actor_sigma_params,
        actor_optimizer=actor_optimizer,
        critic_params=critic_params,
        target_entropy=target_entropy,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        shared_mu_sigma=args.shared_mu_sigma,
        n_contexts=n_contexts
    )

    # Load weights if requested
    if args.load_agent:
        agent = agent.load(args.load_agent)
    else:
        if args.load_critic:
            agent.set_critic_weights(args.load_critic)
        if args.load_actor:
            agent.policy.set_weights(np.load(args.load_actor))

    agent.set_logger(single_logger)
    agent.models_summary()

    core = VecCore(agent, mdp)
    env_names = mdp.get_attr("env_name")

    # Initialize metrics
    metrics = {k: {m: [] for m in [
        "MinReturn","MaxReturn","AverageReturn",
        "AverageDiscountedReturn","SuccessRate","LogAlpha"
    ]} for k in env_names}
    metrics["all_metaworld"] = {"SuccessRate": []}

    # --- Initial random evaluation ---
    if args.start_epoch == 0:
        core.eval = False
        core.learn(
            n_steps=initial_replay_size,
            n_steps_per_fit=initial_replay_size,
            render=args.render_train
        )

        core.eval = True
        total_test_epis = n_contexts * n_episodes_test
        dataset, ds_info = core.evaluate(
            n_episodes=total_test_epis,
            render=args.render_eval if exp_id == 0 else False
        )

        # Compute per-context stats
        per_ctx = defaultdict(list)
        for idx_obs, _, reward, _, done, _ in dataset:
            per_ctx[idx_obs[0]].append((reward, done))

        sum_sr = 0
        for c, key in enumerate(env_names):
            eps = split_episodes(per_ctx[c])
            returns = [sum(ep) for ep in eps]
            disc_returns = [sum(r * (gamma_eval**i) for i,r in enumerate(ep)) for ep in eps]
            sr = sum(1 for ep in eps if max(ep) >= 0) / len(eps)
            sum_sr += sr
            metrics[key]["AverageReturn"].append(np.mean(returns))
            metrics[key]["AverageDiscountedReturn"].append(np.mean(disc_returns))
            metrics[key]["SuccessRate"].append(sr)
            metrics["all_metaworld"]["SuccessRate"].append(sum_sr / n_contexts)
        if args.wandb:
            wandb.log({f"all_metaworld/SuccessRate": metrics["all_metaworld"]["SuccessRate"][0]}, step=0)

    # --- Main training loop with parallel eval ---
    for epoch in trange(args.start_epoch, n_epochs):
        # Training
        core.eval = False
        core.learn(
            n_steps=n_steps,
            n_steps_per_fit=train_frequency,
            render=args.render_train
        )

        # Evaluation
        core.eval = True
        total_test_epis = n_contexts * n_episodes_test
        dataset, ds_info = core.evaluate(
            n_episodes=total_test_epis,
            render=(args.render_eval if (epoch % args.render_interval == 0 and exp_id == 0) else False)
        )

        per_ctx = defaultdict(list)
        for idx_obs, _, reward, _, done, _ in dataset:
            per_ctx[idx_obs[0]].append((reward, done))

        sum_sr = 0
        for c, key in enumerate(env_names):
            eps = split_episodes(per_ctx[c])
            returns = [sum(ep) for ep in eps]
            disc_returns = [sum(r * (gamma_eval**i) for i,r in enumerate(ep)) for ep in eps]
            sr = sum(1 for ep in eps if max(ep) >= 0) / len(eps)
            sum_sr += sr
            metrics[key]["AverageReturn"].append(np.mean(returns))
            metrics[key]["AverageDiscountedReturn"].append(np.mean(disc_returns))
            metrics[key]["SuccessRate"].append(sr)
            if args.wandb:
                wandb.log({
                    f"{key}/AverageReturn": metrics[key]["AverageReturn"][ -1],
                    f"{key}/SuccessRate": metrics[key]["SuccessRate"][ -1],
                }, step=epoch+1)
        all_sr = sum_sr / n_contexts
        metrics["all_metaworld"]["SuccessRate"].append(all_sr)
        if args.wandb:
            wandb.log({f"all_metaworld/SuccessRate": all_sr}, step=epoch+1)

        # Checkpoint
        if (epoch+1) % args.rl_checkpoint_interval == 0:
            weights = agent.policy.get_weights()
            np.save(os.path.join(save_dir, f"actor/actor_weights_{epoch+1}.npy"), weights)
            critic_w = agent.get_critic_weights()
            for k,v in critic_w.items():
                np.save(os.path.join(save_dir, f"critic/{k}_{epoch+1}.npy"), v)
            agent.save(os.path.join(save_dir, f"agent/agent_{epoch+1}"), full_save=True)

    if args.wandb:
        wandb.finish()

    # Final save
    w = agent.policy.get_weights()
    np.save(os.path.join(save_dir, "actor/actor_weights.npy"), w)
    cw = agent.get_critic_weights()
    for k,v in cw.items():
        np.save(os.path.join(save_dir, f"critic/{k}.npy"), v)
    agent.save(os.path.join(save_dir, "agent/agent_final"), full_save=True)

    return metrics

if __name__ == '__main__':
    args = argparser()
    if args.seed is not None:
        assert len(args.seed) == args.n_exp
    alg_name = "mixture_orthogonal_experts" if args.orthogonal else "mixture_experts"
    results_dir = os.path.join(args.results_dir, args.exp_type, alg_name)
    logger = Logger(args.exp_name, results_dir=results_dir, log_console=True, use_timestamp=args.use_timestamp)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + MTSAC.__name__)
    save_dir = logger.path
    with open(os.path.join(save_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    logger.info(vars(args))
    out = run_experiment(args, save_dir, seed=args.seed[0])
    for key,value in out.items():
        if key == 'all_metaworld':
            np.save(os.path.join(save_dir, 'all_SuccessRate.npy'), value['SuccessRate'])
        else:
            for mk,mv in value.items():
                np.save(os.path.join(save_dir, f'{key}_{mk}.npy'), mv)
