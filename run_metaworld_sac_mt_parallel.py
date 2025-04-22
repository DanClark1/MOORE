import metaworld
# mushroomrl
from mushroom_rl.core import Logger
# deep learning frameworks
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
from datetime import datetime

# Helper to split a flat trajectory into episodes

def split_episodes(traj):
    episodes, current = [], []
    for reward, done in traj:
        current.append(reward)
        if done:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes

# Unified save & load utilities with timestamped filenames
def save_checkpoint(agent, save_dir, epoch=None):
    """
    Save actor, critic, and full agent with timestamp and optional epoch.
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f"_{epoch}_{ts}" if epoch is not None else f"_{ts}"
    # actor
    actor_w = agent.policy.get_weights()
    np.save(os.path.join(save_dir, f"actor/actor_weights{suffix}.npy"), actor_w)
    # critic
    critic_w = agent.get_critic_weights()
    for key, val in critic_w.items():
        np.save(os.path.join(save_dir, f"critic/{key}{suffix}.npy"), val)
    # full agent
    agent.save(os.path.join(save_dir, f"agent/agent{suffix}"), full_save=True)

# Load existing checkpoints if found (ignores timestamp suffix)
def load_checkpoint(agent, load_dir):
    """
    Attempt to load the most recent agent checkpoint from load_dir/agent.
    """
    agent_dir = os.path.join(load_dir, "agent")
    if not os.path.isdir(agent_dir):
        return agent
    files = [f for f in os.listdir(agent_dir) if f.startswith("agent_")]
    if not files:
        return agent
    epochs = []
    for f in files:
        parts = f.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            epochs.append((int(parts[1]), f))
    if not epochs:
        return agent
    latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
    path = os.path.join(agent_dir, latest_file.rstrip('.npy'))
    return agent.load(path)

# The function to run a single experiment

def run_experiment(args, save_dir, exp_id=0, seed=None):
    import matplotlib
    matplotlib.use('Agg')
    np.random.seed(seed)

    # initialize logger and directories
    single_logger = Logger(f"seed_{exp_id}", results_dir=save_dir, log_console=True)
    save_dir = single_logger.path
    os.makedirs(save_dir, exist_ok=True)
    for sub in ("actor", "critic", "agent"):
        os.makedirs(os.path.join(save_dir, sub), exist_ok=True)

    # prepare MDP
    benchmark = getattr(metaworld, args.exp_type)()
    mdp = SubprocVecEnv([
        make_env(env_name=env_name,
                 env_cls=env_cls,
                 train_tasks=benchmark.train_tasks,
                 horizon=args.horizon,
                 gamma=args.gamma,
                 normalize_reward=args.normalize_reward,
                 sample_task_per_episode=args.sample_task_per_episode)
        for env_name, env_cls in benchmark.train_classes.items()
    ])
    n_contexts = mdp.num_envs

    # extract settings
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    n_episodes_test = args.n_episodes_test
    train_freq = args.train_frequency
    gamma_eval = args.gamma_eval

    # networks and hyperparams (same as before)
    # [setup actor_params, critic_params, optimizers...]
    # omitted for brevity

    # create agent
    agent = MTSAC(
        mdp_info=mdp.info,
        batch_size=args.batch_size,
        initial_replay_size=args.initial_replay_size,
        max_replay_size=int(args.max_replay_size),
        warmup_transitions=args.warmup_transitions,
        tau=args.tau,
        lr_alpha=args.lr_alpha,
        actor_params=actor_params,
        actor_mu_params=actor_mu_params,
        actor_sigma_params=actor_sigma_params,
        actor_optimizer={'class': optim.Adam, 'params': {'lr': args.lr_actor, 'betas': (0.9, 0.999)}},
        critic_params=critic_params,
        target_entropy=args.target_entropy,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        shared_mu_sigma=args.shared_mu_sigma,
        n_contexts=n_contexts
    )

    # load checkpoint if requested
    if args.load_agent or args.load_actor or args.load_critic:
        agent = load_checkpoint(agent, args.load_agent or save_dir)
    single_logger.info("Loaded checkpoint if available.")

    agent.set_logger(single_logger)
    agent.models_summary()

    # prepare core
    core = VecCore(agent, mdp)
    env_names = mdp.get_attr("env_name")

    # init metrics dict
    metrics = {k: {m: [] for m in [
        "MinReturn","MaxReturn","AverageReturn",
        "AverageDiscountedReturn","SuccessRate","LogAlpha"
    ]} for k in env_names}
    metrics["all_metaworld"] = {"SuccessRate": []}

    # initial replay-fill and eval
    core.eval = False
    core.learn(n_steps=args.initial_replay_size, n_steps_per_fit=args.initial_replay_size, render=args.render_train)

    core.eval = True
    total_test = n_contexts * n_episodes_test
    data, _ = core.evaluate(n_episodes=total_test, render=args.render_eval if exp_id==0 else False)

    # compute and log initial metrics
    per_ctx = defaultdict(list)
    for idx_obs, _, reward, _, done, _ in data:
        per_ctx[idx_obs[0]].append((reward, done))
    sum_sr = 0
    for c, key in enumerate(env_names):
        eps = split_episodes(per_ctx[c])
        returns = [sum(ep) for ep in eps]
        disc = [sum(r*(gamma_eval**i) for i,r in enumerate(ep)) for ep in eps]
        sr = sum(1 for ep in eps if max(ep)>=0)/len(eps)
        sum_sr += sr
        metrics[key]["AverageReturn"].append(np.mean(returns))
        metrics[key]["AverageDiscountedReturn"].append(np.mean(disc))
        metrics[key]["SuccessRate"].append(sr)
    metrics["all_metaworld"]["SuccessRate"].append(sum_sr/n_contexts)
    if args.wandb:
        wandb.log({"all_metaworld/SuccessRate": metrics["all_metaworld"]["SuccessRate"][0]}, step=0)

    # main loop with parallel eval and timestamped checkpoints
    for epoch in trange(args.start_epoch, n_epochs):
        # training
        core.eval = False
        core.learn(n_steps=n_steps, n_steps_per_fit=train_freq, render=args.render_train)
        # evaluation
        core.eval = True
        data, _ = core.evaluate(n_episodes=total_test,
                                render=(args.render_eval if (epoch % args.render_interval==0 and exp_id==0) else False))
        per_ctx = defaultdict(list)
        for idx_obs, _, reward, _, done, _ in data:
            per_ctx[idx_obs[0]].append((reward, done))
        sum_sr = 0
        for c, key in enumerate(env_names):
            eps = split_episodes(per_ctx[c])
            returns = [sum(ep) for ep in eps]
            disc = [sum(r*(gamma_eval**i) for i,r in enumerate(ep)) for ep in eps]
            sr = sum(1 for ep in eps if max(ep)>=0)/len(eps)
            sum_sr += sr
            metrics[key]["AverageReturn"].append(np.mean(returns))
            metrics[key]["AverageDiscountedReturn"].append(np.mean(disc))
            metrics[key]["SuccessRate"].append(sr)
            if args.wandb:
                wandb.log({f"{key}/AverageReturn":metrics[key]["AverageReturn"][-1],
                           f"{key}/SuccessRate":metrics[key]["SuccessRate"][-1]}, step=epoch+1)
        all_sr = sum_sr/n_contexts
        metrics["all_metaworld"]["SuccessRate"].append(all_sr)
        if args.wandb:
            wandb.log({"all_metaworld/SuccessRate":all_sr}, step=epoch+1)

        # checkpoint
        if (epoch+1) % args.rl_checkpoint_interval == 0:
            save_checkpoint(agent, save_dir, epoch+1)

    # final save
    save_checkpoint(agent, save_dir)
    if args.wandb:
        wandb.finish()
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
    out = run_experiment(args, save_dir, exp_id=0, seed=args.seed[0] if args.seed else None)
    for key, value in out.items():
        if key == 'all_metaworld':
            np.save(os.path.join(save_dir, 'all_SuccessRate.npy'), value['SuccessRate'])
        else:
            for mk, mv in value.items():
                np.save(os.path.join(save_dir, f'{key}_{mk}.npy'), mv)
