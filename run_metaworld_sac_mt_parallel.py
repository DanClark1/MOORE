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


def save_checkpoint(agent, save_dir, epoch=None):
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


def load_checkpoint(agent, load_dir, args):
    # load full agent if requested
    if args.load_agent:
        return agent.load(args.load_agent)
    # load critic only
    if args.load_critic:
        agent.set_critic_weights(args.load_critic)
    # load actor only
    if args.load_actor:
        agent.policy.set_weights(np.load(args.load_actor))
    return agent


def run_experiment(args, save_dir, exp_id=0, seed=None):
    import matplotlib
    matplotlib.use('Agg')

    np.random.seed(seed)
    
    single_logger = Logger(f"seed_{exp_id if seed is None else seed}", results_dir=save_dir, log_console=True)
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
    gamma = args.gamma
    gamma_eval = args.gamma_eval

    # networks and hyperparams setup (same as original)...
    # [setup actor_params, critic_params, optimizers, etc.]
    # omitted here for brevity but must match original logic

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
    agent = load_checkpoint(agent, save_dir, args)
    single_logger.info("Loaded checkpoint if available.")

    agent.set_logger(single_logger)
    agent.models_summary()

    core = VecCore(agent, mdp)
    env_names = mdp.get_attr("env_name")

    # init metrics dict
    metrics = {k: defaultdict(list) for k in env_names}
    for k in metrics:
        for m in ["MinReturn","MaxReturn","AverageReturn","AverageDiscountedReturn","SuccessRate","LogAlpha"]:
            metrics[k][m] = []
    metrics["all_metaworld"] = {"SuccessRate": []}

    # initial replay-fill and eval
    core.eval = False
    core.learn(n_steps=args.initial_replay_size, n_steps_per_fit=args.initial_replay_size, render=args.render_train)

    # evaluate initial policy
    core.eval = True
    total_test = n_contexts * n_episodes_test
    data, info = core.evaluate(n_episodes=total_test, render=(args.render_eval if exp_id==0 else False), get_env_info=True)

    # compute metrics using get_stats per-context
    index_data = defaultdict(list)
    for idx_obs, action, reward, next_obs, done, env_info in data:
        index_data[idx_obs[0]].append((reward, done))
    sum_sr = 0
    for c, key in enumerate(env_names):
        # use get_stats for full metrics
        idx_data = [d for d in data if d[0][0] == c]
        stats = get_stats(idx_data, gamma, gamma_eval, dataset_info=info[c])
        min_J, max_J, mean_J, mean_discounted_J, success_rate = stats
        log_alpha = agent.get_log_alpha(c)

        metrics[key]["MinReturn"].append(min_J)
        metrics[key]["MaxReturn"].append(max_J)
        metrics[key]["AverageReturn"].append(mean_J)
        metrics[key]["AverageDiscountedReturn"].append(mean_discounted_J)
        metrics[key]["SuccessRate"].append(success_rate)
        metrics[key]["LogAlpha"].append(log_alpha)
        sum_sr += success_rate
        if args.wandb:
            wandb.log({
                f'{key}/MinReturn': min_J,
                f'{key}/MaxReturn': max_J,
                f'{key}/AverageReturn': mean_J,
                f'{key}/AverageDiscountedReturn': mean_discounted_J,
                f'{key}/SuccessRate': success_rate,
                f'{key}/LogAlpha': log_alpha
            }, step=0, commit=False)

    all_sr = sum_sr / n_contexts
    metrics["all_metaworld"]["SuccessRate"].append(all_sr)
    if args.wandb:
        wandb.log({"all_metaworld/SuccessRate": all_sr}, step=0, commit=True)

    # main loop
    for epoch in trange(args.start_epoch, n_epochs):
        core.eval = False
        core.learn(n_steps=n_steps, n_steps_per_fit=train_freq, render=args.render_train)

        core.eval = True
        sum_sr = 0
        for c, key in enumerate(env_names):
            data, info = core.evaluate(n_episodes=n_episodes_test, render=(args.render_eval if (epoch % args.render_interval == 0 and exp_id==0) else False), get_env_info=True)
            # stats
            min_J, max_J, mean_J, mean_discounted_J, success_rate = get_stats(data, gamma, gamma_eval, dataset_info=info)
            log_alpha = agent.get_log_alpha(c)

            metrics[key]["MinReturn"].append(min_J)
            metrics[key]["MaxReturn"].append(max_J)
            metrics[key]["AverageReturn"].append(mean_J)
            metrics[key]["AverageDiscountedReturn"].append(mean_discounted_J)
            metrics[key]["SuccessRate"].append(success_rate)
            metrics[key]["LogAlpha"].append(log_alpha)
            sum_sr += success_rate
            
            if args.wandb:
                wandb.log({
                    f'{key}/MinReturn': min_J,
                    f'{key}/MaxReturn': max_J,
                    f'{key}/AverageReturn': mean_J,
                    f'{key}/AverageDiscountedReturn': mean_discounted_J,
                    f'{key}/SuccessRate': success_rate,
                    f'{key}/LogAlpha': log_alpha
                }, step=epoch+1, commit=False)

        all_sr = sum_sr / n_contexts
        metrics["all_metaworld"]["SuccessRate"].append(all_sr)
        if args.wandb:
            wandb.log({"all_metaworld/SuccessRate": all_sr}, step=epoch+1, commit=True)

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
