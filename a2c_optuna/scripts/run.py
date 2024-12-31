import functools
import json
import os
import time

import gym
import gym.vector
import numpy as np
import optuna
import torch

import wandb
from a2c_optuna.agents.pg_agent import PGAgent
from a2c_optuna.infrastructure import pytorch_util as ptu
from a2c_optuna.infrastructure import utils
from a2c_optuna.infrastructure.logger import prepare_trajs_as_videos

MAX_NVIDEO = 2


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id, render_mode=None)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def run_training_loop(config: utils.AttrDict, training_callback=None):
    wandb_project = config.pop("wandb_project")
    wandb_tags = [config.env_name]
    study_name = config.get("study_name", None)
    if study_name is not None:
        wandb_tags.append(study_name)
    wandb.init(
        name=config.get("exp_name", None),
        project=wandb_project,
        config=config,
        group=study_name,
        tags=wandb_tags,
        reinit=True,
    )
    print("Run training with config:")
    print(json.dumps(config, indent=4))

    # set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    ptu.init_gpu(use_gpu=True, gpu_id=config.which_gpu)

    # make the gym environment
    env = gym.vector.AsyncVectorEnv(
        [make_env(config.env_name, config.seed + i) for i in range(config.num_envs)]
    )
    env_for_recording = make_env(config.env_name, config.seed)()

    discrete = isinstance(env.single_action_space, gym.spaces.Discrete)

    max_ep_len = config.ep_len or env.call("spec")[0].max_episode_steps
    assert max_ep_len is not None, "env must have max episode length"
    print(f"Will use a maximum of {max_ep_len} steps")
    ob_dim = env.single_observation_space.shape[0]
    ac_dim = env.single_action_space.n if discrete else env.single_action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.metadata["render_fps"]

    # initialize agent
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=config.n_layers,
        layer_size=config.layer_size,
        gamma=config.discount,
        learning_rate=config.learning_rate,
        use_baseline=config.use_baseline,
        use_reward_to_go=config.use_reward_to_go,
        normalize_advantages=config.normalize_advantages,
        value_td_lambda=config.value_td_lambda,
        baseline_learning_rate=config.learning_rate,
        baseline_gradient_steps=config.baseline_gradient_steps,
        gae_lambda=config.gae_lambda,
    )

    total_envsteps = 0
    start_time = time.time()
    eval_reward = 0.0
    assert (
        config.batch_size % config.num_envs == 0
    ), "Batch size must be divisible by num_envs"
    steps_in_batch = config.batch_size // config.num_envs

    assert (
        config.eval_episodes % config.num_envs == 0
    ), "Eval episodes must be divisible by num_envs"
    eval_episodes_per_env = config.eval_episodes // config.num_envs
    assert (
        config.video_log_freq % config.scalar_log_freq == 0
    ), "Video log freq must be divisible by scalar log freq"
    assert agent.critic is not None, "Critic must be used to estimate advantages"
    for itr in range(config.n_iter):
        print(f"\n********** Iteration {itr} ************")
        trajs = utils.sample_trajectories_vectorized(
            env, agent.actor, agent.critic, steps_in_batch
        )
        total_envsteps += steps_in_batch * config.num_envs

        train_info: dict = agent.update(
            trajs["observations"],
            trajs["actions"],
            trajs["rewards"],
            trajs["value_bootstraps"],
            trajs["dones"],
        )

        if itr % config.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs = utils.sample_n_trajectories_vectorized_for_eval(
                env,
                agent.actor,
                eval_episodes_per_env,
                deterministic_predict=config.deterministic_eval,
            )
            eval_reward = utils.compute_average_return(eval_trajs)
            logs = utils.compute_metrics(eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            # perform the logging

            for key, value in logs.items():
                print("{} : {}".format(key, value))
            wandb.log(logs, step=itr)
            print("Done logging...\n\n")

            if config.video_log_freq != -1 and itr % config.video_log_freq == 0:
                print("\nCollecting video rollouts...")
                video_trajs = utils.sample_n_trajectories(
                    env_for_recording,
                    agent.actor,
                    MAX_NVIDEO,
                    max_ep_len,
                    render=True,
                    deterministic_predict=config.deterministic_eval,
                )
                videos = prepare_trajs_as_videos(video_trajs, MAX_NVIDEO)
                wandb.log({"EvalRollouts": wandb.Video(videos, fps=fps)}, step=itr)
        if training_callback is not None:
            training_callback(itr, eval_reward)

    return eval_reward


def objective(config: utils.AttrDict, trial: optuna.Trial):
    hyper_params = get_hyper_parameters(trial)

    config.update(hyper_params)
    config.update(trial.study.user_attrs)
    config["trial_number"] = trial.number

    def training_callback(iteration, eval_return):
        trial.report(eval_return, step=iteration)
        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.TrialPruned()

    result = run_training_loop(config, training_callback)
    wandb.finish(quiet=True)
    return result


def get_hyper_parameters(trial: optuna.Trial):
    size_category = trial.suggest_categorical("size", ["small", "medium", "large"])
    if size_category == "small":
        layer_size = 64
        n_layers = 2
    elif size_category == "medium":
        layer_size = 128
        n_layers = 3
    else:
        layer_size = 256
        n_layers = 3
    hyperparams = {
        "n_layers": n_layers,
        "layer_size": layer_size,
        "size_category": size_category,
        "discount": trial.suggest_float("discount", 0.95, 0.99, step=0.04),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 1e-4]),
        "use_baseline": trial.suggest_categorical("use_baseline", [True]),
        "use_reward_to_go": trial.suggest_categorical("use_reward_to_go", [True]),
        "normalize_advantages": trial.suggest_categorical(
            "normalize_advantages", [True]
        ),
        "baseline_gradient_steps": trial.suggest_categorical(
            "baseline_gradient_steps", [5, 25, 50]
        ),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0, 0.97, 1]),
        "value_td_lambda": trial.suggest_categorical("value_td_lambda", [0, 0.97, 1]),
        "batch_size": trial.suggest_categorical("batch_size", [25000, 50000]),
    }
    return hyperparams


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--study_storage",
        type=str,
        required=True,
    )
    parser.add_argument("--wandb_project", required=True, type=str)

    parser.add_argument("--study_name", type=str)
    parser.add_argument("--sampler", type=str, default="TPESampler")
    parser.add_argument("--pruner", type=str, default="HyperbandPruner")
    parser.add_argument("--n_startup_trials", type=int, default=5)
    parser.add_argument("--pruner_min_resource", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--exp_config", type=str)
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    args = parser.parse_args()
    assert (
        args.exp_config or args.study_name
    ), "Either exp_config or study_name must be provided"
    config = vars(args)
    config = utils.AttrDict(config)
    if args.study_name:
        sampler_kwargs = {}
        if args.sampler == "TPESampler":
            sampler_kwargs["n_startup_trials"] = args.n_startup_trials
        pruner_kwargs = {}
        if args.pruner == "HyperbandPruner":
            pruner_kwargs["min_resource"] = args.pruner_min_resource
        elif args.pruner == "MedianPruner":
            pruner_kwargs["n_startup_trials"] = args.n_startup_trials
        study = optuna.load_study(
            study_name=args.study_name,
            sampler=getattr(optuna.samplers, args.sampler)(**sampler_kwargs),
            pruner=getattr(optuna.pruners, args.pruner)(**pruner_kwargs),
            storage=args.study_storage,
        )
        config.pop("study_storage")

        study.optimize(functools.partial(objective, config), n_trials=args.n_trials)  # type: ignore
    else:
        with open(args.exp_config) as f:
            exp_config = json.load(f)
        config.update(exp_config)
        config["exp_name"] = os.path.basename(args.exp_config).replace(".json", "")
        run_training_loop(config)


if __name__ == "__main__":
    main()
