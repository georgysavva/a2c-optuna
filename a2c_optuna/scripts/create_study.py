import optuna


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study_storage",
        type=str,
        required=True,
    )
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--study", type=str, required=True)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_iter", "-n", type=int, default=1000)
    parser.add_argument("--eval_episodes", "-ee", type=int, default=8)
    parser.add_argument("--num_envs", "-en", type=int, default=8)
    parser.add_argument(
        "--deterministic_eval", "-de", action="store_true", default=True
    )
    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--video_log_freq", type=int, default=50)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    args = parser.parse_args()
    study_name = args.env_name + "-" + args.study
    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=args.study_storage
    )
    study.set_user_attr("env_name", args.env_name)
    study.set_user_attr("n_iter", args.n_iter)
    study.set_user_attr("eval_episodes", args.eval_episodes)
    study.set_user_attr("deterministic_eval", args.deterministic_eval)
    study.set_user_attr("ep_len", args.ep_len)
    study.set_user_attr("num_envs", args.num_envs)
    study.set_user_attr("video_log_freq", args.video_log_freq)
    study.set_user_attr("scalar_log_freq", args.scalar_log_freq)
    study.set_user_attr("seed", args.seed)


if __name__ == "__main__":
    main()
