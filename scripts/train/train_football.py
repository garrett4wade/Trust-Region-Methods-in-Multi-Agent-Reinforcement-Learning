#!/usr/bin/env python
import sys
import os

sys.path.append("../")
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from configs.config import get_config
from envs.football.football_env import FootballEnvironment, map_agent_registry
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from runners.separated.football_runner import FootballRunner as Runner
"""Train script for football."""


def make_train_env(all_args):

    def get_env_fn(rank):

        def init_env():
            assert all_args.env_name == "football"
            env = FootballEnvironment(
                seed=all_args.seed + rank * 1000,
                share_reward=all_args.share_reward,
                number_of_left_players_agent_controls=map_agent_registry[
                    all_args.map_name][0],
                number_of_right_players_agent_controls=0,
                rewards="scoring,checkpoints",
                env_name=all_args.map_name,
            )
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):

    def get_env_fn(rank):

        def init_env():
            assert all_args.env_name == 'football'
            env = FootballEnvironment(
                seed=all_args.seed * 50000 + rank * 10000,
                share_reward=all_args.share_reward,
                number_of_left_players_agent_controls=map_agent_registry[
                    all_args.map_name][0],
                number_of_right_players_agent_controls=0,
                rewards="scoring",
                env_name=all_args.map_name,
            )
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument(
        '--map_name',
        type=str,
        default='academy_3_vs_1_with_keeper',
    )
    parser.add_argument("--share_reward", action='store_true')
    parser.add_argument("--use_single_network",
                        action='store_true',
                        default=False)
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed = all_args.runing_id
    else:
        all_args.seed = np.random.randint(1000, 10000)
    print("seed is :", all_args.seed)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +
        "/results"
    ) / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name / str(
        all_args.seed)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [
            int(str(folder.name).split('run')[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith('run')
        ]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" +
        str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = map_agent_registry[all_args.map_name][0]

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    # run experiments
    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    runner.writter.export_scalars_to_json(str(runner.log_dir +
                                              '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
