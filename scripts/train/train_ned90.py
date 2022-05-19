#!/usr/bin/env python
import sys
import os

import wandb

sys.path.append("../")
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from configs.config import get_config
from envs.ned90.ned90_env import MockNED90Environment, NED90Environment
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from runners.separated.ned90_runner import Ned90Runner as Runner

GAME_CONFIG = {
    "win_reward": 100,
    "fail_reward": -100,
    "attack_reward": 100,
    "defence_reward": 80,
    "step_reward": -0.2,
    "out_circle_reward": -3,
    "shock_reward": 10,
    "use_combo_reward": True
}

action_reward = {3: 1, 4: 1, 5: 2}


def make_train_env(all_args):

    def get_env_fn(rank):

        def init_env():
            assert all_args.env_name == "ned90"
            env = MockNED90Environment(
                broker_addr='frlcpu001',
                inbound_port=27777,
                outbound_port=27778,
                game_config=GAME_CONFIG,
                action_rewards=action_reward,
                env_id=rank,
                is_opponent=False,  # self-play by default
            )
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.num_env_steps = int(1e8)
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
    ) / all_args.env_name / all_args.algorithm_name / all_args.experiment_name / str(
        all_args.seed)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project="ned90-mappo",
                         group=all_args.experiment_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) + "_seed" +
                         str(all_args.seed),
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
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
    eval_envs = None
    num_agents = 1
    all_args.num_agents = num_agents

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
    if not all_args.eval_only and not all_args.use_render:
        runner.run()
    else:
        assert all_args.model_dir is not None
        if all_args.eval_only:
            runner.eval(0, render=False)
        else:
            assert all_args.n_render_rollout_threads == 1
            runner.eval(0, render=True)

    # post process
    envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        if hasattr(runner, writter):
            runner.writter.export_scalars_to_json(
                str(runner.log_dir + '/summary.json'))
            runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
