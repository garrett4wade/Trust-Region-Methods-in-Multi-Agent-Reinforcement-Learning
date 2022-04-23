import time
import os
import numpy as np
from itertools import chain
import torch
import wandb
from tensorboardX import SummaryWriter
from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):

    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.share_policy = self.all_args.share_policy
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.all_args.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        if self.all_args.algorithm_name == "happo":
            from algorithms.happo_trainer import HAPPO as TrainAlgo
            from algorithms.happo_policy import HAPPO_Policy as Policy
        elif self.all_args.algorithm_name == "hatrpo":
            from algorithms.hatrpo_trainer import HATRPO as TrainAlgo
            from algorithms.hatrpo_policy import HATRPO_Policy as Policy
        else:
            raise NotImplementedError

        if not self.share_policy:
            self.policy = []
            for agent_id in range(self.num_agents):
                share_observation_space = self.envs.share_observation_space[
                    agent_id] if self.use_centralized_V else self.envs.observation_space[
                        agent_id]
                # policy network
                po = Policy(self.all_args,
                            self.envs.observation_space[agent_id],
                            share_observation_space,
                            self.envs.action_space[agent_id],
                            device=self.device)
                self.policy.append(po)
        else:
            share_observation_space = self.envs.share_observation_space[
                0] if self.use_centralized_V else self.envs.observation_space[0]
            self.policy = Policy(self.all_args,
                                 self.envs.observation_space[0],
                                 share_observation_space,
                                 self.envs.action_space[0],
                                 device=self.device)

        if self.model_dir is not None:
            self.restore()

        self.buffer = []
        for agent_id in range(self.num_agents):
            # buffer
            share_observation_space = self.envs.share_observation_space[
                agent_id] if self.use_centralized_V else self.envs.observation_space[
                    agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)

        if self.share_policy:
            self.trainer = TrainAlgo(self.all_args,
                                     self.policy,
                                     device=self.device)
        else:
            self.trainer = []
            for agent_id in range(self.num_agents):
                # algorithm
                tr = TrainAlgo(self.all_args,
                               self.policy[agent_id],
                               device=self.device)
                self.trainer.append(tr)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            trainer = self.trainer[
                agent_id] if not self.share_policy else self.trainer
            trainer.prep_rollout()
            next_value = trainer.policy.get_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value,
                                                  trainer.value_normalizer)

    def train(self):
        train_infos = []
        # random update order

        action_dim = self.buffer[0].actions.shape[-1]
        factor = np.ones(
            (self.episode_length, self.n_rollout_threads, action_dim),
            dtype=np.float32)

        distillation_agents = []

        for agent_id in torch.randperm(self.num_agents):
            if self.share_policy:
                trainer = self.trainer
            else:
                trainer = self.trainer[agent_id]
            trainer.prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            if self.all_args.algorithm_name == "hatrpo":
                old_actions_logprob, _, _, _, _ = trainer.policy.actor.evaluate_actions(
                    self.buffer[agent_id].obs[:-1].reshape(
                        -1, *self.buffer[agent_id].obs.shape[2:]),
                    self.buffer[agent_id].rnn_states[0:1].reshape(
                        -1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]),
                    self.buffer[agent_id].masks[:-1].reshape(
                        -1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id].active_masks[:-1].reshape(
                        -1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                old_actions_logprob, _, _ = trainer.policy.actor.evaluate_actions(
                    self.buffer[agent_id].obs[:-1].reshape(
                        -1, *self.buffer[agent_id].obs.shape[2:]),
                    self.buffer[agent_id].rnn_states[0:1].reshape(
                        -1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]),
                    self.buffer[agent_id].masks[:-1].reshape(
                        -1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id].active_masks[:-1].reshape(
                        -1, *self.buffer[agent_id].active_masks.shape[2:]))
            train_info = trainer.train(agent_id, self.buffer,
                                       distillation_agents)

            if self.all_args.algorithm_name == "hatrpo":
                new_actions_logprob, _, _, _, _ = trainer.policy.actor.evaluate_actions(
                    self.buffer[agent_id].obs[:-1].reshape(
                        -1, *self.buffer[agent_id].obs.shape[2:]),
                    self.buffer[agent_id].rnn_states[0:1].reshape(
                        -1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]),
                    self.buffer[agent_id].masks[:-1].reshape(
                        -1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id].active_masks[:-1].reshape(
                        -1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                new_actions_logprob, _, new_actor_output = trainer.policy.actor.evaluate_actions(
                    self.buffer[agent_id].obs[:-1].reshape(
                        -1, *self.buffer[agent_id].obs.shape[2:]),
                    self.buffer[agent_id].rnn_states[0:1].reshape(
                        -1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]),
                    self.buffer[agent_id].masks[:-1].reshape(
                        -1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id].active_masks[:-1].reshape(
                        -1, *self.buffer[agent_id].active_masks.shape[2:]))
                if self.all_args.share_policy:
                    new_value, _ = trainer.policy.critic(
                        self.buffer[agent_id].share_obs[:-1].reshape(
                            -1, *self.buffer[agent_id].share_obs.shape[2:]),
                        self.buffer[agent_id].rnn_states_critic[0:1].reshape(
                            -1,
                            *self.buffer[agent_id].rnn_states_critic.shape[2:]
                        ),
                        self.buffer[agent_id].masks[:-1].reshape(
                            -1, *self.buffer[agent_id].masks.shape[2:]),
                    )
                    distillation_agents.append(agent_id)
                    new_value = _t2n(new_value).reshape(
                        self.episode_length, self.n_rollout_threads, 1)
                    new_actor_output = _t2n(new_actor_output).reshape(
                        self.episode_length, self.n_rollout_threads,
                        new_actor_output.shape[-1])
                else:
                    new_value = None
                    new_actor_output = None

            factor = factor * _t2n(
                torch.exp(new_actions_logprob - old_actions_logprob).reshape(
                    self.episode_length, self.n_rollout_threads, action_dim))
            train_infos.append(train_info)
            self.buffer[agent_id].after_update(new_value, new_actor_output)

        return train_infos

    def save(self):
        if self.share_policy:
            torch.save(self.policy.actor.state_dict(),
                       str(self.save_dir) + "/actor.pt")
            torch.save(self.policy.critic.state_dict(),
                       str(self.save_dir) + "/critic.pt")
        else:
            for agent_id in range(self.num_agents):
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(
                    policy_actor.state_dict(),
                    str(self.save_dir) + "/actor_agent" + str(agent_id) +
                    ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(
                    policy_critic.state_dict(),
                    str(self.save_dir) + "/critic_agent" + str(agent_id) +
                    ".pt")

    def restore(self):
        if self.share_policy:
            actor_state_dict = torch.load(
                os.path.join(str(self.model_dir), "actor.pt"))
            self.policy.actor.load_state_dict(actor_state_dict)
            critic_state_dict = torch.load(
                os.path.joint(str(self.model_dir), "critic.pt"))
            self.policy.critic.load_state_dict(critic_state_dict)
        else:
            for agent_id in range(self.num_agents):
                policy_actor_state_dict = torch.load(
                    str(self.model_dir) + '/actor_agent' + str(agent_id) +
                    '.pt')
                self.policy[agent_id].actor.load_state_dict(
                    policy_actor_state_dict)
                policy_critic_state_dict = torch.load(
                    str(self.model_dir) + '/critic_agent' + str(agent_id) +
                    '.pt')
                self.policy[agent_id].critic.load_state_dict(
                    policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.all_args.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v},
                                             total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.all_args.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)},
                                             total_num_steps)
