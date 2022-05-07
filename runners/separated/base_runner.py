import time
import os
import numpy as np
from itertools import chain
import torch
import wandb
from tensorboardX import SummaryWriter
# from utils.separated_buffer import SeparatedReplayBuffer
from utils.shared_buffer import SharedReplayBuffer
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
        self.autoregressive = self.all_args.autoregressive
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
                    agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
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

        self.buffer = SharedReplayBuffer(self.all_args, self.num_agents, self.envs.observation_space[0],
                                         share_observation_space, self.envs.action_space[0])
        # for agent_id in range(self.num_agents):
        #     # buffer
        #     share_observation_space = self.envs.share_observation_space[
        #         agent_id] if self.use_centralized_V else self.envs.observation_space[
        #             agent_id]
        #     bu = SeparatedReplayBuffer(self.all_args,
        #                                self.envs.observation_space[agent_id],
        #                                share_observation_space,
        #                                self.envs.action_space[agent_id])
        #     self.buffer.append(bu)

        if self.share_policy:
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        else:
            self.trainer = []
            for agent_id in range(self.num_agents):
                # algorithm
                tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
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
        if not self.share_policy:
            nex_values = []
            for agent_id in range(self.num_agents):
                trainer = self.trainer[agent_id]
                trainer.prep_rollout()
                next_value = trainer.policy.get_values(self.buffer.share_obs[-1, :, agent_id],
                                                       self.buffer.rnn_states_critic[-1, :, agent_id],
                                                       self.buffer.masks[-1, :, agent_id])
                assert next_value.shape == (self.n_rollout_threads, 1)
                nex_values.append(next_value)
            next_value = torch.stack(nex_values, dim=1)
            self.buffer.compute_returns(next_value, [trainer.value_normalizer for trainer in self.trainer])
        else:
            next_value = self.trainer.policy.get_values(self.buffer.share_obs[-1].flatten(end_dim=1),
                                                        self.buffer.rnn_states_critic[-1].flatten(end_dim=1),
                                                        self.buffer.masks[-1].flatten(end_dim=1))
            next_value = next_value.view(self.n_rollout_threads, self.num_agents, 1)
            self.buffer.compute_returns(next_value, self.trainer.value_normalizer)

    def train(self):
        for _ in range(self.all_args.sample_reuse):
            train_infos = []

            log_factor = torch.zeros_like(self.buffer.factor[:, :, 0])
            distill_value_targets = []
            distill_actor_output_targets = []

            # random update order
            for agent_id in torch.randperm(self.num_agents):
                if self.share_policy:
                    trainer = self.trainer
                else:
                    trainer = self.trainer[agent_id]
                trainer.prep_training()
                self.buffer.update_factor(agent_id, log_factor.exp())
                available_actions = None if self.buffer.available_actions is None \
                    else self.buffer.available_actions[:-1, :, agent_id].flatten(end_dim=1)

                with torch.no_grad():
                    if self.all_args.algorithm_name == "hatrpo":
                        old_actions_logprob, _, _, _, _ = trainer.policy.actor.evaluate_actions(
                            self.buffer.obs[:-1, :, agent_id].flatten(end_dim=1),
                            self.buffer.rnn_states[0:1, :, agent_id].flatten(end_dim=1),
                            self.buffer.actions[:, :, agent_id].flatten(end_dim=1),
                            self.buffer.masks[:-1, :, agent_id].flatten(end_dim=1),
                            available_actions,
                            self.buffer.active_masks[:-1, :, agent_id].flatten(end_dim=1),
                        )
                    else:
                        old_actions_logprob, _, _ = trainer.policy.actor.evaluate_actions(
                            self.buffer.obs[:-1, :, agent_id].flatten(end_dim=1),
                            self.buffer.rnn_states[0:1, :, agent_id].flatten(end_dim=1),
                            self.buffer.actions[:, :, agent_id].flatten(end_dim=1),
                            self.buffer.masks[:-1, :, agent_id].flatten(end_dim=1),
                            available_actions,
                            self.buffer.active_masks[:-1, :, agent_id].flatten(end_dim=1),
                            agent_actions=self.buffer.agent_actions[:, :, agent_id].flatten(
                                end_dim=1) if self.all_args.autoregressive else None,
                            execution_masks=self.buffer.execution_masks[:, :, agent_id].flatten(
                                end_dim=1) if self.all_args.autoregressive else None,
                        )
                train_info = trainer.train(agent_id, self.buffer, distill_value_targets, distill_actor_output_targets)

                with torch.no_grad():
                    if self.all_args.algorithm_name == "hatrpo":
                        new_actions_logprob, _, _, _, _ = trainer.policy.actor.evaluate_actions(
                            self.buffer.obs[:-1, :, agent_id].flatten(end_dim=1),
                            self.buffer.rnn_states[0:1, :, agent_id].flatten(end_dim=1),
                            self.buffer.actions[:, :, agent_id].flatten(end_dim=1),
                            self.buffer.masks[:-1, :, agent_id].flatten(end_dim=1),
                            available_actions,
                            self.buffer.active_masks[:-1, :, agent_id].flatten(end_dim=1),
                        )
                    else:
                        new_actions_logprob, _, new_actor_output = trainer.policy.actor.evaluate_actions(
                            self.buffer.obs[:-1, :, agent_id].flatten(end_dim=1),
                            self.buffer.rnn_states[0:1, :, agent_id].flatten(end_dim=1),
                            self.buffer.actions[:, :, agent_id].flatten(end_dim=1),
                            self.buffer.masks[:-1, :, agent_id].flatten(end_dim=1),
                            available_actions,
                            self.buffer.active_masks[:-1, :, agent_id].flatten(end_dim=1),
                            agent_actions=self.buffer.agent_actions[:, :, agent_id].flatten(
                                end_dim=1) if self.all_args.autoregressive else None,
                            execution_masks=self.buffer.execution_masks[:, :, agent_id].flatten(
                                end_dim=1) if self.all_args.autoregressive else None,
                        )
                        if self.all_args.share_policy:
                            new_value, _ = trainer.policy.critic(
                                self.buffer.share_obs[:-1, :, agent_id].flatten(end_dim=1),
                                self.buffer.rnn_states_critic[0:1, :, agent_id].flatten(end_dim=1),
                                self.buffer.masks[:-1, :, agent_id].flatten(end_dim=1),
                            )
                            new_value = new_value.view(self.episode_length, self.n_rollout_threads, 1)
                            new_actor_output = new_actor_output.view(self.episode_length, self.n_rollout_threads,
                                                                     new_actor_output.shape[-1])
                            distill_value_targets.append(new_value)
                            distill_actor_output_targets.append(new_actor_output)
                        else:
                            new_value = None
                            new_actor_output = None

                    log_factor += (new_actions_logprob - old_actions_logprob).view(self.episode_length,
                                                                                   self.n_rollout_threads, -1)
                train_infos.append(train_info)
        self.buffer.after_update()
        return train_infos

    def save(self):
        if self.share_policy:
            torch.save(self.policy.actor.state_dict(), str(self.save_dir) + "/actor.pt")
            torch.save(self.policy.critic.state_dict(), str(self.save_dir) + "/critic.pt")
        else:
            for agent_id in range(self.num_agents):
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        if self.share_policy:
            actor_state_dict = torch.load(os.path.join(str(self.model_dir), "actor.pt"))
            self.policy.actor.load_state_dict(actor_state_dict)
            critic_state_dict = torch.load(os.path.joint(str(self.model_dir), "critic.pt"))
            self.policy.critic.load_state_dict(critic_state_dict)
        else:
            for agent_id in range(self.num_agents):
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.all_args.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.all_args.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
