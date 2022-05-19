import time
import numpy as np
from functools import reduce
import torch
import wandb
import uuid
from collections import defaultdict
from runners.separated.base_runner import Runner
from utils.timing import Timing
from utils.util import save_frames_as_gif


def _t2n(x):
    return x.detach().cpu().numpy()


class Ned90Runner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(Ned90Runner, self).__init__(config)

    def run(self):

        def to_tensor(x):
            return torch.from_numpy(x).to(self.device)

        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps
                       ) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            assert not self.use_linear_lr_decay
            timing = Timing()

            train_env_info = defaultdict(list)

            for step in range(self.episode_length):
                # Sample actions
                with timing.add_time("inference"):
                    (values, actions, action_log_probs, rnn_states,
                     rnn_states_critic, agent_actions,
                     execution_masks) = self.collect(step)

                with timing.add_time("envstep"):
                    # Obser reward and next obs
                    (obs, share_obs, rewards, dones, infos,
                     available_actions) = self.envs.step(actions.cpu().numpy())
                    (obs, share_obs, rewards,
                     dones) = map(to_tensor, (obs, share_obs, rewards, dones))

                    for (done, info) in zip(dones, infos):
                        if done.all():
                            for k, v in info[0].items():
                                train_env_info[k].append(v)

                with timing.add_time("buffer"):
                    data = obs, share_obs, rewards, dones, infos, None, \
                        values, actions, action_log_probs, \
                        rnn_states, rnn_states_critic, agent_actions, execution_masks

                    # insert data into buffer
                    self.insert(data)

            # compute return and update network
            with timing.add_time("gae"):
                self.compute()
            with timing.add_time("train"):
                train_infos = self.train()

            print(timing)
            # post process
            total_num_steps = (
                episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.experiment_name, episode, episodes,
                            total_num_steps, self.num_env_steps,
                            int(total_num_steps / (end - start))))

                assert self.env_name == "ned90"
                if len(train_env_info) > 0:
                    train_env_info = {
                        k: np.mean(v)
                        for k, v in train_env_info.items()
                    }
                    print("Average training info is {}".format(train_env_info))
                    if self.all_args.use_wandb:
                        wandb.log(train_env_info, step=total_num_steps)
                    else:
                        self.writter.add_scalars("train_env_info",
                                                 train_env_info,
                                                 total_num_steps)

                self.log_train(train_infos, total_num_steps)

            # eval is disabled
    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.share_obs[0] = torch.from_numpy(share_obs).to(self.device)
        self.buffer.obs[0] = torch.from_numpy(obs).to(self.device)
        if not all([a is None for a in available_actions]):
            self.buffer.available_actions[0] = torch.from_numpy(
                available_actions).to(self.device)

    @torch.no_grad()
    def collect(self, step):
        agent_actions = execution_masks = None
        assert not self.autoregressive and self.all_args.num_agents == 1
        assert self.share_policy
        trainer = self.trainer
        trainer.prep_rollout()
        (value, action, action_log_prob, rnn_state,
         rnn_state_critic) = trainer.policy.get_actions(
             self.buffer.share_obs[step].flatten(end_dim=1),
             self.buffer.obs[step].flatten(end_dim=1),
             self.buffer.rnn_states[step].flatten(end_dim=1),
             self.buffer.rnn_states_critic[step].flatten(end_dim=1),
             self.buffer.masks[step].flatten(end_dim=1),
             None,  # available_actions is None
         )

        def _cast(x):
            return x.view(self.n_rollout_threads, self.num_agents,
                          *x.shape[1:])

        values = _cast(value)
        actions = _cast(action)
        action_log_probs = _cast(action_log_prob)
        rnn_states = _cast(rnn_state)
        rnn_states_critic = _cast(rnn_state_critic)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, agent_actions, execution_masks

    def insert(self, data):
        # [n_rollout_threads, num_agents, *]
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, agent_actions, execution_masks = data

        dones = dones.unsqueeze(-1)
        dones_env = dones.all(1, keepdim=True).float()
        masks = 1 - dones_env

        active_masks = 1 - dones
        active_masks = active_masks * (1 - dones_env) + dones_env

        bad_masks = torch.tensor(
            [[[0.0] if info[agent_id]['bad_transition'] else [1.0]
              for agent_id in range(self.num_agents)] for info in infos],
            dtype=torch.float32,
            device=self.device)

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            bad_masks,
            active_masks,
            available_actions,
            agent_actions=agent_actions,
            execution_masks=execution_masks,
        )

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id][
                "average_step_rewards"] = self.buffer.rewards[:, :,
                                                              agent_id].mean(
                                                              ).item()
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.all_args.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v},
                                             total_num_steps)
