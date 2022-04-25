import time
import numpy as np
from functools import reduce
import torch
import wandb
from runners.separated.base_runner import Runner
from utils.timing import Timing


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):

        def to_tensor(x):
            return torch.from_numpy(np.array(x)).to(dtype=torch.float32,
                                                    device=self.device)

        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps
                       ) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            assert not self.use_linear_lr_decay
            timing = Timing()

            for step in range(self.episode_length):
                # Sample actions
                with timing.add_time("inference"):
                    (values, actions, action_log_probs, rnn_states,
                     rnn_states_critic) = self.collect(step)

                with timing.add_time("envstep"):
                    # Obser reward and next obs
                    (obs, share_obs, rewards, dones, infos,
                     available_actions) = self.envs.step(actions.cpu().numpy())
                    (obs, share_obs, rewards, dones, available_actions) = map(
                        to_tensor,
                        (obs, share_obs, rewards, dones, available_actions))

                with timing.add_time("buffer"):
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                        values, actions, action_log_probs, \
                        rnn_states, rnn_states_critic

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
                    "\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.all_args.map_name, self.algorithm_name,
                            self.experiment_name, episode, episodes,
                            total_num_steps, self.num_env_steps,
                            int(total_num_steps / (end - start))))

                assert self.env_name == "StarCraft2"
                battles_won = []
                battles_game = []
                incre_battles_won = []
                incre_battles_game = []

                for i, info in enumerate(infos):
                    if 'battles_won' in info[0].keys():
                        battles_won.append(info[0]['battles_won'])
                        incre_battles_won.append(info[0]['battles_won'] -
                                                 last_battles_won[i])
                    if 'battles_game' in info[0].keys():
                        battles_game.append(info[0]['battles_game'])
                        incre_battles_game.append(info[0]['battles_game'] -
                                                  last_battles_game[i])

                incre_win_rate = np.sum(incre_battles_won) / np.sum(
                    incre_battles_game) if np.sum(
                        incre_battles_game) > 0 else 0.0
                print("incre win rate is {}.".format(incre_win_rate))
                if self.all_args.use_wandb:
                    wandb.log({"incre_win_rate": incre_win_rate},
                              step=total_num_steps)
                else:
                    self.writter.add_scalars(
                        "incre_win_rate", {"incre_win_rate": incre_win_rate},
                        total_num_steps)

                last_battles_game = battles_game
                last_battles_won = battles_won
                # modified

                for agent_id in range(self.num_agents):
                    train_infos[agent_id][
                        'dead_ratio'] = 1 - self.buffer.active_masks[:, :, agent_id].sum(
                        ) / (self.num_agents * reduce(
                            lambda x, y: x * y,
                            list(self.buffer.active_masks[:, :,
                                                          agent_id].shape)))
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.share_obs[0] = torch.from_numpy(share_obs).to(self.device)
        self.buffer.obs[0] = torch.from_numpy(obs).to(self.device)
        self.buffer.available_actions[0] = torch.from_numpy(
            available_actions).to(self.device)

    @torch.no_grad()
    def collect(self, step):
        if not self.share_policy:
            value_collector = []
            action_collector = []
            action_log_prob_collector = []
            rnn_state_collector = []
            rnn_state_critic_collector = []
            for agent_id in range(self.num_agents):
                trainer = self.trainer[agent_id]
                trainer.prep_rollout()
                (value, action, action_log_prob, rnn_state,
                 rnn_state_critic) = trainer.policy.get_actions(
                     self.buffer.share_obs[step, :, agent_id],
                     self.buffer.obs[step, :, agent_id],
                     self.buffer.rnn_states[step, :, agent_id],
                     self.buffer.rnn_states_critic[step, :, agent_id],
                     self.buffer.masks[step, :, agent_id],
                     self.buffer.available_actions[step, :, agent_id],
                 )
                value_collector.append(value)
                action_collector.append(action)
                action_log_prob_collector.append(action_log_prob)
                rnn_state_collector.append(rnn_state)
                rnn_state_critic_collector.append(rnn_state_critic)
            # [self.envs, agents, dim]
            values = torch.stack(value_collector, 1)
            actions = torch.stack(action_collector, 1)
            action_log_probs = torch.stack(action_log_prob_collector, 1)
            rnn_states = torch.stack(rnn_state_collector, 1)
            rnn_states_critic = torch.stack(rnn_state_critic_collector, 1)
        else:
            trainer = self.trainer
            trainer.prep_rollout()
            (value, action, action_log_prob, rnn_state,
             rnn_state_critic) = trainer.policy.get_actions(
                 self.buffer.share_obs[step].flatten(end_dim=1),
                 self.buffer.obs[step].flatten(end_dim=1),
                 self.buffer.rnn_states[step].flatten(end_dim=1),
                 self.buffer.rnn_states_critic[step].flatten(end_dim=1),
                 self.buffer.masks[step].flatten(end_dim=1),
                 self.buffer.available_actions[step].flatten(end_dim=1),
             )

            def _cast(x):
                return x.view(self.n_rollout_threads, self.num_agents,
                              *x.shape[1:])

            values = _cast(value)
            actions = _cast(action)
            action_log_probs = _cast(action_log_prob)
            rnn_states = _cast(rnn_state)
            rnn_states_critic = _cast(rnn_state_critic)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        # [n_rollout_threads, num_agents, *]
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones = dones.unsqueeze(-1)
        dones_env = dones.all(1, keepdim=True)
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

    @torch.no_grad()
    def eval(self, total_num_steps):

        def to_tensor(x):
            return torch.from_numpy(np.array(x)).to(dtype=torch.float32,
                                                    device=self.device)

        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        running_rewards = torch.zeros(self.n_eval_rollout_threads,
                                      device=self.device,
                                      dtype=torch.float32)

        (eval_obs, eval_share_obs,
         eval_available_actions) = map(to_tensor, self.eval_envs.reset())

        eval_rnn_states = torch.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N,
             self.hidden_size),
            dtype=torch.float32,
            device=self.device)
        eval_masks = torch.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1),
            dtype=torch.float32,
            device=self.device)
        while True:
            if not self.share_policy:
                eval_actions_collector = []
                for agent_id in range(self.num_agents):
                    trainer = self.trainer[agent_id]
                    trainer.prep_rollout()
                    (eval_actions, temp_rnn_state) = trainer.policy.act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id],
                        deterministic=True)
                    eval_rnn_states[:, agent_id] = temp_rnn_state
                    eval_actions_collector.append(eval_actions)
                eval_actions = torch.stack(eval_actions_collector, 1)
            else:
                trainer = self.trainer
                eval_actions, eval_rnn_states = trainer.policy.act(
                    eval_obs.flatten(end_dim=1),
                    eval_rnn_states.flatten(end_dim=1),
                    eval_masks.flatten(end_dim=1),
                    eval_available_actions.flatten(end_dim=1),
                    deterministic=True)
                eval_actions = eval_actions.view(self.n_eval_rollout_threads,
                                                 self.num_agents,
                                                 *eval_actions.shape[1:])
                eval_rnn_states = eval_rnn_states.view(
                    self.n_eval_rollout_threads, self.num_agents,
                    *eval_rnn_states.shape[1:])

            # Obser reward and next obs
            (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos,
             eval_available_actions) = self.eval_envs.step(
                 eval_actions.cpu().numpy())
            (eval_obs, eval_share_obs,
             eval_rewards, eval_dones, eval_available_actions) = map(
                 to_tensor, (eval_obs, eval_share_obs, eval_rewards,
                             eval_dones, eval_available_actions))
            running_rewards += eval_rewards.squeeze(-1).mean(-1)

            eval_dones_env = eval_dones.all(1)

            eval_masks[:] = 1 - eval_dones.unsqueeze(-1).all(1, keepdim=True)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(running_rewards[eval_i].item())
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards,
                                                dtype=np.float32)
                eval_env_infos = {
                    'eval_average_episode_rewards': eval_episode_rewards
                }
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won / eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.all_args.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate},
                              step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate",
                                             {"eval_win_rate": eval_win_rate},
                                             total_num_steps)
                break
