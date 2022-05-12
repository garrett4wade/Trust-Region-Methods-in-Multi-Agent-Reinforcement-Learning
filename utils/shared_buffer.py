import torch
import numpy as np
from collections import defaultdict
from utils.util import check, get_shape_from_obs_space, get_shape_from_act_space


class SharedReplayBuffer(object):

    def __init__(self, args, num_agents, obs_space, share_obs_space,
                 act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.rnn_hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        self.num_agents = num_agents

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.device = device = torch.device("cuda:0")

        self.share_obs = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents,
             *share_obs_shape),
            dtype=torch.float32,
            device=device)
        self.obs = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents,
             *obs_shape),
            dtype=torch.float32,
            device=device)

        self.rnn_states = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents,
             self.recurrent_N, self.rnn_hidden_size),
            dtype=torch.float32,
            device=device)
        self.rnn_states_critic = torch.zeros_like(self.rnn_states)

        self.value_preds = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=torch.float32,
            device=device)
        self.returns = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=torch.float32,
            device=device)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = torch.ones(
                (self.episode_length + 1, self.n_rollout_threads, num_agents,
                 act_space.n),
                dtype=torch.float32,
                device=device)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = torch.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents,
             act_shape),
            dtype=torch.float32,
            device=device)
        self.action_log_probs = torch.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents,
             act_shape),
            dtype=torch.float32,
            device=device)
        self.rewards = torch.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1),
            dtype=torch.float32,
            device=device)

        self.masks = torch.ones(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=torch.float32,
            device=device)
        self.bad_masks = torch.ones_like(self.masks)
        self.active_masks = torch.ones_like(self.masks)

        self.factor = torch.zeros_like(self.action_log_probs)

        self.autoregressive = args.autoregressive
        if args.autoregressive:
            self.agent_actions = torch.zeros(
                (self.episode_length, self.n_rollout_threads, num_agents,
                 num_agents, 1),
                dtype=torch.int32,
                device=device)
            self.execution_masks = torch.zeros(
                (self.episode_length, self.n_rollout_threads, num_agents,
                 num_agents),
                dtype=torch.float32,
                device=device)

        self.step = 0

    def update_factor(self, agent_idx, factor):
        self.factor[:, :, agent_idx] = factor

    def insert(
        self,
        share_obs,
        obs,
        rnn_states,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks=None,
        active_masks=None,
        available_actions=None,
        agent_actions=None,
        execution_masks=None,
    ):
        self.share_obs[self.step + 1] = share_obs
        self.obs[self.step + 1] = obs
        self.rnn_states[self.step + 1] = rnn_states
        self.rnn_states_critic[self.step + 1] = rnn_states_critic
        self.actions[self.step] = actions
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.rewards[self.step] = rewards
        self.masks[self.step + 1] = masks
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions

        if self.autoregressive:
            self.agent_actions[self.step] = agent_actions
            self.execution_masks[self.step] = execution_masks

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1]
        self.obs[0] = self.obs[-1]
        self.rnn_states[0] = self.rnn_states[-1]
        self.rnn_states_critic[0] = self.rnn_states_critic[-1]
        self.masks[0] = self.masks[-1]
        self.bad_masks[0] = self.bad_masks[-1]
        self.active_masks[0] = self.active_masks[-1]
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1]

    def compute_returns(self, next_value, value_normalizers=None):
        """
        use proper time limits, the difference of use or not is whether use bad_mask
        """
        self.value_preds[-1] = next_value
        if self._use_popart or self._use_valuenorm:
            if isinstance(value_normalizers, list):
                value_preds = []
                for agent_id, value_normalizer in enumerate(value_normalizers):
                    value_preds.append(
                        value_normalizer.denormalize(
                            self.value_preds[:, :, agent_id]))
                value_preds = torch.stack(value_preds, -2)
            else:
                value_preds = value_normalizers.denormalize(self.value_preds)
        else:
            value_preds = self.value_preds
        if self._use_proper_time_limits:
            if self._use_gae:
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    delta = self.rewards[step] + self.gamma * value_preds[
                        step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[
                        step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + value_preds[step]
            else:
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma *
                        self.masks[step + 1] +
                        self.rewards[step]) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]) * value_preds[step]
        else:
            if self._use_gae:
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    delta = self.rewards[step] + self.gamma * value_preds[
                        step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[
                        step + 1] * gae
                    self.returns[step] = gae + value_preds[step]
            else:
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[
                        step + 1] * self.gamma * self.masks[
                            step + 1] + self.rewards[step]

    def feed_forward_generator(
        self,
        agent_idx,
        advantages,
        num_mini_batch,
        values_after_update=None,
        actor_outputs_after_update=None,
    ):
        mini_batch_size = None
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size)
        sampler = [
            rand[i * mini_batch_size:(i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        share_obs = self.share_obs[:-1, :, agent_idx].flatten(end_dim=1)
        obs = self.obs[:-1, :, agent_idx].flatten(end_dim=1)
        rnn_states = self.rnn_states[:-1, :, agent_idx].flatten(end_dim=1)
        rnn_states_critic = self.rnn_states_critic[:-1, :, agent_idx].flatten(
            end_dim=1)
        actions = self.actions[:, :, agent_idx].flatten(end_dim=1)
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1, :,
                                                       agent_idx].flatten(
                                                           end_dim=1)
        value_preds = self.value_preds[:-1, :, agent_idx].flatten(end_dim=1)
        returns = self.returns[:-1, :, agent_idx].flatten(end_dim=1)
        masks = self.masks[:-1, :, agent_idx].flatten(end_dim=1)
        active_masks = self.active_masks[:-1, :, agent_idx].flatten(end_dim=1)
        action_log_probs = self.action_log_probs[:, :,
                                                 agent_idx].flatten(end_dim=1)
        if self.factor is not None:
            # factor = self.factor.reshape(-1,1)
            factor = self.factor[:, :, agent_idx].flatten(end_dim=1)
        if self.autoregressive:
            agent_actions = self.agent_actions[:, :, agent_idx].flatten(
                end_dim=1)  # [B, n_agents, 1]
            execution_masks = self.execution_masks[:, :, agent_idx].flatten(
                end_dim=1)  # [B, n_agents]
        if values_after_update is not None:
            new_values = []
            for value_after_update in values_after_update:
                new_values.append(value_after_update.flatten(end_dim=1))
        else:
            new_values = None
        if actor_outputs_after_update is not None:
            new_actor_outputs = []
            for actor_output_after_update in actor_outputs_after_update:
                new_actor_outputs.append(
                    actor_output_after_update.flatten(end_dim=1))
        else:
            new_actor_outputs = None
        advantages = advantages.flatten(end_dim=1)
        assert advantages.shape[-1] == 1 and len(
            advantages.shape) == 2, advatanges.shape

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]
            if new_values is not None:
                new_value_batch = [
                    new_value[indices] for new_value in new_values
                ]
            else:
                new_value_batch = None
            if new_actor_outputs is not None:
                new_actor_output_batch = [
                    new_actor_output[indices]
                    for new_actor_output in new_actor_outputs
                ]
            else:
                new_actor_output_batch = None

            if self.autoregressive:
                agent_actions_batch = agent_actions[indices]
                execution_masks_batch = execution_masks[indices]
            else:
                agent_actions_batch = execution_masks_batch = None

            if self.factor is None:
                yield (share_obs_batch, obs_batch, rnn_states_batch,
                       rnn_states_critic_batch, actions_batch,
                       value_preds_batch, return_batch, masks_batch,
                       active_masks_batch, old_action_log_probs_batch,
                       adv_targ, available_actions_batch, None,
                       new_value_batch, new_actor_output_batch,
                       agent_actions_batch, execution_masks_batch)
            else:
                factor_batch = factor[indices]
                yield (share_obs_batch, obs_batch, rnn_states_batch,
                       rnn_states_critic_batch, actions_batch,
                       value_preds_batch, return_batch, masks_batch,
                       active_masks_batch, old_action_log_probs_batch,
                       adv_targ, available_actions_batch, factor_batch,
                       new_value_batch, new_actor_output_batch,
                       agent_actions_batch, execution_masks_batch)

    def recurrent_generator(self,
                            agent_idx,
                            advantages,
                            num_mini_batch,
                            data_chunk_length,
                            values_after_update=None,
                            actor_outputs_after_update=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        num_chunks = episode_length // data_chunk_length
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        def _cast(x):
            x = x.reshape(num_chunks, x.shape[0] // num_chunks, *x.shape[1:])
            x = x.transpose(1, 0)
            return x.flatten(start_dim=1, end_dim=2)

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, episode_length,
                                             data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")

        rand = torch.randperm(data_chunks)
        sampler = [
            rand[i * mini_batch_size:(i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        share_obs = _cast(self.share_obs[:-1, :, agent_idx])
        obs = _cast(self.obs[:-1, :, agent_idx])

        actions = _cast(self.actions[:, :, agent_idx])
        action_log_probs = _cast(self.action_log_probs[:, :, agent_idx])
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1, :, agent_idx])
        returns = _cast(self.returns[:-1, :, agent_idx])
        masks = _cast(self.masks[:-1, :, agent_idx])
        active_masks = _cast(self.active_masks[:-1, :, agent_idx])
        if self.factor is not None:
            factor = _cast(self.factor[:, :, agent_idx])
        if self.autoregressive:
            agent_actions = _cast(
                self.agent_actions[:, :, agent_idx])  # [T, bs, n_agents, 1]
            execution_masks = _cast(
                self.execution_masks[:, :, agent_idx])  # [T, bs, n_agents]
        if values_after_update is not None:
            new_values = [
                _cast(value_after_update)
                for value_after_update in values_after_update
            ]
        if actor_outputs_after_update is not None:
            new_actor_outputs = [
                _cast(actor_output_after_update)
                for actor_output_after_update in actor_outputs_after_update
            ]
        # rnn_states = _cast(self.rnn_states[:-1, :, agent_idx])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1, :, agent_idx])
        rnn_states = _cast(self.rnn_states[:-1, :, agent_idx])
        rnn_states_critic = _cast(self.rnn_states_critic[:-1, :, agent_idx])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1, :,
                                                             agent_idx])

        for indices in sampler:
            # These are all from_numpys of size (N, L, Dim)
            share_obs_batch = share_obs[:, indices]
            obs_batch = obs[:, indices]

            actions_batch = actions[:, indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[:, indices]
            if self.factor is not None:
                factor_batch = factor[:, indices]
            if self.autoregressive:
                agent_actions_batch = agent_actions[:, indices]
                execution_masks_batch = execution_masks[:, indices]
            if values_after_update is not None:
                new_value_batch = [
                    new_value[:, indices] for new_value in new_values
                ]
            if actor_outputs_after_update is not None:
                new_actor_output_batch = [
                    new_actor_output[:, indices]
                    for new_actor_output in new_actor_outputs
                ]
            value_preds_batch = value_preds[:, indices]
            return_batch = returns[:, indices]
            masks_batch = masks[:, indices]
            active_masks_batch = active_masks[:, indices]
            old_action_log_probs_batch = action_log_probs[:, indices]
            adv_targ = advantages[:, indices]

            # States is just a (N, -1) from_numpy
            rnn_states_batch = rnn_states[0, indices]
            rnn_states_critic_batch = rnn_states_critic[0, indices]

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = share_obs_batch.flatten(end_dim=1)
            obs_batch = obs_batch.flatten(end_dim=1)
            actions_batch = actions_batch.flatten(end_dim=1)
            if self.available_actions is not None:
                available_actions_batch = available_actions_batch.flatten(
                    end_dim=1)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = factor_batch.flatten(end_dim=1)
            if self.autoregressive:
                agent_actions_batch = agent_actions_batch.flatten(end_dim=1)
                execution_masks_batch = execution_masks_batch.flatten(
                    end_dim=1)
            else:
                agent_actions_batch = execution_masks_batch = None
            if values_after_update is not None:
                new_value_batch = [
                    x.flatten(end_dim=1) for x in new_value_batch
                ]
            else:
                new_value_batch = None
            if actor_outputs_after_update is not None:
                new_actor_output_batch = [
                    x.flatten(end_dim=1) for x in new_actor_output_batch
                ]
            else:
                new_actor_output_batch = None
            value_preds_batch = value_preds_batch.flatten(end_dim=1)
            return_batch = return_batch.flatten(end_dim=1)
            masks_batch = masks_batch.flatten(end_dim=1)
            active_masks_batch = active_masks_batch.flatten(end_dim=1)
            old_action_log_probs_batch = old_action_log_probs_batch.flatten(
                end_dim=1)
            adv_targ = adv_targ.flatten(end_dim=1)
            if self.factor is not None:
                yield (share_obs_batch, obs_batch, rnn_states_batch,
                       rnn_states_critic_batch, actions_batch,
                       value_preds_batch, return_batch, masks_batch,
                       active_masks_batch, old_action_log_probs_batch,
                       adv_targ, available_actions_batch, factor_batch,
                       new_value_batch, new_actor_output_batch,
                       agent_actions_batch, execution_masks_batch)
            else:
                yield (share_obs_batch, obs_batch, rnn_states_batch,
                       rnn_states_critic_batch, actions_batch,
                       value_preds_batch, return_batch, masks_batch,
                       active_masks_batch, old_action_log_probs_batch,
                       adv_targ, available_actions_batch, None,
                       new_value_batch, new_actor_output_batch,
                       agent_actions_batch, execution_masks_batch)
