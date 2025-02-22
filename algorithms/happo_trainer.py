import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt
from algorithms.utils.util import check


class HAPPO():
    """
    Trainer class for HAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (HAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, policy, device=torch.device("cpu")):

        self.share_policy = args.share_policy
        self.actor_distill_coef = args.actor_distill_coef
        self.critic_distill_coef = args.critic_distill_coef
        self.num_agents = args.num_agents
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.no_factor = args.no_factor

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        for x in sample:
            if x is not None and not isinstance(x, list):
                assert isinstance(x, torch.Tensor) and (x.device == torch.device('cuda:0'))
            elif isinstance(x, list) and len(x) > 0:
                for y in x:
                    assert isinstance(y, torch.Tensor) and (y.device == torch.device('cuda:0'))
        (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch,
         return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch,
         factor_batch, distill_value_targets, distill_actor_output_targets, agent_actions_batch,
         execution_masks_batch) = sample

        if self.no_factor:
            factor_batch = 1

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, actor_output = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            agent_actions=agent_actions_batch,
            execution_masks=execution_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True) *
                                  active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss - dist_entropy * self.entropy_coef

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            policy_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        if self.critic_distill_coef > 0 and len(distill_value_targets) > 0:
            distill_v_loss = 0
            self.policy.distill_critic_optimizer.zero_grad()
            v = self.policy.get_values(share_obs_batch, rnn_states_critic_batch, masks_batch)
            for d_vt in distill_value_targets:
                assert self.share_policy
                assert d_vt.shape[-1] == 1
                distill_v_loss += self.critic_distill_coef * ((d_vt - v)**2).mean()
            distill_v_loss.backward()
            self.policy.distill_critic_optimizer.step()

        if self.actor_distill_coef > 0 and len(distill_actor_output_targets) > 0:
            distill_pi_loss = 0
            self.policy.distill_actor_optimizer.zero_grad()
            _, _, a = self.policy.actor.evaluate_actions(obs_batch,
                                                         rnn_states_batch,
                                                         actions_batch,
                                                         masks_batch,
                                                         available_actions_batch,
                                                         active_masks_batch,
                                                         agent_actions=agent_actions_batch,
                                                         execution_masks=execution_masks_batch)
            for d_at in distill_actor_output_targets:
                assert self.share_policy
                assert d_at.shape[-1] == actor_output.shape[-1]
                distill_pi_loss += self.actor_distill_coef * ((d_at - a)**2).mean()
            distill_pi_loss.backward()
            self.policy.distill_actor_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def _generate_adv(self, agent_id, buffer):
        if self._use_popart:
            advantages = buffer.returns[:-1, :, agent_id] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1, :, agent_id])
        else:
            advantages = buffer.returns[:-1, :, agent_id] - buffer.value_preds[:-1, :, agent_id]

        x = advantages.clone()
        dim = tuple(range(len(x.shape)))
        mask = buffer.active_masks[:-1, :, agent_id]
        x = x * mask
        factor = mask.sum(dim=dim, keepdim=True)
        x_sum = x.sum(dim=dim, keepdim=True)
        x_sum_sq = x.square().sum(dim=dim, keepdim=True)
        mean = x_sum / factor
        meansq = x_sum_sq / factor
        var = meansq - mean**2
        # var *= factor / (factor - 1)
        return (x - mean) / (var.sqrt() + 1e-5)

    def train(self, agent_id, buffer, distill_value_targets, distill_actor_output_targets, update_actor=True):
        advantages = self._generate_adv(agent_id, buffer)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            assert not self._use_naive_recurrent
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    agent_id,
                    advantages,
                    self.num_mini_batch,
                    self.data_chunk_length,
                    values_after_update=distill_value_targets,
                    actor_outputs_after_update=distill_actor_output_targets,
                )
            else:
                data_generator = buffer.feed_forward_generator(
                    agent_id,
                    advantages,
                    self.num_mini_batch,
                    values_after_update=distill_value_targets,
                    actor_outputs_after_update=distill_actor_output_targets,
                )

            for sample in data_generator:
                (value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm,
                 imp_weights) = self.ppo_update(sample, update_actor=update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
