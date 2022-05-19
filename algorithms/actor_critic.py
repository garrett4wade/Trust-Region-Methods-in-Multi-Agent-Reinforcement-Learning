import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_obs_space


class Actor(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 obs_space,
                 action_space,
                 device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.args = args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.autoregressive = args.autoregressive
        if self.autoregressive:
            self.act_dim = action_space.n
        self.base = base(args, obs_shape, act_dim=action_space.n if self.autoregressive else None)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size,
                                self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size,
                            self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self,
                obs,  # [T, bs, obs_dim]
                rnn_states,
                masks,
                available_actions=None,
                agent_actions=None,  # [bs, n_agents, 1]
                execution_masks=None,  # [bs, n_agents]
                deterministic=False):
        if self.autoregressive:
            assert agent_actions.shape[-2:] == (self.args.num_agents, 1)
            assert execution_masks.shape[-1] == self.args.num_agents
            assert len(agent_actions.shape) == 3
            assert len(execution_masks.shape) == 2
            agent_actions = agent_actions.squeeze(-1) * execution_masks
            agent_actions = F.one_hot(agent_actions.long(), self.act_dim).float()
            agent_actions = agent_actions.flatten(start_dim=-2, end_dim=-1)
            assert agent_actions.shape[-1] == self.args.num_agents * self.act_dim
        else:
            agent_actions = None

        actor_features = self.base(obs, agent_actions)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states,
                                                  masks)

        actions, action_log_probs = self.act(actor_features, available_actions,
                                             deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self,
                         obs,
                         rnn_states,
                         action,
                         masks,
                         available_actions=None,
                         active_masks=None,
                         agent_actions=None,
                         execution_masks=None):
        if self.autoregressive:
            assert agent_actions.shape[-2:] == (self.args.num_agents, 1)
            assert execution_masks.shape[-1] == self.args.num_agents
            assert len(agent_actions.shape) == 3
            assert len(execution_masks.shape) == 2
            agent_actions = agent_actions.squeeze(-1) * execution_masks
            agent_actions = F.one_hot(agent_actions.long(), self.act_dim).float()
            agent_actions = agent_actions.flatten(start_dim=-2, end_dim=-1)
            assert agent_actions.shape[-1] == self.args.num_agents * self.act_dim
        else:
            agent_actions = None

        actor_features = self.base(obs, agent_actions)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states,
                                                  masks)

        if self.args.algorithm_name == "hatrpo":
            action_log_probs, dist_entropy, action_mu, action_std, all_probs = self.act.evaluate_actions_trpo(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks
                if self._use_policy_active_masks else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy, actor_output = self.act.evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks
                if self._use_policy_active_masks else None)

            return action_log_probs, dist_entropy, actor_output


class Critic(nn.Module):
    """
    Critic network class for HAPPO. Outputs value function predictions given centralized input (HAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size,
                                self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states,
                                                   masks)
        values = self.v_out(critic_features)

        return values, rnn_states
