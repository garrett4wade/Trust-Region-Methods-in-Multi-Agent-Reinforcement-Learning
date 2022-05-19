from typing import List, Dict
import dataclasses
import time
import zmq
import gym
import logging
import pickle
import numpy as np
import random

logger = logging.getLogger("env-ned90")


def random_env_id():
    return str(hash(random.random()))


class MockNED90Environment:

    def __init__(self, **kwargs):
        pass

    @property
    def action_space(self):
        return [gym.spaces.MultiDiscrete([9, 16])]

    @property
    def observation_space(self):
        return [
            gym.spaces.Box(
                low=np.array([-np.inf for _ in range(712)]),
                high=np.array([np.inf for _ in range(712)]),
            )
        ]

    @property
    def share_observation_space(self):
        return self.observation_space

    @property
    def n_agents(self):
        return 1

    @property
    def num_agents(self):
        return 1

    def reset(self):
        obs = np.random.randn(1, 712)
        return obs, obs, None

    def step(self, actions):
        assert actions.shape == (1, 2)
        assert 0 <= actions[0, 0].item() < 9, actions[0]
        assert 0 <= actions[0, 1].item() < 16, actions[1]
        obs = np.random.randn(1, 712)
        r = np.random.randn(1, 1)
        done = np.zeros(
            1, dtype=np.uint8) if np.random.rand() < 0.05 else np.ones(
                1, dtype=np.uint8)
        info = dict(episode_length=np.random.randn(),
                    episode_return=np.random.randn(),
                    episode_win=np.random.randn(),
                    episode_trim=np.random.randn(),
                    perries=np.random.randn(),
                    perriables=np.random.randn(),
                    bad_transition=False)
        return obs, obs, r, done, [info], None


class NED90Environment(MockNED90Environment):
    # for single agent
    def __init__(self,
                 broker_addr,
                 inbound_port,
                 outbound_port,
                 game_config: dict,
                 action_rewards: dict,
                 env_id: int,
                 is_opponent=False):
        self.broker_addr = broker_addr
        self.inbound_port = inbound_port
        self.outbound_port = outbound_port
        self.inbound_socket = None
        self.outbound_socket = None
        self.win_record = 0
        self.total_record = 0
        # self.env_id = random_env_id() # generate a unique env_id, port hard encoding
        self.env_id = env_id
        # even id fight odd id

        # action reward only apply on the second head of action
        # format: {action number: reward per action}
        self.config = game_config
        self.action_rewards = action_rewards

        if "win_reward" in self.config:
            self.win_reward = self.config["win_reward"]
        else:
            self.win_reward = 100
        self.buffer = None

        self.is_opponent = is_opponent

    # dynamic sockets
    def init_sockets(self):
        self.close_sockets()
        self.context = zmq.Context()
        # self.init_socket = self.context.socket(zmq.REQ)
        self.inbound_socket = self.context.socket(zmq.SUB)
        self.outbound_socket = self.context.socket(zmq.PUSH)

        self.inbound_socket.connect(
            f"tcp://{self.broker_addr}:{self.inbound_port}")
        self.outbound_socket.connect(
            f"tcp://{self.broker_addr}:{self.outbound_port}")

        print(f"connected to broker, env id {self.env_id}.", flush=True)
        self.inbound_socket.setsockopt(zmq.RCVTIMEO, 3000)
        self.inbound_socket.setsockopt(zmq.SUBSCRIBE,
                                       pickle.dumps(self.env_id))

    def close_sockets(self):
        if self.inbound_socket and self.outbound_socket:
            self.inbound_socket.close()
            self.outbound_socket.close()

    def reset(self):
        # retry until reset
        while True:
            res = self.__reset(self.config)
            if res:
                return res

    def __reset(self, config=None):
        self.init_sockets()
        print(f"reset, env id {self.env_id}")
        if self.is_opponent:
            config = None
        reset_msg = [self.env_id, "reset", config]
        self.outbound_socket.send_multipart(
            [pickle.dumps(x) for x in reset_msg])
        try:
            _, obs, reward, done = [
                pickle.loads(x) for x in self.inbound_socket.recv_multipart()
            ]
            self.buffer = (obs, reward)
        except zmq.Again:
            return False
        self.__step_count = np.ones(1, dtype=np.int32)
        self.__episode_return = np.zeros(1, dtype=np.float32)
        self.__perries = np.zeros(1, dtype=np.int32)
        self.__perriables = np.zeros(1, dtype=np.int32)
        print(f"reset success, {self.env_id}", flush=True)
        obs = np.array([obs], dtype=np.float32)
        return obs, obs, None

    def step(self, actions: np.ndarray):
        assert actions.shape[0] == 1, actions.shape
        trim = False

        def win(reward):
            if reward > self.win_reward / 2:
                # if reward > float(self.win_reward/400):
                return np.ones(1, dtype=np.float32)
            else:
                return np.zeros(1, dtype=np.float32)

        for i in range(self.agent_count):
            if actions[i] is not None:
                action = actions[i].x[np.newaxis, :]
            else:
                action = np.array([[0, 0]])

            action_msg = [self.env_id, "action", action]
            self.outbound_socket.send_multipart(
                [pickle.dumps(x) for x in action_msg])
        # set a timeout to this recv, if timeout, kill and reset ("or done?")
        try:
            _, obs, reward, done = [
                pickle.loads(x) for x in self.inbound_socket.recv_multipart()
            ]
            self.buffer = (obs, reward)
        except zmq.Again:
            if self.buffer:
                obs, reward = self.buffer
                done = True
            else:
                raise Exception("something is very wrong")

        if self.__step_count[0] > 5000:
            print(f"maximum step count 5000, reset")
            done = True

        # adhoc action reward
        if action[0][1] in self.action_rewards.keys():
            reward += self.action_rewards[action[0][1]]

        # action statistics
        if action[0][1] == 5:
            self.__perries += 1

        if action[0][1] == 3 or action[0][1] == 4:
            self.__perriables += 1
        ##

        if done and abs(reward) < self.win_reward / 2:
            trim = True

        self.__step_count += 1
        self.__episode_return += reward

        info = dict(episode_length=self.__step_count.item(),
                    episode_return=self.__episode_return.item(),
                    episode_win=win(reward).item(),
                    episode_trim=bool(trim),
                    perries=float(self.__perries / self.__step_count),
                    perriables=float(self.__perriables / self.__step_count),
                    bad_transition=False)

        if done:
            self.buffer = None
            if win(reward)[0]:
                self.win_record += 1
            self.total_record += 1
            print(
                f"episode done, step count {self.__step_count}, env_id {self.env_id}, last reward {reward}",
                flush=True)
            print(
                f"win/total = {self.win_record}/{self.total_record}, winrate {float(self.win_record/self.total_record)}, \
                    this win {win(reward)}",
                flush=True)
        if self.__step_count[0] % 100 == 0:
            print(f"step finished, step count {self.__step_count}", flush=True)

        obs = np.array([obs], dtype=np.float32)
        return (obs, obs, np.array([[float(reward)]], dtype=np.float32),
                np.array([bool(done)], dtype=np.uint8), [info], None)

    def render(self) -> None:
        raise NotImplementedError(
            'Rendering the NED90 environment is by default disabled.')

    def __del__(self):
        pass
