# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        # added by 51
        self.obs_dim = obs_shape
        self.act_dim = action_shape

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, curr_rewards, rewards, next_obses, not_dones

    def sample_batch_with_history(self, max_hist_len=5):  # added by 51
        """
        :param max_hist_len: the length of experiences before current experience
        :return:
        """
        hist_queue = deque([], maxlen=self.batch_size)
        hist2_queue = deque([], maxlen=self.batch_size)
        hist_act_queue = deque([], maxlen=self.batch_size)
        hist_act2_queue = deque([], maxlen=self.batch_size)

        # History
        obs_buffer = np.zeros([max_hist_len, *self.obs_dim])
        obs2_buffer = np.zeros([max_hist_len, *self.obs_dim])
        act_buffer = np.zeros([max_hist_len, *self.act_dim])
        act2_buffer = np.zeros([max_hist_len, *self.act_dim])
        hist_obs_len = max_hist_len * np.ones(self.batch_size*max_hist_len)
        hist_obs2_len = max_hist_len * np.ones(self.batch_size*max_hist_len)

        # Extract history experiences before sampled index
        max_index = (self.capacity if self.full else self.idx) - max_hist_len
        idxs = np.random.randint(max_hist_len, max_index, size=self.batch_size)
        real_idxs = idxs.copy()
        for i, idx in enumerate(idxs):
            hist_start_id = idx - max_hist_len
            if hist_start_id < 0:
                hist_start_id = 0
            # If exist done before the last experience (not including the done in id), start from the index next to the done.
            len1 = len(np.where(self.not_dones[hist_start_id:idx] == 0)[0])
            if len1 != 0:
                where1 = np.where(self.not_dones[hist_start_id:idx] == 0)[0][-1]  #
                hist_start_id = hist_start_id + where1 + 1
                # update the index ,shift the index to avoid the done flag
                real_idxs[i] = hist_start_id + max_hist_len
                if (hist_start_id+max_hist_len)>self.idx:
                    print("error 51")

            for j in range(max_hist_len):
                obs_buffer[j] = self.obses[hist_start_id + j]
                obs2_buffer[j] = self.next_obses[hist_start_id + j]
                act_buffer[j] = self.actions[hist_start_id + j]
                act2_buffer[j] = self.actions[hist_start_id + 1 + j]

            hist_queue.append(obs_buffer)
            hist2_queue.append(obs2_buffer)
            hist_act_queue.append(act_buffer)
            hist_act2_queue.append(act2_buffer)

        history_obs = np.concatenate(list(hist_queue), axis=0)
        hist_obs = history_obs
        history2_obs = np.concatenate(list(hist_queue), axis=0)
        hist_obs2 = history2_obs
        hist_act = np.concatenate(list(hist_act_queue), axis=0)
        hist_act2 = np.concatenate(list(hist_act_queue), axis=0)

        batch = dict(obs=self.obses[real_idxs],
                     obs2=self.next_obses[real_idxs],
                     act=self.actions[real_idxs],
                     rew=self.rewards[real_idxs],
                     done=self.not_dones[real_idxs],
                     hist_obs=hist_obs,
                     hist_act=hist_act,
                     hist_obs2=hist_obs2,
                     hist_act2=hist_act2,
                     hist_obs_len=hist_obs_len,
                     hist_obs2_len=hist_obs2_len)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            self.idx = end


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
