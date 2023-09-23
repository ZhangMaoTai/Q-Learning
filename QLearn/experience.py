import random
import numpy as np
import torch
from collections import deque
from utils.const import *


class BasicBuffer:
    def __init__(self,
                 max_size,
                 batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

        if len(self.buffer) >= self.max_size:
            return self.seperate()
        else:
            return None

    def seperate(self):
        buffer_num = len(self.buffer)
        data = []

        for i in range(0, buffer_num, self.batch_size):
            data.append(
                self.collate(self.buffer[i: i + self.batch_size])
            )

        self.free()
        return data

    def collate(self, experiences):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for exp in experiences:
            state, action, reward, next_state, done = exp
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (
            torch.cat(state_batch),
            torch.tensor(action_batch),
            torch.tensor(reward_batch),
            torch.cat(next_state_batch),
            done_batch
        )

    def __len__(self):
        return len(self.buffer)

    def free(self):
        self.buffer = []


# class PrioritizedBuffer:
#
#     def __init__(self, max_size, alpha=0.6, beta=0.4):
#         self.sum_tree = SumTree(max_size)
#         self.alpha = alpha
#         self.beta = beta
#         self.current_length = 0
#
#     def push(self, state, action, reward, next_state, done):
#         priority = 1.0 if self.current_length is 0 else self.sum_tree.tree.max()
#         self.current_length = self.current_length + 1
#         #priority = td_error ** self.alpha
#         experience = (state, action, np.array([reward]), next_state, done)
#         self.sum_tree.add(priority, experience)
#
#     def sample(self, batch_size):
#         batch_idx, batch, IS_weights = [], [], []
#         segment = self.sum_tree.total() / batch_size
#         p_sum = self.sum_tree.tree[0]
#
#         for i in range(batch_size):
#             a = segment * i
#             b = segment * (i + 1)
#
#             s = random.uniform(a, b)
#             idx, p, data = self.sum_tree.get(s)
#
#             batch_idx.append(idx)
#             batch.append(data)
#             prob = p / p_sum
#             IS_weight = (self.sum_tree.total() * prob) ** (-self.beta)
#             IS_weights.append(IS_weight)
#
#         state_batch = []
#         action_batch = []
#         reward_batch = []
#         next_state_batch = []
#         done_batch = []
#
#         for transition in batch:
#             state, action, reward, next_state, done = transition
#             state_batch.append(state)
#             action_batch.append(action)
#             reward_batch.append(reward)
#             next_state_batch.append(next_state)
#             done_batch.append(done)
#
#         return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), batch_idx, IS_weights
#
#     def update_priority(self, idx, td_error):
#         priority = td_error ** self.alpha
#         self.sum_tree.update(idx, priority)
#
#     def __len__(self):
#         return self.current_length

