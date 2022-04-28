import random

from collections import deque, namedtuple

import numpy
import torch


class Buffer:

    def __init__(self, device, buffer_size, batch_size, seed):
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed

        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        experience_to_add = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience_to_add)

    def batch(self):
        if len(self.memory) < self.batch_size:
            return None

        experiences = random.sample(self.memory, k=self.batch_size)

        states_list = [e.state for e in experiences if e is not None]
        states = torch.from_numpy(numpy.vstack(states_list)).float().to(self.device)

        actions_list = [e.action for e in experiences if e is not None]
        actions = torch.from_numpy(numpy.vstack(actions_list)).long().to(self.device)

        rewards_list = [e.reward for e in experiences if e is not None]
        rewards = torch.from_numpy(numpy.vstack(rewards_list)).float().to(self.device)

        next_states_list = [e.next_state for e in experiences if e is not None]
        next_states = torch.from_numpy(numpy.vstack(next_states_list)).float().to(self.device)

        dones_list = [e.done for e in experiences if e is not None]
        dones = torch.from_numpy(numpy.vstack(dones_list).astype(numpy.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones
