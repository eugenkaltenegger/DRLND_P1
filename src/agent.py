import torch
import numpy
import random
from torch import optim
from torch.nn import functional

from src.model import QNetwork
from src.buffer import Buffer


class Agent:

    def __init__(self, device, state_size, action_size, layers, buffer_size, batch_size, batch_frequency, epsilon_start,
                 epsilon_end, epsilon_factor, seed=42, learning_rate=0.0005, gamma=0.99, tau=0.001):
        self.device = device

        self.learning_rate = learning_rate

        self.state_size = state_size
        self.action_size = action_size
        self.layers = layers

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_factor = epsilon_factor
        self.epsilon = self.epsilon_start

        self.seed = seed
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.batch_frequency = batch_frequency
        self.current_step = 0
        self.gamma = gamma  # discount factor
        self.tau = tau

        # Q-Network
        self.q_network_local = QNetwork(self.state_size, self.action_size, self.layers).to(self.device)
        self.q_network_global = QNetwork(self.state_size, self.action_size, self.layers).to(self.device)

        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.buffer = Buffer(self.device, self.buffer_size, self.batch_size, seed=42)

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

        self.current_step = (self.current_step + 1) % self.batch_frequency
        if self.current_step == 0:
            batch = self.buffer.batch()
            if batch is not None:
                self.learn(batch)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        # TODO: check if required
        # self.q_network_local.train()

        if random.random() > self.epsilon:
            action = numpy.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(numpy.arange(self.action_size))

        return action

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_factor * self.epsilon)  # update epsilon

    def learn(self, experiences, gamma=None):
        if gamma is None:
            gamma = self.gamma

        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.q_network_global(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.q_network_local(states).gather(1, actions)

        # Compute loss
        loss = functional.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_network_local, self.q_network_global)

    def soft_update(self, local_model, target_model, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        checkpoint = {'input_size': self.state_size,
                      'output_size': self.action_size,
                      'hidden_layers': self.layers,
                      'state_dict': self.q_network_local.state_dict()}
        torch.save(checkpoint, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.q_network_local = QNetwork(checkpoint['input_size'],
                                        checkpoint['output_size'],
                                        checkpoint['hidden_layers']).to(self.device)
        self.q_network_local.load_state_dict(checkpoint['state_dict'])
