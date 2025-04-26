"""
This module implements the Generator and Solver for Maze-Gen in the DQN architecture.

Author: Deja S.
Version: 1.0.4
Edited: 25-04-2025
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from random import choice, sample


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append([state, action, reward, next_state, done])
        else:
            self.memory[self.position] = [state, action, reward, next_state, done]
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return self.memory
        return sample(self.memory, batch_size)

    def get_length(self):
        return len(self.memory)


class GeneratorNetwork(nn.Module):
    def __init__(self, input_size, maze_size, replay_capacity=10000):
        super(GeneratorNetwork, self).__init__()
        self.maze_size = maze_size
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, maze_size * maze_size)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

    def generate_maze(self, input_vector):
        flat_array = self.forward(input_vector).detach().cpu().numpy()
        maze = flat_array.reshape((self.maze_size, self.maze_size))
        maze = (maze + 1) / 2
        maze = np.round(maze).astype(int)

        # Set start and goal points randomly
        start_x = choice(range(1, self.maze_size - 1, 2))
        start_y = choice(range(1, self.maze_size - 1, 2))
        goal_x = choice(range(1, self.maze_size - 1, 2))
        goal_y = choice(range(1, self.maze_size - 1, 2))

        maze[start_x, start_y] = 2
        maze[goal_x, goal_y] = 3

        return maze, (start_x, start_y), (goal_x, goal_y)

    def add_to_replay(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)


class SolverNetwork(nn.Module):
    def __init__(self, state_size, action_size, replay_capacity=10000):
        super(SolverNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def get_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(4)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            q_values = self.forward(state_tensor)
            return torch.argmax(q_values).item()

    def add_to_replay(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)