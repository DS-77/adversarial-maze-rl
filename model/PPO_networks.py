"""
This module implements the Generator and Solver for Maze-Gen in the PPO architecture.

Version: 1.0.7
Edited: 27-04-2025
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def act(self, state, memory=None, training=True):
        state = torch.FloatTensor(state).to(self.actor[0].weight.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        if training:
            action = dist.sample()
            action_logprob = dist.log_prob(action)

            if memory is not None:
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(action_logprob)

            return action.item()
        else:
            # just take the best action
            return torch.argmax(action_probs).item()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class GeneratorPPO:
    def __init__(self, input_size, maze_size, device, lr=0.002, betas=(0.9, 0.999), gamma=0.99, n_epoches=4, eps_clip=0.2):
        self.maze_size = maze_size
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.n_epoches = n_epoches

        self.generator = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, maze_size * maze_size),
            nn.Tanh()
        )

        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.generator.to(device)

    def generate_maze(self, input_vector):
        flat_array = self.generator(input_vector).detach().cpu().numpy()
        maze = flat_array.reshape((self.maze_size, self.maze_size))
        maze = (maze + 1) / 2
        maze = np.round(maze).astype(int)

        # Set start and goal points randomly
        from random import choice
        start_x = choice(range(1, self.maze_size - 1, 2))
        start_y = choice(range(1, self.maze_size - 1, 2))
        goal_x = choice(range(1, self.maze_size - 1, 2))
        goal_y = choice(range(1, self.maze_size - 1, 2))

        maze[start_x, start_y] = 2
        maze[goal_x, goal_y] = 3

        return maze, (start_x, start_y), (goal_x, goal_y)

    def update(self, input_vector, reward_signal):
        # Forward pass
        flat_array = self.generator(input_vector)

        # negative reward signal * log prob
        loss = -reward_signal * torch.log(torch.clamp(torch.abs(flat_array), min=1e-5)).mean()

        # print(loss)
        # exit()

        # Update generator
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class SolverPPO:
    def __init__(self, state_dim, action_dim, device, lr=0.0003, betas=(0.9, 0.999), gamma=0.99,
                 n_epoches=4, eps_clip=0.2, hidden_dim=64):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.n_epoches = n_epoches

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy.to(device)
        self.policy_old.to(device)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        return self.policy_old.act(state, memory)

    def update(self, memory, device):
        # Monte Carlo estimate of rewards
        rewards = []

        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):

            # print(reward)

            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalising the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_states = torch.cat(memory.states).to(device).detach()
        old_actions = torch.cat(memory.actions).to(device).detach()
        old_logprobs = torch.cat(memory.logprobs).to(device).detach()

        # Optimise policy for N epochs
        for _ in range(self.n_epoches):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr_1 = ratios * advantages
            surr_2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr_1, surr_2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss.mean().item()