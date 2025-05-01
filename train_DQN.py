"""
This module trains the DQN version of the generator and solver agents for Maze-Gen.

Author: Deja S.
Version: 1.0.3
Edited: 25-04-2025
"""

import os
import tqdm
import torch
import numpy as np
import argparse as ap
import torch.nn as nn
from datetime import date
import torch.optim as optim
import matplotlib.pyplot as plt
from model.enviornment import Environment
from torch.utils.tensorboard import SummaryWriter
from basic_maze_gen import solve_maze_gen, maze_viz
from model.DQN_networks import GeneratorNetwork, SolverNetwork, ReplayBuffer

generator_iteration_rewards = []
solver_iteration_rewards = []

def save_models(generator, solver, path_prefix, iteration):
    torch.save(generator.state_dict(), f"{path_prefix}/generator_ep{iteration}.pth")
    torch.save(solver.state_dict(), f"{path_prefix}/solver_ep{iteration}.pth")


def train_generator(generator, solver_network, generator_optimizer, memory, num_solver_episodes,
                    num_episodes, maze_size, device, max_steps_per_episode=100):
    episode_rewards = []

    for episode in range(num_episodes):
        input_vector = torch.randn(128).to(device)
        maze_array, start, goal = generator.generate_maze(input_vector)
        env = Environment(maze=maze_array)
        env.set_start_goal(start, goal)

        # Check if the maze is solvable
        if not env.is_solvable():
            reward_signal = -2.0
            print(f"[GEN] Ep {episode}: Unsolvable maze generated. Penalising generator.")
        else:
            total_solver_steps = 0
            success = False

            for _ in range(num_solver_episodes):
                state = env.reset()
                done = False
                steps = 0

                while not done and steps < max_steps_per_episode:
                    action = solver_network.get_action(state, epsilon=0.2)
                    _, reward, done, _ = env.step(action)
                    next_state = env.get_encoded_state()
                    memory.add(state, action, reward, next_state, done)
                    state = next_state
                    steps += 1

                total_solver_steps += steps
                if done and reward > 0:
                    success = True

            # If solver fails, give some reward
            if not success:
                reward_signal = 1.0
            else:
                reward_signal = 3 + total_solver_steps / (num_solver_episodes * max_steps_per_episode)

        # Generator update step
        logits = generator.forward(input_vector)
        log_probs = torch.log(torch.clamp(torch.abs(logits), min=1e-5))
        generator_loss = -reward_signal * log_probs.mean()

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        print(f"[GEN] Ep {episode}: Loss {generator_loss.item():.4f}, Reward Signal: {reward_signal:.2f}")
        episode_rewards.append(reward_signal)

    return episode_rewards



def train_solver(env, solver_network, memory, num_episodes, optimiser, device,
                 epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.995,
                 discount_factor=0.95, batch_size=32, max_steps=100):

    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(num_episodes):
        env.reset()
        state = env.get_encoded_state()
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = solver_network.get_action(state, epsilon)
            _, reward, done, _ = env.step(action)
            next_state = env.get_encoded_state()

            memory.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

            if memory.get_length() >= batch_size:
                batch = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = np.array(states)
                next_states = np.array(next_states)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.tensor(rewards).float().to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.tensor(dones).float().to(device)

                q_values = solver_network(states)
                next_q_values = solver_network(next_states)
                max_next_q = torch.max(next_q_values, dim=1)[0]
                targets = rewards + (1 - dones) * discount_factor * max_next_q

                q_selected = q_values[torch.arange(len(actions)), actions]
                loss = nn.MSELoss()(q_selected, targets)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"[SOL] Ep {episode}: Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Steps: {step_count}")
        episode_rewards.append(total_reward)

    return episode_rewards



def main():
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('--gen_episode_num', default=50, type=int, help="Number of generator training episodes.")
    parser.add_argument('--sol_episode_num', default=10, type=int, help="Number of solver training episodes.")
    parser.add_argument('--train_iters', default=20, type=int, help="Number of alternated training rounds.")
    parser.add_argument('--replay_size', default=10000, type=int, help="Replay buffer size.")
    parser.add_argument('--mini_batch_size', default=32, type=int, help="Mini-batch size for solver.")
    parser.add_argument('--maze_size', default=15, type=int, help="Size of the maze.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    opts = parser.parse_args()

    # Required Directories
    output_directory_root = "./runs"
    exp_dir = f"{output_directory_root}/exp_{date.today()}_DQN"
    checkpoint_path = f"{exp_dir}/weights"
    log_path = f"{exp_dir}/logs"
    sample_imgs = f"{exp_dir}/maze_images"

    if not os.path.exists(output_directory_root):
        os.mkdir(output_directory_root)

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    if not os.path.exists(sample_imgs):
        os.mkdir(sample_imgs)

    # Hyperparameters
    gen_ep_num = opts.gen_episode_num
    sol_ep_num = opts.sol_episode_num
    replay_buffer_size = opts.replay_size
    batch_size = opts.mini_batch_size
    base_maze_size = opts.maze_size
    lr = opts.lr
    train_iters = opts.train_iters

    # Setup
    state_size = 4 * base_maze_size * base_maze_size
    action_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Models
    maze_generator = GeneratorNetwork(128, base_maze_size, replay_capacity=replay_buffer_size).to(device)
    maze_solver = SolverNetwork(state_size, action_size, replay_capacity=replay_buffer_size).to(device)

    gen_optimiser = optim.Adam(maze_generator.parameters(), lr=lr)
    sol_optimiser = optim.Adam(maze_solver.parameters(), lr=lr)

    memory = ReplayBuffer(capacity=replay_buffer_size)

    print(f"Starting self-play training with {train_iters} iterations...\n")

    writer = SummaryWriter(log_dir=log_path)

    for i in tqdm.tqdm(range(train_iters)):
        print(f"\n====== Training Iteration {i + 1}/{train_iters} ======")

        # Adapt the maze size
        # adaptive_maze_size = base_maze_size + (i // 5) * 2
        # adaptive_maze_size = adaptive_maze_size if adaptive_maze_size % 2 == 1 else adaptive_maze_size + 1
        adaptive_maze_size = base_maze_size

        # Generator Training
        print(f"\n====== Training Generator ======")
        generator_rewards = train_generator(
            generator=maze_generator,
            solver_network=maze_solver,
            generator_optimizer=gen_optimiser,
            memory=memory,
            num_solver_episodes=sol_ep_num,
            num_episodes=gen_ep_num,
            maze_size=adaptive_maze_size,
            device=device
        )

        avg_gen_reward = np.mean(generator_rewards[-gen_ep_num:])
        generator_iteration_rewards.append(avg_gen_reward)

        if (i + 1) % 5 == 0:
            # Plotting the episode rewards
            plt.figure(figsize=(12, 6))
            plt.plot(generator_rewards, label="Generator Rewards per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Solver Rewards vs. Episodes")
            plt.legend()
            solver_plot_path = f"{log_path}/generator_rewards_{i + 1}.png"
            plt.savefig(solver_plot_path)
            plt.close()

        # # Generate a maze to test the solver on (old way)
        # print(f"\n====== Testing Solver ======")
        # test_input = torch.randn(128).to(device)
        # test_maze_array, start, goal = maze_generator.generate_maze(test_input)
        # test_env = Environment(test_maze_array)
        # test_env.set_start_goal(start, goal)
        #
        # # Solver Test
        # success_count = 0
        # total_steps = 0
        # reward = 0
        # for _ in tqdm.tqdm(range(sol_ep_num)):
        #     test_env.reset()
        #     state = test_env.get_encoded_state()
        #     done = False
        #     steps = 0
        #     while not done and steps < 100:
        #         action = maze_solver.get_action(state, epsilon=0.1)
        #         _, reward, done, _ = test_env.step(action)
        #         next_state = test_env.get_encoded_state()
        #         state = next_state
        #         steps += 1
        #     if done and reward > 0:
        #         success_count += 1
        #     total_steps += steps
        #
        # solver_success_rate = success_count / sol_ep_num
        # avg_steps = total_steps / sol_ep_num

        print(f"\n====== Testing Solver ======")
        test_input = torch.randn(128).to(device)
        test_maze_array, start, goal = maze_generator.generate_maze(test_input)
        test_env = Environment(test_maze_array)
        test_env.set_start_goal(start, goal)

        # Test solvability -- DFS Solver
        path, solve_steps = solve_maze_gen(test_maze_array, start, goal)

        if (i + 1) % 5 == 0:
            maze_viz(test_maze_array, iter=i+1, path=sample_imgs)

        if test_env.is_solvable():
            print("Generated maze not solvable, skipping test.")
            continue
        else:
            success_count = 0
            total_steps = 0
            reward = 0
            max_steps = 2 * (test_env.maze.shape[1] + test_env.maze.shape[0])

            for _ in tqdm.tqdm(range(sol_ep_num)):
                test_env.reset()
                state = test_env.get_encoded_state()
                done = False
                steps = 0

                while not done and steps < max_steps:
                    action = maze_solver.get_action(state, epsilon=0.01)
                    _, reward, done, _ = test_env.step(action)
                    next_state = test_env.get_encoded_state()
                    state = next_state
                    steps += 1

                if done and reward > 0:
                    success_count += 1
                total_steps += steps

            solver_success_rate = success_count / sol_ep_num
            avg_steps = total_steps / sol_ep_num

            # Logging results
            writer.add_scalar("Generator/RewardSignal", gen_ep_num, i + 1)
            writer.add_scalar("Solver/SuccessRate", solver_success_rate, i + 1)
            writer.add_scalar("Solver/AvgSteps", avg_steps, i + 1)

        # Solver Training
        print(f"\n====== Training Solver ======")
        solver_rewards = train_solver(
            env=test_env,
            solver_network=maze_solver,
            memory=memory,
            num_episodes=sol_ep_num,
            optimiser=sol_optimiser,
            device=device,
            batch_size=batch_size
        )

        avg_sol_reward = np.mean(solver_rewards)
        solver_iteration_rewards.append(avg_sol_reward)

        if (i + 1) % 5 == 0:
            # Plotting the episode rewards
            plt.figure(figsize=(12, 6))
            plt.plot(solver_rewards, label="Solver Rewards per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Solver Rewards vs. Episodes")
            plt.legend()
            solver_plot_path = f"{log_path}/solver_rewards_{i + 1}.png"
            plt.savefig(solver_plot_path)
            plt.close()

        # Save models
        if (i + 1) % 5 == 0:
            save_models(maze_generator, maze_solver, path_prefix=checkpoint_path, iteration=i + 1)

    writer.close()

    # Plot the rewards vs episode for both the generator and the solver
    plt.figure(figsize=(12, 6))
    plt.plot(generator_iteration_rewards, label="Generator Avg Reward per Iteration")
    plt.plot(solver_iteration_rewards, label="Solver Avg Reward per Iteration")
    plt.xlabel("Training Iteration")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards per Iteration")
    plt.legend()
    iteration_plot_path = f"{log_path}/average_rewards_per_iteration.png"
    plt.savefig(iteration_plot_path)
    plt.close()

if __name__ == "__main__":
    main()