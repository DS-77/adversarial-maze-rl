"""
This module trains the PPO version of the generator and solver agents for Maze-Gen.

Version: 1.0.2
Edited: 26-04-2025
"""

import os
import torch
import numpy as np
import argparse as ap
from tqdm import tqdm
from datetime import date
from basic_maze_gen import maze_viz
from model.enviornment import Environment
from torch.utils.tensorboard import SummaryWriter
from model.PPO_networks import GeneratorPPO, SolverPPO, Memory


def save_models(generator, solver, path_prefix, iteration):
    torch.save(generator.generator.state_dict(), f"{path_prefix}/generator_ep_{iteration}.pth")
    torch.save(solver.policy.state_dict(), f"{path_prefix}/solver_ep_{iteration}.pth")


def evaluate_solver(env, solver_ppo, episodes=10, max_steps=100):
    success_count = 0
    total_steps = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = solver_ppo.policy_old.act(state, None, training=False)
            _, reward, done, _ = env.step(action)
            next_state = env.get_encoded_state()
            state = next_state
            steps += 1

        if done and reward > 0:
            success_count += 1
        total_steps += steps

    success_rate = success_count / episodes
    avg_steps = total_steps / episodes

    return success_rate, avg_steps


def train_solver(env, solver_ppo, num_episodes, device, max_steps=100):
    solver_memory = Memory()
    avg_reward = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = solver_ppo.select_action(state, solver_memory)
            _, reward, done, _ = env.step(action)
            next_state = env.get_encoded_state()

            # Record step in memory
            solver_memory.rewards.append(reward)
            solver_memory.is_terminals.append(done)

            state = next_state
            episode_reward += reward
            steps += 1

        avg_reward += episode_reward

        # Update PPO policy
        solver_ppo.update(solver_memory, device)
        solver_memory.clear_memory()

    avg_reward = avg_reward / num_episodes
    return avg_reward


def train_generator_with_solver_feedback(generator_ppo, solver_ppo, num_episodes,
                                         num_solver_episodes, maze_size, device, max_steps=100):
    for episode in range(num_episodes):
        # Generate a maze using the generator
        input_vector = torch.randn(128).to(device)
        maze_array, start, goal = generator_ppo.generate_maze(input_vector)
        env = Environment(maze=maze_array)
        env.set_start_goal(start, goal)

        # Evaluate the solver's performance
        success_rate, avg_steps = evaluate_solver(env, solver_ppo, episodes=3, max_steps=max_steps)

        # Calculate reward based on solver performance
        if success_rate == 0:
            reward_signal = 0.1
        elif success_rate == 1.0 and avg_steps < max_steps * 0.5:
            reward_signal = 0.5
        else:
            reward_signal = 1.0

        # Update generator based on reward
        loss = generator_ppo.update(input_vector, reward_signal)

        # Train solver on this maze
        solver_reward = train_solver(env, solver_ppo, num_solver_episodes, device, max_steps)

        print(f"[GEN] Ep {episode}: Loss {loss:.4f}, Reward: {reward_signal:.2f}, "
              f"Solver Success: {success_rate:.2f}, Avg Steps: {avg_steps:.1f}")

    return loss


def main():
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('--gen_episode_num', default=30, type=int, help="Number of generator training episodes.")
    parser.add_argument('--sol_episode_num', default=8, type=int, help="Number of solver training episodes per maze.")
    parser.add_argument('--train_iters', default=15, type=int, help="Number of alternated training rounds.")
    parser.add_argument('--maze_size', default=15, type=int, help="Size of the maze.")
    parser.add_argument('--gen_lr', default=1e-3, type=float, help="Generator learning rate.")
    parser.add_argument('--sol_lr', default=3e-4, type=float, help="Solver learning rate.")
    opts = parser.parse_args()

    # Required Directories
    output_directory_root = "./runs"
    exp_dir = f"{output_directory_root}/exp_{date.today()}_PPO"
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
    base_maze_size = opts.maze_size
    gen_lr = opts.gen_lr
    sol_lr = opts.sol_lr
    train_iters = opts.train_iters

    # Setup
    state_size = 4 * base_maze_size * base_maze_size
    action_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialise models
    generator_ppo = GeneratorPPO(input_size=128, maze_size=base_maze_size, device=device, lr=gen_lr)
    solver_ppo = SolverPPO(state_dim=state_size, action_dim=action_size, device=device, lr=sol_lr)

    writer = SummaryWriter(log_dir=log_path)

    for i in tqdm(range(train_iters)):
        print(f"\n====== Training Iteration {i + 1}/{train_iters} ======")

        adaptive_maze_size = base_maze_size + (i // 5) * 2
        adaptive_maze_size = adaptive_maze_size if adaptive_maze_size % 2 == 1 else adaptive_maze_size + 1
        # adaptive_maze_size = 15

        generator_ppo.maze_size = adaptive_maze_size

        # Train generator using solver feedback
        gen_loss = train_generator_with_solver_feedback(
            generator_ppo=generator_ppo,
            solver_ppo=solver_ppo,
            num_episodes=gen_ep_num,
            num_solver_episodes=sol_ep_num,
            maze_size=adaptive_maze_size,
            device=device
        )

        # Save models
        save_models(generator_ppo, solver_ppo, path_prefix=checkpoint_path, iteration=i + 1)

        # Generate a test maze to evaluate solver
        test_input = torch.randn(128).to(device)
        test_maze, test_start, test_goal = generator_ppo.generate_maze(test_input)
        test_env = Environment(test_maze)
        test_env.set_start_goal(test_start, test_goal)

        # Evaluate solver on test maze
        success_rate, avg_steps = evaluate_solver(test_env, solver_ppo, episodes=10)

        # Log metrics
        writer.add_scalar("Generator/Loss", gen_loss, i + 1)
        writer.add_scalar("Solver/SuccessRate", success_rate, i + 1)
        writer.add_scalar("Solver/AvgSteps", avg_steps, i + 1)
        writer.add_scalar("Training/MazeSize", adaptive_maze_size, i + 1)

        print(f"[EVAL] Test maze: Success rate: {success_rate:.2f}, Avg steps: {avg_steps:.1f}")
        if (i + 1) % 5 == 0:
            # Save maze occasionally
            maze_viz(test_maze, iter=i + 1, path=sample_imgs)

    writer.close()


if __name__ == "__main__":
    main()