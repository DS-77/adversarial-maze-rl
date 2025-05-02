"""
This module is used to evaluate all four Maze generation approaches against three metrics: solvability, Triviality, and
difficulty.
"""

import os
import tqdm
import argparse
import torch.cuda
import numpy as np
import matplotlib as plt
from random import choice, sample
from model.PPO_networks import GeneratorPPO
from model.DQN_networks import GeneratorNetwork
from metrics.metrics import triviality, difficulty
from basic_maze_gen import gen_maze_DFS, gen_maze_Prims, maze_viz, solve_maze_gen


class RandomPolicyMaze:
    def __init__(self, maze_size):
        self.maze_size = maze_size

    def generate_maze(self):
        maze = np.zeros((self.maze_size, self.maze_size), dtype=int)
        # num_paths = int(self.maze_size * self.maze_size * 0.4)
        # path_indices = sample(range(self.maze_size * self.maze_size), num_paths)

        for i in range(1, self.maze_size - 1):
            if i % 2 != 0:
                maze[i, 1] = 1
                maze[i, self.maze_size - 2] = 1

        for j in range(1, self.maze_size - 1):
            if j % 2 != 0:
                maze[1, j] = 1
                maze[self.maze_size - 2, j] = 1

            # Randomly carve out paths
        for _ in range(int(self.maze_size * self.maze_size * 0.3)):
            x = choice(range(1, self.maze_size - 1))
            y = choice(range(1, self.maze_size - 1))
            maze[x, y] = 1

        valid_coordinates = []
        for i in range(1, self.maze_size - 1):
            for j in range(1, self.maze_size - 1):
                if maze[i, j] == 1:
                    valid_coordinates.append((i, j))

        start_x, start_y = choice(valid_coordinates)
        goal_x, goal_y = choice(valid_coordinates)

        # ensure start and goal are different
        while (goal_x, goal_y) == (start_x, start_y):
            goal_x, goal_y = choice(valid_coordinates)

        maze[start_x, start_y] = 2
        maze[goal_x, goal_y] = 3

        return maze, (start_x, start_y), (goal_x, goal_y)


def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn_weights", required=True, type=str, help="The path to DQN pre-trained weights.")
    parser.add_argument("--ppo_weights", required=True, type=str, help="The path to PPO pre-trained weights.")
    opts = parser.parse_args()

    # Weights path
    dqn_weights_path = opts.dqn_weights
    ppo_weights_path = opts.ppo_weights

    assert os.path.exists(dqn_weights_path)
    assert os.path.exists(ppo_weights_path)

    # Evaluation parameters
    maze_size = [15, 21, 31]
    iterations = 100

    # Evaluation results
    DFS_solve_steps = []
    DFS_sol_rate = 0
    DFS_diff = []
    DFS_triv = []

    prim_solve_steps = []
    prim_sol_rate = 0
    prim_diff = []
    prim_triv = []

    DQN_solve_steps = []
    DQN_sol_rate = 0
    DQN_diff = []
    DQN_triv = []

    PPO_solve_steps = []
    PPO_sol_rate = 0
    PPO_diff = []
    PPO_triv = []

    random_solve_steps = []
    random_sol_rate = 0
    random_diff = []
    random_triv = []

    # Load in Models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DQN_generator = GeneratorNetwork(128, maze_size[0], replay_capacity=10000).to(device)
    DQN_generator.load_state_dict(torch.load(dqn_weights_path))

    PPO_generator = GeneratorPPO(128, maze_size[0], device)
    state_dict = torch.load(ppo_weights_path, map_location=next(PPO_generator.generator.parameters()).device)
    PPO_generator.generator.load_state_dict(state_dict)
    random_generator = RandomPolicyMaze(maze_size[0])

    for i in tqdm.tqdm(range(iterations)):
        # Generate mazes
        dfs_maze, dfs_start, dfs_goal = gen_maze_DFS(maze_size[0])
        prim_maze, prim_start, prim_goal = gen_maze_Prims(maze_size[0])

        dqn_input_vector = torch.randn(128).to(device)
        dqn_maze, dqn_start, dqn_goal = DQN_generator.generate_maze(dqn_input_vector)

        ppo_input_vector = torch.randn(128).to(device)
        ppo_maze, ppo_start, ppo_goal = PPO_generator.generate_maze(ppo_input_vector)

        random_maze, random_start, random_goal = random_generator.generate_maze()

        # Triviality
        DFS_triv.append(triviality(dfs_maze))
        prim_triv.append(triviality(prim_maze))
        DQN_triv.append(triviality(dfs_maze))
        PPO_triv.append(triviality(ppo_maze))
        random_triv.append(triviality(random_maze))

        # Difficulty
        DFS_diff.append(difficulty(dfs_maze, dfs_goal))
        prim_diff.append(difficulty(prim_maze, prim_goal))
        DQN_diff.append(difficulty(dqn_maze, dqn_goal))
        PPO_diff.append(difficulty(ppo_maze, ppo_goal))
        random_diff.append(difficulty(random_maze, random_goal))

        # Solvability
        dfs_path, dfs_steps = solve_maze_gen(dfs_maze, dfs_start, dfs_goal)
        if dfs_path is not None:
            DFS_sol_rate += 1
            DFS_solve_steps.append(dfs_steps)

        prim_path, prim_steps = solve_maze_gen(prim_maze, prim_start, prim_goal)
        if prim_path is not None:
            prim_sol_rate += 1
            prim_solve_steps.append(prim_steps)

        dqn_path, dqn_steps = solve_maze_gen(dqn_maze, dqn_start, dqn_goal)
        if dqn_path is not None:
            DQN_sol_rate += 1
            DQN_solve_steps.append(dqn_steps)

        ppo_path, ppo_steps = solve_maze_gen(ppo_maze, ppo_start, ppo_goal)
        if ppo_path is not None:
            PPO_sol_rate += 1
            PPO_solve_steps.append(ppo_steps)

        random_path, random_steps = solve_maze_gen(random_maze, random_start, random_goal)
        if random_path is not None:
            random_sol_rate += 1
            random_solve_steps.append(random_steps)

    # print results

    print("Maze Analysis")
    print("=" * 80)

    print("DFS Metrics:")
    print("-" * 80)
    print(f"Average Triviality: {np.mean(DFS_triv):.3f}")
    print(f"Difficulty: {np.mean(DFS_diff):.3f}")
    print(f"Solvability Rate: {DFS_sol_rate / iterations if DFS_sol_rate != 0 else 0:.3f}")
    print(f"Average Solve Steps: {np.mean(DFS_solve_steps):.3f}")

    print()
    print("Prim Metrics:")
    print("-" * 80)
    print(f"Average Triviality: {np.mean(prim_triv):.3f}")
    print(f"Difficulty: {np.mean(prim_diff):.3f}")
    print(f"Solvability Rate: {prim_sol_rate / iterations if prim_sol_rate != 0 else 0:.3f}")
    print(f"Average Solve Steps: {np.mean(prim_solve_steps):.3f}")

    print()
    print("DQN Metrics:")
    print("-" * 80)
    print(f"Average Triviality: {np.mean(DQN_triv):.3f}")
    print(f"Difficulty: {np.mean(DQN_diff):.3f}")
    print(f"Solvability Rate: {DQN_sol_rate / iterations if DQN_sol_rate != 0 else 0:.3f}")
    print(f"Average Solve Steps: {np.mean(DQN_solve_steps):.3f}")

    print()
    print("PPO Metrics:")
    print("-" * 80)
    print(f"Average Triviality: {np.mean(PPO_triv):.3f}")
    print(f"Difficulty: {np.mean(PPO_diff):.3f}")
    print(f"Solvability Rate: {PPO_sol_rate / iterations if PPO_sol_rate != 0 else 0:.3f}")
    print(f"Average Solve Steps: {np.mean(PPO_solve_steps):.3f}")

    print()
    print("Random Policy Metrics:")
    print("-" * 80)
    print(f"Average Triviality: {np.mean(random_triv):.3f}")
    print(f"Difficulty: {np.mean(random_diff):.3f}")
    print(f"Solvability Rate: {random_sol_rate / iterations if random_sol_rate != 0 else 0:.3f}")
    print(f"Average Solve Steps: {np.mean(random_solve_steps):.3f}")

if __name__ == "__main__":

    main()