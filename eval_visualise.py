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
from model.PPO_networks import GeneratorPPO
from model.DQN_networks import GeneratorNetwork
from metrics.metrics import triviality, difficulty
from basic_maze_gen import gen_maze_DFS, gen_maze_Prims, maze_viz, solve_maze_gen


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

    # Load in Models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DQN_generator = GeneratorNetwork(128, maze_size[0], replay_capacity=10000).to(device)
    DQN_generator.load_state_dict(torch.load(dqn_weights_path))

    PPO_generator = GeneratorPPO(128, maze_size[0], device)
    state_dict = torch.load(ppo_weights_path, map_location=next(PPO_generator.generator.parameters()).device)
    PPO_generator.generator.load_state_dict(state_dict)

    for i in tqdm.tqdm(range(iterations)):
        # Generate mazes
        dfs_maze, dfs_start, dfs_goal = gen_maze_DFS(maze_size[0])
        prim_maze, prim_start, prim_goal = gen_maze_Prims(maze_size[0])

        dqn_input_vector = torch.randn(128).to(device)
        dqn_maze, dqn_start, dqn_goal = DQN_generator.generate_maze(dqn_input_vector)

        ppo_input_vector = torch.randn(128).to(device)
        ppo_maze, ppo_start, ppo_goal = PPO_generator.generate_maze(ppo_input_vector)

        # Triviality
        DFS_triv.append(triviality(dfs_maze))
        prim_triv.append(triviality(prim_maze))
        DQN_triv.append(triviality(dfs_maze))
        PPO_triv.append(triviality(ppo_maze))

        # Difficulty
        DFS_diff.append(difficulty(dfs_maze, dfs_goal))
        prim_diff.append(difficulty(prim_maze, prim_goal))
        DQN_diff.append(difficulty(dqn_maze, dqn_goal))
        PPO_diff.append(difficulty(ppo_maze, ppo_goal))

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

    # print results

    print("Maze Analysis")
    print("=" * 80)

    print("DFS Metrics:")
    print("-" * 80)
    print(f"Average Triviality: {np.mean(DFS_triv):.3f}")
    print(f"Difficulty: {np.mean(DFS_diff):.3f}")
    print(f"Solvabilty Rate: {DFS_sol_rate / iterations if DFS_sol_rate != 0 else 0:.3f}")
    print(f"Average Solve Steps: {np.mean(DFS_solve_steps):.3f}")

    print("Prim Metrics:")
    print("-" * 80)
    print(f"Average Triviality: {np.mean(prim_triv):.3f}")
    print(f"Difficulty: {np.mean(prim_diff):.3f}")
    print(f"Solvabilty Rate: {prim_sol_rate / iterations if prim_sol_rate != 0 else 0:.3f}")
    print(f"Average Solve Steps: {np.mean(prim_solve_steps):.3f}")

    print("DQN Metrics:")
    print("-" * 80)
    print(f"Average Triviality: {np.mean(DQN_triv):.3f}")
    print(f"Difficulty: {np.mean(DQN_diff):.3f}")
    print(f"Solvabilty Rate: {DQN_sol_rate / iterations if DQN_sol_rate != 0 else 0:.3f}")
    print(f"Average Solve Steps: {np.mean(DQN_solve_steps):.3f}")

    print("PPO Metrics:")
    print("-" * 80)
    print(f"Average Triviality: {np.mean(PPO_triv):.3f}")
    print(f"Difficulty: {np.mean(PPO_diff):.3f}")
    print(f"Solvabilty Rate: {PPO_sol_rate / iterations if PPO_sol_rate != 0 else 0:.3f}")
    print(f"Average Solve Steps: {np.mean(PPO_solve_steps):.3f}")

if __name__ == "__main__":

    main()