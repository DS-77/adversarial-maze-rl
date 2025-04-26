"""
This module is the baseline experiment used to establish the constraints in which the RL adversarial models will build
their policy.

Author: Deja S.
Created: 31-03-2025
Edited: 02-04-2025
Version: 1.0.0
"""
import numpy as np
from metrics.metrics import triviality, difficulty
from basic_maze_gen import gen_maze_DFS, gen_maze_Prims, solve_maze_gen


if __name__ == "__main__":

    # DFS: 15
    dfs_steps_avg_fifteen = []
    dfs_trivial_avg_fifteen = []
    dfs_dif_avg_fifteen = []

    for i in range(100):
        maze, start, end = gen_maze_DFS(15)
        path, steps = solve_maze_gen(maze, start, end)

        temp_triv = triviality(maze)
        dif = difficulty(maze, end)

        dfs_steps_avg_fifteen.append(steps)
        dfs_trivial_avg_fifteen.append(temp_triv)
        dfs_dif_avg_fifteen.append(dif)

     # DFS: 21
    dfs_steps_avg_twenty_one = []
    dfs_trivial_avg_twenty_one = []
    dfs_dif_avg_twenty_one = []

    for i in range(100):
        maze, start, end = gen_maze_DFS(21)
        path, steps = solve_maze_gen(maze, start, end)

        temp_triv = triviality(maze)
        dif = difficulty(maze, end)

        dfs_steps_avg_twenty_one.append(steps)
        dfs_trivial_avg_twenty_one.append(temp_triv)
        dfs_dif_avg_twenty_one.append(dif)

    # DFS 31
    dfs_steps_avg_thirty_one = []
    dfs_trivial_avg_thirty_one = []
    dfs_dif_avg_thirty_one = []

    for i in range(100):
        maze, start, end = gen_maze_DFS(31)
        path, steps = solve_maze_gen(maze, start, end)

        temp_triv = triviality(maze)
        dif = difficulty(maze, end)

        dfs_steps_avg_thirty_one.append(steps)
        dfs_trivial_avg_thirty_one.append(temp_triv)
        dfs_dif_avg_thirty_one.append(dif)

    # Prim: 15
    prim_steps_avg_fifteen = []
    prim_trivial_avg_fifteen = []
    prim_dif_avg_fifteen = []

    for i in range(100):
        maze, start, end = gen_maze_Prims(15)
        path, steps = solve_maze_gen(maze, start, end)

        temp_triv = triviality(maze)
        dif = difficulty(maze, end)

        prim_steps_avg_fifteen.append(steps)
        prim_trivial_avg_fifteen.append(temp_triv)
        prim_dif_avg_fifteen.append(dif)

    # Prim: 21
    prim_steps_avg_twenty_one = []
    prim_trivial_avg_twenty_one = []
    prim_dif_avg_twenty_one = []

    for i in range(100):
        maze, start, end = gen_maze_Prims(21)
        path, steps = solve_maze_gen(maze, start, end)

        temp_triv = triviality(maze)
        dif = difficulty(maze, end)

        prim_steps_avg_twenty_one.append(steps)
        prim_trivial_avg_twenty_one.append(temp_triv)
        prim_dif_avg_twenty_one.append(dif)

    # Prim 31
    prim_steps_avg_thirty_one = []
    prim_trivial_avg_thirty_one = []
    prim_dif_avg_thirty_one = []

    for i in range(100):
        maze, start, end = gen_maze_Prims(31)
        path, steps = solve_maze_gen(maze, start, end)

        temp_triv = triviality(maze)
        dif = difficulty(maze, end)

        prim_steps_avg_thirty_one.append(steps)
        prim_trivial_avg_thirty_one.append(temp_triv)
        prim_dif_avg_thirty_one.append(dif)

    # Print results
    print("DFS Results:")
    print("=" * 80)
    print("DFS with size 15 Maze:")
    print(f"Steps: {np.mean(dfs_steps_avg_fifteen):.3}")
    print(f"Triviality: {np.mean(dfs_trivial_avg_fifteen):.3}")
    print(f"Difficulty: {np.mean(dfs_dif_avg_fifteen):.3}")
    print("-" * 80)
    print("DFS with size 21 Maze:")
    print(f"Steps: {np.mean(dfs_steps_avg_twenty_one):.3}")
    print(f"Triviality: {np.mean(dfs_trivial_avg_twenty_one):.3}")
    print(f"Difficulty: {np.mean(dfs_dif_avg_twenty_one):.3}")
    print("-" * 80)
    print("DFS with size 31 Maze:")
    print(f"Steps: {np.mean(dfs_steps_avg_thirty_one):.3}")
    print(f"Triviality: {np.mean(dfs_trivial_avg_thirty_one):.3}")
    print(f"Difficulty: {np.mean(dfs_dif_avg_thirty_one):.3}")

    print()

    print("Prim Results:")
    print("=" * 80)
    print("Prim with size 15 Maze:")
    print(f"Steps: {np.mean(prim_steps_avg_fifteen):.3}")
    print(f"Triviality: {np.mean(prim_trivial_avg_fifteen):.3}")
    print(f"Difficulty: {np.mean(prim_dif_avg_fifteen):.3}")
    print("-" * 80)
    print("Prim with size 21 Maze:")
    print(f"Steps: {np.mean(prim_steps_avg_twenty_one):.3}")
    print(f"Triviality: {np.mean(prim_trivial_avg_twenty_one):.3}")
    print(f"Difficulty: {np.mean(prim_dif_avg_twenty_one):.3}")
    print("-" * 80)
    print("Prim with size 31 Maze:")
    print(f"Steps: {np.mean(prim_steps_avg_thirty_one):.3}")
    print(f"Triviality: {np.mean(prim_trivial_avg_thirty_one):.3}")
    print(f"Difficulty: {np.mean(prim_dif_avg_thirty_one):.3}")