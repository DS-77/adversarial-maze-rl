"""
This module generates a Maze using Prim and DFS algorithm. It also provides a solution.

Created: 28-03-2025
Edited: 29-03-2025
Version: 1.0.0
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, randrange, randint, choice


def maze_viz(maze, solution=None):
    plt.imshow(maze, cmap='gray')
    plt.title("Simple Maze")
    plt.show()

def gen_maze_DFS(size: int) -> np.array:
    # Check if the size is odd and positive
    if size % 2 == 0 and size < 0:
        print("ERROR: The size must be odd and positive.")

    # Create an array of the given size
    temp_maze = np.zeros((size, size), dtype=int)
    # temp_maze = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])

    # Determine the goal point
    # 0: Top, 1: Right, 2: Bottom, 3: Left
    edge = choice([0, 1, 2, 3])

    if edge == 0:  # Top Edge
        goal_x = randrange(1, size - 1, 2)
        goal_y = 0
    elif edge == 1:  # Right Edge
        goal_x = size - 1
        goal_y = randrange(1, size - 1, 2)
    elif edge == 2:  # Bottom Edge
        goal_x = randrange(1, size - 1, 2)
        goal_y = size - 1
    else:  # Left Edge (edge == 3)
        goal_x = 0
        goal_y = randrange(1, size - 1, 2)

    # Define the traversal function
    def traverse(x:int , y:int):
        temp_maze[y, x] = 1

        d = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        shuffle(d)

        for (tx, ty) in d:
            nx, ny = x + tx, y + ty
            # if temp_maze[ty][tx]: continue
            if 0 < nx < size - 1 and 0 < ny < size - 1 and temp_maze[ny, nx] == 0:
                temp_maze[y + ty // 2, x + tx // 2] = 1
                traverse(nx, ny)

    # Get random starting point
    start_x = randrange(1, size - 1, 2)
    start_y = randrange(1, size - 1, 2)

    print(f"Start Point: {start_x}, {start_y}")
    print(f"End Point: {goal_x}, {goal_y}")

    # Walk the grid
    traverse(start_x, start_y)

    # Set the start and goal state in the maze
    temp_maze[start_x][start_y] = 3
    temp_maze[goal_x][goal_y] = 3

    return temp_maze, (start_x, start_y), (goal_x, goal_y)

def solve_maze_gen(maze) -> np.array:
    pass


def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--size", type=int, required=True, help="The size of the maze. Must be odd and positive.")
    parser.add_argument("-s", "--solve", action="store_true")
    opts = parser.parse_args()

    # Generate maze
    maze, maze_start, maze_goal = gen_maze_DFS(opts.size)

    print(maze)

    # View maze
    maze_viz(maze)

if __name__ == "__main__":
    main()