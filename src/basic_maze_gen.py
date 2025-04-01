"""
This module generates a Maze using Prim and DFS algorithm. It also provides a solution.

Created: 28-03-2025
Edited: 31-03-2025
Version: 1.0.2
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, randrange, randint, choice


def maze_viz(maze, solution=None):

    # if the solution is provided, highlight the path
    if solution is not None:
        for x, y in solution:
            maze[x][y] = 4

    plt.imshow(maze, cmap='gray')
    plt.title("Simple Maze")
    plt.show()

def gen_maze_DFS(size: int) -> np.array:
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
    temp_maze[goal_x][goal_y] = 1

    return temp_maze, (start_x, start_y), (goal_x, goal_y)

def check_valid_move(maze, visited, x, y):
    return (0 <= x < len(maze) and 0 <= y < len(maze[0]) and
            maze[x][y] == 1 and not visited[x][y])

def solver_DFS(maze, visited, path, x, y, goal):

    if (x, y) == goal:
        path.append((x, y))
        return True

    # Add current coordinate to visited stack
    visited[x][y] = True
    path.append((x, y))

    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for move in moves:
        nx, ny = move[0] + x, move[1] + y
        if check_valid_move(maze, visited, nx, ny):
            if solver_DFS(maze, visited, path, nx, ny, goal):
                return True

    # Backtrack
    path.pop()
    return False

def solve_maze_gen(maze, start_point, goal_point) -> np.array:
    # Path stack
    path = []

    # Visited stack
    visited = [[False for _ in range(len(maze[0]))] for _ in range(len(maze))]

    # generate solver
    if solver_DFS(maze, visited, path, start_point[0], start_point[1], goal_point):
        print("Found a path!")
        return path
    else:
        return None

def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--size", type=int, required=True, help="The size of the maze. Must be odd and positive.")
    parser.add_argument("-s", "--solve", action="store_true")
    parser.add_argument("-a", "--algorithm", type=str, help="The name of the generation algorithm: 'prim', 'kruskal', and 'DFS'")
    opts = parser.parse_args()

    # Check maze size
    if opts.size % 2 == 0 or opts.size <= 0:
        print("ERROR: The size must be odd and positive.")
        exit()

    # Maze
    maze = None
    maze_start = None
    maze_goal = None

    # Generate maze
    if opts.algorithm == "prim":
        pass
    elif opts.algorithm == "DFS":
        maze, maze_start, maze_goal = gen_maze_DFS(opts.size)
        # print(maze)

    elif opts.algorithm == "kruskal":
        pass

    else:
        print("ERROR: Unknown algorithm.")
        exit()

    if opts.solve :
        path = solve_maze_gen(maze, maze_start, maze_goal)
        # print(path)

        # View maze
        maze_viz(maze, path)
    else:
        maze_viz(maze)

if __name__ == "__main__":
    main()