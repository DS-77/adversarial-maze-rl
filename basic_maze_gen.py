"""
This module generates a Maze using Prim and DFS algorithm. It also provides a solution.

Author: Deja S.
Created: 28-03-2025
Edited: 31-03-2025
Version: 1.0.2
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, randrange, randint, choice
from metrics.metrics import triviality, difficulty, A_star_solver


def maze_viz(maze, solution=None, iter=None ,path=None):

    # if the solution is provided, highlight the path
    if solution is not None:
        for x, y in solution:
            maze[x][y] = 4

    plt.imshow(maze, cmap='gray')
    plt.title("Simple Maze")
    if path is not None:
        plt.savefig(f"{path}/maze_gen_{iter}.png")
        plt.close()
    else:
        plt.show()


def gen_maze_DFS(size) -> np.array:
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
    def traverse(x , y):
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

    # print(f"Start Point: {start_x}, {start_y}")
    # print(f"End Point: {goal_x}, {goal_y}")

    # Walk the grid
    traverse(start_x, start_y)

    # Set the start and goal state in the maze
    temp_maze[start_y][start_x] = 2
    temp_maze[goal_y][goal_x] = 3

    return temp_maze, (start_x, start_y), (goal_x, goal_y)

# def gen_maze_kruskal(size):
#     pass


def gen_maze_Prims(size) -> tuple[np.array, tuple[int, int], tuple[int, int]]:
    temp_maze = np.zeros((size, size), dtype=int)

    edge = choice([0, 1, 2, 3])
    if edge == 0:
        goal_x = choice(range(1, size - 1, 2))
        goal_y = 0
    elif edge == 1:
        goal_x = size - 1
        goal_y = choice(range(1, size - 1, 2))
    elif edge == 2:
        goal_x = choice(range(1, size - 1, 2))
        goal_y = size - 1
    else:
        goal_x = 0
        goal_y = choice(range(1, size - 1, 2))

    # Choose a random starting cell
    start_x = choice(range(1, size - 1, 2))
    start_y = choice(range(1, size - 1, 2))

    temp_maze[start_y, start_x] = 1

    # Initialise a list to hold frontier cells
    frontier = []
    for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
        nx, ny = start_x + dx, start_y + dy
        if 0 < nx < size - 1 and 0 < ny < size - 1 and temp_maze[ny, nx] == 0:
            frontier.append((nx, ny))

    while frontier:
        # Choose a random frontier cell
        cell_x, cell_y = choice(frontier)
        frontier.remove((cell_x, cell_y))

        # Find adjacent maze cells
        neighbors = []
        for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
            nx, ny = cell_x + dx, cell_y + dy
            if 0 < nx < size - 1 and 0 < ny < size - 1 and temp_maze[ny, nx] == 1:
                neighbors.append((nx, ny))

        if not neighbors:
            continue  # Skip if no neighbors (this is a fix for potential index errors)

        # Choose a random neighbour to connect to
        neighbor_x, neighbor_y = choice(neighbors)

        # Carve a path between the frontier cell
        temp_maze[cell_y, cell_x] = 1
        temp_maze[(cell_y + neighbor_y) // 2, (cell_x + neighbor_x) // 2] = 1

        # Add new frontier cells
        for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
            nx, ny = cell_x + dx, cell_y + dy
            if 0 < nx < size - 1 and 0 < ny < size - 1 and temp_maze[ny, nx] == 0:
                if (nx, ny) not in frontier:
                    frontier.append((nx, ny))

    temp_maze[goal_y, goal_x] = 3
    temp_maze[start_y, start_x] = 2

    # print(f"Start Point: ({start_x}, {start_y})")
    # print(f"End Point: ({goal_x}, {goal_y})")

    return temp_maze, (start_x, start_y), (goal_x, goal_y)


def check_valid_move(maze, visited, y, x) -> bool:
    # Note the order: y, x for numpy array indexing
    return (0 <= y < maze.shape[0] and 0 <= x < maze.shape[1] and
            (maze[y, x] == 1 or maze[y, x] == 3) and not visited[y][x])

def solver_DFS(maze, visited, path, start_x, start_y, goal, steps) -> bool:
    y, x = start_y, start_x
    goal_x, goal_y = goal

    if (x, y) == goal:
        path.append((x, y))
        return True

    # Add current coordinate to visited stack
    visited[y][x] = True
    path.append((x, y))
    steps[0] += 1

    # Four possible moves: right, down, left, up
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for dy, dx in moves:
        ny, nx = y + dy, x + dx
        if check_valid_move(maze, visited, ny, nx):
            if solver_DFS(maze, visited, path, nx, ny, goal, steps):
                return True

    # Backtrack
    path.pop()
    steps[0] -= 1
    return False


def solve_maze_gen(maze, start_point, goal_point) -> list:
    # Path stack
    path = []

    # Create visited array with same shape as maze
    visited = [[False for _ in range(maze.shape[1])] for _ in range(maze.shape[0])]

    # init counter
    steps = [0]

    # Unpack the start and goal points
    start_x, start_y = start_point
    goal_x, goal_y = goal_point

    # generate solver
    if solver_DFS(maze, visited, path, start_x, start_y, goal_point, steps):
        # print(f"Found a path in {steps[0]} steps!")
        return path, steps
    else:
        # print("No path found")
        return None, steps

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
        maze, maze_start, maze_goal = gen_maze_Prims(opts.size)

    elif opts.algorithm == "DFS":
        maze, maze_start, maze_goal = gen_maze_DFS(opts.size)
        # print(maze)

    elif opts.algorithm == "kruskal":
        # TODO: Fix this at some point
        # maze, maze_start, maze_goal = gen_maze_kruskal(opts.size)
        pass
    else:
        print("ERROR: Unknown algorithm.")
        exit()

    if opts.solve :
        path, steps = solve_maze_gen(maze, maze_start, maze_goal)
        print(path)

        # View maze
        maze_viz(maze, solution=path)
    else:
        # View maze
        maze_viz(maze)

if __name__ == "__main__":
    main()