"""
This module contains functions that measure difficulty, solvability, and triviality of a given maze.

Author: Deja S.
Created: 31-03-2025
Edited: 02-04-2025
Version: 1.0.0
"""

import numpy as np
import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def __len__(self):
        return len(self._queue)

    def is_empty(self):
        return len(self._queue) == 0

def count_deadends(maze) -> int:
    rows, cols = maze.shape
    deadends = 0

    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1:
                neighbors = 0
                # Check neighbours
                if r > 0 and maze[r - 1, c] == 1:
                    neighbors += 1
                if r < rows - 1 and maze[r + 1, c] == 1:
                    neighbors += 1
                if c > 0 and maze[r, c - 1] == 1:
                    neighbors += 1
                if c < cols - 1 and maze[r, c + 1] == 1:
                    neighbors += 1

                if neighbors == 1:
                    deadends += 1

    return deadends

def getNeighbours(maze, row, col) -> list:
    rows = len(maze)
    cols = len(maze[0])
    neighbours = []

    # Possible directions
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for dr, dc in moves:
        new_row = row + dr
        new_col = col + dc

        # Check for boundaries and walls
        if 0 <= new_row < rows and 0 <= new_col < cols and maze[new_row][new_col] == 0:
            neighbours.append((new_row, new_col))

    return neighbours

def get_all_nodes(maze) -> list:
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0

    nodes = []
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:
                nodes.append((r, c))
    return nodes

def manhattan_distance(current_point, goal_point) -> float:
    return abs(current_point[0] - goal_point[0]) + abs(current_point[1] - goal_point[1])

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]

def A_star_solver(maze, start, goal) -> tuple[list, int]:
    open_set = PriorityQueue()
    came_from = {}
    g_score = {node: float('inf') for node in get_all_nodes(maze)}
    g_score[start] = 0
    f_score = {node: float('inf') for node in get_all_nodes(maze)}
    f_score[start] = manhattan_distance(start, goal)

    open_set.push(start, f_score[start])
    steps = 0

    while not open_set.is_empty():
        steps += 1
        current = open_set.pop()

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, steps

        for neighbour in getNeighbours(maze, current[0], current[1]):
            tentative_g_score = g_score[current] + 1

            if tentative_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g_score
                f_score[neighbour] = tentative_g_score + manhattan_distance(neighbour, goal)
                open_set.push(neighbour, f_score[neighbour])

    return None, steps


def total_maze_length(maze) -> int:
    return np.sum(maze == 1)

def count_deadends(maze, end) -> int:
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    deadends = 0
    deadend_points = []

    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 1:
                neighbors = 0
                # Check neighbours
                if r > 0 and maze[r-1][c] == 1:
                    neighbors += 1
                if r < rows - 1 and maze[r+1][c] == 1:
                    neighbors += 1
                if c > 0 and maze[r][c-1] == 1:
                    neighbors += 1
                if c < cols - 1 and maze[r][c+1] == 1:
                    neighbors += 1

                if neighbors == 1 and end != (r, c):
                    deadends += 1
                    deadend_points.append((r, c))
    # print(deadend_points)
    return deadends

def difficulty(maze, end) -> float:
    # Higher score indicates higher difficulty.
    w1 = 0.3
    w2 = 0.3
    w3 = 0.3
    score = 0

    branches = triviality(maze)
    deadend = count_deadends(maze, end)
    maze_length = total_maze_length(maze)

    # print(f"Branches: {branches}")
    # print(f"Dead Ends: {deadend}")
    # print(f"Total Maze Length: {maze_length}")

    score = w1 * maze_length + w2 * branches + w3 * deadend

    return score


def triviality(maze) -> int:
    # Lower score indicate more triviality.
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    branch_count = 0

    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 1:
                neighbor_paths = 0
                # Check neighbours
                if r > 0 and maze[r - 1][c] == 1:
                    neighbor_paths += 1
                if r < rows - 1 and maze[r + 1][c] == 1:
                    neighbor_paths += 1
                if c > 0 and maze[r][c - 1] == 1:
                    neighbor_paths += 1
                if c < cols - 1 and maze[r][c + 1] == 1:
                    neighbor_paths += 1

                if neighbor_paths == 3:
                    branch_count += 1

    return branch_count