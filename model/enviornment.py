"""
This module defines the maze environment for Maze-Gen.
"""


import numpy as np
from collections import deque

class Environment:
    def __init__(self, maze):
        self.maze = maze
        self.start = None
        self.goal = None
        self.current_position = None
        for r in range(len(maze)):
            for c in range(len(maze[0])):
                if maze[r][c] == 1:
                    pass
                elif maze[r][c] == 2:
                    self.start = (r, c)
                elif maze[r][c] == 3:
                    self.goal = (r, c)

    def reset(self):
        self.current_position = self.start
        return self.get_encoded_state()

    def step(self, action):
        if self.current_position is None:
            self.reset()

        row, col = self.current_position
        if action == 0:  # Up
            next_row = row - 1
            next_col = col
        # Down
        elif action == 1:
            next_row = row + 1
            next_col = col
        # Left
        elif action == 2:
            next_row = row
            next_col = col - 1
        # Right
        else:
            next_row = row
            next_col = col + 1

        # Check if the next position is valid
        if (0 <= next_row < len(self.maze) and
                0 <= next_col < len(self.maze[0]) and
                self.maze[next_row][next_col] != 0):

            self.current_position = (next_row, next_col)

            # Check if reached goal
            if self.maze[next_row][next_col] == 3:
                return self.current_position, 1.0, True, {}
            else:
                return self.current_position, -0.01, False, {}

        else:
            # Penalty for hitting wall
            return self.current_position, -0.1, False, {}

    def render(self):
        print(np.array2string(self.maze, max_line_width=50))

    def set_start_goal(self, start, goal):
      self.start = start
      self.goal = goal
      self.maze[start[0]][start[1]] = 2
      self.maze[goal[0]][goal[1]] = 3

    def get_encoded_state(self):
        """
        Returns a 4-channel encoded version of the maze: 0 -> walls, 1 -> agent, 2 -> start, 3 -> goal
        """
        channels = 4
        h, w = self.maze.shape
        state = np.zeros((channels, h, w), dtype=np.float32)

        for r in range(h):
            for c in range(w):
                if self.maze[r, c] == 0:
                    state[0, r, c] = 1
                elif self.maze[r, c] == 2:
                    state[2, r, c] = 1
                elif self.maze[r, c] == 3:
                    state[3, r, c] = 1

        if self.current_position:
            r, c = self.current_position
            state[1, r, c] = 1

        return state.flatten()

    def get_neighbours(self, position):
        h, w = self.maze.shape
        x, y = position
        neighbours = []

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                neighbours.append((nx, ny))

        return neighbours

    def is_solvable(self):
        # Basically a BFS solver -> safer for solvability check
        visited = set()
        queue = deque()

        queue.append(self.start)
        visited.add(self.start)

        while queue:
            current = queue.popleft()
            if current == self.goal:
                return True

            for neighbour in self.get_neighbours(current):
                if neighbour not in visited:
                    x, y = neighbour
                    if self.maze[x][y] == 0:
                        visited.add(neighbour)
                        queue.append(neighbour)

        return False