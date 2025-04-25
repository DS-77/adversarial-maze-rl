import numpy as np
from random import choice

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
        return self.start

    def step(self, action):
        row, col = self.current_position
        if action == 0: # Up
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

        if (0 <= next_row < len(self.maze) and
            0 <= next_col < len(self.maze[0]) and
            self.maze[next_row][next_col] == 1):
            self.current_position = (next_row, next_col)
            if self.maze[next_row][next_col] == 3:
                return self.current_position, 1, True
            else:
                return self.current_position, 0,

        else:
            return self.current_position, -1, False

    def render(self):
        print(np.array2string(self.maze, max_line_width=50))

    def set_start_goal(self, start, goal):
      self.start = start
      self.goal = goal
      self.maze[start[0]][start[1]] = 2
      self.maze[goal[0]][goal[1]] = 3