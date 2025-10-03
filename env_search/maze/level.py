"""Utilities for maze envs."""
import numpy as np

INT_TO_OBJ_TYPES = " XSGP"
OBJ_TYPES_TO_INT = {obj: i for i, obj in enumerate(INT_TO_OBJ_TYPES)}
NUM_OBJS = len(INT_TO_OBJ_TYPES)


class MazeLevel:
    """Utilities for handling maze envs.

    Mostly useful when evaluating the envs individually.

    Args:
        data: 2D numpy array of integers where the integers represent the object
            types. The array must be of shape (env_height, env_width).
    """

    def __init__(self, data):
        self.data = data

    @property
    def env_height(self):
        return self.data.shape[0]

    @property
    def env_width(self):
        return self.data.shape[1]

    @staticmethod
    def str_to_number(env_str):
        """Converts list of strings (each string is a row) to numpy array."""
        np_env = np.zeros((len(env_str), len(env_str[0])))
        for x, row in enumerate(env_str):
            # row = row.strip()
            for y, tile in enumerate(row):
                np_env[x, y] = OBJ_TYPES_TO_INT[tile]
        return np_env

    @staticmethod
    def number_to_str(np_env):
        env_str = []
        n_row, n_col = np_env.shape
        for i in range(n_row):
            curr_row = []
            for j in range(n_col):
                curr_row.append(INT_TO_OBJ_TYPES[np_env[i, j]])
            env_str.append("".join(curr_row))
        return env_str

    def to_str_grid(self):
        """Converts level to grid of characters."""
        return [[INT_TO_OBJ_TYPES[x] for x in row] for row in self.data]

    def to_str(self):
        """Converts level to single string."""
        return "\n".join(["".join(row) for row in self.to_str_grid()])
