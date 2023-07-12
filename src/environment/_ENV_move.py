"""
It is mostly based on the Board object from python-chess. Some modifications are made to make it easier for the algorithm to converge:

- There is a maximum of 25 moves, after that the environment resets
- Our Agent only plays white: the white player is the one who moves first
- The Black player is part of the environment and returns random moves
- The reward structure is not based on winning/losing/drawing but on capturing black pieces:
   - pawn capture: +1
   - knight capture: +3
   - bishop capture: +3
   - rook capture: +5
   - queen capture: +9
- Our state is represent by an 8x8x8 array: 8x8 for the chess board and 8 for the planes
- Plane 0 represents pawns
- Plane 1 represents rooks
- Plane 2 represents knights
- Plane 3 represents bishops
- Plane 4 represents queens
- Plane 5 represents kings
- Plane 6 represents 1/fullmove number (needed for markov property)
- Plane 7 represents can-claim-draw
- White pieces have the value 1, black pieces are -1
"""

import pprint
import numpy as np


class Board(object):

    def __init__(self):
        self.state = (0, 0)
        self.reward_space = np.zeros(shape=(8, 8)) - 1
        self.terminal_state = (7, 5)

    def step(self, action):
        reward = self.reward_space[self.state[0], self.state[1]]
        if self.state == self.terminal_state:
            episode_end = True
            return 0, episode_end
        else:
            episode_end = False
            old_state = self.state
            new_state = (self.state[0] + action[0], self.state[1] + action[1])  # step
            self.state = old_state if np.min(new_state) < 0 or np.max(new_state) > 7 else new_state
            return reward, episode_end

    def render(self):
        visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())
        visual_board[self.state[0]][self.state[1]] = "[S]"
        visual_board[self.terminal_state[0]][self.terminal_state[1]] = "[F]"
        self.visual_board = visual_board
