import pprint
import torch
from ..environment import GridWorld
from ..agent import Piece
from ..agent import Agent
from ..environment import ChessBoard
from ..config import Config as cfg
from abc import ABCMeta, abstractmethod

class BaseLearn(metaclass=ABCMeta):

    ARROWS = {
        'king': "↑ ↗ → ↘ ↓ ↙ ← ↖",
        'knight': "↑↗ ↗→ →↘ ↓↘ ↙↓ ←↙ ←↖ ↖↑",
        'bishop': "↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖",
        'rook': "↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ←"
    }

    @abstractmethod
    def __init__(self, **kwargs):
        
        self._init_config(**kwargs)

        if cfg.comm.piece is not None:
            self.agent = Piece(piece=cfg.comm.piece)
            self.env = GridWorld()

        if cfg.comm.FEN is not None:
            self.agent = Agent()
            self.env = ChessBoard(FEN=cfg.comm.FEN)

        if cfg.comm.capture:
            self.agent = Agent()
            self.env = ChessBoard()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_config(self, **kwargs):
        # Update the configuration with any keyword arguments passed to the method
        for key, value in kwargs.items():
            # list all subconfigurations in a dictionary
            subconfigs = {
                "comm": cfg.comm,
                "poli": cfg.poli,
            }
            for subconfig_name, subconfig in subconfigs.items():
                if hasattr(subconfig, key):
                    setattr(subconfig, key, value)
                    break
            else:  # if no break, attribute was not found in any subconfig
                print(f"Warning: Invalid config key: {key} - this will be ignored.")
        
    def visualize_policy(self):
        greedy_policy = self.agent.policy.argmax(axis=2)
        policy_visualization = {idx: arrow for idx, arrow in enumerate(self.ARROWS[self.agent.piece].split(" "))}
        visual_board = [[policy_visualization[greedy_policy[row, col].item()] for col in range(greedy_policy.shape[1])]
                        for row in range(greedy_policy.shape[0])]
        visual_board[self.env.terminal_state[0]][self.env.terminal_state[1]] = "F"
        pprint.pprint(visual_board)

    def visualize_action_function(self):
        print(torch.max(self.agent.action_function, dim=2)[0].to(torch.int))

    