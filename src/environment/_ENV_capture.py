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

import chess
import numpy as np

mapper = {}
mapper["p"] = 0 # pawn
mapper["r"] = 1 # rook
mapper["n"] = 2 # knight
mapper["b"] = 3 # bishop
mapper["q"] = 4 # queen
mapper["k"] = 5 # king
mapper["P"] = 0
mapper["R"] = 1
mapper["N"] = 2
mapper["B"] = 3
mapper["Q"] = 4
mapper["K"] = 5


class Board(object):

    def __init__(self, FEN=None):
        """
        Chess Board Environment
        Args:
            FEN: str
                Starting FEN notation, if None then start in the default chess position
        """
        self.FEN = FEN
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_action_space()
        self.layer_board = np.zeros(shape=(8, 8, 8))
        self.init_layer_board()

    def init_action_space(self):
        """
        Initialize the action space
        Returns:

        """
        self.action_space = np.zeros(shape=(64, 64))

    def init_layer_board(self):
        """
        Initalize the numerical representation of the environment
        Returns:

        """
        self.layer_board = np.zeros(shape=(8, 8, 8))
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self.board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = mapper[piece.symbol()]
            self.layer_board[layer, row, col] = sign
        if self.board.turn:
            self.layer_board[6, :, :] = 1 / self.board.fullmove_number
        if self.board.can_claim_draw():
            self.layer_board[7, :, :] = 1

    def step(self, action):
        """
        Run a step
        Args:
            action: tuple of 2 integers
                Move from, Move to

        Returns:
            epsiode end: Boolean
                Whether the episode has ended
            reward: int
                Difference in material value after the move
        """
        piece_balance_before = self.get_material_value()
        self.board.push(action)
        self.init_layer_board()
        piece_balance_after = self.get_material_value()
        if self.board.result() == "*":
            opponent_move = self.get_random_action()
            self.board.push(opponent_move)
            self.init_layer_board()
            capture_reward = piece_balance_after - piece_balance_before
            if self.board.result() == "*":
                reward = 0 + capture_reward
                episode_end = False
            else:
                reward = 0 + capture_reward
                episode_end = True
        else:
            capture_reward = piece_balance_after - piece_balance_before
            reward = 0 + capture_reward
            episode_end = True
        if self.board.is_game_over():
            reward = 0
            episode_end = True
        return episode_end, reward

    def get_random_action(self):
        """
        Sample a random action
        Returns: move
            A legal python chess move.

        """
        legal_moves = [x for x in self.board.generate_legal_moves()]
        legal_moves = np.random.choice(legal_moves)
        return legal_moves

    def project_legal_moves(self):
        """
        Create a mask of legal actions
        Returns: np.ndarray with shape (64,64)
        """
        self.action_space = np.zeros(shape=(64, 64))
        moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
        for move in moves:
            self.action_space[move[0], move[1]] = 1
        return self.action_space

    def get_material_value(self):
        """
        Sums up the material balance using Reinfield values
        Returns: The material balance on the board
        """
        pawns = 1 * np.sum(self.layer_board[0, :, :])
        rooks = 5 * np.sum(self.layer_board[1, :, :])
        minor = 3 * np.sum(self.layer_board[2:4, :, :])
        queen = 9 * np.sum(self.layer_board[4, :, :])
        return pawns + rooks + minor + queen

    def reset(self):
        """
        Reset the environment
        Returns:

        """
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_layer_board()
        self.init_action_space()
