import numpy as np
import chess
import torch

class ChessBoard:
    """
    A class representing a Chess Board Environment. It can be initialized with a given FEN string, or default to
    the standard initial chess board setup.

    Attributes:
        FEN (str): The Forsyth–Edwards Notation (FEN) representation of the current board state.
        board (chess.Board): The chess board.
        action_space (np.array): An array representing the possible actions.
        layer_board (np.array): A numerical representation of the environment.
    """

    mapper = {"p": 0, "r": 1, "n": 2, "b": 3, "q": 4, "k": 5,
          "P": 0, "R": 1, "N": 2, "B": 3, "Q": 4, "K": 5}

    def __init__(self, FEN=None):
        """
        Initializes the Chess Board environment with given FEN string or standard initial chess board setup.

        Args:
            FEN (str): Starting FEN notation. If None, starts in the default chess position.
        """

        self.FEN = FEN
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_action_space()
        self.layer_board = np.zeros(shape=(8, 8, 8))
        self.init_layer_board()

    def init_action_space(self):
        """
        Initializes the action space of the environment as a zero 2D array of size 64x64.
        """

        self.action_space = np.zeros(shape=(64, 64))

    def init_layer_board(self):
        """
        Initializes the numerical representation of the environment as an 8x8x8 zero array.
        Fills in the appropriate layers based on the current board state.
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
            layer = self.mapper[piece.symbol()]
            self.layer_board[layer, row, col] = sign
        if self.board.turn:
            self.layer_board[6, :, :] = 1 / self.board.fullmove_number
        if self.board.can_claim_draw():
            self.layer_board[7, :, :] = 1

    def step(self, action):
        """
        Executes an action and updates the environment state.

        Args:
            action (tuple): Tuple of 2 integers, representing the move from and move to square indices.

        Returns:
            episode_end (bool): Indicates if the game has ended.
            reward (int): The reward gained from the move, calculated as the difference in material value after the move.
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
        Gets a random legal move from the current board state.

        Returns:
            legal_moves (chess.Move): A randomly chosen legal move.
        """

        legal_moves = [x for x in self.board.generate_legal_moves()]
        legal_moves = np.random.choice(legal_moves)
        return legal_moves

    def project_legal_moves(self):
        """
        Projects the legal moves to the action space. The action space is updated with 1's at the indices
        corresponding to legal moves.

        Returns:
            action_space (np.array): The updated action space as a numpy array.
        """

        self.action_space = torch.zeros((64, 64))
        moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
        for move in moves:
            self.action_space[move[0], move[1]] = 1
        return self.action_space.numpy()

    def get_material_value(self):
        """
        Calculates the total material value on the board using Reinfeld values.

        Returns:
            value (int): The material balance on the board.
        """

        pawns = 1 * np.sum(self.layer_board[0, :, :])
        rooks = 5 * np.sum(self.layer_board[1, :, :])
        minor = 3 * np.sum(self.layer_board[2:4, :, :])
        queen = 9 * np.sum(self.layer_board[4, :, :])
        return pawns + rooks + minor + queen

    def reset(self):
        """
        Resets the environment to its initial state defined by the provided FEN string or the default chess board setup.
        """
        
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_layer_board()
        self.init_action_space()


