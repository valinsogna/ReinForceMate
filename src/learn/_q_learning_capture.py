import torch
import numpy as np
from chess.pgn import Game
from ._base import BaseLearn
import pandas as pd
from tqdm.auto import tqdm
from ..config import Config as cfg

class Q_Learning(BaseLearn):
    """
    A Q-Learning agent which learns to play capture chess.

    Attributes:
        memory: list
            A list to store past game experiences for training.
        reward_trace: list
            A list to track the reward obtained after each move.
        sampling_probs: list
            A list to store sampling probabilities for each experience in memory.
    """

    def __init__(self, **kwargs):
        """
        Initialize Q-Learning agent.

        Args:
            agent (Agent): The agent playing the chess game as white.
            env (Environment): The environment including the python-chess board.
            memsize (int): maximum amount of games to retain in-memory.
        """

        super().__init__(**kwargs)

        self.memory = []
        self.reward_trace = []
        self.memory = []
        self.sampling_probs = []

    # Run the Q-learning algorithm
    def learn(self, iters=100, c=10):
        """
        Run the Q-learning algorithm.

        Args:
            iters (int, optional): Amount of games to train. Default is 100.
            c (int, optional): Update the network every c games. Default is 10.

        Returns:
            pgn (str): PGN string describing final game.
        """

        # Iterate over the training games
        for k in tqdm(range(iters), desc="Training Iterations"):
            greedy = True if k == iters - 1 else False
            self.env.reset()
            self.play_game(k, greedy=greedy)

        # Generate the PGN string describing the final game
        pgn = Game.from_board(self.env.board)

        # Plot the rolling average of the rewards
        reward_smooth = pd.DataFrame(self.reward_trace)
        reward_smooth.rolling(window=10, min_periods=0).mean().plot()

        return pgn

    # Play a game of chess
    def play_game(self, k, greedy=False, maxiter=25):
        """
        Play a game of chess.

        Args:
            k (int): Game iteration number.
            greedy (bool, optional): Whether to play greedily. Default is False.
            maxiter (int, optional): Maximum turns in a game. Default is 25.

        Returns:
            Board: The final board state after playing the game.
        """

        episode_end = False
        turncount = 0

        # Here we determine the exploration rate. k is divided by 250 to slow down the exploration rate decay.
        eps = max(0.05, 1 / (1 + (k / 250))) if not greedy else 0.

        # Play a game of chess
        while not episode_end:
            state = self.env.layer_board

            # Determine whether to explore or exploit
            explore = np.random.uniform(0, 1) < eps

            if explore:
                move = self.env.get_random_action()
                move_from = move.from_square
                move_to = move.to_square
            else:
                # Get action values and legal action space
                action_values = self.agent.get_action_values(torch.tensor(state).unsqueeze(0))
                action_values = action_values.view(64, 64)  # equivalent to reshape in PyTorch
                action_space = torch.tensor(self.env.project_legal_moves()).to(action_values.device)
                action_values = action_values * action_space

                # Determine the move with the highest action value
                move_from = torch.argmax(action_values).item() // 64
                move_to = torch.argmax(action_values).item() % 64

                # Check if the move is valid and choose randomly if not
                moves = [x for x in self.env.board.generate_legal_moves() if
                        x.from_square == move_from and x.to_square == move_to]

                if len(moves) == 0:
                    move = self.env.get_random_action()
                    move_from = move.from_square
                    move_to = move.to_square
                else:
                    move = np.random.choice(moves)

            episode_end, reward = self.env.step(move)
            new_state = self.env.layer_board

            # Remove the oldest experience if memory is full
            if len(self.memory) > cfg.comm.memsize:
                self.memory.pop(0)
                self.sampling_probs.pop(0)

            turncount += 1

            if turncount > maxiter:
                episode_end = True
                reward = 0

            if episode_end:
                new_state = new_state * 0

            # Append the current experience to memory
            self.memory.append([state, (move_from, move_to), reward, new_state])
            self.sampling_probs.append(1)

            self.reward_trace.append(reward)

            # Update the agent
            self.update_agent(turncount)

        return self.env.board

    # Get a sample from memory for experience replay.
    def sample_memory(self, turncount):
        """
        Get a sample from memory for experience replay.

        Args:
            turncount (int): Turn count.

        Returns:
            tuple: A mini-batch of experiences (list) and indices of chosen experiences.
        """

        # Initialize an empty mini-batch
        minibatch = []

        # Exclude the most recent experiences from memory and sampling probabilities
        memory = self.memory[:-turncount]
        probs = self.sampling_probs[:-turncount]

        # Calculate the sampling probabilities for the remaining experiences
        sample_probs = [probs[n] / np.sum(probs) for n in range(len(probs))]

        # Sample indices from the memory based on the sampling probabilities
        indices = np.random.choice(range(len(memory)), min(1028, len(memory)), replace=True, p=sample_probs)

        # Retrieve the corresponding experiences and add them to the mini-batch
        for i in indices:
            minibatch.append(memory[i])

        return minibatch, indices


    # Update the Q-Learning agent.
    def update_agent(self, turncount):
        """
        Update the Q-Learning agent.

        Args:
            turncount (int): Turn count.
        """

        # Check if turncount is within the memory size
        if turncount < len(self.memory):
            # Sample minibatch and corresponding indices
            minibatch, indices = self.sample_memory(turncount)
            # Update the network and get TD errors
            td_errors = self.agent.network_update(minibatch)
            # Update the sampling probabilities based on TD errors
            for n, i in enumerate(indices):
                self.sampling_probs[i] = np.abs(td_errors[n])

                
