import torch
import torch.nn as nn
from chess.pgn import Game
import pandas as pd

class Reinforce(object):
    def __init__(self, agent, env):
        """
        Reinforce object to learn capture chess
        Args:
            agent: The agent playing the chess game as white
            env: The environment including the python-chess board
            memsize: maximum amount of games to retain in-memory
        """
        self.agent = agent
        self.env = env
        self.reward_trace = []
        self.action_value_mem = []

    def learn(self, iters=100, c=10):
        """
        Run the Q-learning algorithm. Play greedy on the final iter
        Args:
            iters: int
                amount of games to train
            c: int
                update the network every c games

        Returns: pgn (str)
            pgn string describing final game

        """
        for k in range(iters):
            self.env.reset()
            states, actions, rewards, action_spaces = self.play_game(k)
            self.reinforce_agent(states, actions, rewards, action_spaces)

        pgn = Game.from_board(self.env.board)
        reward_smooth = pd.DataFrame(self.reward_trace)
        reward_smooth.rolling(window=10, min_periods=0).mean().plot()

        return pgn

    def play_game(self, k, maxiter=25):
        """
        Play a game of capture chess
        Args:
            k: int
                game count, determines epsilon (exploration rate)
            greedy: Boolean
                if greedy, no exploration is done
            maxiter: int
                Maximum amount of steps per game

        Returns:

        """
        episode_end = False
        turncount = 0

        states = []
        actions = []
        rewards = []
        action_spaces = []

        # Play a game of chess
        while not episode_end:
            state = self.env.layer_board
            action_space = self.env.project_legal_moves()  # The environment determines which moves are legal
            action_probs = self.agent.model(torch.Tensor(state).unsqueeze(0), torch.zeros(1, 1),
                                             torch.Tensor(action_space).view(1, 4096))
            self.action_value_mem.append(action_probs)
            action_probs = action_probs / action_probs.sum()
            move = torch.multinomial(action_probs.squeeze(), num_samples=1).item()
            move_from = move // 64
            move_to = move % 64
            moves = [x for x in self.env.board.generate_legal_moves() if
                     x.from_square == move_from and x.to_square == move_to]
            assert len(moves) > 0  # should not be possible
            if len(moves) > 1:
                move = np.random.choice(moves)  # If there are multiple max-moves, pick a random one.
            elif len(moves) == 1:
                move = moves[0]

            episode_end, reward = self.env.step(move)
            new_state = self.env.layer_board
            turncount += 1
            if turncount > maxiter:
                episode_end = True
                reward = 0
            if episode_end:
                new_state = new_state * 0

            states.append(state)
            actions.append((move_from, move_to))
            rewards.append(reward)
            action_spaces.append(torch.Tensor(action_space).view(1, 4096))

        self.reward_trace.append(torch.sum(torch.Tensor(rewards)))

        return states, actions, rewards, action_spaces

    def reinforce_agent(self, states, actions, rewards, action_spaces):
        """
        Update the agent using experience replay. Set the sampling probs with the td error
        Args:
            turncount: int
                Amount of turns played. Only sample the memory of there are sufficient samples
        Returns:

        """
        self.agent.policy_gradient_update(states, actions, rewards, action_spaces)

## This is an optimized version of the code above:
# import torch
# import torch.nn as nn
# from chess.pgn import Game
# import pandas as pd

# class Reinforce(object):
#     def __init__(self, agent, env):
#         """
#         Reinforce object to learn capture chess
#         Args:
#             agent: The agent playing the chess game as white
#             env: The environment including the python-chess board
#             memsize: maximum amount of games to retain in-memory
#         """
#         self.agent = agent
#         self.env = env
#         self.reward_trace = []
#         self.action_value_mem = []

#     def learn(self, iters=100, c=10):
#         """
#         Run the Q-learning algorithm. Play greedy on the final iter
#         Args:
#             iters: int
#                 amount of games to train
#             c: int
#                 update the network every c games

#         Returns: pgn (str)
#             pgn string describing final game

#         """
#         for k in range(iters):
#             self.env.reset()
#             states, actions, rewards, action_spaces = self.play_game(k)
#             self.reinforce_agent(states, actions, rewards, action_spaces)

#         pgn = Game.from_board(self.env.board)
#         reward_smooth = pd.DataFrame(self.reward_trace)
#         reward_smooth.rolling(window=10, min_periods=0).mean().plot()

#         return pgn

#     def play_game(self, k, maxiter=25):
#         """
#         Play a game of capture chess
#         Args:
#             k: int
#                 game count, determines epsilon (exploration rate)
#             greedy: Boolean
#                 if greedy, no exploration is done
#             maxiter: int
#                 Maximum amount of steps per game

#         Returns:

#         """
#         episode_end = False
#         turncount = 0

#         states = []
#         actions = []
#         rewards = []
#         action_space = torch.Tensor(self.env.project_legal_moves()).view(1, 4096)

#         # Play a game of chess
#         while not episode_end:
#             state = torch.from_numpy(self.env.layer_board)
#             action_probs = self.agent.model(state.unsqueeze(0), torch.zeros(1, 1), action_space)
#             self.action_value_mem.append(action_probs)
#             action_probs /= action_probs.sum()
#             move = torch.multinomial(action_probs.squeeze(), num_samples=1).item()
#             move_from = move // 64
#             move_to = move % 64
#             moves = [x for x in self.env.board.generate_legal_moves() if
#                      x.from_square == move_from and x.to_square == move_to]
#             assert len(moves) > 0  # should not be possible
#             if len(moves) > 1:
#                 move = np.random.choice(moves)  # If there are multiple max-moves, pick a random one.
#             elif len(moves) == 1:
#                 move = moves[0]

#             episode_end, reward = self.env.step(move)
#             new_state = torch.from_numpy(self.env.layer_board)
#             turncount += 1
#             if turncount > maxiter:
#                 episode_end = True
#                 reward = 0
#             if episode_end:
#                 new_state.zero_()

#             states.append(state)
#             actions.append((move_from, move_to))
#             rewards.append(reward)

#         self.reward_trace.append(torch.sum(torch.Tensor(rewards)))

#         return states, actions, rewards, action_space

#     def reinforce_agent(self, states, actions, rewards, action_space):
#         """
#         Update the agent using experience replay. Set the sampling probs with the td error
#         Args:
#             turncount: int
#                 Amount of turns played. Only sample the memory if there are sufficient samples
#         Returns:

#         """
#         self.agent.policy_gradient_update(states, actions, rewards, action_space)
