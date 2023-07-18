import torch
from torch.distributions import Categorical
from chess.pgn import Game
import pandas as pd

class Q_learning_capture(object):

    def __init__(self, agent, env, memsize=1000):
        """
        Reinforce object to learn capture chess
        Args:
            agent: The agent playing the chess game as white
            env: The environment including the python-chess board
            memsize: maximum amount of games to retain in-memory
        """
        self.agent = agent
        self.env = env
        self.memory = []
        self.memsize = memsize
        self.reward_trace = []
        self.memory = []
        self.sampling_probs = []

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
            if k % c == 0:
                print("iter", k)
                self.agent.fix_model()
            greedy = True if k == iters - 1 else False
            self.env.reset()
            self.play_game(k, greedy=greedy)

        pgn = Game.from_board(self.env.board)
        reward_smooth = pd.DataFrame(self.reward_trace)
        reward_smooth.rolling(window=10, min_periods=0).mean().plot()

        return pgn

    def play_game(self, k, greedy=False, maxiter=25):
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

        # Here we determine the exploration rate. k is divided by 250 to slow down the exploration rate decay.
        eps = max(0.05, 1 / (1 + (k / 250))) if not greedy else 0.

        # Play a game of chess
        while not episode_end:
            state = torch.tensor(self.env.layer_board, dtype=torch.float32)
            explore = torch.rand(1).item() < eps  # determine whether to explore
            if explore:
                move = self.env.get_random_action()
                move_from = move.from_square
                move_to = move.to_square
            else:
                action_values = self.agent.get_action_values(state.unsqueeze(0))
                action_values = action_values.view(64, 64)
                action_space = self.env.project_legal_moves()  # The environment determines which moves are legal
                action_values = action_values * action_space
                move_from = torch.argmax(action_values).item() // 64
                move_to = torch.argmax(action_values).item() % 64
                moves = [x for x in self.env.board.generate_legal_moves() if
                         x.from_square == move_from and x.to_square == move_to]
                if len(moves) == 0:  # If all legal moves have negative action value, explore.
                    move = self.env.get_random_action()
                    move_from = move.from_square
                    move_to = move.to_square
                else:
                    move = torch.tensor(moves[0], dtype=torch.long)  # If there are multiple max-moves, pick a random one.

            episode_end, reward = self.env.step(move)
            new_state = torch.tensor(self.env.layer_board, dtype=torch.float32)
            if len(self.memory) > self.memsize:
                self.memory.pop(0)
                self.sampling_probs.pop(0)
            turncount += 1
            if turncount > maxiter:
                episode_end = True
                reward = 0
            if episode_end:
                new_state *= 0
            self.memory.append([state.numpy(), (move_from, move_to), reward, new_state.numpy()])
            self.sampling_probs.append(1)

            self.reward_trace.append(reward)

            self.update_agent(turncount)

        return self.env.board

    def sample_memory(self, turncount):
        """
        Get a sample from memory for experience replay
        Args:
            turncount: int
                turncount limits the size of the minibatch

        Returns: tuple
            a mini-batch of experiences (list)
            indices of chosen experiences

        """
        minibatch = []
        memory = self.memory[:-turncount]
        probs = self.sampling_probs[:-turncount]
        sample_probs = [probs[n] / sum(probs) for n in range(len(probs))]
        indices = torch.multinomial(torch.tensor(sample_probs, dtype=torch.float32), min(1028, len(memory)), replacement=True)
        for i in indices:
            minibatch.append(memory[i])

        return minibatch, indices

    def update_agent(self, turncount):
        """
        Update the agent using experience replay. Set the sampling probs with the td error
        Args:
            turncount: int
                Amount of turns played. Only sample the memory of there are sufficient samples
        Returns:

        """
        if turncount < len(self.memory):
            minibatch, indices = self.sample_memory(turncount)
            td_errors = self.agent.network_update(minibatch)
            for n, i in enumerate(indices):
                self.sampling_probs[i] = abs(td_errors[n].item())


# This is an optimized version:
# import torch
# from torch.distributions import Categorical
# from chess.pgn import Game
# import pandas as pd

# class Q_learning(object):

#     def __init__(self, agent, env, memsize=1000):
#         """
#         Reinforce object to learn capture chess
#         Args:
#             agent: The agent playing the chess game as white
#             env: The environment including the python-chess board
#             memsize: maximum amount of games to retain in-memory
#         """
#         self.agent = agent
#         self.env = env
#         self.memory = []
#         self.memsize = memsize
#         self.reward_trace = []
#         self.sampling_probs = []

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
#             if k % c == 0:
#                 print("iter", k)
#                 self.agent.fix_model()
#             greedy = True if k == iters - 1 else False
#             self.env.reset()
#             self.play_game(k, greedy=greedy)

#         pgn = Game.from_board(self.env.board)
#         reward_smooth = pd.DataFrame(self.reward_trace)
#         reward_smooth.rolling(window=10, min_periods=0).mean().plot()

#         return pgn

#     def play_game(self, k, greedy=False, maxiter=25):
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

#         # Here we determine the exploration rate. k is divided by 250 to slow down the exploration rate decay.
#         eps = max(0.05, 1 / (1 + (k / 250))) if not greedy else 0.

#         # Play a game of chess
#         while not episode_end:
#             state = torch.tensor(self.env.layer_board, dtype=torch.float32)
#             explore = torch.rand(1).item() < eps  # determine whether to explore
#             if explore:
#                 move = self.env.get_random_action()
#                 move_from = move.from_square
#                 move_to = move.to_square
#             else:
#                 action_values = self.agent.get_action_values(state.unsqueeze(0))
#                 action_values = action_values.view(64, 64)
#                 action_space = self.env.project_legal_moves()  # The environment determines which moves are legal
#                 action_values = action_values * action_space
#                 move_from, move_to = torch.divmod(torch.argmax(action_values), 64)
#                 moves = [x for x in self.env.board.generate_legal_moves() if
#                          x.from_square == move_from and x.to_square == move_to]
#                 if len(moves) == 0:  # If all legal moves have negative action value, explore.
#                     move = self.env.get_random_action()
#                     move_from = move.from_square
#                     move_to = move.to_square
#                 else:
#                     move = torch.tensor(moves[0], dtype=torch.long)  # If there are multiple max-moves, pick a random one.

#             episode_end, reward = self.env.step(move)
#             new_state = torch.tensor(self.env.layer_board, dtype=torch.float32)
#             if len(self.memory) > self.memsize:
#                 self.memory.pop(0)
#                 self.sampling_probs.pop(0)
#             turncount += 1
#             if turncount > maxiter:
#                 episode_end = True
#                 reward = 0
#             if episode_end:
#                 new_state *= 0
#             self.memory.append([state.numpy(), (move_from.item(), move_to.item()), reward, new_state.numpy()])
#             self.sampling_probs.append(1)

#             self.reward_trace.append(reward)

#             self.update_agent(turncount)

#         return self.env.board

#     def sample_memory(self, turncount):
#         """
#         Get a sample from memory for experience replay
#         Args:
#             turncount: int
#                 turncount limits the size of the minibatch

#         Returns: tuple
#             a mini-batch of experiences (list)
#             indices of chosen experiences

#         """
#         minibatch = []
#         memory = self.memory[:-turncount]
#         probs = self.sampling_probs[:-turncount]
#         sample_probs = torch.tensor(probs) / sum(probs)
#         indices = torch.multinomial(sample_probs, min(1028, len(memory)), replacement=True)
#         for i in indices:
#             minibatch.append(memory[i])

#         return minibatch, indices

#     def update_agent(self, turncount):
#         """
#         Update the agent using experience replay. Set the sampling probs with the td error
#         Args:
#             turncount: int
#                 Amount of turns played. Only sample the memory of there are sufficient samples
#         Returns:

#         """
#         if turncount < len(self.memory):
#             minibatch, indices = self.sample_memory(turncount)
#             td_errors = self.agent.network_update(minibatch)
#             for n, i in enumerate(indices):
#                 self.sampling_probs[i] = abs(td_errors[n].item())
