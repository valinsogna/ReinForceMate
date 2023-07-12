import torch
from chess.pgn import Game
import pandas as pd

class ActorCritic(object):

    def __init__(self, actor, critic, env):
        """
        ActorCritic object to learn capture chess
        Args:
            actor: Policy Gradient Agent
            critic: Q-learning Agent
            env: The environment including the python-chess board
            memsize: maximum amount of games to retain in-memory
        """
        self.actor = actor
        self.critic = critic
        self.env = env
        self.reward_trace = []
        self.action_value_mem = []
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
                self.critic.fix_model()
            self.env.reset()
            end_state = self.play_game(k)

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

        # Play a game of chess
        state = self.env.layer_board
        while not episode_end:
            state = self.env.layer_board
            action_space = self.env.project_legal_moves()  # The environment determines which moves are legal
            action_probs = self.actor.model(torch.tensor(state).unsqueeze(0), torch.zeros((1, 1)),
                                             torch.tensor(action_space).reshape(1, 4096))
            self.action_value_mem.append(action_probs)
            # print(action_probs)
            # print(torch.max(action_probs))
            action_probs = action_probs / action_probs.sum()
            move = torch.multinomial(action_probs.squeeze(), num_samples=1).item()
            move_from = move // 64
            move_to = move % 64
            moves = [x for x in self.env.board.generate_legal_moves() if \
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

            self.memory.append([state, (move_from, move_to), reward, new_state, action_space.reshape(1, 4096)])
            self.sampling_probs.append(1)
            self.reward_trace.append(reward)

        self.update_actorcritic(turncount)

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
        indices = np.random.choice(range(len(memory)), min(1028, len(memory)), replace=False, p=sample_probs)
        for i in indices:
            minibatch.append(memory[i])

        return minibatch, indices

    def update_actorcritic(self, turncount):
        """Actor critic"""

        if turncount < len(self.memory):

            # Get a sample
            minibatch, indices = self.sample_memory(turncount)

            # Update critic and find td errors for prioritized experience replay
            td_errors = self.critic.network_update(minibatch)

            # Get a Q value from the critic
            states = [x[0] for x in minibatch]
            actions = [x[1] for x in minibatch]
            Q_est = self.critic.get_action_values(torch.stack(states, dim=0))
            action_spaces = [x[4] for x in minibatch]

            self.actor.policy_gradient_update(states, actions, Q_est, action_spaces, actor_critic=True)

            # Update sampling probs
            for n, i in enumerate(indices):
                self.sampling_probs[i] = abs(td_errors[n])

    def update_critic(self, turncount):
        """
        Update the agent using experience replay. Set the sampling probs with the td error
        Args:
            turncount: int
                Amount of turns played. Only sample the memory if there are sufficient samples
        Returns:

        """
        if turncount < len(self.memory):
            minibatch, indices = self.sample_memory(turncount)
            td_errors = self.critic.network_update(minibatch)

            for n, i in enumerate(indices):
                self.sampling_probs[i] = abs(td_errors[n])

# This is an optimized version of the actor critic code:
# import torch
# from chess.pgn import Game
# import pandas as pd

# class ActorCritic(object):

#     def __init__(self, actor, critic, env):
#         self.actor = actor
#         self.critic = critic
#         self.env = env
#         self.reward_trace = []
#         self.action_value_mem = []
#         self.memory = []
#         self.sampling_probs = []

#     def learn(self, iters=100, c=10):
#         for k in range(iters):
#             if k % c == 0:
#                 self.critic.fix_model()
#             self.env.reset()
#             end_state = self.play_game(k)

#         pgn = Game.from_board(self.env.board)
#         reward_smooth = pd.DataFrame(self.reward_trace)
#         reward_smooth.rolling(window=10, min_periods=0).mean().plot()

#         return pgn

#     def play_game(self, k, greedy=False, maxiter=25):
#         episode_end = False
#         turncount = 0

#         state = self.env.layer_board
#         while not episode_end:
#             state = self.env.layer_board
#             action_space = self.env.project_legal_moves()
#             action_probs = self.actor.model(torch.tensor(state).unsqueeze(0),
#                                             torch.zeros((1, 1)),
#                                             torch.tensor(action_space).reshape(1, 4096))
#             self.action_value_mem.append(action_probs)

#             action_probs = action_probs / action_probs.sum()
#             move = torch.multinomial(action_probs.squeeze(), num_samples=1).item()
#             move_from = move // 64
#             move_to = move % 64
#             moves = [x for x in self.env.board.generate_legal_moves() if
#                      x.from_square == move_from and x.to_square == move_to]
#             assert len(moves) > 0

#             if len(moves) > 1:
#                 move = moves[np.random.choice(len(moves))]
#             elif len(moves) == 1:
#                 move = moves[0]

#             episode_end, reward = self.env.step(move)
#             new_state = self.env.layer_board
#             turncount += 1
#             if turncount > maxiter:
#                 episode_end = True
#                 reward = 0
#             if episode_end:
#                 new_state = new_state * 0

#             self.memory.append([state, (move_from, move_to), reward, new_state, action_space.reshape(1, 4096)])
#             self.sampling_probs.append(1)
#             self.reward_trace.append(reward)

#         self.update_actorcritic(turncount)

#         return self.env.board

#     def sample_memory(self, turncount):
#         minibatch = []
#         memory = self.memory[:-turncount]
#         probs = self.sampling_probs[:-turncount]
#         sample_probs = torch.tensor(probs) / sum(probs)
#         indices = torch.multinomial(sample_probs, min(1028, len(memory)), replacement=False)
#         for i in indices:
#             minibatch.append(memory[i])

#         return minibatch, indices

#     def update_actorcritic(self, turncount):
#         if turncount < len(self.memory):
#             minibatch, indices = self.sample_memory(turncount)
#             td_errors = self.critic.network_update(minibatch)

#             states = [x[0] for x in minibatch]
#             actions = [x[1] for x in minibatch]
#             Q_est = self.critic.get_action_values(torch.stack(states, dim=0))
#             action_spaces = [x[4] for x in minibatch]

#             self.actor.policy_gradient_update(states, actions, Q_est, action_spaces, actor_critic=True)

#             self.sampling_probs[indices] = torch.abs(td_errors)

#     def update_critic(self, turncount):
#         if turncount < len(self.memory):
#             minibatch, indices = self.sample_memory(turncount)
#             td_errors = self.critic.network_update(minibatch)

#             self.sampling_probs[indices] = torch.abs(td_errors)

