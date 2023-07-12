import numpy as np
import pprint


class Temporal_difference(object):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    
    def sarsa_td(self, n_episodes=1000, alpha=0.01, gamma=0.9):
        """
        Run the sarsa control algorithm (TD0), finding the optimal policy and action function
        :param n_episodes: int, amount of episodes to train
        :param alpha: learning rate
        :param gamma: discount factor of future rewards
        :return: finds the optimal policy for move chess
        """
        for k in range(n_episodes):
            state = (0, 0)
            self.env.state = state
            episode_end = False
            epsilon = max(1 / (1 + k), 0.05)
            while not episode_end:
                state = self.env.state
                action_index = self.agent.apply_policy(state, epsilon)
                action = self.agent.action_space[action_index]
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.agent.apply_policy(successor_state, epsilon)

                action_value = self.agent.action_function[state[0], state[1], action_index]
                successor_action_value = self.agent.action_function[successor_state[0],
                                                                    successor_state[1], successor_action_index]

                q_update = alpha * (reward + gamma * successor_action_value - action_value)

                self.agent.action_function[state[0], state[1], action_index] += q_update
                self.agent.policy = self.agent.action_function.copy()


    def visualize_policy(self):
        """
        Gives you are very ugly visualization of the policy
        Returns: None

        """
        greedy_policy = self.agent.policy.argmax(axis=2)
        policy_visualization = {}
        if self.agent.piece == 'king':
            arrows = "↑ ↗ → ↘ ↓ ↙ ← ↖"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        elif self.agent.piece == 'knight':
            arrows = "↑↗ ↗→ →↘ ↓↘ ↙↓ ←↙ ←↖ ↖↑"
            visual_row = ["[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]"]
        elif self.agent.piece == 'bishop':
            arrows = "↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        elif self.agent.piece == 'rook':
            arrows = "↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ←"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        arrowlist = arrows.split(" ")
        for idx, arrow in enumerate(arrowlist):
            policy_visualization[idx] = arrow
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())

        for row in range(greedy_policy.shape[0]):
            for col in range(greedy_policy.shape[1]):
                idx = greedy_policy[row, col]

                visual_board[row][col] = policy_visualization[idx]

        visual_board[self.env.terminal_state[0]][self.env.terminal_state[1]] = "F"
        pprint.pprint(visual_board)


    
    def TD_zero(self, epsilon=0.1, alpha=0.05, max_steps=1000, lamb=0.9):
        """
        Find the value function of move chess states
        :param epsilon: exploration rate
        :param alpha: learning rate
        :param max_steps: max amount of steps in an episode
        """
        state = (0, 0)
        self.env.state = state
        states = []
        actions = []
        rewards = []
        episode_end = False
        count_steps = 0
        while not episode_end:
            count_steps += 1
            states.append(state)
            action_index = self.agent.apply_policy(state, epsilon=epsilon)
            action = self.agent.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            suc_state = self.env.state
            self.agent.value_function[state[0], state[1]] = self.agent.value_function[state[0], state[1]] + alpha * (
                    reward + lamb * self.agent.value_function[suc_state[0], suc_state[1]] - self.agent.value_function[
                state[0], state[1]])
            state = self.env.state

            if count_steps > max_steps:
                episode_end = True