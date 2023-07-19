import torch
import pprint


class Temporal_difference_lambda(object):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def sarsa_lambda(self, n_episodes=1000, alpha=0.05, gamma=0.9, lamb=0.8):
        """
        Run the sarsa control algorithm (TD lambda), finding the optimal policy and action function
        :param n_episodes: int, amount of episodes to train
        :param alpha: learning rate
        :param gamma: discount factor of future rewards
        :param lamb: lambda parameter describing the decay over n-step returns
        :return: finds the optimal move chess policy
        """
        for k in range(n_episodes):
            self.agent.E = torch.zeros(self.agent.action_function.shape)
            state = (0, 0)
            self.env.state = state
            episode_end = False
            epsilon = max(1 / (1 + k), 0.2)
            action_index = self.agent.apply_policy(state, epsilon)
            action = self.agent.action_space[action_index]
            while not episode_end:
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.agent.apply_policy(successor_state, epsilon)

                action_value = self.agent.action_function[state[0], state[1], action_index]
                if not episode_end:
                    successor_action_value = self.agent.action_function[successor_state[0],
                                                                        successor_state[1], successor_action_index]
                else:
                    successor_action_value = 0
                delta = reward + gamma * successor_action_value - action_value
                self.agent.E[state[0], state[1], action_index] += 1
                self.agent.action_function = self.agent.action_function + alpha * delta * self.agent.E
                self.agent.E = gamma * lamb * self.agent.E
                state = successor_state
                action = self.agent.action_space[successor_action_index]
                action_index = successor_action_index
                self.agent.policy = self.agent.action_function.clone()

    def run_episode(self, episode_number, alpha=0.2, gamma=0.9, lamb=0.8):
        """
        Run the sarsa control algorithm (TD lambda), finding the optimal policy and action function
        :param n_episodes: int, amount of episodes to train
        :param alpha: learning rate
        :param gamma: discount factor of future rewards
        :param lamb: lambda parameter describing the decay over n-step returns
        :return: finds the optimal move chess policy
        """
        reward_step = []
        self.agent.E = torch.zeros(self.agent.action_function.shape)
        state = (0, 0)
        self.env.state = state
        episode_end = False
        epsilon = max(1 / (1 + episode_number), 0.2)
        action_index = self.agent.apply_policy(state, epsilon)
        action = self.agent.action_space[action_index]
        while not episode_end:
            reward, episode_end = self.env.step(action)
            reward_step.append(reward)
            successor_state = self.env.state
            successor_action_index = self.agent.apply_policy(successor_state, epsilon)

            action_value = self.agent.action_function[state[0], state[1], action_index]
            if not episode_end:
                successor_action_value = self.agent.action_function[successor_state[0],
                                                                    successor_state[1], successor_action_index]
            else:
                successor_action_value = 0
            delta = reward + gamma * successor_action_value - action_value
            self.agent.E[state[0], state[1], action_index] += 1
            self.agent.action_function = self.agent.action_function + alpha * delta * self.agent.E
            self.agent.E = gamma * lamb * self.agent.E
            state = successor_state
            action = self.agent.action_space[successor_action_index]
            action_index = successor_action_index
            self.agent.policy = self.agent.action_function.clone()

        return sum(reward_step)
    
    

    def visualize_policy(self):
        """
        Gives you are very ugly visualization of the policy
        Returns: None

        """
        greedy_policy = self.agent.policy.argmax(axis=2)
        # print(greedy_policy)
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
                idx = greedy_policy[row, col].item()

                visual_board[row][col] = policy_visualization[idx]

        visual_board[self.env.terminal_state[0]][self.env.terminal_state[1]] = "F"
        pprint.pprint(visual_board)

    def visualize_action_function(self):
        print(torch.max(self.agent.action_function, dim=2)[0].to(torch.int))

    def TD_lambda(self, epsilon=0.1, alpha=0.05, gamma=0.9, max_steps=1000, lamb=0.9):
        self.agent.E = torch.zeros(self.agent.value_function.shape)
        state = (0, 0)
        self.env.state = state
        states = []
        actions = []
        rewards = []
        count_steps = 0
        episode_end = False
        while not episode_end:
            count_steps += 1
            states.append(state)
            action_index = self.agent.apply_policy(state, epsilon=epsilon)
            action = self.agent.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            suc_state = self.env.state
            delta = reward + lamb * self.agent.value_function[suc_state[0], suc_state[1]] - self.agent.value_function[
                state[0],
                state[1]]
            self.agent.E[state[0], state[1]] += 1

            # Note to self: vectorize code below.
            self.agent.value_function = self.agent.value_function + alpha * delta * self.E
            self.agent.E = gamma * lamb * self.agent.E
            state = self.env.state

            if count_steps > max_steps:
                episode_end = True