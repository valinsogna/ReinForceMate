import torch
import pprint


class Temporal_difference(object):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def expected_sarsa(self, n_episodes=1000, alpha=0.01, gamma=0.9):
        """
        Run the Expected SARSA control algorithm, finding the optimal policy and action function
        :param n_episodes: int, number of episodes to train
        :param alpha: learning rate
        :param gamma: discount factor of future rewards
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

                expected_action_value = torch.sum(
                    self.agent.action_function[successor_state[0], successor_state[1]] * self.agent.policy[
                        successor_state[0], successor_state[1]]
                )
                action_value = self.agent.action_function[state[0], state[1], action_index]

                q_update = alpha * (reward + gamma * expected_action_value - action_value)

                self.agent.action_function[state[0], state[1], action_index] += q_update.item()

                # Update the policy based on the updated action function
                self.agent.policy = torch.softmax(self.agent.action_function, dim=2)

    def visualize_policy(self):
        """
        Returns: None
        """
        greedy_policy = self.agent.policy.argmax(axis=2)
        policy_visualization = {}
        if self.agent.piece == 'king':
            arrows = "↑ ↗ → ↘ ↓ ↙ ← ↖"
            visual_row = [". ", ". ", ". ", ". ", ". ", ". ", ". ", ". "]
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
        for _ in range(8):
            visual_board.append(visual_row.copy())

        for row in range(greedy_policy.shape[0]):
            for col in range(greedy_policy.shape[1]):
                idx = greedy_policy[row, col].item()
                visual_board[row][col] = policy_visualization[idx]

        visual_board[self.env.terminal_state[0]][self.env.terminal_state[1]] = "F"
        pprint.pprint(visual_board)

    def visualize_action_function(self):
        print("Value function for this policy:")
        print(torch.max(self.agent.action_function, dim=2)[0].to(torch.int))

