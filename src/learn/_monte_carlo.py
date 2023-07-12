import numpy as np
import pprint


class Monte_carlo(object):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def monte_carlo_learning(self, epsilon=0.1):
        """
        Learn move chess through monte carlo control
        :param epsilon: exploration rate
        :return:
        """
        state = (0, 0)
        self.env.state = state

        # Play out an episode
        states, actions, rewards = self.play_episode(state, epsilon=epsilon)

        first_visits = []
        for idx, state in enumerate(states):
            action_index = actions[idx]
            if (state, action_index) in first_visits:
                continue
            r = np.sum(rewards[idx:])
            if (state, action_index) in self.agent.Returns.keys():
                self.agent.Returns[(state, action_index)].append(r)
            else:
                self.agent.Returns[(state, action_index)] = [r]
            self.agent.action_function[state[0], state[1], action_index] = \
                np.mean(self.agent.Returns[(state, action_index)])
            first_visits.append((state, action_index))
        # Update the policy. In Monte Carlo Control, this is greedy behavior with respect to the action function
        self.agent.policy = self.agent.action_function.copy()

    def monte_carlo_evaluation(self, epsilon=0.1, first_visit=True):
        """
        Find the value function of states using MC evaluation
        :param epsilon: exploration rate
        :param first_visit: Boolean, count only from first occurence of state
        :return:
        """
        state = (0, 0)
        self.env.state = state
        states, actions, rewards = self.play_episode(state, epsilon=epsilon)

        visited_states = set()
        for idx, state in enumerate(states):
            if state not in visited_states and first_visit:
                self.agent.N[state[0], state[1]] += 1
                n = self.agent.N[state[0], state[1]]
                forward_rewards = np.sum(rewards[idx:])
                expected_rewards = self.agent.value_function[state[0], state[1]]
                delta = forward_rewards - expected_rewards
                self.agent.value_function[state[0], state[1]] += ((1 / n) * delta)
                visited_states.add(state)
            elif not first_visit:
                self.agent.N[state[0], state[1]] += 1
                n = self.agent.N[state[0], state[1]]
                forward_rewards = np.sum(rewards[idx:])
                expected_rewards = self.agent.value_function[state[0], state[1]]
                delta = forward_rewards - expected_rewards
                self.agent.value_function[state[0], state[1]] += ((1 / n) * delta)
                visited_states.add(state)
            elif state in visited_states and first_visit:
                continue

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
        