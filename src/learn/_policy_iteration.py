import torch
import pprint


class Policy_iteration(object):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        
    def evaluate_state(self, state, gamma=0.9, synchronous=True):
        """
        Calculates the value of a state based on the successor states and the immediate rewards.
        Args:
            state: tuple of 2 integers 0-7 representing the state
            gamma: float, discount factor
            synchronous: Boolean

        Returns: The expected value of the state under the current policy.

        """
        greedy_action_value = torch.max(self.agent.policy[state[0], state[1], :])
        greedy_indices = [i for i, a in enumerate(self.agent.policy[state[0], state[1], :]) if
                          a == greedy_action_value]  # List of all greedy actions
        prob = 1 / len(greedy_indices)  # probability of an action occuring
        state_value = 0
        for i in greedy_indices:
            self.env.state = state  # reset state to the one being evaluated
            reward, episode_end = self.env.step(self.agent.action_space[i])
            if synchronous:
                successor_state_value = self.agent.value_function_prev[self.env.state]
            else:
                successor_state_value = self.agent.value_function[self.env.state]
            state_value += (prob * (
                    reward + gamma * successor_state_value))  # sum up rewards and discounted successor state value
        return state_value

    def evaluate_policy(self, gamma=0.9, synchronous=True):
        self.agent.value_function_prev = self.agent.value_function.clone()  # For synchronous updates
        for row in range(self.agent.value_function.shape[0]):
            for col in range(self.agent.value_function.shape[1]):
                self.agent.value_function[row, col] = self.evaluate_state((row, col), gamma=gamma,
                                                                          synchronous=synchronous)

    def improve_policy(self):
        """
        Finds the greedy policy w.r.t. the current value function
        """

        self.agent.policy_prev = self.agent.policy.clone()
        for row in range(self.agent.action_function.shape[0]):
            for col in range(self.agent.action_function.shape[1]):
                for action in range(self.agent.action_function.shape[2]):
                    self.env.state = (row, col)  # reset state to the one being evaluated
                    reward, episode_end = self.env.step(self.agent.action_space[action])
                    successor_state_value = 0 if episode_end else self.agent.value_function[self.env.state]
                    self.agent.policy[row, col, action] = reward + successor_state_value

                max_policy_value = torch.max(self.agent.policy[row, col, :])
                max_indices = [i for i, a in enumerate(self.agent.policy[row, col, :]) if a == max_policy_value]
                for idx in max_indices:
                    self.agent.policy[row, col, idx] = 1

    def policy_iteration(self, eps=0.1, gamma=0.9, iteration=1, k=32, synchronous=True):
        """
        Finds the optimal policy
        Args:
            eps: float, exploration rate
            gamma: float, discount factor
            iteration: the iteration number
            k: (int) maximum amount of policy evaluation iterations
            synchronous: (Boolean) whether to use synchronous are asynchronous back-ups 

        Returns:

        """
        policy_stable = True
        # print("\n\n______iteration:", iteration, "______")
        # print("\n policy:")
        # self.visualize_policy()

        print("")
        value_delta_max = 0
        for _ in range(k):
            self.evaluate_policy(gamma=gamma, synchronous=synchronous)
            value_delta = torch.max(torch.abs(self.agent.value_function_prev - self.agent.value_function))
            value_delta_max = value_delta
            if value_delta_max < eps:
                break
        print("Value function for this policy:")
        print(torch.round(self.agent.value_function).to(torch.int))
        action_function_prev = self.agent.action_function.clone()
        # print("\n Improving policy:")
        self.improve_policy()
        policy_stable = self.agent.compare_policies() < 1
        # print("policy diff:", policy_stable)

        if not policy_stable and iteration < 1000:
            iteration += 1
            self.policy_iteration(iteration=iteration)
        elif policy_stable:
            print("Optimal policy found in", iteration, "steps of policy evaluation")
        else:
            print("failed to converge.")

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
                idx = greedy_policy[row, col].item()

                visual_board[row][col] = policy_visualization[idx]

        visual_board[self.env.terminal_state[0]][self.env.terminal_state[1]] = "F"
        pprint.pprint(visual_board)
