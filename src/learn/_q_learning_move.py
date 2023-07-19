import torch
import pprint


class Q_learning_move(object):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def q_learning(self, n_episodes=1000, alpha=0.05, gamma=0.9):
        """
        Run Q-learning (also known as sarsa-max, finding the optimal policy and value function
        :param n_episodes: int, amount of episodes to train
        :param alpha: learning rate
        :param gamma: discount factor of future rewards
        :return: finds the optimal move chess policy
        """
        for k in range(n_episodes):
            state = (0, 0)
            self.env.state = state
            episode_end = False
            epsilon = max(1 / (k + 1), 0.1)
            while not episode_end:
                action_index = self.agent.apply_policy(state, epsilon)
                action = self.agent.action_space[action_index]
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.agent.apply_policy(successor_state, -1)

                action_value = self.agent.action_function[state[0], state[1], action_index]
                if not episode_end:
                    successor_action_value = self.agent.action_function[successor_state[0],
                                                                        successor_state[1], successor_action_index]
                else:
                    successor_action_value = 0

                av_new = self.agent.action_function[state[0], state[1], action_index] + alpha * (reward +
                                                                                                 gamma *
                                                                                                 successor_action_value
                                                                                                 - action_value)
                self.agent.action_function[state[0], state[1], action_index] = av_new
                self.agent.policy = self.agent.action_function.clone()
                state = successor_state


    """
    decay epsilon
    Decaying Epsilon (Epsilon-Greedy with Decay): 
    Another approach is to start with a higher exploration rate 
    (e.g., 1.0, meaning the agent always explores) and gradually 
    decrease it over time as the agent becomes more experienced. 
    This is based on the intuition that early in training, the agent 
    should explore more to discover better actions, but as it learns, 
    it should exploit more to take advantage of its knowledge. 
    
    Common decay strategies include linear or exponential decay. 
    For example, you could decrease epsilon by a small amount after each 
    episode or after a certain number of time steps.


    """
    def run_episode(self, episode_number, alpha=0.05, gamma=0.9):
        """
        Run Q-learning (also known as sarsa-max, finding the optimal policy and value function
        :param n_episodes: int, amount of episodes to train
        :param alpha: learning rate
        :param gamma: discount factor of future rewards
        :return: finds the optimal move chess policy
        """
        reward_step = []
        state = (0, 0)
        self.env.state = state
        episode_end = False
        epsilon = max(1 / (episode_number + 1), 0.1)
        while not episode_end:
            action_index = self.agent.apply_policy(state, epsilon)
            action = self.agent.action_space[action_index]
            reward, episode_end = self.env.step(action)
            reward_step.append(reward)
            successor_state = self.env.state
            successor_action_index = self.agent.apply_policy(successor_state, -1)

            action_value = self.agent.action_function[state[0], state[1], action_index]
            if not episode_end:
                successor_action_value = self.agent.action_function[successor_state[0],
                                                                    successor_state[1], successor_action_index]
            else:
                successor_action_value = 0

            av_new = self.agent.action_function[state[0], state[1], action_index] + alpha * (reward +
                                                                                                gamma *
                                                                                                successor_action_value
                                                                                                - action_value)
            self.agent.action_function[state[0], state[1], action_index] = av_new
            self.agent.policy = self.agent.action_function.clone()
            state = successor_state

        return sum(reward_step)



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
        

    def visualize_action_function(self):
        # print(self.agent.action_function.max(axis=2).round().astype(int))
        print(torch.max(self.agent.action_function, dim=2)[0].to(torch.int))