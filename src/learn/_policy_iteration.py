import torch
import pprint
from ._base import BaseLearn
from ..config import Config as cfg
import itertools

class PolicyIteration(BaseLearn):
    """
    Class for implementing the Policy Iteration Reinforcement Learning algorithm.

    Args:
        kwargs: Arguments required by the parent BaseLearn class.
    """

    def __init__(self, **kwargs):
        """
        Initializes the PolicyIteration class.

        Args:
            kwargs (dict): Arguments needed for the parent BaseLearn class.
        """

        super().__init__(**kwargs)
        self.cumulative_rewards = []
        
    # Calculates the value of a state based on the successor states and the immediate rewards
    def evaluate_state(self, state, gamma=0.9, synchronous=True):
        """
        Calculates the value of a state based on the successor states and the immediate rewards.

        Args:
            state (tuple): Tuple of 2 integers 0-7 representing the state.
            gamma (float): Discount factor. Defaults to 0.9.
            synchronous (bool): If True, performs synchronous updates. Defaults to True.

        Returns:
            float: The expected value of the state under the current policy.
        """

        # Find the greedy action value and all corresponding greedy actions
        greedy_action_value = torch.max(self.agent.policy[state[0], state[1], :])
        greedy_indices = [i for i, a in enumerate(self.agent.policy[state[0], state[1], :]) if
                        a == greedy_action_value]  # List of all greedy actions
        prob = 1 / len(greedy_indices)  # probability of an action occurring
        state_value = 0

        for i in greedy_indices:
            self.env.state = state  # Reset state to the one being evaluated
            reward, episode_end = self.env.step(self.agent.action_space[i])

            if synchronous:
                successor_state_value = self.agent.value_function_prev[self.env.state]
            else:
                successor_state_value = self.agent.value_function[self.env.state]
            
            state_value += (prob * (reward + gamma * successor_state_value))  # Sum up rewards and discounted successor state value

        return state_value

    def evaluate_policy(self, gamma=0.9, synchronous=True):
        """
        Evaluates the current policy over all states.

        Args:
            gamma (float): Discount factor. Defaults to 0.9.
            synchronous (bool): If True, performs synchronous updates. Defaults to True.
        """

        self.agent.value_function_prev = self.agent.value_function.clone()  # For synchronous updates
        for row in range(self.agent.value_function.shape[0]):
            for col in range(self.agent.value_function.shape[1]):
                self.agent.value_function[row, col] = self.evaluate_state((row, col), gamma=gamma,
                                                                          synchronous=synchronous)

    def improve_policy(self):
        """
        Finds the greedy policy with respect to the current value function.

        Returns:
            float: The total reward under the updated policy.
        """

        rewards = 0

        self.agent.policy_prev = self.agent.policy.clone()  # Store the previous policy for comparison

        # Iterate over each state in the action function
        for row in range(self.agent.action_function.shape[0]):
            for col in range(self.agent.action_function.shape[1]):

                # Iterate over each action in the action function for the current state
                for action in range(self.agent.action_function.shape[2]):
                    self.env.state = (row, col)  # Reset state to the one being evaluated
                    reward, episode_end = self.env.step(self.agent.action_space[action])
                    rewards += reward  # Accumulate the rewards obtained

                    # Calculate the value of the successor state
                    successor_state_value = 0 if episode_end else self.agent.value_function[self.env.state]

                    # Update the policy with the sum of the immediate reward and the successor state value
                    self.agent.policy[row, col, action] = reward + successor_state_value

                # Find the maximum policy value for the current state
                max_policy_value = torch.max(self.agent.policy[row, col, :])

                # Find the indices of all actions with the maximum policy value
                max_indices = [i for i, a in enumerate(self.agent.policy[row, col, :]) if a == max_policy_value]

                # Set the policy value of the maximum indices to 1, making them the greedy actions
                for idx in max_indices:
                    self.agent.policy[row, col, idx] = 1

        return rewards  # Return the total reward obtained under the updated policy

    def policy_iteration(self, eps=0.1, gamma=0.9, iteration=1, k=32, synchronous=True):
        """
        Performs the policy iteration process to find the optimal policy.

        Args:
            eps (float): Threshold for policy evaluation. Defaults to 0.1.
            gamma (float): Discount factor. Defaults to 0.9.
            iteration (int): Current iteration number. Defaults to 1.
            k (int): Maximum number of policy evaluation iterations. Defaults to 32.
            synchronous (bool): If True, performs synchronous updates. Defaults to True.
        """

        policy_stable = True  # Flag to indicate if the policy is stable

        print("")
        value_delta_max = 0

        # Policy Evaluation: Update the value function using the current policy
        for _ in range(k):
            self.evaluate_policy(gamma=gamma, synchronous=synchronous)  # Evaluate the current policy
            value_delta = torch.max(torch.abs(self.agent.value_function_prev - self.agent.value_function))
            value_delta_max = value_delta
            if value_delta_max < eps:
                break

        action_function_prev = self.agent.action_function.clone()

        # Policy Improvement: Update the policy based on the updated value function
        self.improve_policy()

        # Check if the policy has changed
        policy_stable = self.agent.compare_policies() < 1

        if not policy_stable and iteration < 1000:
            # If the policy is not stable and the maximum iteration limit is not reached, continue policy iteration
            iteration += 1
            self.policy_iteration(iteration=iteration)
        elif policy_stable:
            # If the policy is stable, the optimal policy is found
            print("Optimal policy found in", iteration, "steps of policy evaluation")
        else:
            # If the policy does not converge within the maximum number of iterations, print a failure message
            print("failed to converge.")

    def calculate_cumulative_rewards(self, num_episodes=1):
        """
        Calculates the cumulative rewards over a number of episodes.

        Args:
            num_episodes (int): Number of episodes to consider. Defaults to 1.

        Returns:
            list: Cumulative rewards over the episodes.
        """

        for k in range(num_episodes):
            self.env.reset()
            episode_reward = 0
            episode_end = False
            iteration = 0

            # Run a single episode
            while not episode_end:
                iteration += 1
                state = self.env.state
                action_index = self.agent.apply_policy(state, epsilon=0)  # No exploration (greedy)
                action = self.agent.action_space[action_index]
                reward, episode_end = self.env.step(action)
                episode_reward += reward

            self.cumulative_rewards.append(episode_reward)  # Store the cumulative reward for the episode

        return self.cumulative_rewards

    def run_episode(self, eps=0.1, gamma=0.9, iteration=1, k=32, synchronous=True):
        """
        Runs the policy iteration for a single episode and calculates cumulative rewards.

        Args:
            eps (float): Threshold for policy evaluation. Defaults to 0.1.
            gamma (float): Discount factor. Defaults to 0.9.
            iteration (int): Current iteration number. Defaults to 1.
            k (int): Maximum number of policy evaluation iterations. Defaults to 32.
            synchronous (bool): If True, performs synchronous updates. Defaults to True.

        Returns:
            list: Cumulative rewards over the episode.
        """

        self.evaluate_policy(gamma=gamma,synchronous=synchronous)
        self.improve_policy()
        return self.calculate_cumulative_rewards()