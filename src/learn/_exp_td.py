import torch
import pprint
from ._base import BaseLearn
from ..config import Config as cfg
from typing import Tuple

class ExpectedTemporalDifference(BaseLearn):
    """
    ExpectedTemporalDifference is a subclass of BaseLearn.
    It uses the Expected SARSA control algorithm for reinforcement learning.

    The Expected SARSA control algorithm learns the optimal policy and the action-value function
    based on the expectation over possible next actions, as opposed to Q-learning which uses the maximum.

    """

    def __init__(self, **kwargs):
        """
        Inherits the parent class (BaseLearn) and initializes the ExpectedTemporalDifference class.
        """
        super().__init__(**kwargs)

    def expected_sarsa(self, n_episodes=1000, alpha=0.01, gamma=0.9):
        """
        Run the Expected SARSA control algorithm, finding the optimal policy and action function.

        Args:
            n_episodes (int): Number of episodes to train.
            alpha (float): The learning rate.
            gamma (float): The discount factor of future rewards.

        """
        # For each episode until the maximum number of episodes
        for k in range(n_episodes):
            # Initialize the state and the environment state
            state = (0, 0)
            self.env.state = state
            
            # Initialize the episode end flag
            episode_end = False
            
            # Set the epsilon value, which determines the balance between exploration and exploitation
            epsilon = max(1 / (1 + k), 0.05)

            # Until the episode ends
            while not episode_end:
                # Get the current state
                state = self.env.state
                
                # Get the action index according to the current policy
                action_index = self.agent.apply_policy(state, epsilon)
                
                # Get the action corresponding to the action index
                action = self.agent.action_space[action_index]
                
                # Perform the action and receive a reward and a flag indicating whether the episode has ended
                reward, episode_end = self.env.step(action)
                
                # Get the new state after performing the action
                successor_state = self.env.state
                
                # Calculate the expected action value
                expected_action_value = torch.sum(
                    self.agent.action_function[successor_state[0], successor_state[1]] * self.agent.policy[
                        successor_state[0], successor_state[1]]
                )
                
                # Get the current action value
                action_value = self.agent.action_function[state[0], state[1], action_index]
                
                # Calculate the update for the action-value function
                q_update = alpha * (reward + gamma * expected_action_value - action_value)
                
                # Update the action-value function
                self.agent.action_function[state[0], state[1], action_index] += q_update.item()
                
                # Update the policy based on the updated action-value function
                self.agent.policy = torch.softmax(self.agent.action_function, dim=2)

    def run_episode(self, episode_number, alpha=0.2, gamma=0.9):
        """
        Run the Expected SARSA control algorithm, finding the optimal policy and action function.

        Args:
            episode_number (int): The current episode number.
            alpha (float): The learning rate.
            gamma (float): The discount factor of future rewards.

        Returns:
            float: The sum of rewards for the episode.

        """

        reward_step = []  # Initialize list to keep track of rewards at each step in the episode

        state = (0, 0)  # Initialize state
        self.env.state = state  # Set the state of the environment

        episode_end = False  # Initialize flag to track if the episode has ended

        # Set exploration-exploitation tradeoff value, decreases over time to a minimum of 0.05
        epsilon = max(1 / (1 + episode_number), 0.05) 

        # Run until the episode ends
        while not episode_end:

            # Get the current state
            state = self.env.state

            # Determine the action to take based on the current policy
            action_index = self.agent.apply_policy(state, epsilon)

            # Get the action corresponding to the chosen index
            action = self.agent.action_space[action_index]

            # Perform the action in the environment, get reward and check if episode ended
            reward, episode_end = self.env.step(action)

            # Store the reward
            reward_step.append(reward)

            # Get the state resulting from the action
            successor_state = self.env.state

            # Calculate the expected value of the successor state using the action function and policy
            expected_action_value = torch.sum(
                self.agent.action_function[successor_state[0], successor_state[1]] * self.agent.policy[
                    successor_state[0], successor_state[1]]
            )

            # Get the value of the current state-action pair
            action_value = self.agent.action_function[state[0], state[1], action_index]

            # Compute the update for the action function
            q_update = alpha * (reward + gamma * expected_action_value - action_value)

            # Update the action function
            self.agent.action_function[state[0], state[1], action_index] += q_update.item()

            # Update the policy based on the updated action function
            self.agent.policy = torch.softmax(self.agent.action_function, dim=2)

        # Return the total reward for the episode
        return sum(reward_step)

