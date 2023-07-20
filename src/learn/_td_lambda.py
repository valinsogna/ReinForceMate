import torch
import pprint
from ._base import BaseLearn
from ..config import Config as cfg
from typing import Tuple

class TemporalDifferenceLambda(BaseLearn):
    """
    Implements the SARSA(λ) algorithm to find the optimal move in a chess game.

    Args:
        piece (str): The piece to move. Can be any valid chess piece in FEN notation.
        FEN (str): The Forsyth–Edwards Notation string representation of the game state.
        kwargs: Any additional keyword arguments.
    """

    def __init__(self, **kwargs):
        """
        Initializes the TemporalDifferenceLambda class.

        Args:
            piece (str, optional): The piece to move. Can be any valid chess piece in FEN notation.
            FEN (str, optional): The Forsyth–Edwards Notation string representation of the game state.
            kwargs (dict): Any additional keyword arguments.
        """

        super().__init__(**kwargs)

    def sarsa_lambda(self, n_episodes=1000, alpha=0.05, gamma=0.9, lamb=0.8):
        """
        Executes the SARSA(λ) control algorithm to find the optimal policy and action function.

        Args:
            n_episodes (int, optional): The number of episodes to train. Defaults to 1000.
            alpha (float, optional): The learning rate. Defaults to 0.05.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.9.
            lamb (float, optional): The lambda parameter describing the decay over n-step returns. Defaults to 0.8.

        Returns:
            None. The optimal move chess policy is determined by this method.
        """

        for k in range(n_episodes):
            # Initialize eligibility traces
            self.agent.E = torch.zeros(self.agent.action_function.shape)  
            # Initialize the starting state
            state = (0, 0)  
            self.env.state = state
            episode_end = False
            # Calculate the epsilon value for epsilon-greedy policy
            epsilon = max(1 / (1 + k), 0.2)  
            # Select an action index using epsilon-greedy policy
            action_index = self.agent.apply_policy(state, epsilon)  
            # Convert the action index to an action
            action = self.agent.action_space[action_index]  

            # Run a single episode
            while not episode_end:
                # Take a step in the environment with the selected action
                reward, episode_end = self.env.step(action) 
                # Get the successor state 
                successor_state = self.env.state  
                # Select the next action index using epsilon-greedy policy
                successor_action_index = self.agent.apply_policy(successor_state, epsilon)  

                # Get the current action value
                action_value = self.agent.action_function[state[0], state[1], action_index] 
                if not episode_end:
                    # Get the action value for the successor state
                    successor_action_value = self.agent.action_function[successor_state[0], successor_state[1], successor_action_index]  
                else:
                    # If the episode ends, set the successor action value to 0
                    successor_action_value = 0  

                # Calculate the TD-error
                delta = reward + gamma * successor_action_value - action_value  
                # Update the eligibility trace
                self.agent.E[state[0], state[1], action_index] += 1  
                # Update the action function using the eligibility trace
                self.agent.action_function = self.agent.action_function + alpha * delta * self.agent.E 
                # Decay the eligibility trace 
                self.agent.E = gamma * lamb * self.agent.E
                # Update the state to the successor state  
                state = successor_state  
                # Update the action to the next action
                action = self.agent.action_space[successor_action_index]  
                # Update the action index to the next action index
                action_index = successor_action_index  
                # Update the policy based on the updated action function
                self.agent.policy = self.agent.action_function.clone()  

    def run_episode(self, episode_number, alpha=0.2, gamma=0.9, lamb=0.8):
        """
        Runs a single episode of the SARSA(λ) control algorithm to find the optimal policy and action function.

        Args:
            episode_number (int): The current episode number.
            alpha (float, optional): The learning rate. Defaults to 0.2.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.9.
            lamb (float, optional): The lambda parameter describing the decay over n-step returns. Defaults to 0.8.

        Returns:
            float: The total reward for the episode.
        """

        # Stores the rewards obtained at each step in the episode
        reward_step = []  
        # Initialize eligibility traces
        self.agent.E = torch.zeros(self.agent.action_function.shape) 
        # Initialize the starting state 
        state = (0, 0)  
        self.env.state = state
        episode_end = False
        # Calculate the epsilon value for epsilon-greedy policy
        epsilon = max(1 / (1 + episode_number), 0.2)  
        # Select an action index using epsilon-greedy policy
        action_index = self.agent.apply_policy(state, epsilon)
        # Convert the action index to an action 
        action = self.agent.action_space[action_index]  

        # Run a single episode
        while not episode_end:
            # Take a step in the environment with the selected action
            reward, episode_end = self.env.step(action) 
            # Store the reward obtained at the current step 
            reward_step.append(reward)  
            # Get the successor state
            successor_state = self.env.state  
            # Select the next action index using epsilon-greedy policy
            successor_action_index = self.agent.apply_policy(successor_state, epsilon)  

            # Get the current action value
            action_value = self.agent.action_function[state[0], state[1], action_index]  
            if not episode_end:
                # Get the action value for the successor state
                successor_action_value = self.agent.action_function[successor_state[0], successor_state[1], successor_action_index]  
            else:
                # If the episode ends, set the successor action value to 0
                successor_action_value = 0  

            # Calculate the TD-error
            delta = reward + gamma * successor_action_value - action_value 
            # Update the eligibility trace 
            self.agent.E[state[0], state[1], action_index] += 1 
            # Update the action function using the eligibility trace
            self.agent.action_function = self.agent.action_function + alpha * delta * self.agent.E
            # Decay the eligibility trace  
            self.agent.E = gamma * lamb * self.agent.E
            # Update the state to the successor state  
            state = successor_state  
            # Update the action to the next action
            action = self.agent.action_space[successor_action_index]
            # Update the action index to the next action index  
            action_index = successor_action_index  
            # Update the policy based on the updated action function
            self.agent.policy = self.agent.action_function.clone()  

        return sum(reward_step)  # Return the total reward for the episode

    def TD_lambda(self, epsilon=0.1, alpha=0.05, gamma=0.9, max_steps=1000, lamb=0.9):
        """
        Executes the TD(λ) algorithm to find the optimal policy and action function.

        Args:
            epsilon (float, optional): The probability of selecting a random action. Defaults to 0.1.
            alpha (float, optional): The learning rate. Defaults to 0.05.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.9.
            max_steps (int, optional): The maximum number of steps per episode. Defaults to 1000.
            lamb (float, optional): The lambda parameter describing the decay over n-step returns. Defaults to 0.9.

        Returns:
            None. The optimal move chess policy is determined by this method.
        """

        # Initialize the eligibility traces to zeros
        self.agent.E = torch.zeros(self.agent.value_function.shape)
        # Initialize the starting state  
        state = (0, 0)  
        self.env.state = state
        # List to store the visited states in the episode
        states = []
        # List to store the taken actions in the episode  
        actions = []  
        # List to store the obtained rewards in the episode
        rewards = []  
        # Counter for the number of steps in the episode
        count_steps = 0  
        episode_end = False

        # Run the TD(λ) algorithm
        while not episode_end:
            count_steps += 1
            # Append the current state to the list of visited states
            states.append(state)  
            # Select an action index using epsilon-greedy policy
            action_index = self.agent.apply_policy(state, epsilon=epsilon)
            # Convert the action index to an action  
            action = self.agent.action_space[action_index]  
            # Append the selected action to the list of taken actions
            actions.append(action)  
            # Take a step in the environment with the selected action, obtaining the reward and the episode end flag
            reward, episode_end = self.env.step(action)
            # Append the obtained reward to the list of rewards  
            rewards.append(reward)  
            # Get the successor state
            suc_state = self.env.state  
            # Calculate the TD-error
            delta = reward + lamb * self.agent.value_function[suc_state[0], suc_state[1]] - self.agent.value_function[state[0], state[1]]  
            # Update the eligibility trace for the current state
            self.agent.E[state[0], state[1]] += 1  

            # Update the action function and eligibility traces

            # Update the action function using the eligibility trace and the TD-error
            self.agent.value_function = self.agent.value_function + alpha * delta * self.E
            # Decay the eligibility trace  
            self.agent.E = gamma * lamb * self.agent.E  

            # Update the state to the successor state
            state = self.env.state  

            if count_steps > max_steps:
                # If the maximum number of steps is reached, end the episode
                episode_end = True  

        # Return None as the optimal policy and action function are determined by this method

