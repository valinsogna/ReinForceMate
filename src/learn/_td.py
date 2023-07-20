import torch
from ._base import BaseLearn
from ..config import Config as cfg
import pprint

class TemporalDifference(BaseLearn):
    """
    Implements the SARSA TD(0) algorithm to find the optimal move in a chess game.

    Args:
        kwargs: Any additional keyword arguments.
    """

    def __init__(self, **kwargs):
        """
        Initializes the TemporalDifference class.

        Args:
            kwargs (dict): Any additional keyword arguments.
        """

        super().__init__(**kwargs)
    
    def sarsa_td(self, n_episodes=1000, alpha=0.01, gamma=0.9):
        """
        Executes the SARSA TD(0) control algorithm to find the optimal policy and action function.

        Args:
            n_episodes (int, optional): The number of episodes to train. Defaults to 1000.
            alpha (float, optional): The learning rate. Defaults to 0.01.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.9.

        Returns:
            list: A list of cumulative rewards for each episode.
        """

        # List to store cumulative rewards for each episode
        cumulative_reward = []  

        # Run SARSA TD(0) algorithm for the specified number of episodes
        for k in range(n_episodes):
            # Initialize the starting state
            state = (0, 0)  
            # Reset the environment to the starting state
            self.env.state = state  
            episode_end = False
            # Calculate the exploration rate (epsilon)
            epsilon = max(1 / (1 + k), 0.05)  
            # List to store rewards obtained in the current episode
            reward_step = []  

            # Run a single episode until episode termination
            while not episode_end:
                # Get the current state
                state = self.env.state  
                # Select an action index using epsilon-greedy policy
                action_index = self.agent.apply_policy(state, epsilon) 
                # Convert the action index to an action 
                action = self.agent.action_space[action_index]  

                # Take a step in the environment with the selected action, obtaining the reward and the episode end flag
                reward, episode_end = self.env.step(action) 
                # Append the obtained reward to the list of rewards 
                reward_step.append(reward)  

                # Get the successor state
                successor_state = self.env.state  
                # Select an action index for the successor state
                successor_action_index = self.agent.apply_policy(successor_state, epsilon)  

                # Get the action value of the current state-action pair
                action_value = self.agent.action_function[state[0], state[1], action_index]  
                # Get the action value of the successor state-action pair
                successor_action_value = self.agent.action_function[successor_state[0], successor_state[1], successor_action_index]  

                # Calculate the TD update
                q_update = alpha * (reward + gamma * successor_action_value - action_value)  

                # Update the action value using the TD update
                self.agent.action_function[state[0], state[1], action_index] += q_update.item() 
                # Update the policy based on the updated action function 
                self.agent.policy = self.agent.action_function.clone()  

            # Append the cumulative reward of the episode to the list
            cumulative_reward.append(sum(reward_step))  

        # Return the list of cumulative rewards for each episode
        return cumulative_reward  
    
    def run_episode(self, episode_number, alpha=0.2, gamma=0.9):
        """
        Runs a single episode of the SARSA TD(0) control algorithm to find the optimal policy and action function.

        Args:
            episode_number (int): The current episode number.
            alpha (float, optional): The learning rate. Defaults to 0.2.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.9.

        Returns:
            float: The total reward for the episode.
        """

        # Initialize the starting state
        state = (0, 0)  
        # Reset the environment to the starting state
        self.env.state = state  
        episode_end = False
        # Calculate the exploration rate (epsilon)
        epsilon = max(1 / (1 + episode_number), 0.05)
        # List to store rewards obtained in the current episode  
        reward_step = []  

        # Run a single episode until episode termination
        while not episode_end:
            # Get the current state
            state = self.env.state  
            # Select an action index using epsilon-greedy policy
            action_index = self.agent.apply_policy(state, epsilon)  
            # Convert the action index to an action
            action = self.agent.action_space[action_index]  
            
            # Take a step in the environment with the selected action, obtaining the reward and the episode end flag
            reward, episode_end = self.env.step(action)
            # Append the obtained reward to the list of rewards  
            reward_step.append(reward)  

            # Get the successor state
            successor_state = self.env.state  
            # Select an action index for the successor state
            successor_action_index = self.agent.apply_policy(successor_state, epsilon)  

            # Get the action value of the current state-action pair
            action_value = self.agent.action_function[state[0], state[1], action_index]  
            # Get the action value of the successor state-action pair
            successor_action_value = self.agent.action_function[successor_state[0], successor_state[1], successor_action_index]  
            
            # Calculate the TD update
            q_update = alpha * (reward + gamma * successor_action_value - action_value)  
            
            # Update the action value using the TD update
            self.agent.action_function[state[0], state[1], action_index] += q_update.item()
            # Update the policy based on the updated action function  
            self.agent.policy = self.agent.action_function.clone()  
        
        return sum(reward_step)  # Return the total reward for the episode
            
    def TD_zero(self, epsilon=0.1, alpha=0.05, max_steps=1000, lamb=0.9):
        """
        Executes the TD(0) algorithm to find the value function of chess move states.

        Args:
            epsilon (float, optional): The probability of selecting a random action. Defaults to 0.1.
            alpha (float, optional): The learning rate. Defaults to 0.05.
            max_steps (int, optional): The maximum number of steps per episode. Defaults to 1000.
            lamb (float, optional): The lambda parameter describing the decay over n-step returns. Defaults to 0.9.

        Returns:
            None. The optimal move chess policy is determined by this method. Prints the number of steps taken.
        """

        # Initialize the starting state
        state = (0, 0)  
        # Reset the environment to the starting state
        self.env.state = state  
        # List to store visited states
        states = []  
        # List to store taken actions
        actions = []  
        episode_end = False
        # Counter to keep track of the number of steps taken in the episode
        count_steps = 0  

        # Run the TD(0) algorithm until episode termination or maximum steps reached
        while not episode_end:
            # Increment the step counter
            count_steps += 1  
            # Append the current state to the list of visited states
            states.append(state)  
            # Select an action index using epsilon-greedy policy
            action_index = self.agent.apply_policy(state, epsilon=epsilon)  
            # Convert the action index to an action
            action = self.agent.action_space[action_index]
            # Append the taken action to the list of taken actions  
            actions.append(action)  
            # Take a step in the environment with the selected action, obtaining the reward and the episode end flag
            reward, episode_end = self.env.step(action) 
            # Get the successor state
            suc_state = self.env.state  

            # Perform the TD(0) update on the value function for the current state
            self.agent.value_function[state[0], state[1]] = self.agent.value_function[state[0], state[1]] + alpha * (
                    reward + lamb * self.agent.value_function[suc_state[0], suc_state[1]] - self.agent.value_function[state[0], state[1]])

            # Update the current state with the successor state
            state = self.env.state  

            if count_steps > max_steps:
                # Terminate the episode if the maximum steps allowed is reached
                episode_end = True  

        # Print the number of steps taken in the episode
        print(count_steps) 
