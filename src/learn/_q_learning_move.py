from ._base import BaseLearn
from ..config import Config as cfg
from typing import List

class Q_LearningMove(BaseLearn):
    """
    Implements the Q-learning algorithm (also known as SARSA-max) to find the optimal move in a chess game.

    """
    def __init__(self, **kwargs):
        """
        Initializes the Q_LearningMove class.

        Args:
            kwargs (dict): Any additional keyword arguments.
        """

        super().__init__(**kwargs)

    def q_learning(self, n_episodes=1000, alpha=0.05, gamma=0.9):
        """
        Executes the Q-learning algorithm to find the optimal policy and value function.

        Args:
            n_episodes (int, optional): The number of episodes to train. Defaults to 1000.
            alpha (float, optional): The learning rate. Defaults to 0.05.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.9.

        Returns:
            None. The optimal move chess policy is determined by this method.
        """

        for k in range(n_episodes):
            # Initialize the starting state
            state = (0, 0) 
            self.env.state = state
            episode_end = False
            # Calculate the epsilon value for epsilon-greedy policy
            epsilon = max(1 / (k + 1), 0.1)  

            # Run a single episode
            while not episode_end:
                # Select an action index using epsilon-greedy policy
                action_index = self.agent.apply_policy(state, epsilon)  
                # Convert the action index to an action
                action = self.agent.action_space[action_index]  
                # Take a step in the environment with the selected action
                reward, episode_end = self.env.step(action) 
                # Get the successor state 
                successor_state = self.env.state  
                # Apply a greedy policy for the successor state
                successor_action_index = self.agent.apply_policy(successor_state, -1)  

                # Get the current action value
                action_value = self.agent.action_function[state[0], state[1], action_index]  
                if not episode_end:
                    # Get the action value for the successor state
                    successor_action_value = self.agent.action_function[successor_state[0], successor_state[1], successor_action_index]  
                else:
                    # If the episode ends, set the successor action value to 0
                    successor_action_value = 0  

                # Update the action value based on the Q-learning update rule
                av_new = action_value + alpha * (reward + gamma * successor_action_value - action_value)
                self.agent.action_function[state[0], state[1], action_index] = av_new

                # Update the policy based on the updated action function
                self.agent.policy = self.agent.action_function.clone() 
                # Update the state to the successor state
                state = successor_state  



    def run_episode(self, episode_number, alpha=0.2, gamma=0.9):
        """
        Runs a single episode of the Q-learning algorithm to find the optimal policy and value function.

        Args:
            episode_number (int): The current episode number.
            alpha (float, optional): The learning rate. Defaults to 0.2.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.9.

        Returns:
            float: The total reward for the episode.
        """

        # Stores the rewards obtained at each step in the episode
        reward_step = []  
        # Initialize the starting state
        state = (0, 0)  
        self.env.state = state
        episode_end = False
        # Calculate the epsilon value for epsilon-greedy policy
        epsilon = max(1 / (episode_number + 1), 0.1)  

        # Run a single episode
        while not episode_end:
            # Select an action index using epsilon-greedy policy
            action_index = self.agent.apply_policy(state, epsilon)  
            # Convert the action index to an action
            action = self.agent.action_space[action_index] 
            # Take a step in the environment with the selected action 
            reward, episode_end = self.env.step(action) 
            # Store the reward obtained at the current step 
            reward_step.append(reward)  
            # Get the successor state
            successor_state = self.env.state  
            # Apply a greedy policy for the successor state
            successor_action_index = self.agent.apply_policy(successor_state, -1)  

            # Get the current action value
            action_value = self.agent.action_function[state[0], state[1], action_index]  
            if not episode_end:
                # Get the action value for the successor state
                successor_action_value = self.agent.action_function[successor_state[0], successor_state[1], successor_action_index]  
            else:
                # If the episode ends, set the successor action value to 0
                successor_action_value = 0  

            # Update the action value based on the Q-learning update rule
            av_new = action_value + alpha * (reward + gamma * successor_action_value - action_value)
            self.agent.action_function[state[0], state[1], action_index] = av_new

            # Update the policy based on the updated action function
            self.agent.policy = self.agent.action_function.clone()
            # Update the state to the successor state  
            state = successor_state 

        # Return the total reward for the episode
        return sum(reward_step)  

