import torch


class GridWorld:
    """
    A class representing a grid world environment, commonly used in reinforcement learning. 
    It consists of a 2D grid (8x8) with a starting state (0,0) and a terminal state (4,6).
    
    Attributes:
        state (tuple): A tuple representing the current state in the form (row, col).
        reward_space (torch.Tensor): A PyTorch tensor representing the reward for each state in the grid world.
        terminal_state (tuple): The terminal state of the grid world. An episode ends when this state is reached.
        visual_board (list): A list of lists used to visually represent the grid world.
    """

    def __init__(self):
        """
        Initializes the grid world with the starting state (0,0), and the terminal state (4,6).
        The reward for all states is initialized to -1.
        """

        self.state = (0, 0)
        self.reward_space = torch.zeros(((8, 8))) - 1
        self.terminal_state = (4,6)#(7, 5)

    def step(self, action):
        """
        Performs an action in the grid world which results in moving to a new state and receiving a reward.

        Args:
            action (tuple): The action to be performed. It is a tuple indicating the movement in the form (row_change, col_change).

        Returns:
            reward (int): The reward for the performed action.
            episode_end (bool): A flag indicating whether the episode has ended (True) or not (False).
        """

        reward = self.reward_space[self.state[0], self.state[1]]
        if self.state == self.terminal_state:
            episode_end = True
            return 0, episode_end
        else:
            episode_end = False
            old_state = self.state
            new_state = (self.state[0] + action[0], self.state[1] + action[1])  # step
            self.state = old_state if min(new_state) < 0 or max(new_state) > 7 else new_state
            return reward, episode_end

    def render(self):
        """
        Updates the visual representation of the grid world.
        The current state is marked by '[S]' and the terminal state is marked by '[F]'.
        """

        visual_row = [" ", " ", " ", " ", " ", " ", " ", " "]
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())
        visual_board[self.state[0]][self.state[1]] = "[S]"
        visual_board[self.terminal_state[0]][self.terminal_state[1]] = "[F]"
        self.visual_board = visual_board


    def reset(self):
        """
        Resets the grid world to its initial state (0,0).

        Returns:
            state (tuple): The initial state of the grid world.
        """

        self.state = (0, 0)
        return self.state
    
    def get_state(self):
        """
        Returns the current state of the grid world.

        Returns:
            state (tuple): The current state in the grid world.
        """
        
        return self.state


