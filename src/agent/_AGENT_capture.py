"""
The agent is no longer a single piece, it's a chess player
Its action space consist of 64x64=4096 actions:
- There are 8x8 = 64 piece from where a piece can be picked up
- And another 64 pieces from where a piece can be dropped.
Of course, only certain actions are legal.
Which actions are legal in a certain state is part of the environment (in RL, anything outside 
the control of the agent is considered part of the environment). 
We can use the python-chess package to select legal moves.
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def policy_gradient_loss(Returns):
    def modified_crossentropy(action, action_probs):
        cost = (torch.nn.functional.cross_entropy(action_probs, action, reduction='none') * Returns)
        return torch.mean(cost)

    return modified_crossentropy

class Agent(nn.Module):

    def __init__(self, network='linear', gamma=0.99, lr=0.001, verbose=True):
        """
        Agent class for capture chess
        Args:
            network: str
                type of network architecture ('linear' or 'conv')
            gamma: float
                discount factor for future rewards
            lr: float
                learning rate for the optimizer
            verbose: bool
                whether to print debug information
        """
        super(Agent, self).__init__()
        self.network = network
        self.gamma = gamma
        self.lr = lr
        self.verbose = verbose
        self.init_network()
        self.weight_memory = []
        self.long_term_mean = []

    def init_network(self):
        """
        Initialize the neural network based on the specified network architecture
        """
        if self.network == 'linear':
            self.model = nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 4096)
            )
        elif self.network == 'conv':
            self.model = nn.Sequential(
                nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 4096)
            )
        elif self.network == 'conv_pg':
            self.model = nn.Sequential(
                nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 4096),
                nn.Softmax(dim=1)
            )
        else:
            raise ValueError("Invalid network type specified.")

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.0, weight_decay=0.0, nesterov=False)

    def forward(self, x):
        """
        Forward pass of the neural network
        Args:
            x: torch.Tensor
                input tensor to the network
        Returns:
            torch.Tensor
                output tensor from the network
        """
        return self.model(x)

    def get_action_values(self, state):
        """
        Get the action values (Q-values) for a given state
        Args:
            state: torch.Tensor
                input state tensor
        Returns:
            torch.Tensor
                action values (Q-values) for the state
        """
        with torch.no_grad():
            return self.forward(state)

    def network_update(self, minibatch):
        """
        Perform a network update (i.e., backpropagation) based on a minibatch of experiences
        Args:
            minibatch: list
                minibatch of experiences [(state, action, reward, next_state), ...]
        Returns:
            torch.Tensor
                TD errors for the minibatch
        """
        states, actions, rewards, next_states = zip(*minibatch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards).float()
        next_states = torch.stack(next_states)

        current_action_values = self.get_action_values(states)
        next_action_values = self.get_action_values(next_states).max(dim=1).values

        targets = rewards + self.gamma * next_action_values
        td_errors = targets - current_action_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        loss = nn.MSELoss()(current_action_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(), targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_errors

    def fix_model(self):
        """
        Create a fixed model for bootstrapping
        """
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.0, weight_decay=0.0, nesterov=False)
        self.fixed_model = copy.deepcopy(self)
        self.fixed_model.optimizer = optimizer

    # def fix_model(self):
    #     """
    #     Create a fixed model for bootstrapping
    #     """
    #     optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.0, weight_decay=0.0, nesterov=False)
    #     self.fixed_model = Agent(network=self.network, gamma=self.gamma, lr=self.lr, verbose=self.verbose)
    #     self.fixed_model.load_state_dict(self.state_dict())
    #     self.fixed_model.optimizer = optimizer



