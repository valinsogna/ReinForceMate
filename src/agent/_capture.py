import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ..config import Config as cfg

class Agent(nn.Module):
    """
    This is an implementation of a Q-learning agent with a simple linear network.

    Attributes
    ----------
    device : torch.device
        The device (cpu or gpu) where tensors will be allocated.
    gamma : float
        Discount factor for future rewards.
    lr : float
        Learning rate for optimizer.
    verbose : int
        Level of verbosity (0 is least verbose, higher numbers mean more verbose).
    model : torch.nn.Module
        The neural network model used to estimate Q-values.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    weight_memory : list
        Keeps track of model's weights throughout training.
    long_term_mean : list
        Keeps track of model's long-term means throughout training.

    Methods
    -------
    init_optimizer():
        Initialize the optimizer (Stochastic Gradient Descent).

    create_model():
        Initialize a linear neural network model.

    init_linear_network():
        Initialize the linear neural network and its optimizer.

    network_update(minibatch):
        Update the Q-network using samples from the minibatch.

    get_action_values(state):
        Get action values for given state by forward propagating through the model.

    policy_gradient_update(states, actions, rewards, action_spaces, actor_critic=False):
        Update the policy network using the policy gradient method.
    """
    
    def __init__(self, verbose=0):
        super(Agent, self).__init__()
        self.verbose = verbose
        self.init_linear_networks()
        self.weight_memory = []
        self.long_term_mean = []
        self.loss_fn = nn.MSELoss()

    def init_optimizer(self, parameters):
        return torch.optim.SGD(parameters, lr=cfg.comm.lr, momentum=0.0)

    def create_model(self):
        """
        Initializes a linear neural network model.

        Returns
        -------
        model : torch.nn.Module
            The initialized linear neural network model.
        """
        
        model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(8*8*8, 4096),
                nn.ReLU(),
            ).to(self.device)
        return model

    def init_linear_networks(self):
    	"""
    	Initialize linear networks.

    	This method creates multiple models and optimizers, with the count equal to the number of heads 
    	specified in the configuration file (cfg.comm.num_heads).

    	It first creates the models by calling `self.create_model()`. Then, it creates an optimizer for 
    	each model by calling `self.init_optimizer()` with the model parameters.

    	The created models and optimizers are stored in the instance variables `self.models` and 
    	`self.optimizers` respectively.

    	Raises:
        	TypeError: An error occurs if `cfg.comm.num_heads` is not an integer.

    	Returns:
        	None
    	"""
    
        self.models = [self.create_model() for _ in range(cfg.comm.num_heads)]
        self.optimizers = [self.init_optimizer(model.parameters()) for model in self.models]

    # Update the Q-network using samples from the minibatch
    def network_update(self, minibatch):
        """
        Update the Q-network using samples from the minibatch.

        Args:
            minibatch: list
                The minibatch contains the states, moves, rewards, and new states.

        Returns:
            td_errors: list
                List of temporal difference errors.
        """

        head_idx = np.random.randint(cfg.comm.num_heads)
        model = self.models[head_idx]
        optimizer = self.optimizers[head_idx]

        # Convert minibatch data to tensors and move them to the device
        states = torch.tensor(np.array([sample[0] for sample in minibatch]), 
                            dtype=torch.float32, device=self.device)
        moves = torch.tensor(np.array([sample[1] for sample in minibatch]), 
                            dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array([sample[2] for sample in minibatch]), 
                            dtype=torch.float32, device=self.device)
        new_states = torch.tensor(np.array([sample[3] for sample in minibatch]), 
                                dtype=torch.float32, device=self.device)

        # Compute the target Q-values using the Bellman equation
        with torch.no_grad():
            q_target = rewards + cfg.comm.gamma * torch.max(model(new_states), dim=1).values

        # Compute the Q-values for the current states
        q_state = model(states)

        # Map move_from, move_to to a single index
        single_index_moves = moves[:, 0] * 64 + moves[:, 1]

        td_errors = []
        # Create a clone of q_state for modification
        q_state_clone = q_state.clone()
        for idx, move in enumerate(single_index_moves):
            td_errors.append(q_state[idx, move] - q_target[idx])
            q_state_clone[idx, move] = q_target[idx]

        q_state_clone = q_state_clone.view(len(minibatch), -1)

        optimizer.zero_grad()
        # Use q_state_clone as the target in MSE loss calculation
        loss = self.loss_fn(q_state_clone, q_state)  
        loss.backward()
        optimizer.step()

        return [td_error.item() for td_error in td_errors]

    # Get action values for a given state by forward propagating through the model
    def get_action_values(self, state):
        """
        Get action values for given state by forward propagating through the model.

        Args:
            state: tensor
                The state for which action values are needed.

        Returns:
            action_values: tensor
                The action values as estimated by the model.
        """

        # Clone and move the state to the device, and add a batch dimension
        state = state.clone().detach().to(self.device).unsqueeze(0).float()

        # Forward propagate through all models and take the mean
        action_values = torch.mean(torch.stack([model(state) for model in self.models]), dim=0)

        # Add a small amount of random noise for exploration
        action_values += torch.randn_like(action_values) * 1e-9

        return action_values.detach()
        
    # Update the policy network using the policy gradient method
    def policy_gradient_update(self, states, actions, rewards, action_spaces, actor_critic=False):
        """
        Update the policy network using the policy gradient method.

        Args:
            states: tensor
                States visited during the episode.
            actions: tensor
                Actions taken during the episode.
            rewards: tensor
                Rewards received during the episode.
            action_spaces: list
                Action spaces during the episode.
            actor_critic: bool, optional
                Whether to use actor-critic method for updates. Default is False (use only actor updates).
        """

        # Convert data to tensors and move them to the device
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        n_steps = len(states)
        Returns = []
        targets = torch.zeros((n_steps, 64, 64), device=self.device)
        for t in range(n_steps):
            action = actions[t]
            targets[t, action[0], action[1]] = 1
            R = torch.sum(rewards[t:] * (self.gamma ** torch.arange(len(rewards[t:]))))
            Returns.append(R)

        train_returns = torch.stack(Returns)
        targets = targets.reshape((n_steps, 4096))

        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()


    
