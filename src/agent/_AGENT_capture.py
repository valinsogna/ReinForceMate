import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def policy_gradient_loss(Returns):
    def modified_crossentropy(action_probs, action):
        cost = (torch.nn.functional.cross_entropy(action_probs, action, reduction='none') * Returns)
        return torch.mean(cost)

    return modified_crossentropy

class Agent:
    def __init__(self, gamma=0.5, network='linear', lr=0.01, verbose=0):
        self.gamma = gamma
        self.network = network
        self.lr = lr
        self.verbose = verbose
        self.init_network()
        self.weight_memory = []
        self.long_term_mean = []

    def init_network(self):
        if self.network == 'linear':
            self.init_linear_network()
        elif self.network == 'conv':
            self.init_conv_network()
        elif self.network == 'conv_pg':
            self.init_conv_pg()

    def fix_model(self):
        self.fixed_model = self.model.clone()

    def init_linear_network(self):
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*8, 4096)
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def init_conv_network(self):
        self.model = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(64, 4096)
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def init_conv_pg(self):
        self.model = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(64, 4096),
            nn.Softmax(dim=1)
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def network_update(self, minibatch):
        states, moves, rewards, new_states = zip(*minibatch)

        states = torch.stack(states)
        moves = torch.tensor(moves)
        rewards = torch.tensor(rewards)
        new_states = torch.stack(new_states)

        q_target = rewards + self.gamma * self.fixed_model(new_states).max(dim=1).values

        q_state = self.model(states)
        q_state = q_state.view(len(minibatch), 64, 64)

        td_errors = q_state.gather(1, moves.unsqueeze(1)).squeeze(1) - q_target
        q_state.scatter_(1, moves.unsqueeze(1), q_target.unsqueeze(1))

        q_state = q_state.view(len(minibatch), 4096)

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(self.model(states), q_state)
        loss.backward()
        self.optimizer.step()

        return td_errors.detach().numpy()

    def get_action_values(self, state):
        state = torch.unsqueeze(state, 0)
        return self.fixed_model(state).detach().numpy() + np.random.randn() * 1e-9

    def policy_gradient_update(self, states, actions, rewards, action_spaces, actor_critic=False):
        n_steps = len(states)
        Returns = []
        targets = torch.zeros((n_steps, 64, 64))
        for t in range(n_steps):
            action = actions[t]
            targets[t, action[0], action[1]] = 1
            if actor_critic:
                R = rewards[t][action[0] * 64 + action[1]]
            else:
                R = np.sum([r * self.gamma ** i for i, r in enumerate(rewards[t:])])
            Returns.append(R)

        if not actor_critic:
            mean_return = np.mean(Returns)
            self.long_term_mean.append(mean_return)
            train_returns = np.stack(Returns, axis=0) - np.mean(self.long_term_mean)
        else:
            train_returns = np.stack(Returns, axis=0)

        targets = targets.view(n_steps, 4096)

        self.weight_memory.append(self.model.state_dict())
        self.optimizer.zero_grad()
        loss = policy_gradient_loss(torch.tensor(train_returns))(self.model(states), targets)
        loss.backward()
        self.optimizer.step()

## This is an optimized version of the above class
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np


# def policy_gradient_loss(Returns):
#     def modified_crossentropy(action_probs, action):
#         cost = (nn.functional.cross_entropy(action_probs, action, reduction='none') * Returns)
#         return torch.mean(cost)

#     return modified_crossentropy


# class Agent:
#     def __init__(self, gamma=0.5, network='linear', lr=0.01, verbose=0):
#         self.gamma = gamma
#         self.network = network
#         self.lr = lr
#         self.verbose = verbose
#         self.init_network()
#         self.init_target_network()
#         self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

#     def init_network(self):
#         if self.network == 'linear':
#             self.model = nn.Linear(8 * 8 * 8, 4096)
#         elif self.network == 'conv':
#             self.model = nn.Sequential(
#                 nn.Conv2d(8, 1, kernel_size=1),
#                 nn.Flatten(),
#                 nn.Linear(64, 4096)
#             )
#         elif self.network == 'conv_pg':
#             self.model = nn.Sequential(
#                 nn.Conv2d(8, 1, kernel_size=1),
#                 nn.Flatten(),
#                 nn.Linear(64, 4096),
#                 nn.Softmax(dim=1)
#             )

#     def init_target_network(self):
#         self.target_model = self.model.clone().eval()

#     def network_update(self, minibatch):
#         states, moves, rewards, new_states = zip(*minibatch)

#         states = torch.stack(states)
#         moves = torch.tensor(moves)
#         rewards = torch.tensor(rewards)
#         new_states = torch.stack(new_states)

#         q_target = rewards + self.gamma * self.target_model(new_states).max(dim=1).values

#         q_state = self.model(states)
#         q_state = q_state.view(len(minibatch), 64, 64)

#         td_errors = q_state.gather(1, moves.unsqueeze(1)).squeeze(1) - q_target

#         q_state.scatter_(1, moves.unsqueeze(1), q_target.unsqueeze(1))
#         q_state = q_state.view(len(minibatch), 4096)

#         self.optimizer.zero_grad()
#         loss = nn.MSELoss()(self.model(states), q_state)
#         loss.backward()
#         self.optimizer.step()

#         return td_errors.detach().numpy()

#     def get_action_values(self, state):
#         state = torch.unsqueeze(state, 0)
#         return self.target_model(state).detach().cpu().numpy() + np.random.randn() * 1e-9

#     def policy_gradient_update(self, states, actions, rewards, action_spaces, actor_critic=False):
#         n_steps = len(states)
#         Returns = []
#         targets = torch.zeros((n_steps, 64, 64))
#         for t in range(n_steps):
#             action = actions[t]
#             targets[t, action[0], action[1]] = 1
#             if actor_critic:
#                 R = rewards[t][action[0] * 64 + action[1]]
#             else:
#                 R = np.sum([r * self.gamma ** i for i, r in enumerate(rewards[t:])])
#             Returns.append(R)

#         if not actor_critic:
#             mean_return = np.mean(Returns)
#             train_returns = np.stack(Returns, axis=0) - mean_return
#         else:
#             train_returns = np.stack(Returns, axis=0)

#         targets = targets.view(n_steps, 4096)

#         self.optimizer.zero_grad()
#         loss = policy_gradient_loss(torch.tensor(train_returns))(self.model(states), targets)
#         loss.backward()
#         self.optimizer.step()
