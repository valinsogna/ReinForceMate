import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyGradientLoss(nn.Module):
    def __init__(self):
        super(PolicyGradientLoss, self).__init__()

    def forward(self, action, action_probs, Returns):
        cost = (F.cross_entropy(action_probs, action, reduction='none') * Returns)
        return cost.mean()

class Agent(object):
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
        self.fixed_model = nn.Sequential(*list(self.model.children()))
        self.fixed_model.load_state_dict(self.model.state_dict())

    def init_linear_network(self):
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 4096)
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def init_conv_network(self):
        self.model = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(64, 4096)
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def init_conv_pg(self):
        self.model = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(64, 4096),
            nn.Softmax(dim=1)
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_fn = PolicyGradientLoss()

    def network_update(self, minibatch):
        states, moves, rewards, new_states = [], [], [], []
        td_errors = []
        episode_ends = []
        for sample in minibatch:
            states.append(sample[0])
            moves.append(sample[1])
            rewards.append(sample[2])
            new_states.append(sample[3])

            if torch.equal(sample[3], sample[3] * 0):
                episode_ends.append(0)
            else:
                episode_ends.append(1)

        q_target = torch.tensor(rewards) + torch.tensor(episode_ends) * self.gamma * torch.max(
            self.fixed_model(torch.stack(new_states)).detach(), dim=1).values

        q_state = self.model(torch.stack(states))
        td_errors = q_state[range(len(moves)), moves[:, 0], moves[:, 1]] - q_target
        q_state[range(len(moves)), moves[:, 0], moves[:, 1]] = q_target
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_state, q_state.detach())
        loss.backward()
        self.optimizer.step()

        return td_errors.detach().numpy()

    def get_action_values(self, state):
        with torch.no_grad():
            return self.fixed_model(state) + torch.randn_like(self.fixed_model(state)) * 1e-9

    def policy_gradient_update(self, states, actions, rewards, action_spaces, actor_critic=False):
        n_steps = len(states)
        Returns = []
        targets = torch.zeros((n_steps, 64, 64))
        for t in range(n_steps):
            action = actions[t]
            targets[t, action[0], action[1]] = 1
            if actor_critic:
                R = rewards[t, action[0] * 64 + action[1]]
            else:
                R = torch.sum(torch.tensor([r * self.gamma ** i for i, r in enumerate(rewards[t:])]))
            Returns.append(R)

        if not actor_critic:
            mean_return = torch.mean(torch.stack(Returns))        
            self.long_term_mean.append(mean_return)
            train_returns = torch.stack(Returns) - torch.mean(torch.stack(self.long_term_mean))
        else:
            train_returns = torch.stack(Returns)
        
        targets = targets.view(n_steps, -1)
        self.weight_memory.append(self.model.state_dict())
        self.optimizer.zero_grad()
        output = self.model(torch.stack(states))
        loss = self.loss_fn(output, targets, train_returns.unsqueeze(1))
        loss.backward()
        self.optimizer.step()
