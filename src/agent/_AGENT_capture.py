from keras.models import Model, clone_model
from keras.layers import Input, Conv2D, Dense, Reshape, Dot, Activation, Multiply
from keras.optimizers import SGD
import numpy as np
import keras.backend as K
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


def policy_gradient_loss(Returns):
    def modified_crossentropy(action, action_probs):
        cost = (K.categorical_crossentropy(action, action_probs, from_logits=False, axis=1) * Returns)
        return K.mean(cost)

    return modified_crossentropy

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
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.lr, momentum=0.0, nesterov=False)
        self.fixed_model = clone_model(self.model)
        self.fixed_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        self.fixed_model.set_weights(self.model.get_weights())

    def init_linear_network(self):
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.lr, momentum=0.0, nesterov=False)
        input_layer = Input(shape=(8, 8, 8), name='board_layer')
        reshape_input = Reshape((512,))(input_layer)
        output_layer = Dense(4096)(reshape_input)
        self.model = Model(inputs=[input_layer], outputs=[output_layer])
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def init_conv_network(self):
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.lr, momentum=0.0, nesterov=False)
        input_layer = Input(shape=(8, 8, 8), name='board_layer')
        inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_last")(input_layer)  # 1,8,8
        inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_last")(input_layer)  # 1,8,8
        flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
        flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)
        output_dot_layer = Dot(axes=1)([flat_1, flat_2])
        output_layer = Reshape(target_shape=(4096,))(output_dot_layer)
        self.model = Model(inputs=[input_layer], outputs=[output_layer])
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def init_conv_pg(self):
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.lr, momentum=0.0, nesterov=False)
        input_layer = Input(shape=(8, 8, 8), name='board_layer')
        R = Input(shape=(1,), name='Rewards')
        legal_moves = Input(shape=(4096,), name='legal_move_mask')
        inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_last")(input_layer)  # 1,8,8
        inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_last")(input_layer)  # 1,8,8
        flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
        flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)
        output_dot_layer = Dot(axes=1)([flat_1, flat_2])
        output_layer = Reshape(target_shape=(4096,))(output_dot_layer)
        softmax_layer = Activation('softmax')(output_layer)
        legal_softmax_layer = Multiply()([legal_moves, softmax_layer])
        self.model = Model(inputs=[input_layer, R, legal_moves], outputs=[legal_softmax_layer])
        self.model.compile(optimizer=optimizer, loss=policy_gradient_loss(R))

    def network_update(self, minibatch):
        states, moves, rewards, new_states = [], [], [], []
        td_errors = []
        episode_ends = []
        for sample in minibatch:
            states.append(sample[0])
            moves.append(sample[1])
            rewards.append(sample[2])
            new_states.append(sample[3])
            if np.array_equal(sample[3], sample[3] * 0):
                episode_ends.append(0)
            else:
                episode_ends.append(1)
        q_target = np.array(rewards) + np.array(episode_ends) * self.gamma * np.max(
            self.fixed_model.predict(np.stack(new_states, axis=0)), axis=1)
        q_state = self.model.predict(np.stack(states, axis=0))
        q_state = np.reshape(q_state, (len(minibatch), 64, 64))
        for idx, move in enumerate(moves):
            td_errors.append(q_state[idx, move[0], move[1]] - q_target[idx])
            q_state[idx, move[0], move[1]] = q_target[idx]
        q_state = np.reshape(q_state, (len(minibatch), 4096))
        self.model.fit(x=np.stack(states, axis=0), y=q_state, epochs=1, verbose=0)
        return td_errors

    def get_action_values(self, state):
        return self.fixed_model.predict(state) + np.random.randn() * 1e-9

    def policy_gradient_update(self, states, actions, rewards, action_spaces, actor_critic=False):
        n_steps = len(states)
        Returns = []
        targets = np.zeros((n_steps, 64, 64))
        for t in range(n_steps):
            action = actions[t]
            targets[t, action[0], action[1]] = 1
            if actor_critic:
                R = rewards[t, action[0] * 64 + action[1]]
            else:
                R = np.sum([r * self.gamma ** i for i, r in enumerate(rewards[t:])])
            Returns.append(R)
        if not actor_critic:
            mean_return = np.mean(Returns)
            self.long_term_mean.append(mean_return)
            train_returns = np.stack(Returns, axis=0) - np.mean(self.long_term_mean)
        else:
            train_returns = np.stack(Returns, axis=0)
        targets = targets.reshape((n_steps, 4096))
        self.model.fit(x=[np.stack(states, axis=0), train_returns, np.stack(action_spaces, axis=0)], y=targets, epochs=1,
                       verbose=0)

# class Agent(object):

#     def __init__(self, gamma=0.5, network='linear', lr=0.01, verbose=0):
#         """
#         Agent that plays the white pieces in capture chess
#         Args:
#             gamma: float
#                 Temporal discount factor
#             network: str
#                 'linear' or 'conv'
#             lr: float
#                 Learning rate, ideally around 0.1
#         """
#         self.gamma = gamma
#         self.network = network
#         self.lr = lr
#         self.verbose = verbose
#         self.init_network()
#         self.weight_memory = []
#         self.long_term_mean = []

#     def init_network(self):
#         """
#         Initialize the network
#         Returns:

#         """
#         if self.network == 'linear':
#             self.init_linear_network()
#         elif self.network == 'conv':
#             self.init_conv_network()
#         elif self.network == 'conv_pg':
#             self.init_conv_pg()

#     def fix_model(self):
#         """
#         The fixed model is the model used for bootstrapping
#         Returns:
#         """
#         optimizer = SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False)
#         self.fixed_model = clone_model(self.model)
#         self.fixed_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
#         self.fixed_model.set_weights(self.model.get_weights())

#     def init_linear_network(self):
#         """
#         Initialize a linear neural network
#         Returns:

#         """
#         optimizer = SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False)
#         input_layer = Input(shape=(8, 8, 8), name='board_layer')
#         reshape_input = Reshape((512,))(input_layer)
#         output_layer = Dense(4096)(reshape_input)
#         self.model = Model(inputs=[input_layer], outputs=[output_layer])
#         self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

#     def init_conv_network(self):
#         """
#         Initialize a convolutional neural network
#         Returns:

#         """
#         optimizer = SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False)
#         input_layer = Input(shape=(8, 8, 8), name='board_layer')
#         inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
#         inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
#         flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
#         flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)
#         output_dot_layer = Dot(axes=1)([flat_1, flat_2])
#         output_layer = Reshape(target_shape=(4096,))(output_dot_layer)
#         self.model = Model(inputs=[input_layer], outputs=[output_layer])
#         self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

#     def init_conv_pg(self):
#         """
#         Convnet net for policy gradients
#         Returns:

#         """
#         optimizer = SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False)
#         input_layer = Input(shape=(8, 8, 8), name='board_layer')
#         R = Input(shape=(1,), name='Rewards')
#         legal_moves = Input(shape=(4096,), name='legal_move_mask')
#         inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
#         inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
#         flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
#         flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)
#         output_dot_layer = Dot(axes=1)([flat_1, flat_2])
#         output_layer = Reshape(target_shape=(4096,))(output_dot_layer)
#         softmax_layer = Activation('softmax')(output_layer)
#         legal_softmax_layer = Multiply()([legal_moves, softmax_layer])  # Select legal moves
#         self.model = Model(inputs=[input_layer, R, legal_moves], outputs=[legal_softmax_layer])
#         self.model.compile(optimizer=optimizer, loss=policy_gradient_loss(R))

#     def network_update(self, minibatch):
#         """
#         Update the Q-network using samples from the minibatch
#         Args:
#             minibatch: list
#                 The minibatch contains the states, moves, rewards and new states.

#         Returns:
#             td_errors: np.array
#                 array of temporal difference errors

#         """

#         # Prepare separate lists
#         states, moves, rewards, new_states = [], [], [], []
#         td_errors = []
#         episode_ends = []
#         for sample in minibatch:
#             states.append(sample[0])
#             moves.append(sample[1])
#             rewards.append(sample[2])
#             new_states.append(sample[3])

#             # Episode end detection
#             if np.array_equal(sample[3], sample[3] * 0):
#                 episode_ends.append(0)
#             else:
#                 episode_ends.append(1)

#         # The Q target
#         q_target = np.array(rewards) + np.array(episode_ends) * self.gamma * np.max(
#             self.fixed_model.predict(np.stack(new_states, axis=0)), axis=1)

#         # The Q value for the remaining actions
#         q_state = self.model.predict(np.stack(states, axis=0))  # batch x 64 x 64

#         # Combine the Q target with the other Q values.
#         q_state = np.reshape(q_state, (len(minibatch), 64, 64))
#         for idx, move in enumerate(moves):
#             td_errors.append(q_state[idx, move[0], move[1]] - q_target[idx])
#             q_state[idx, move[0], move[1]] = q_target[idx]
#         q_state = np.reshape(q_state, (len(minibatch), 4096))

#         # Perform a step of minibatch Gradient Descent.
#         self.model.fit(x=np.stack(states, axis=0), y=q_state, epochs=1, verbose=0)

#         return td_errors

#     def get_action_values(self, state):
#         """
#         Get action values of a state
#         Args:
#             state: np.ndarray with shape (8,8,8)
#                 layer_board representation

#         Returns:
#             action values

#         """
#         return self.fixed_model.predict(state) + np.random.randn() * 1e-9

#     def policy_gradient_update(self, states, actions, rewards, action_spaces, actor_critic=False):
#         """
#         Update parameters with Monte Carlo Policy Gradient algorithm
#         Args:
#             states: (list of tuples) state sequence in episode
#             actions: action sequence in episode
#             rewards: rewards sequence in episode

#         Returns:

#         """
#         n_steps = len(states)
#         Returns = []
#         targets = np.zeros((n_steps, 64, 64))
#         for t in range(n_steps):
#             action = actions[t]
#             targets[t, action[0], action[1]] = 1
#             if actor_critic:
#                 R = rewards[t, action[0] * 64 + action[1]]
#             else:
#                 R = np.sum([r * self.gamma ** i for i, r in enumerate(rewards[t:])])
#             Returns.append(R)

#         if not actor_critic:
#             mean_return = np.mean(Returns)
#             self.long_term_mean.append(mean_return)
#             train_returns = np.stack(Returns, axis=0) - np.mean(self.long_term_mean)
#         else:
#             train_returns = np.stack(Returns, axis=0)
#         # print(train_returns.shape)
#         targets = targets.reshape((n_steps, 4096))
#         self.weight_memory.append(self.model.get_weights())
#         self.model.fit(x=[np.stack(states, axis=0),
#                           train_returns,
#                           np.concatenate(action_spaces, axis=0)
#                           ],
#                        y=[np.stack(targets, axis=0)],
#                        verbose=self.verbose
#                        )


# """
# The agent is no longer a single piece, it's a chess player
# Its action space consist of 64x64=4096 actions:
# - There are 8x8 = 64 piece from where a piece can be picked up
# - And another 64 pieces from where a piece can be dropped.
# Of course, only certain actions are legal.
# Which actions are legal in a certain state is part of the environment (in RL, anything outside 
# the control of the agent is considered part of the environment). 
# We can use the python-chess package to select legal moves.
# """
# import copy
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np


# def policy_gradient_loss(Returns):
#     def modified_crossentropy(action, action_probs):
#         cost = (torch.nn.functional.cross_entropy(action_probs, action, reduction='none') * Returns)
#         return torch.mean(cost)

#     return modified_crossentropy

# class Agent(nn.Module):

#     def __init__(self, network='linear', gamma=0.99, lr=0.001, verbose=True):
#         """
#         Agent class for capture chess
#         Args:
#             network: str
#                 type of network architecture ('linear' or 'conv')
#             gamma: float
#                 discount factor for future rewards
#             lr: float
#                 learning rate for the optimizer
#             verbose: bool
#                 whether to print debug information
#         """
#         super(Agent, self).__init__()
#         self.network = network
#         self.gamma = gamma
#         self.lr = lr
#         self.verbose = verbose
#         self.init_network()
#         self.weight_memory = []
#         self.long_term_mean = []

#     def init_network(self):
#         """
#         Initialize the neural network based on the specified network architecture
#         """
#         if self.network == 'linear':
#             self.model = nn.Sequential(
#                 nn.Linear(64, 256),
#                 nn.ReLU(),
#                 nn.Linear(256, 256),
#                 nn.ReLU(),
#                 nn.Linear(256, 4096)
#             )
#         elif self.network == 'conv':
#             self.model = nn.Sequential(
#                 nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(64 * 8 * 8, 4096)
#             )
#         elif self.network == 'conv_pg':
#             self.model = nn.Sequential(
#                 nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(64 * 8 * 8, 4096),
#                 nn.Softmax(dim=1)
#             )
#         else:
#             raise ValueError("Invalid network type specified.")

#         self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.0, weight_decay=0.0, nesterov=False)

#     def forward(self, x):
#         """
#         Forward pass of the neural network
#         Args:
#             x: torch.Tensor
#                 input tensor to the network
#         Returns:
#             torch.Tensor
#                 output tensor from the network
#         """
#         return self.model(x)

#     def get_action_values(self, state):
#         """
#         Get the action values (Q-values) for a given state
#         Args:
#             state: torch.Tensor
#                 input state tensor
#         Returns:
#             torch.Tensor
#                 action values (Q-values) for the state
#         """
#         with torch.no_grad():
#             return self.forward(state)

#     def network_update(self, minibatch):
#         """
#         Perform a network update (i.e., backpropagation) based on a minibatch of experiences
#         Args:
#             minibatch: list
#                 minibatch of experiences [(state, action, reward, next_state), ...]
#         Returns:
#             torch.Tensor
#                 TD errors for the minibatch
#         """
#         states, actions, rewards, next_states = zip(*minibatch)

#         states = torch.stack(states)
#         actions = torch.stack(actions)
#         rewards = torch.tensor(rewards).float()
#         next_states = torch.stack(next_states)

#         current_action_values = self.get_action_values(states)
#         next_action_values = self.get_action_values(next_states).max(dim=1).values

#         targets = rewards + self.gamma * next_action_values
#         td_errors = targets - current_action_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

#         loss = nn.MSELoss()(current_action_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(), targets.detach())

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         return td_errors

#     def fix_model(self):
#         """
#         Create a fixed model for bootstrapping
#         """
#         optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.0, weight_decay=0.0, nesterov=False)
#         self.fixed_model = copy.deepcopy(self)
#         self.fixed_model.optimizer = optimizer



