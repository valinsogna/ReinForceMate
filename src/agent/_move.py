import torch


class Piece(object):
    """
    A chess piece agent class that uses different movement strategies based on the piece type.

    Attributes:
        piece (str): The type of the chess piece.
        value_function (Tensor): A value function represented by a PyTorch Tensor.
        value_function_prev (Tensor): The previous value function represented by a PyTorch Tensor.
        N (Tensor): Tensor to store data for computing average return.
        E (Tensor): Tensor to store data for computing average return.
        Returns (dict): Dictionary to store returns for each state-action pair.
        action_function (Tensor): A Tensor to represent the action value function.
        policy (Tensor): A Tensor to represent the policy.
        policy_prev (Tensor): A Tensor to represent the previous policy.
        action_space (list): A list of possible actions.
    """
    def __init__(self, piece='king'):
        """
        Initializes the chess piece agent.

        Args:
            piece (str, optional): The type of the chess piece. Default is 'king'.
        """

        self.piece = piece
        self.init_actionspace()
        self.value_function = torch.zeros((8, 8))
        self.value_function_prev = self.value_function.clone()
        self.N = torch.zeros((8, 8))
        self.E = torch.zeros((8, 8))
        self.Returns = {}
        self.action_function = torch.zeros((8, 8, len(self.action_space)))
        self.policy = torch.zeros(self.action_function.shape)
        self.policy_prev = self.policy.clone()

    def apply_policy(self, state, epsilon):
        """
        Apply the policy of the agent.

        Args:
            state (tuple): A tuple representing the state, length 2.
            epsilon (float): The probability of exploration. 0 for greedy behavior, 1 for pure exploration.

        Returns:
            int: The selected action for the state under the current policy.
        """

        greedy_action_value = torch.max(self.policy[state[0], state[1], :])
        greedy_indices = [i for i, a in enumerate(self.policy[state[0], state[1], :]) if
                          a == greedy_action_value]
        action_index = greedy_indices[torch.randint(0, len(greedy_indices), size=(1,))]
        if torch.rand(1) < epsilon:
            action_index = torch.randint(0, len(self.action_space), size=(1,))
        return action_index


    def compare_policies(self):
        """
        Compare the current policy with the previous one.

        Returns:
            Tensor: The sum of absolute differences between the current and previous policy.
        """

        return torch.sum(torch.abs(self.policy - self.policy_prev))


    def init_actionspace(self):
        """
        Initializes the action space for the piece based on its type.

        Raises:
            AssertionError: If the piece is not supported.
        """

        assert self.piece in ["king", "rook", "bishop",
                              "knight"], f"{self.piece} is not a supported piece try another one"
        if self.piece == 'king':
            self.action_space = [(-1, 0),  # north
                                 (-1, 1),  # north-west
                                 (0, 1),  # west
                                 (1, 1),  # south-west
                                 (1, 0),  # south
                                 (1, -1),  # south-east
                                 (0, -1),  # east
                                 (-1, -1),  # north-east
                                 ]
        elif self.piece == 'rook':
            self.action_space = []
            for amplitude in range(1, 8):
                self.action_space.append((-amplitude, 0))  # north
                self.action_space.append((0, amplitude))  # east
                self.action_space.append((amplitude, 0))  # south
                self.action_space.append((0, -amplitude))  # west
        elif self.piece == 'knight':
            self.action_space = [(-2, 1),  # north-north-west
                                 (-1, 2),  # n-w-w
                                 (1, 2),  # s-w-w
                                 (2, 1),  # s-s-w
                                 (2, -1),  # s-s-e
                                 (1, -2),  # s-e-e
                                 (-1, -2),  # n-e-e
                                 (-2, -1)]  # n-n-e
        elif self.piece == 'bishop':
            self.action_space = []
            for amplitude in range(1, 8):
                self.action_space.append((-amplitude, amplitude))  # north-west
                self.action_space.append((amplitude, amplitude))  # south-west
                self.action_space.append((amplitude, -amplitude))  # south-east
                self.action_space.append((-amplitude, -amplitude))  # north-east


    def get_action(self, state):
        """
        Returns the action for a given state according to the current policy.

        Args:
            state (tuple): A tuple representing the state.

        Returns:
            int: The index of the selected action from the action space.
        """
        
        return self.apply_policy(state, epsilon=0)  # Use greedy policy (epsilon=0) to select the action


