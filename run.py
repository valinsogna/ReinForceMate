from src.environment._ENV_move import Board
from src.agent._AGENT_move import Piece
from src.learn._monte_carlo import Monte_carlo
from src.learn._q_learning_move import Q_learning_move
from src.learn._td import Temporal_difference
from src.learn._td_lambda import Temporal_difference_lambda
from src.learn._policy_iteration import Policy_iteration

env = Board()
p = Piece(piece='rook')
r = Policy_iteration(p,env)

r.policy_iteration(k=1,gamma=1,synchronous=True)