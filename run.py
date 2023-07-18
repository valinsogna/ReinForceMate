from src.environment._ENV_move import Board
from src.agent._AGENT_move import Piece
from src.learn._monte_carlo import Monte_carlo
from src.learn._q_learning_move import Q_learning_move
from src.learn._td import Temporal_difference
from src.learn._td_lambda import Temporal_difference_lambda
from src.learn._policy_iteration import Policy_iteration
from src.learn._exp_td import Expected_temporal_difference
import torch

env = Board()
p = Piece(piece='king')  # king", "rook", "bishop" or "knight"

# env.render()

# r = Policy_iteration(agent=p,env=env)
# r.policy_iteration(k=1,gamma=1,synchronous=True)

# r = Temporal_difference(p, env)
# r.sarsa_td(n_episodes=1000, alpha=0.2, gamma=0.9)
# r.visualize_policy() # controllare il risultato
# r.visualize_action_function()
# # r.TD_zero(epsilon=0.1, alpha=0.05, lamb=0.9)

r = Expected_temporal_difference(agent=p, env=env)
r.expected_sarsa(n_episodes=1000, alpha=0.2, gamma=0.9)
r.visualize_policy() # controllare il risultato
r.visualize_action_function()



# r = Temporal_difference_lambda(agent=p, env=env)
# r.sarsa_lambda(n_episodes=10000,alpha=0.2,gamma=0.9)
# # r.visualize_policy() # controllare il risultato
# r.visualize_action_function()

# r = Q_learning_move(agent=p, env=env)
# r.q_learning(n_episodes=1000, alpha=0.2, gamma=0.9)
# # r.visualize_policy() # controllare il risultato
# r.visualize_action_function()

# r = Monte_carlo(agent=p, env=env)
# for k in range(1000):
#     eps = 0.5
#     r.monte_carlo_learning(epsilon=eps)
# r.visualize_policy() # controllare il risultato

# print(torch.max(r.agent.action_function, dim=2)[0].to(torch.int))