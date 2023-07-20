import torch
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Common:
    n_episodes: int = 100
    alpha: float = 0.05
    gamma: float = 0.9
    epsilon: float = 0.1
    max_steps: int = 1000
    lamb: float = 0.9
    memsize: int = 1000
    lr = 0.07
    verbose: int = 0
    piece: str = None
    FEN: str = None
    capture: bool = False

@dataclass
class PolicyIteration(Common):
    """
    Policy Iteration
    """
    iteration: int = 1
    k: int = 32
    synchronous: bool = True

@dataclass
class Config:
    comm: Common = Common()
    poli: PolicyIteration = PolicyIteration()
