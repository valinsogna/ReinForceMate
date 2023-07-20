# README

## Overview

This script contains a suite of reinforcement learning algorithms, all applied to a game of chess. The game is viewed as a Markov Decision Process (MDP), and the algorithms learn a policy to play the game. The algorithms included are Policy Iteration, Temporal Difference, Expected Temporal Difference, Temporal Difference Lambda, and Q-Learning.

## Getting Started

You need Python installed on your computer to run the script. If you don't have Python installed, you can download it from the [official website](https://www.python.org/downloads/).

Also, this script relies on the following Python libraries:
- torch
- matplotlib
- numpy
- tqdm
- chess

You can install them using pip:

```
pip install torch matplotlib numpy tqdm chess
```

## Installation
You can clone the ReinForceMate repository and install the package using the following commands:

```bash
> git clone https://github.com/valinsogna/ReinForceMate
> cd ReinForceMate
> ./install.sh
```

## Usage

The script is structured into different sections, each applying a different algorithm.

First, the script imports the necessary modules from the ReinForceMate package, along with the other necessary libraries.

```python
from ReinForceMate import Q_LearningMove
from ReinForceMate import TemporalDifference
from ReinForceMate import TemporalDifferenceLambda
from ReinForceMate import PolicyIteration
from ReinForceMate import ExpectedTemporalDifference
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
```

The script then applies several reinforcement learning algorithms to learn a policy for playing chess.

For example, to apply the Policy Iteration algorithm:

```python
r = PolicyIteration(piece='bishop')
policy_iter_rewards = r.run_episode()
```

The `run_episode` method plays a full game of chess using the current policy and returns the total reward obtained.

After each algorithm has been run, the script will visualize the learnt policy, for example:

```python
r.visualize_policy()
```

The script also evaluates the performance of the algorithms. It plays a certain number of episodes with each algorithm and measures the time it takes to complete. The average cumulative reward for each algorithm is then plotted:

```python
for result in results:
    plt.plot(np.cumsum(result['rewards']) / np.arange(1, n_of_episodes+1), label=result['name'])
plt.legend()
plt.title('Average Cumulative Reward')
plt.show()
```

The performance of different piece types with Policy Iteration is evaluated in a similar way:

```python
algorithms = [
    PolicyIteration(piece='king'),
    PolicyIteration(piece='rook'),
    PolicyIteration(piece='knight'),
    PolicyIteration(piece='bishop')
]
```

Finally, Q-Learning performance is evaluated for different values of alpha:

```python
for alpha in tqdm([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]):
    td = Q_LearningMove(piece='king')
    for episode in range(n_of_episodes):
        reward = td.run_episode(episode, alpha=alpha)
    rewards.append(reward)
```

Please refer to the source code comments for a more in-depth understanding of the workings of each algorithm.

## Customization

You can customize the algorithms by modifying the parameters they accept. For example, you can change the `piece` parameter in the `PolicyIteration` instantiation to apply the algorithm to a different chess piece:

```python
r = PolicyIteration(piece='rook')
```

You can also change the number of episodes played in the evaluation stage by modifying the `n_of_episodes` variable:

```python
n_of_episodes = 200
```

For Q-Learning, you can change the values of alpha to experiment with:

```python
for alpha in tqdm([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
```

## Additional Details

This script also generates csv files with performance results and chess piece values, as well as a PGN file containing the game played during the Q-Learning stage. Please ensure that the directory you are running the script in allows file writing to access these generated files.
