import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym

from algorithms.TMQN import TMQN
from policies.TM import Policy
config = {'nr_of_clauses': 1000, 'T': 750, 's': 10, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 25, 'gamma': 0.80, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 3, 'test_freq': 5, 'threshold_score': 50, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': True, 'number_of_state_bits_ta': 3, 'dynamic_memory_max_size': 25}

env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)
