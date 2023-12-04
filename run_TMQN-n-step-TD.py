import numpy as np
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.TMQN_n_step_TD import TMQN
from policies.TM import Policy

config = {'n_step_TD': 10, 'nr_of_clauses': 1000, 'T': 750, 's': 15, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 25, 'gamma': 0.85, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 5, 'threshold_score': 450, "save": True, "balance_feedback": False, "min_feedback_p": 0.25, 'dynamic_memory': False, 'number_of_state_bits_ta': 6, "dynamic_memory_max_size": 6, "seed":42}


env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)



