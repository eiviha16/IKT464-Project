import gymnasium as gym
import random
from algorithms.TMQN import TMQN
#from policies.DNN import QNet
from policies.TM import Policy
import numpy as np
import torch
config = {'nr_of_clauses': 1000, 'T': 750, 's': 10, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 25, 'gamma': 0.75, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 50, 'threshold_score': 450, "save": True}

#set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)
