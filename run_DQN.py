import gymnasium as gym
import random
from algorithms.DQN import DQN
from policies.Policy import Policy
import torch
import numpy as np

config = {'gamma': 0.98, 'c': 30, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2,'hidden_size': 128, 'learning_rate': 0.001, 'test_freq': 5, 'threshold_score': 450, "save": True}

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

env = gym.make("CartPole-v1")


agent = DQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)