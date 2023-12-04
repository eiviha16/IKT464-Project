import gymnasium as gym
import random
from algorithms.DQN_n_step_TD import DQN
#from policies.DNN import QNet
from policies.Policy import Policy
import torch
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

config = {'n_step_TD': 10, 'gamma': 0.98, 'c': 30, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2,'hidden_size': 128, 'learning_rate': 0.001, 'test_freq': 5, 'threshold_score': 450, "save": True}

#set random seed


env = gym.make("CartPole-v1")


agent = DQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)