import gymnasium as gym
import random
from algorithms.TMQN import TMQN
#from policies.DNN import QNet
from policies.TM import Policy
config = {'nr_of_clauses': 1_000, 'T': 2_000, 's': 2.5, 'y_max': 60, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature':1000, 'gamma': 0.98, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 250, 'threshold_score': 450, "save": True}

#set random seed
random.seed(42)
env = gym.make("CartPole-v1")


agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)
