import gymnasium as gym
import random
from algorithms.DQN import DQN
#from policies.DNN import QNet
from policies.Policy import Policy
config = {'gamma': 0.98, 'c': 30, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2,'hidden_size': 128, 'learning_rate': 0.001, 'test_freq': 250, 'threshold_score': 450, "save": True}

#set random seed
random.seed(42)
env = gym.make("CartPole-v1")


agent = DQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)
