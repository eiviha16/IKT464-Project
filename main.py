import gymnasium as gym
import random
from algorithms.DQN import DQN
config = {'gamma': 0.9, 'tau': 0.9, 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.1, 'buffer_size': 2000, 'batch_size': 16, }

#set random seed
random.seed(42)

env = gym.make("CartPole-v1", render_mode='human')
policy = ''

agent = DQN(env, policy, config)
