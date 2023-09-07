import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from misc.replay_buffer import Replay_buffer
# https://towardsdatascience.com/deep-q-networks-theory-and-implementation-37543f60dd67
# https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html


class DQN:
    def __init__(self, env, model, config):
        self.env = env
        self.action_space = env.action_space.shape[0]
        self.obs_space = env.observation_space.shape[0]

        self.gamma = config['gamma']  # discount factor
        self.tau = config['tau']  # soft update coefficent

        self.exploration_prob_init = config['exploration_prob_init']
        self.exploration_prob_decay = config['exploration_prob_init']

        self.buffer = []
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.replay_buffer = Replay_buffer()

        self.model = model

    def predict_action(self, obs):
        pass

    def update_learning_rate(self):
        pass

    def predict(replay_batch):
        pass

    def train(self):
        self.replay_buffer.sample()
        with torch.no_grad():
            next_q_vals = self.predict(self.replay_buffer.sampled_obs) #next_obs?
            next_q_vals, _ = torch.max(next_q_vals, dim=1)
            next_q_vals = next_q_vals.reshape(-1, 1)
            #Temporal Difference
            target_q_vals = self.replay_buffer.sampled_rewards + (1 - self.replay_buffer.sampled_dones * self.gamma * next_q_vals)

        cur_q_vals = self.predict(self.replay_buffer.sampled_obs)
        


        loss = F.smooth_l1_loss(target_q_vals, cur_q_vals)
    def test(self):
        pass
