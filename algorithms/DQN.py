import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from misc.replay_buffer import Replay_buffer
# https://towardsdatascience.com/deep-q-networks-theory-and-implementation-37543f60dd67
# https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html


class DQN:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.shape[0]
        self.obs_space_size = env.observation_space.shape[0]
        self.policy = Policy(self.action_space_size, self.obs_space_size, config['hidden_size'])

        self.gamma = config['gamma']  # discount factor
        self.tau = config['tau']  # soft update coefficent

        self.exploration_prob = config['exploration_prob_init']
        self.exploration_prob_decay = config['exploration_prob_decay']

        self.buffer = []
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.replay_buffer = Replay_buffer()


    def action(self, cur_obs):
        if np.random.random() < self.exploration_prob:
            return np.random.choice(range(self.action_space_size))
        q_vals = self.model.predict(cur_obs)
        return np.argmax(q_vals)

    def update_exploration_prob(self):
        self.exploration_prob = self.exploration_prob * np.exp(-self.exploration_prob_decay)

    def train(self):
        self.replay_buffer.sample()
        with torch.no_grad():
            next_q_vals = self.policy.predict(self.replay_buffer.next_obs) #next_obs?
            next_q_vals, _ = torch.max(next_q_vals, dim=1)
            next_q_vals = next_q_vals.reshape(-1, 1)
            #Temporal Difference
            target_q_vals = self.replay_buffer.sampled_rewards + (1 - self.replay_buffer.sampled_dones * self.gamma * next_q_vals)

        cur_q_vals = self.policy.predict(self.replay_buffer.cur_obs)

        #Huber loss, mix between MSE and something else
        self.policy.optimizer.zero_grad()
        loss = F.smooth_l1_loss(target_q_vals, cur_q_vals)
        loss.backward()
        self.policy.optimizer.step()

    def learn(self, nr_of_episodes):
        nr_of_steps = 0
        episode_rewards = [0 for i in range(nr_of_episodes)]

        for episode in range(nr_of_episodes):
            cur_obs = self.env.reset()
            cur_obs = np.array([cur_obs])
            while True:
                nr_of_steps += 1
                action = self.action(cur_obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.replay_buffer.save_experience(action,cur_obs, next_obs, reward, done, nr_of_steps)
                episode_rewards[episode] += reward
                if done:
                    self.update_exploration_prob()
                    break

            if nr_of_steps >= self.batch_size:
                self.train()

        print(f'Rewards: {episode_rewards[-1]}')

    def test(self):
        pass
