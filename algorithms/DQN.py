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
        self.action_space_size = env.action_space.n.size
        self.obs_space_size = env.observation_space.shape[0]
        self.policy = Policy(self.obs_space_size, self.action_space_size, config)

        self.gamma = config['gamma']  # discount factor
        self.tau = config['tau']  # soft update coefficent

        self.exploration_prob = config['exploration_prob_init']
        self.exploration_prob_decay = config['exploration_prob_decay']

        self.epochs = config['epochs']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']

        self.replay_buffer = Replay_buffer(self.buffer_size, self.batch_size)

    def action(self, cur_obs):
        if np.random.random() < self.exploration_prob:
            return torch.tensor(np.random.choice(range(self.action_space_size + 1)))
        q_vals = self.policy.predict(cur_obs)
        return torch.argmax(q_vals)

    def update_exploration_prob(self):
        self.exploration_prob = self.exploration_prob * np.exp(-self.exploration_prob_decay)
        #print(f'Exploration probability: {self.exploration_prob}')
    def get_q_val_for_action(self, q_vals):
        #indices = np.vstack(self.replay_buffer.actions)
        indices = np.array(self.replay_buffer.sampled_actions)
        #indices = torch.tensor(actions)

        #selected_q_vals = torch.index_select(q_vals, dim=0, index=indices)
        selected_q_vals = q_vals[range(q_vals.shape[0]), indices]

        return selected_q_vals

    def train(self):
        for epoch in range(self.epochs):
            self.replay_buffer.clear_cache()
            self.replay_buffer.sample()
            with torch.no_grad():
                next_q_vals = self.policy.predict(self.replay_buffer.sampled_next_obs) #next_obs?
                next_q_vals, _ = torch.max(next_q_vals, dim=1)
                #next_q_vals = next_q_vals.reshape(-1, 1)
                #Temporal Difference
                target_q_vals = torch.tensor(self.replay_buffer.sampled_rewards) + (1 - torch.tensor(self.replay_buffer.sampled_dones)) * self.gamma * next_q_vals

            cur_q_vals = self.policy.predict(self.replay_buffer.sampled_cur_obs)
            cur_q_vals = self.get_q_val_for_action(cur_q_vals)
            #cur_q_vals, _ = torch.max(cur_q_vals, dim=1)
            #Its learning but not the correct policy so the values being compared are wrong.
            #Huber loss, mix between MSE and something else
            #print(self.policy.hidden_layer.weight[0, 0])
            self.policy.optimizer.zero_grad()
            #print(f'Values: {target_q_vals[0]}, {cur_q_vals[0]}')
            loss = F.smooth_l1_loss(target_q_vals, cur_q_vals)
            #print(f'Loss: {loss}')
            loss.backward()
            self.policy.optimizer.step()

    def learn(self, nr_of_episodes):
        nr_of_steps = 0
        actions_nr = [0, 0]
        for episode in range(nr_of_episodes):
            cur_obs, _ = self.env.reset(seed=42)
            episode_reward = 0

            while True:
                action = self.action(cur_obs).numpy()
                actions_nr[action] += 1
                next_obs, reward, done, truncated, _ = self.env.step(action)
                if truncated:
                    self.replay_buffer.save_experience(action, cur_obs, next_obs, reward, int(truncated), nr_of_steps)
                else:
                    self.replay_buffer.save_experience(action, cur_obs, next_obs, reward, int(done), nr_of_steps)
                episode_reward += reward
                cur_obs = next_obs
                nr_of_steps += 1

                if done or truncated:
                    self.update_exploration_prob()
                    break
            if nr_of_steps >= self.batch_size:
                self.train()
            if episode % 100 == 0:
                print(f'Actions: {actions_nr}')
                print(f'Reward: {episode_reward}')
                #print(self.policy.output_layer.weight[0, 0])
        print(f'Rewards: {episode_reward}')

    def test(self):
        pass
