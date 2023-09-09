import numpy as np

class Replay_buffer:
    def __init__(self):
        self.cur_obs = []
        self.next_obs = []
        self.rewards = []
        self.dones = []
        self.actions = []

        self.sampled_cur_obs = []
        self.sampled_next_obs = []
        self.sampled_rewards = []
        self.sampled_dones = []
        self.sampled_actions = []

        self.indices = []
        self.sample = []

    def sample(self):

        self.sample = np.sample(self.indices)
        self.sampled_actions = [self.actions[i] for i in self.sample]
        self.sampled_cur_obs = [self.cur_obs[i] for i in self.sample]
        self.sampled_next_obs = [self.next_obs[i] for i in self.sample]
        self.sampled_rewards = [self.rewards[i] for i in self.sample]
        self.sampled_dones = [self.dones[i] for i in self.sample]

    def save_experience(self, action, cur_obs, next_obs, reward, done, i):
        self.actions[i] = action
        self.cur_obs[i] = cur_obs
        self.next_obs[i] = next_obs
        self.rewards[i] = reward
        self.dones[i] = done
