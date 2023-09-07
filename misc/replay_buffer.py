import numpy as np

class Replay_buffer:
    def __init__(self):
        self.observation = []
        self.reward = []
        self.done = []
        self.action = []

        self.sampled_obs = []
        self.sampled_rewards = []
        self.sampled_dones = []
        self.sampled_actions = []

        self.indices = []
        self.sample = []

    def sample(self):
        #np.random.shuffle(self.buffer)
        #return self.buffer[0:self.batch_size]
        self.sample = np.sample(self.indices)
        self.sampled_obs = [self.observation[i] for i in self.sample]
        self.sampled_rewards = [self.reward[i] for i in self.sample]
        self.sampled_dones = [self.done[i] for i in self.sample]
        self.sampled_actions = [self.action[i] for i in self.sample]