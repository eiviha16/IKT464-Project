import numpy as np
import random


class Replay_buffer:
    def __init__(self, buffer_size, batch_size, n=5):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.actions = []  # 0 for i in range(buffer_size)]
        self.cur_obs = []  # 0 for i in range(buffer_size)]
        self.next_obs = []  # 0 for i in range(buffer_size)]
        self.rewards = []  # 0 for i in range(buffer_size)]
        self.dones = []  # 0 for i in range(buffer_size)]

        self._actions = [[], []]  # 0 for i in range(buffer_size)]
        self._cur_obs = [[], []]  # 0 for i in range(buffer_size)]
        self._next_obs = [[], []]  # 0 for i in range(buffer_size)]
        self._rewards = [[], []]  # 0 for i in range(buffer_size)]
        self._dones = [[], []]  # 0 for i in range(buffer_size)]

        self.sampled_actions = []  # 0 for i in range(batch_size)]
        self.sampled_cur_obs = []  # 0 for i in range(batch_size)]
        self.sampled_next_obs = []  # 0 for i in range(batch_size)]
        self.sampled_rewards = []  # 0 for i in range(batch_size)]
        self.sampled_dones = []  # 0 for i in range(batch_size)]

        self.indices = []  # i for i in range(buffer_size)]
        self.n = n

    def clear_cache(self):
        self.sampled_actions = []
        self.sampled_cur_obs = []
        self.sampled_next_obs = []
        self.sampled_rewards = []
        self.sampled_dones = []

    def split_sample(self):
        weights = [0.5, 0.5]  # [0.5, 0.5] -inc type 2

        # weighting = [weights[self.dones[i]] for i in range(len(self.dones))]

        # try multiplying the weights with the number of dones=0 and dones=1 to account for numeric differences this does cause issues. May struggle with a buffer with no falls
        # sample = random.choices(range(len(self.rewards)), weighting, k=self.batch_size)
        samples = [[], []]
        indexes = [[i for i in range(len(self._rewards[0]))], [i for i in range(len(self._rewards[1]))]]

        for i in range(self.batch_size):
            c = random.choices([0, 1], weights)
            sample = random.choices(indexes[c[0]])
            samples[c[0]].append(sample[0])

        for i in range(len(samples)):
            for j, s in enumerate(samples[i]):
                self.sampled_actions.append(self._actions[i][s])
                self.sampled_cur_obs.append(self._cur_obs[i][s])
                self.sampled_next_obs.append(self._next_obs[i][s])
                self.sampled_rewards.append(self._rewards[i][s])
                self.sampled_dones.append(self._dones[i][s])

    def prioritized_sample(self, feedback, min_feedback_p=0.25, magnitude=3):
        type_I_p = 1 - (feedback['tm1'][0] + feedback['tm2'][0]) / (
                    1 + sum(feedback['tm1']) + sum(feedback['tm2'])) if 1 - (
                    feedback['tm1'][0] + feedback['tm2'][0]) / (1 + sum(feedback['tm1']) + sum(
            feedback['tm2'])) > min_feedback_p else min_feedback_p
        type_II_p = 1 - (feedback['tm1'][1] + feedback['tm2'][1]) / (
                    1 + sum(feedback['tm1']) + sum(feedback['tm2'])) if 1 - (
                    feedback['tm1'][1] + feedback['tm2'][1]) / (1 + sum(feedback['tm1']) + sum(
            feedback['tm2'])) > min_feedback_p else min_feedback_p

        weights = [0.25, 0.75]#[type_I_p, type_II_p]
        #weights = [0.75, 0.25] #[0.5, 0.5] -inc type 2

        # weighting = [weights[self.dones[i]] for i in range(len(self.dones))]

        # try multiplying the weights with the number of dones=0 and dones=1 to account for numeric differences this does cause issues. May struggle with a buffer with no falls
        # sample = random.choices(range(len(self.rewards)), weighting, k=self.batch_size)
        samples = [[], []]
        indexes = [[i for i in range(len(self._rewards[0]))], [i for i in range(len(self._rewards[1]))]]

        for i in range(self.batch_size):
            c = random.choices([0, 1], weights)
            sample = random.choices(indexes[c[0]])
            samples[c[0]].append(sample[0])

        for i in range(len(samples)):
            for j, s in enumerate(samples[i]):
                self.sampled_actions.append(self._actions[i][s])
                self.sampled_cur_obs.append(self._cur_obs[i][s])
                self.sampled_next_obs.append(self._next_obs[i][s])
                self.sampled_rewards.append(self._rewards[i][s])
                self.sampled_dones.append(self._dones[i][s])
        """
        sample = []
        indexes = [i for i in range(len(self.rewards))]
        for i in range(self.batch_size):
            c = random.choices([0, 1], weights)
            while True:
                s = random.choice(indexes)
                if self.dones[s] == c[0]:
                    sample.append(s)
                    break
        """

        """for i, s in enumerate(sample):
            self.sampled_actions.append(self.actions[s])
            self.sampled_cur_obs.append(self.cur_obs[s])
            self.sampled_next_obs.append(self.next_obs[s])
            self.sampled_rewards.append(self.rewards[s])
            self.sampled_dones.append(self.dones[s])"""

    def sample(self):
        sample = random.sample(range(len(self.rewards)), self.batch_size)
        for i, s in enumerate(sample):
            self.sampled_actions.append(self.actions[s])
            self.sampled_cur_obs.append(self.cur_obs[s])
            self.sampled_next_obs.append(self.next_obs[s])
            self.sampled_rewards.append(self.rewards[s])
            self.sampled_dones.append(self.dones[s])

    def sample_n_seq(self):
        sample = random.sample(range(len(self.rewards) - self.n), self.batch_size)
        for i, s in enumerate(sample):
            self.sampled_actions.append(self.actions[s: s + self.n])
            self.sampled_cur_obs.append(self.cur_obs[s: s + self.n])
            self.sampled_next_obs.append(self.next_obs[s: s + self.n])
            self.sampled_rewards.append(self.rewards[s: s + self.n])
            self.sampled_dones.append(self.dones[s: s + self.n])

    def save_experience(self, action, cur_obs, next_obs, reward, done, i):
        if self.buffer_size <= len(self.rewards):
            self.actions.pop(0)
            self.cur_obs.pop(0)
            self.next_obs.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)

        self.actions.append(action)
        self.cur_obs.append(cur_obs)
        self.next_obs.append(next_obs)
        self.rewards.append(reward)
        self.dones.append(done)

    def prioritized_save_experience(self, action, cur_obs, next_obs, reward, done, i):
        if self.buffer_size <= len(self._rewards[done]):
            self._actions[done].pop(0)
            self._cur_obs[done].pop(0)
            self._next_obs[done].pop(0)
            self._rewards[done].pop(0)
            self._dones[done].pop(0)

        self._actions[done].append(action)
        self._cur_obs[done].append(cur_obs)
        self._next_obs[done].append(next_obs)
        self._rewards[done].append(reward)
        self._dones[done].append(done)

    def split_save_experience(self, action, cur_obs, next_obs, reward, done, i):
        if self.buffer_size <= len(self._rewards[action]):
            self._actions[action].pop(0)
            self._cur_obs[action].pop(0)
            self._next_obs[action].pop(0)
            self._rewards[action].pop(0)
            self._dones[action].pop(0)

        self._actions[action].append(action)
        self._cur_obs[action].append(cur_obs)
        self._next_obs[action].append(next_obs)
        self._rewards[action].append(reward)
        self._dones[action].append(done)

if __name__ == '__main__':
    replay_buffer = Replay_buffer(200, 20)
    for i in range(200):
        action = random.random()
        cur_obs = random.random()
        next_obs = random.random()
        rewards = random.random()
        dones = random.randint(0, 1)
        replay_buffer.save_experience(action, cur_obs, next_obs, rewards, dones, i)
    feedback = {}
    feedback['tm1'] = [4, 10]
    feedback['tm2'] = [6, 7]
    replay_buffer.prioritized_sample(feedback)
