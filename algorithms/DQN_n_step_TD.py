import numpy as np
import torch
import torch.nn.functional as F
import os
import yaml
from tqdm import tqdm
import random
from misc.replay_buffer import Replay_buffer
from misc.plot_test_results import plot_test_results


class DQN:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n.size
        self.obs_space_size = env.observation_space.shape[0]
        self.policy = Policy(self.obs_space_size, self.action_space_size, config)

        self.gamma = config['gamma']  # discount factor
        self.exploration_prob = config['exploration_prob_init']
        self.exploration_prob_decay = config['exploration_prob_decay']

        self.epochs = config['epochs']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']

        self.replay_buffer = Replay_buffer(self.buffer_size, self.batch_size, config['n_step_TD'])
        self.test_freq = config['test_freq']
        self.nr_of_test_episodes = 100

        self.run_id = 'run_' + str(len([i for i in os.listdir('./results/DQN-n-step-TD')]) + 1)
        self.threshold_score = config['threshold_score']
        self.has_reached_threshold = False

        self.save = config['save']
        self.best_scores = {'mean': 0, 'std': float('inf')}
        self.config = config
        self.save_path = ''  # f'./results/DQN/{self.run_id}'
        self.make_run_dir()
        self.save_config()

        self.observations = []
        self.q_values = [0, 0]
        self.nr_actions = 0
        self.announce()

    def announce(self):
        print(f'{self.run_id} has been initialized!')

    def make_run_dir(self):
        base_dir = './results'
        algorithm = f'DQN-n-step-TD'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(os.path.join(base_dir, algorithm)):
            os.makedirs(os.path.join(base_dir, algorithm))
        if not os.path.exists(os.path.join(base_dir, algorithm, self.run_id)):
            os.makedirs(os.path.join(base_dir, algorithm, self.run_id))
        self.save_path = os.path.join(base_dir, algorithm, self.run_id)

    def save_config(self):
        with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def temporal_difference(self, next_q_vals):
        return np.array(self.replay_buffer.sampled_rewards) + (
                    1 - np.array(self.replay_buffer.sampled_dones)) * self.gamma * next_q_vals

    def n_step_temporal_difference(self, next_q_vals):
        target_q_vals = []
        for i in range(len(self.replay_buffer.sampled_rewards)):
            target_q_val = 0
            for j in range(len(self.replay_buffer.sampled_rewards[i])):
                target_q_val += (self.gamma ** j) * self.replay_buffer.sampled_rewards[i][j]
                if self.replay_buffer.sampled_dones[i][j]:
                    break
            target_q_val += (1 - self.replay_buffer.sampled_dones[i][j]) * (self.gamma ** j) * next_q_vals[i]
            target_q_vals.append(target_q_val)
        return torch.stack(target_q_vals)

    def action(self, cur_obs):
        if np.random.random() < self.exploration_prob:
            q_vals = torch.tensor([np.random.random() for _ in range(self.action_space_size + 1)])
        else:
            q_vals = self.policy.predict(cur_obs)
            self.q_values[0] += q_vals[0]
            self.q_values[1] += q_vals[1]
            self.nr_actions += 1
        return torch.argmax(q_vals)

    def update_exploration_prob(self):
        self.exploration_prob = self.exploration_prob * np.exp(-self.exploration_prob_decay)

    def get_q_val_for_action(self, q_vals):
        sampled_actions = np.array(self.replay_buffer.sampled_actions)
        indices = sampled_actions[:, 0]
        selected_q_vals = q_vals[range(q_vals.shape[0]), indices]
        return selected_q_vals

    def train(self):
        for epoch in range(self.epochs):
            self.replay_buffer.clear_cache()
            self.replay_buffer.sample_n_seq()
            with torch.no_grad():
                sampled_next_obs = np.array(self.replay_buffer.sampled_next_obs)
                next_q_vals = self.policy.predict(sampled_next_obs[:, -1, :])  # next_obs?
                next_q_vals, _ = torch.max(next_q_vals, dim=1)
                # target_q_vals = self.temporal_difference(next_q_vals)
                target_q_vals = self.n_step_temporal_difference(next_q_vals)

            sampled_cur_obs = np.array(self.replay_buffer.sampled_cur_obs)
            cur_q_vals = self.policy.predict(sampled_cur_obs[:, 0, :])
            cur_q_vals = self.get_q_val_for_action(cur_q_vals)
            self.policy.optimizer.zero_grad()
            loss = F.mse_loss(target_q_vals, cur_q_vals)
            loss.backward()
            self.policy.optimizer.step()

    def learn(self, nr_of_episodes):

        nr_of_steps = 0
        actions_nr = [0, 0]
        for episode in tqdm(range(nr_of_episodes)):
            if self.test_freq:
                if episode % self.test_freq == 0:
                    self.test(nr_of_steps)
                    self.config['nr_of_episodes'] = episode + 1
                    self.config['nr_of_steps'] = nr_of_steps
                    self.save_config()

            cur_obs, _ = self.env.reset(seed=42)
            episode_reward = 0

            while True:
                action = self.action(cur_obs).numpy()
                actions_nr[action] += 1
                next_obs, reward, done, truncated, _ = self.env.step(action)
                self.replay_buffer.save_experience(action, cur_obs, next_obs, reward, int(done), nr_of_steps)
                episode_reward += reward
                cur_obs = next_obs
                nr_of_steps += 1

                if done or truncated:
                    break
            if nr_of_steps - self.config['n_step_TD'] >= self.batch_size:
                self.train()

            self.update_exploration_prob()

        plot_test_results(self.save_path, text={'title': 'DQN with n-step TD'})

    def test(self, nr_of_steps):
        self.q_values = [0, 0]
        self.nr_actions = 0
        exploration_prob = self.exploration_prob
        self.exploration_prob = 0
        episode_rewards = np.array([0 for i in range(self.nr_of_test_episodes)])
        for episode in range(self.nr_of_test_episodes):
            obs, _ = self.env.reset(seed=episode)  # episode)
            while True:
                lucky_number = random.randint(1, self.nr_of_test_episodes)
                if episode % lucky_number == 0:
                    self.observations.append(obs)
                action = self.action(obs).numpy()
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_rewards[episode] += reward
                if done or truncated:
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)

        self.save_results(mean, std, nr_of_steps)
        self.exploration_prob = exploration_prob
        if mean > self.best_scores['mean']:
            self.save_model('best_model')
            self.best_scores['mean'] = mean
            print(f'New best mean after {nr_of_steps} steps: {mean}!')

        self.save_model('last_model')
        if mean >= self.threshold_score:
            self.has_reached_threshold = True
        self.q_values[0] = self.q_values[0] / self.nr_actions
        self.q_values[1] = self.q_values[1] / self.nr_actions
        self.save_q_vals(nr_of_steps)

    def save_model(self, file_name):
        torch.save(self.policy, os.path.join(self.save_path, file_name))

    def save_results(self, mean, std, nr_of_steps):
        file_name = 'test_results.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("mean,std,steps\n")
            file.write(f"{mean},{std},{nr_of_steps}\n")

    def save_q_vals(self, nr_of_steps):
        file_name = 'q_values.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("q1,q2,steps\n")
            file.write(f"{self.q_values[0]},{self.q_values[1]},{nr_of_steps}\n")
