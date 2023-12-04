import numpy as np
import torch
import torch.nn.functional as F
import os
import yaml
from tqdm import tqdm
import random

from misc.replay_buffer import Replay_buffer
from misc.plot_test_results import plot_test_results


class TMQN:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n.size
        self.obs_space_size = env.observation_space.shape[0]
        self.policy = Policy(config)

        self.gamma = config['gamma']  # discount factor
        self.exploration_prob = config['exploration_prob_init']
        self.exploration_prob_decay = config['exploration_prob_decay']

        self.epochs = config['epochs']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']

        self.y_max = config['y_max']
        self.y_min = config['y_min']

        self.replay_buffer = Replay_buffer(self.buffer_size, self.batch_size, config['n_step_TD'])
        self.test_freq = config['test_freq']
        self.nr_of_test_episodes = 100

        self.run_id = 'run_' + str(len([i for i in os.listdir('./results/TMQN-n-step-TD')]) + 1)
        self.threshold_score = config['threshold_score']
        self.has_reached_threshold = False
        self.test_random_seeds = [random.randint(1, 100000) for i in range(self.nr_of_test_episodes)]
        self.save = config['save']
        self.best_scores = {'mean': 0, 'std': float('inf')}
        self.config = config
        self.save_path = ''  # f'./results/DQN/{self.run_id}'
        self.make_run_dir()
        self.save_config()
        self.announce()
        self.q_vals = [0, 0]
        self.nr_actions = 0

    def announce(self):
        print(f'{self.run_id} has been initialized!')

    def make_run_dir(self):
        base_dir = './results'
        algorithm = f'TMQN-n-step-TD'
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

    def action(self, cur_obs):
        if np.random.random() < self.exploration_prob:
            q_vals = np.array([np.random.random() for _ in range(self.action_space_size + 1)])
        else:

            q_vals = self.policy.predict(cur_obs)
            self.q_vals[0] += q_vals[0][0]
            self.q_vals[1] += q_vals[0][1]
        return np.argmax(q_vals)

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
        return target_q_vals

    def update_exploration_prob(self):
        self.exploration_prob = self.exploration_prob * np.exp(-self.exploration_prob_decay)

    def get_q_val_and_obs_for_tm(self, target_q_vals):
        # this should be replaced with something that returns two lists
        # one with the target q_vals for action 1 (tm1)
        # one with the target q_vals for action 2 (tm2)
        tm_1_input, tm_2_input = {'observations': [], 'target_q_vals': []}, {'observations': [], 'target_q_vals': []}
        actions = self.replay_buffer.sampled_actions
        for index, action in enumerate(actions):
            if action[0] == 0:
                tm_1_input['observations'].append(self.replay_buffer.sampled_cur_obs[index][0])
                tm_1_input['target_q_vals'].append(target_q_vals[index])

            elif action[0] == 1:
                tm_2_input['observations'].append(self.replay_buffer.sampled_cur_obs[index][0])
                tm_2_input['target_q_vals'].append(target_q_vals[index])

            else:
                print('Error with get_q_val_for_action')

            # print(f'{action} - {target_q_vals[index]}')

        return tm_1_input, tm_2_input

    def train(self):
        feedback_1_cumulated, feedback_2_cumulated = [0, 0], [0, 0]

        for epoch in range(self.epochs):
            self.replay_buffer.clear_cache()
            self.replay_buffer.sample_n_seq()

            # calculate target_q_vals
            sampled_next_obs = np.array(self.replay_buffer.sampled_next_obs)

            next_q_vals = self.policy.predict(sampled_next_obs[:, -1, :])  # next_obs?
            # should this be done here? or should I use the q_values depending on the action taken.
            # I think it should be.
            next_q_vals = np.max(next_q_vals, axis=1)

            # calculate target q vals
            target_q_vals = self.n_step_temporal_difference(next_q_vals)
            # if target q_vals are lower than 100 it will be considered negative reward. so I need to find a way to normalize the values so that good actions are always over 100 and bad ones under 100.
            tm_1_input, tm_2_input = self.get_q_val_and_obs_for_tm(target_q_vals)
            feedback_1, feedback_2 = self.policy.update(tm_1_input, tm_2_input)

            feedback_1_cumulated[0] += feedback_1[0]
            feedback_1_cumulated[1] += feedback_1[1]
            feedback_2_cumulated[0] += feedback_2[0]
            feedback_2_cumulated[1] += feedback_2[1]
        return feedback_1_cumulated, feedback_2_cumulated

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

            cur_obs, _ = self.env.reset(seed=random.randint(1, 10000))
            episode_reward = 0
            actions = [0, 0]

            while True:
                action = self.action(cur_obs)
                actions[action] += 1
                next_obs, reward, done, truncated, _ = self.env.step(action)

                # might want to not have truncated in my replay buffer
                self.replay_buffer.save_experience(action, cur_obs, next_obs, reward, int(done), nr_of_steps)
                episode_reward += reward
                cur_obs = next_obs
                nr_of_steps += 1

                if done or truncated:
                    break
            self.save_actions(actions, nr_of_steps, 'train_actions.csv')

            if nr_of_steps - self.config['n_step_TD'] >= self.batch_size:
                feedback_1, feedback_2 = self.train()
                self.save_feedback_data(feedback_1, feedback_2, nr_of_steps)

            self.update_exploration_prob()
        plot_test_results(self.save_path, text={'title': 'TMQN with n-step TD'})

    def test(self, nr_of_steps):
        self.q_vals = [0, 0]
        self.nr_actions = 0

        exploration_prob = self.exploration_prob
        self.exploration_prob = 0
        episode_rewards = np.array([0 for i in range(self.nr_of_test_episodes)])

        actions = [0, 0]
        for episode in range(self.nr_of_test_episodes):
            obs, _ = self.env.reset(seed=self.test_random_seeds[episode])  # episode)
            while True:
                self.nr_actions += 1
                action = self.action(obs)
                actions[action] += 1
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_rewards[episode] += reward
                if done or truncated:
                    break
        self.save_actions(actions, nr_of_steps, 'test_actions.csv')
        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)

        self.save_results(mean, std, nr_of_steps)
        self.exploration_prob = exploration_prob
        if mean >= self.best_scores['mean']:
            self.save_model(f'best_model')
            self.best_scores['mean'] = mean
            print(f'New best mean after {nr_of_steps} steps: {mean}!')
        self.save_model('last_model')
        if mean >= self.threshold_score:
            self.has_reached_threshold = True
        self.q_vals[0] = self.q_vals[0] / self.nr_actions
        self.q_vals[1] = self.q_vals[1] / self.nr_actions

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

    def save_feedback_data(self, feedback_1, feedback_2, nr_of_steps):
        file_name = 'feedback.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("1_typeI, 1_typeII, 2_typeI, 2_typeII, steps\n")
            file.write(f"{feedback_1[0]},{feedback_1[1]},{feedback_2[0]},{feedback_2[1]},{nr_of_steps}\n")

    def save_actions(self, actions, nr_of_steps, fp):
        file_name = fp
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("tm1,tm2,steps\n")
            file.write(f"{actions[0]},{actions[1]},{nr_of_steps}\n")

    def save_q_vals(self, nr_of_steps):
        file_name = 'q_values.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("q1,q2,steps\n")
            file.write(f"{self.q_vals[0]},{self.q_vals[1]},{nr_of_steps}\n")
