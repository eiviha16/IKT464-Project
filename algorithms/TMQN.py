import numpy as np
import torch
import torch.nn.functional as F
import os
import yaml
from tqdm import tqdm

from misc.replay_buffer import Replay_buffer
from misc.plot_test_results import plot_test_results


# https://towardsdatascience.com/deep-q-networks-theory-and-implementation-37543f60dd67
# https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

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

        self.replay_buffer = Replay_buffer(self.buffer_size, self.batch_size)
        self.test_freq = config['test_freq']
        self.nr_of_test_episodes = 1

        self.run_id = 'run_' + str(len([i for i in os.listdir('./results/TMQN')]) + 1)
        self.threshold_score = config['threshold_score']
        self.has_reached_threshold = False

        self.save = config['save']
        self.best_scores = {'mean': 0, 'std': float('inf')}
        self.config = config
        self.save_path = ''#f'./results/DQN/{self.run_id}'
        self.make_run_dir()
        self.save_config()

    def make_run_dir(self):
        base_dir = './results'
        algorithm = f'TMQN'
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
            return np.random.choice(range(self.action_space_size + 1))
        q_vals = self.policy.predict(cur_obs)
        return np.argmax(q_vals)

    def update_exploration_prob(self):
        self.exploration_prob = self.exploration_prob * np.exp(-self.exploration_prob_decay)

    def get_q_val_and_obs_for_tm(self, target_q_vals):
        #this should be replaced with something that returns two lists
        #one with the target q_vals for action 1 (tm1)
        #one with the target q_vals for action 2 (tm2)
        tm_1_input, tm_2_input = {'observations': [], 'target_q_vals': []}, {'observations': [], 'target_q_vals': []}
        actions = self.replay_buffer.sampled_actions
        for index, action in enumerate(actions):
            if action == 0:
                tm_1_input['observations'].append(self.replay_buffer.sampled_cur_obs[index])
                tm_1_input['target_q_vals'].append(target_q_vals[index])

                #tm_2_input['observations'].append(self.replay_buffer.sampled_cur_obs[index])
                #tm_2_input['target_q_vals'].append(self.y_max - target_q_vals_min[index])

            elif action == 1:
                tm_2_input['observations'].append(self.replay_buffer.sampled_cur_obs[index])
                tm_2_input['target_q_vals'].append(target_q_vals[index])

                #tm_1_input['observations'].append(self.replay_buffer.sampled_cur_obs[index])
                #tm_1_input['target_q_vals'].append(self.y_max - target_q_vals_min[index])

            else:
                print('Error with get_q_val_for_action')

            print(f'Target q vals: {target_q_vals[index]}')

        return tm_1_input, tm_2_input

    def train(self):
        for epoch in range(self.epochs):
            self.replay_buffer.clear_cache()
            self.replay_buffer.sample()

            #calculate target_q_vals
            next_q_vals = self.policy.predict(np.array(self.replay_buffer.sampled_next_obs)) #next_obs?
                #should this be done here? or should I use the q_values depending on the action taken.
                #I think it should be.
            next_q_vals_max = np.max(next_q_vals, axis=1)
            next_q_vals_min = np.min(next_q_vals, axis=1)
            #Temporal Difference
            target_q_vals_max = np.array(self.replay_buffer.sampled_rewards) + (1 - np.array(self.replay_buffer.sampled_dones)) * self.gamma * next_q_vals_max
            target_q_vals_min = np.array(self.replay_buffer.sampled_rewards) + (1 - np.array(self.replay_buffer.sampled_dones)) * self.gamma * next_q_vals_min
            #if target q_vals are lower than 100 it will be considered negative reward. so I need to find a way to normalize the values so that good actions are always over 100 and bad ones under 100.
            tm_1_input, tm_2_input = self.get_q_val_and_obs_for_tm(target_q_vals_max, target_q_vals_min)
            self.policy.update(tm_1_input, tm_2_input)

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
                action = self.action(cur_obs)
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

        plot_test_results(self.save_path, text={'title': 'TMQN'})

    def test(self, nr_of_steps):
        exploration_prob = self.exploration_prob
        self.exploration_prob = 0
        episode_rewards = np.array([0 for i in range(self.nr_of_test_episodes)])

        for episode in range(self.nr_of_test_episodes):
            obs, _ = self.env.reset(seed=episode)#episode)

            while True:
                action = self.action(obs)
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
        self.save_model('last_model')
        if mean >= self.threshold_score:
            self.has_reached_threshold = True

    def save_model(self, file_name):
        torch.save(self.policy, os.path.join(self.save_path, file_name))

    def save_results(self, mean, std, nr_of_steps):
        file_name = 'test_results.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("mean,std,steps\n")
            file.write(f"{mean},{std},{nr_of_steps}\n")


