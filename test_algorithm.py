import numpy as np
import torch
import random

random.seed(42)
np.random.seed(1)
torch.manual_seed(42)

import gymnasium as gym

fp = 'results/TMQN/run_66/best_model'
policy = torch.load(fp)
env = gym.make("CartPole-v1")
file_name = 'q_valuess.csv'
save_path = './results/TMQN-n-step-TD/run_66'

import os
file_exists = os.path.exists(os.path.join(save_path, file_name))
seeds = [np.random.randint(1, 100000000) for i in range(100)]

def save(q_vals, step):
    with open(os.path.join(save_path, file_name), "a") as file:
        if not os.path.exists(os.path.join(save_path, file_name)):
            file.write("q1,q2,steps\n")
        file.write(f"{q_vals[0][0]},{q_vals[0][1]},{step}\n")

episode_rewards = np.array([0 for _ in range(100)])
actions = 0
_q_vals = [0, 0]
for episode in range(100):
    obs, _ = env.reset(seed=seeds[episode])  # episode)
    print(episode)
    while True:
        try:
            q_vals = policy.predict(obs)

            save(q_vals, actions)
            action = np.argmax(policy.predict(obs))
            actions += 1
            obs, reward, done, truncated, _ = env.step(action)
        except:
            action = torch.argmax(policy.predict(obs))
            obs, reward, done, truncated, _ = env.step(action.detach().numpy())
        episode_rewards[episode] += reward
        if done or truncated:
            break

mean = np.mean(episode_rewards)
std = np.std(episode_rewards)
import os
print(f'Mean reward: {mean}')
print(f'Mean std: {std}')
print(f'Actions: {actions}')
print(f'q values mean: q1 {_q_vals[0] / actions}, q1: {_q_vals[1] / actions}')

