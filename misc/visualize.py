import torch
import os
import gymnasium as gym

path = '../results/DQN'
run = 'run_11'
file = 'best_model'
model = torch.load(os.path.join(path, run, file))

env = gym.make("CartPole-v1", render_mode='human')

observation, info = env.reset(seed=42)

while True:
    q_vals = model.predict(observation)
    action = torch.argmax(q_vals).numpy()
    observation, reward, done, truncated, info = env.step(action)

    if done:
        observation, info = env.reset()
env.close()
exit(0)