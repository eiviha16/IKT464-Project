import gymnasium as gym
import random
import numpy as np
#set random seed
random.seed(42)

env = gym.make("CartPole-v1", render_mode='human')

observations = []
observation, info = env.reset(seed=random.randint(1, 10000))

for _ in range(10_000):
    observations.append(observation)
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=random.randint(1, 10000))
env.close()

np.savetxt('observations4.txt', observations, delimiter=',', fmt='%f')

exit(0)