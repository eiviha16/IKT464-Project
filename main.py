import gymnasium as gym
import random

#set random seed
random.seed(42)
#gymnasium.make("ALE/AirRaid-v5")

#env = gym.make("CartPole-v1", render_mode='human')
#env = gym.make("MountainCar-v0", render_mode='human')
env = gym.make("ALE/AirRaid-v5", render_mode='human')
observation, info = env.reset(seed=42)

for _ in range(10000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
exit(0)