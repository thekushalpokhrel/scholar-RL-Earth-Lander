import gymnasium as gym
import earth_lander
import time

env = gym.make('Earth_Lander-v2', render_mode="human")
observation, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random actions
    observation, _, done, _, _ = env.step(action)
    if done:
        observation, _ = env.reset()
    time.sleep(0.05)
env.close()