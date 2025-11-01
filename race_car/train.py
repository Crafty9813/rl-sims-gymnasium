import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: gym.make("CarRacing-v3")])

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

test_env = gym.make("CarRacing-v3", render_mode="human")

for episode in range(10):
    obs, info = test_env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        test_env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)

test_env.close()
