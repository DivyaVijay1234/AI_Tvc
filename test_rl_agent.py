from stable_baselines3 import PPO
from rocket_tvc_env import RocketTVCEnv

env = RocketTVCEnv()
model = PPO.load("ppo_tvc")

obs = env.reset()
for _ in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        break
