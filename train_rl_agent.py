from stable_baselines3 import PPO
from rocket_tvc_env import RocketTVCEnv

env = RocketTVCEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

model.save("ppo_tvc")
