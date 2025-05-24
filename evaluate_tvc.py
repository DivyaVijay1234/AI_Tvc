from stable_baselines3 import PPO
from rocket_tvc_env import RocketTVCEnv
import time

# Load environment and model
env = RocketTVCEnv()
model = PPO.load("ppo_tvc")

# Run the trained model
obs = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.05)
    if done:
        break
