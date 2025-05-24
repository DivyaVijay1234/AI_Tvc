from stable_baselines3 import PPO
from rocket_tvc_env import RocketTVCEnv
import csv

env = RocketTVCEnv()
model = PPO.load("ppo_tvc_model")

obs, _ = env.reset()
total_reward = 0

with open("tvc_control_log.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Step", "Pitch", "Yaw", "PitchRate", "YawRate", "PitchGimbal", "YawGimbal"])

    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        pitch, yaw, pitch_rate, yaw_rate = obs
        pitch_gimbal, yaw_gimbal = action
        writer.writerow([step, pitch, yaw, pitch_rate, yaw_rate, pitch_gimbal, yaw_gimbal])
        total_reward += reward
        env.render()
        if done:
            break

print(f"✅ Episode finished. Total Reward: {total_reward:.2f}")
print("📁 Control output saved to tvc_control_log.csv")
print("✅ Evaluation complete!")
# Note: The above code assumes that the RocketTVCEnv class has been defined in the rocket_tvc_env.py file.