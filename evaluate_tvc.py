from stable_baselines3 import PPO
from rocket_tvc_env import RocketTVCEnv
import time

# Load environment and model
env = RocketTVCEnv(use_simulator=True)

# Try to load the best available model
model_names = ["final_model", "phase2_model", "phase1_model", "ppo_tvc_model", "minimal_model"]
model = None

for model_name in model_names:
    try:
        model = PPO.load(model_name)
        print(f"✅ Successfully loaded {model_name}")
        break
    except Exception as e:
        print(f"❌ Could not load {model_name}: {str(e)}")
        continue

if model is None:
    print("❌ Could not load any trained model")
    exit(1)

# Run the trained model
obs = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.05)
    if done:
        break
