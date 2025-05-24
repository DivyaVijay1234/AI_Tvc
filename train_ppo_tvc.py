from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from rocket_tvc_env import RocketTVCEnv

def main():
    # Create vectorized env with Monitor wrapper applied to each individual env
    env = make_vec_env(RocketTVCEnv, n_envs=4, monitor_dir="./logs/monitor") 

    # Note: make_vec_env automatically applies Monitor wrapper if you provide monitor_dir

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_tvc")

    eval_env = RocketTVCEnv()
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path="./models/",
                                 log_path="./logs/eval/",
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=False)

    model.learn(total_timesteps=100_000, callback=eval_callback, tb_log_name="ppo_tvc_experiment")

    model.save("ppo_tvc_model")
    print("✅ Training complete and model saved!")

if __name__ == "__main__":
    main()
