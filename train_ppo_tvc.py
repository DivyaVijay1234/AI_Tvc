from stable_baselines3 import PPO
from rocket_tvc_env import RocketTVCEnv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.episode_lengths = []
        self.mean_rewards = []
        self.best_mean_reward = -np.inf
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes
        
    def _on_step(self):
        try:
            if len(self.model.ep_info_buffer) > 0:
                self.rewards.append(self.model.ep_info_buffer[-1]['r'])
                self.episode_lengths.append(self.model.ep_info_buffer[-1]['l'])
                
                # Calculate mean reward
                mean_reward = np.mean(self.rewards[-100:])
                self.mean_rewards.append(mean_reward)
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save("best_model")
                    logging.info(f"New best model saved! Mean reward: {mean_reward:.2f}")
                
                # Periodic save
                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    self.model.save(f"checkpoint_{int(current_time)}")
                    self.last_save_time = current_time
                    logging.info(f"Checkpoint saved at step {self.num_timesteps}")
            
            return True
        except Exception as e:
            logging.error(f"Error in training callback: {str(e)}")
            return False

def make_env():
    """
    Create and wrap the environment
    """
    try:
        env = RocketTVCEnv(use_simulator=True)
        return env
    except Exception as e:
        logging.error(f"Error creating environment: {str(e)}")
        raise

def train_ppo_agent(resume_training=False):
    """
    Train the PPO agent for rocket control
    """
    try:
        # Create and wrap the environment
        env = DummyVecEnv([make_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        # Initialize callbacks
        training_callback = TrainingCallback()
        
        # Create evaluation environment
        eval_env = DummyVecEnv([make_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./best_model",
            log_path="./logs/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Load existing model if resuming training
        if resume_training and os.path.exists("ppo_tvc_model.zip"):
            logging.info("Loading existing model for continued training...")
            model = PPO.load("ppo_tvc_model", env=env)
        else:
            # Create new PPO model with optimized parameters
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=1e-3,
                n_steps=1024,
                batch_size=128,
                n_epochs=5,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log="./logs/",
                policy_kwargs=dict(
                    net_arch=[dict(pi=[64, 64], vf=[64, 64])]
                )
            )
        
        # Create logs directory if it doesn't exist
        os.makedirs("./logs", exist_ok=True)
        
        # Train the model with fewer timesteps initially
        logging.info("\n🚀 Starting PPO Training")
        logging.info("This will take some time...")
        
        try:
            model.learn(
                total_timesteps=100_000,
                callback=[training_callback, eval_callback],
                progress_bar=True
            )
        except KeyboardInterrupt:
            logging.info("\nTraining interrupted by user. Saving model...")
            model.save("ppo_tvc_model_interrupted")
            env.save("vec_normalize_interrupted.pkl")
            return
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            model.save("ppo_tvc_model_error")
            env.save("vec_normalize_error.pkl")
            raise
        
        # Save the final model and normalization stats
        model.save("ppo_tvc_model")
        env.save("vec_normalize.pkl")
        logging.info("\n✅ Training complete! Model saved as 'ppo_tvc_model'")
        
        # Plot training results
        plot_training_results(training_callback)
        
    except Exception as e:
        logging.error(f"Error in training process: {str(e)}")
        raise

def plot_training_results(training_callback):
    """Plot and save training results"""
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(221)
        plt.plot(training_callback.rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(222)
        plt.plot(training_callback.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        # Plot moving average of rewards
        plt.subplot(223)
        window_size = 100
        moving_avg = np.convolve(training_callback.rewards, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        plt.plot(moving_avg)
        plt.title(f'Moving Average Reward (Window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        
        # Plot mean rewards
        plt.subplot(224)
        plt.plot(training_callback.mean_rewards)
        plt.title('Mean Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting results: {str(e)}")

if __name__ == "__main__":
    try:
        # Check if we should resume training
        resume = os.path.exists("ppo_tvc_model.zip")
        train_ppo_agent(resume_training=resume)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise 