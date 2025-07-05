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
    Simplified callback for tracking training progress
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.episode_lengths = []
        self.mean_rewards = []
        self.best_mean_reward = -np.inf
        self.last_save_time = time.time()
        self.save_interval = 600  # Save every 10 minutes
        
    def _on_step(self):
        try:
            # Only process if we have episode info
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                try:
                    self.rewards.append(self.model.ep_info_buffer[-1]['r'])
                    self.episode_lengths.append(self.model.ep_info_buffer[-1]['l'])
                    
                    # Calculate mean reward
                    if len(self.rewards) > 0:
                        mean_reward = np.mean(self.rewards[-100:])
                        self.mean_rewards.append(mean_reward)
                        
                        # Save best model (less frequently)
                        if mean_reward > self.best_mean_reward and len(self.rewards) > 50:
                            self.best_mean_reward = mean_reward
                            self.model.save("best_model")
                            logging.info(f"New best model saved! Mean reward: {mean_reward:.2f}")
                    
                    # Periodic save (less frequently)
                    current_time = time.time()
                    if current_time - self.last_save_time > self.save_interval:
                        self.model.save(f"checkpoint_{int(current_time)}")
                        self.last_save_time = current_time
                        logging.info(f"Checkpoint saved at step {self.num_timesteps}")
                        
                except (KeyError, IndexError) as e:
                    # Ignore missing episode info
                    pass
            
            # Always return True to continue training
            return True
            
        except Exception as e:
            logging.error(f"Error in training callback: {str(e)}")
            # Return True even on error to prevent training from stopping
            return True

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
        
        # Initialize callbacks - simplified for debugging
        training_callback = TrainingCallback()
        
        # Create evaluation environment
        eval_env = DummyVecEnv([make_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./best_model",
            log_path="./logs/",
            eval_freq=50000,  # Much less frequent evaluation
            deterministic=True,
            render=False
        )
        
        # Load existing model if resuming training
        if resume_training and os.path.exists("ppo_tvc_model.zip"):
            logging.info("Loading existing model for continued training...")
            try:
                model = PPO.load("ppo_tvc_model", env=env)
                logging.info("Successfully loaded existing model")
            except Exception as e:
                logging.error(f"Failed to load existing model: {e}")
                logging.info("Creating new model instead...")
                resume_training = False
        
        if not resume_training:
            # Create new PPO model with more conservative parameters for stability
            logging.info("Creating new PPO model with conservative parameters...")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,  # Slightly higher for faster learning
                n_steps=512,         # Much smaller steps per update
                batch_size=32,       # Smaller batch size
                n_epochs=4,          # Fewer epochs per update
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.05,       # Higher entropy for exploration
                verbose=1,
                tensorboard_log="./logs/",
                policy_kwargs=dict(
                    net_arch=[dict(pi=[32, 32], vf=[32, 32])]  # Smaller network for faster training
                )
            )
            logging.info("New model created successfully")
        
        # Create logs directory if it doesn't exist
        os.makedirs("./logs", exist_ok=True)
        
        # Train the model with more timesteps
        logging.info("\n🚀 Starting PPO Training")
        logging.info("Training for 200,000 timesteps...")
        logging.info("This will take some time...")
        
        try:
            # Test environment first
            logging.info("Testing environment...")
            test_obs = env.reset()
            logging.info(f"Environment test successful. Observation shape: {test_obs[0].shape}")
            
            # Add more detailed logging during training
            logging.info("Starting training loop...")
            logging.info("If training appears stuck, it might be:")
            logging.info("1. Episodes ending quickly (normal for early training)")
            logging.info("2. Agent learning slowly (normal for complex tasks)")
            logging.info("3. Environment resets (check logs for details)")
            
            model.learn(
                total_timesteps=200_000,  # Increased from 100,000
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
            logging.error(f"Error type: {type(e).__name__}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
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
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
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