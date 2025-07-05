#!/usr/bin/env python3
"""
Safe training script that starts without callbacks and gradually adds them
"""

from stable_baselines3 import PPO
from rocket_tvc_env import RocketTVCEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_env():
    return RocketTVCEnv(use_simulator=True)

def safe_training():
    """Safe training that starts simple and adds complexity"""
    logger.info("🚀 Starting Safe PPO Training")
    
    # Create environment
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create model with simple parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,  # Small batch
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    logger.info("Phase 1: Training without callbacks (first 10k steps)")
    
    # Phase 1: Train without callbacks
    model.learn(
        total_timesteps=10000,
        progress_bar=True
    )
    
    logger.info("Phase 1 completed! Saving checkpoint...")
    model.save("phase1_model")
    
    logger.info("Phase 2: Training with basic callbacks (next 50k steps)")
    
    # Phase 2: Add simple callbacks
    from stable_baselines3.common.callbacks import EvalCallback
    
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        progress_bar=True
    )
    
    logger.info("Phase 2 completed! Saving checkpoint...")
    model.save("phase2_model")
    
    logger.info("Phase 3: Full training with all callbacks (remaining steps)")
    
    # Phase 3: Add custom callback
    from stable_baselines3.common.callbacks import BaseCallback
    import time
    import numpy as np
    
    class SimpleCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.rewards = []
            
        def _on_step(self):
            try:
                if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                    self.rewards.append(self.model.ep_info_buffer[-1]['r'])
                    
                    if len(self.rewards) % 100 == 0:
                        avg_reward = np.mean(self.rewards[-100:])
                        logger.info(f"Step {self.num_timesteps}: Avg reward = {avg_reward:.2f}")
                        
            except Exception as e:
                logger.error(f"Callback error: {e}")
            
            return True
    
    simple_callback = SimpleCallback()
    
    model.learn(
        total_timesteps=140000,  # Complete the 200k total
        callback=[eval_callback, simple_callback],
        progress_bar=True
    )
    
    logger.info("✅ All training phases completed!")
    model.save("final_model")
    env.save("vec_normalize.pkl")

if __name__ == "__main__":
    safe_training() 