import numpy as np
from rocket_tvc_env import RocketTVCEnv
import matplotlib.pyplot as plt

def test_simulation():
    # Create environment with simulator enabled
    env = RocketTVCEnv(use_simulator=True)
    
    # Run simulation for 1000 steps
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(1000):
        # Simple PD controller for demonstration
        pitch_error = obs[0]  # Current pitch
        yaw_error = obs[1]    # Current yaw
        pitch_rate = obs[2]   # Current pitch rate
        yaw_rate = obs[3]     # Current yaw rate
        
        # PD control law
        pitch_action = -2.0 * pitch_error - 0.5 * pitch_rate
        yaw_action = -2.0 * yaw_error - 0.5 * yaw_rate
        
        # Clip actions to valid range
        action = np.clip([pitch_action, yaw_action], -10.0, 10.0)
        
        # Step environment
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        # Render current state
        env.render()
        
        if done:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break
    
    plt.show()

if __name__ == "__main__":
    test_simulation() 