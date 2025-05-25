from stable_baselines3 import PPO
from rocket_tvc_env import RocketTVCEnv
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def run_rl_simulation():
    # Create environment with simulator enabled
    env = RocketTVCEnv(use_simulator=True)
    
    # Load the trained model
    try:
        model = PPO.load("ppo_tvc_model")
        print("✅ Successfully loaded PPO model")
    except:
        print("❌ Could not load PPO model. Using random actions instead.")
        model = None
    
    # Set random initial conditions
    initial_pitch = np.random.uniform(-0.2, 0.2)  # Random initial pitch (-11.5 to 11.5 degrees)
    initial_yaw = np.random.uniform(-0.2, 0.2)    # Random initial yaw (-11.5 to 11.5 degrees)
    initial_pitch_rate = np.random.uniform(-0.1, 0.1)  # Random initial pitch rate
    initial_yaw_rate = np.random.uniform(-0.1, 0.1)    # Random initial yaw rate
    
    # Reset environment with initial conditions
    obs, _ = env.reset()
    env.simulator.state.orientation[0] = initial_pitch
    env.simulator.state.orientation[1] = initial_yaw
    env.simulator.state.angular_velocity[0] = initial_pitch_rate
    env.simulator.state.angular_velocity[1] = initial_yaw_rate
    
    total_reward = 0
    step = 0
    
    # Lists to store trajectory data
    positions_x = []
    positions_z = []
    orientations = []
    rewards = []
    
    print("\n🚀 Starting Rocket Simulation")
    print(f"Initial Conditions:")
    print(f"Pitch: {np.degrees(initial_pitch):.1f}°")
    print(f"Yaw: {np.degrees(initial_yaw):.1f}°")
    print(f"Pitch Rate: {np.degrees(initial_pitch_rate):.1f}°/s")
    print(f"Yaw Rate: {np.degrees(initial_yaw_rate):.1f}°/s")
    
    # Run simulation
    while step < 1000:  # Maximum 1000 steps
        # Get action from model or random
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        
        # Step environment
        obs, reward, done, _, _ = env.step(action)
        
        # Store data
        positions_x.append(env.simulator.state.position[0])
        positions_z.append(env.simulator.state.position[2])
        orientations.append(env.simulator.state.orientation[0])
        rewards.append(reward)
        
        total_reward += reward
        step += 1
        
        # Render current state
        env.render()
        
        if done:
            break
    
    print(f"\n✅ Simulation finished after {step} steps")
    print(f"Total Reward: {total_reward:.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot trajectory
    plt.subplot(131)
    plt.plot(positions_x, positions_z)
    plt.title('Rocket Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.grid(True)
    
    # Plot orientation
    plt.subplot(132)
    plt.plot(np.degrees(orientations))
    plt.title('Rocket Pitch')
    plt.xlabel('Step')
    plt.ylabel('Pitch (degrees)')
    plt.grid(True)
    
    # Plot rewards
    plt.subplot(133)
    plt.plot(rewards)
    plt.title('Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_rl_simulation()