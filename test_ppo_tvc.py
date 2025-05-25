from stable_baselines3 import PPO
from rocket_tvc_env import RocketTVCEnv
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from datetime import datetime

def run_rl_simulation(num_episodes=5):
    """
    Run multiple episodes of rocket simulation using the trained RL model
    """
    # Create environment with simulator enabled
    env = RocketTVCEnv(use_simulator=True)
    
    # Load the trained model
    try:
        model = PPO.load("ppo_tvc_model")
        print("✅ Successfully loaded PPO model")
    except Exception as e:
        print(f"❌ Could not load PPO model: {str(e)}")
        print("Using random actions instead.")
        model = None
    
    # Create results directory
    results_dir = "simulation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize episode statistics
    episode_rewards = []
    episode_lengths = []
    episode_max_angles = []
    episode_final_positions = []
    
    # Store all trajectories for comparison
    all_trajectories = []
    
    for episode in range(num_episodes):
        print(f"\n🚀 Starting Episode {episode + 1}/{num_episodes}")
        
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
        actions = []
        
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
            actions.append(action)
            
            total_reward += reward
            step += 1
            
            # Render current state
            env.render()
            
            if done:
                break
        
        # Store trajectory for comparison
        all_trajectories.append((positions_x, positions_z))
        
        # Calculate episode statistics
        max_angle = max(np.abs(np.degrees(orientations)))
        final_position = np.sqrt(positions_x[-1]**2 + positions_z[-1]**2)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        episode_max_angles.append(max_angle)
        episode_final_positions.append(final_position)
        
        print(f"\nEpisode {episode + 1} Results:")
        print(f"Steps: {step}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Max Angle: {max_angle:.1f}°")
        print(f"Final Position: {final_position:.2f}m")
        
        # Save episode data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_data = {
            'positions_x': positions_x,
            'positions_z': positions_z,
            'orientations': orientations,
            'rewards': rewards,
            'actions': actions,
            'initial_conditions': {
                'pitch': initial_pitch,
                'yaw': initial_yaw,
                'pitch_rate': initial_pitch_rate,
                'yaw_rate': initial_yaw_rate
            },
            'statistics': {
                'total_reward': total_reward,
                'steps': step,
                'max_angle': max_angle,
                'final_position': final_position
            }
        }
        np.save(f"{results_dir}/episode_{episode+1}_{timestamp}.npy", episode_data)
        
        # Plot episode results
        plot_episode_results(positions_x, positions_z, orientations, rewards, actions, episode + 1)
    
    # Plot trajectory comparison
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_trajectories)))
    for i, (x, z) in enumerate(all_trajectories):
        plt.plot(x, z, '-', color=colors[i], linewidth=2, label=f'Episode {i+1}')
    plt.plot([0, max([max(x) for x, _ in all_trajectories])], 
             [0, max([max(z) for _, z in all_trajectories])], 
             'r--', alpha=0.5, label='Ideal Path')
    plt.title('Trajectory Comparison Across Episodes')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.grid(True)
    plt.legend()
    plt.savefig('simulation_results/trajectory_comparison.png')
    plt.close()
    
    # Print overall statistics
    print("\n📊 Overall Statistics:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Max Angle: {np.mean(episode_max_angles):.1f}° ± {np.std(episode_max_angles):.1f}°")
    print(f"Average Final Position: {np.mean(episode_final_positions):.2f}m ± {np.std(episode_final_positions):.2f}m")
    
    # Plot overall statistics
    plot_overall_statistics(episode_rewards, episode_lengths, episode_max_angles, episode_final_positions)

def plot_episode_results(positions_x, positions_z, orientations, rewards, actions, episode_num):
    """Plot results for a single episode"""
    plt.figure(figsize=(15, 10))
    
    # Plot trajectory
    plt.subplot(231)
    plt.plot(positions_x, positions_z, 'b-', linewidth=2)
    plt.plot([0, positions_x[-1]], [0, positions_z[-1]], 'r--', alpha=0.5, label='Ideal Path')
    plt.title('Rocket Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.grid(True)
    plt.legend()
    
    # Plot orientation
    plt.subplot(232)
    plt.plot(np.degrees(orientations), 'g-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    plt.title('Rocket Pitch')
    plt.xlabel('Step')
    plt.ylabel('Pitch (degrees)')
    plt.grid(True)
    plt.legend()
    
    # Plot rewards
    plt.subplot(233)
    plt.plot(rewards, 'm-', linewidth=2)
    plt.title('Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot actions
    plt.subplot(234)
    actions = np.array(actions)
    plt.plot(actions[:, 0], 'b-', label='Pitch Action', linewidth=2)
    plt.plot(actions[:, 1], 'g-', label='Yaw Action', linewidth=2)
    plt.title('Control Actions')
    plt.xlabel('Step')
    plt.ylabel('Action Value')
    plt.legend()
    plt.grid(True)
    
    # Plot velocity
    plt.subplot(235)
    velocities = np.diff(positions_x) / np.diff(np.arange(len(positions_x)))
    plt.plot(velocities, 'c-', linewidth=2)
    plt.title('Horizontal Velocity')
    plt.xlabel('Step')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    
    # Plot acceleration
    plt.subplot(236)
    accelerations = np.diff(velocities)
    plt.plot(accelerations, 'y-', linewidth=2)
    plt.title('Horizontal Acceleration')
    plt.xlabel('Step')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)
    
    plt.suptitle(f'Episode {episode_num} Results')
    plt.tight_layout()
    plt.savefig(f'simulation_results/episode_{episode_num}_results.png')
    plt.close()

def plot_overall_statistics(episode_rewards, episode_lengths, episode_max_angles, episode_final_positions):
    """Plot overall statistics across all episodes"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards over episodes
    plt.subplot(221)
    plt.plot(episode_rewards, 'b-o', linewidth=2, markersize=8)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot episode lengths
    plt.subplot(222)
    plt.plot(episode_lengths, 'g-o', linewidth=2, markersize=8)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Plot max angles
    plt.subplot(223)
    plt.plot(episode_max_angles, 'r-o', linewidth=2, markersize=8)
    plt.title('Maximum Angles')
    plt.xlabel('Episode')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    
    # Plot final positions
    plt.subplot(224)
    plt.plot(episode_final_positions, 'm-o', linewidth=2, markersize=8)
    plt.title('Final Positions')
    plt.xlabel('Episode')
    plt.ylabel('Distance (m)')
    plt.grid(True)
    
    plt.suptitle('Overall Statistics')
    plt.tight_layout()
    plt.savefig('simulation_results/overall_statistics.png')
    plt.show()

if __name__ == "__main__":
    run_rl_simulation(num_episodes=5)  # Run 5 episodes by default