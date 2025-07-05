import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import matplotlib.animation as animation

@dataclass
class RocketState:
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    orientation: np.ndarray  # [pitch, yaw, roll]
    angular_velocity: np.ndarray  # [pitch_rate, yaw_rate, roll_rate]

class RocketSimulator:
    def __init__(self):
        # Rocket physical parameters
        self.mass = 1.0  # kg
        self.length = 1.0  # m
        self.radius = 0.05  # m
        self.moment_of_inertia = np.array([0.1, 0.1, 0.01])  # kg*m^2
        self.thrust = 20.0  # N
        self.gimbal_max_angle = np.radians(10)  # 10 degrees
        
        # Environmental parameters
        self.gravity = np.array([0, 0, -9.81])  # m/s^2
        self.air_density = 1.225  # kg/m^3
        self.drag_coefficient = 0.5
        
        # Simulation parameters
        self.dt = 0.005  # Smaller time step for slower movement
        self.state = RocketState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3)
        )
        
        # Store last action for visualization
        self.last_action = np.array([0.0, 0.0])
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('🚀 AI-TV Rocket Simulation', fontsize=16, fontweight='bold')
        
        # Initialize visualization elements
        self.rocket_patch = None
        self.thrust_patch = None
        self.thrust_arrow = None
        self.velocity_arrow = None
        self.gravity_arrow = None
        self.angle_text = None
        self.performance_text = None
        self.gimbal_text = None
        
        # Force text creation on first render
        self._text_created = False
        self.step_counter = 0
        self.total_reward = 0
        
    def reset(self):
        """Reset the rocket to initial state"""
        self.state = RocketState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3)
        )
        self.step_counter = 0
        self.total_reward = 0
        return self.get_observation()
    
    def get_observation(self):
        """Get current state observation"""
        return np.concatenate([
            self.state.orientation[:2],  # pitch, yaw
            self.state.angular_velocity[:2]  # pitch_rate, yaw_rate
        ])
    
    def step(self, action):
        """Simulate one time step with given gimbal angles"""
        pitch_gimbal, yaw_gimbal = np.clip(action, -self.gimbal_max_angle, self.gimbal_max_angle)
        
        # Store action for visualization
        self.last_action = np.array([pitch_gimbal, yaw_gimbal])
        
        # Calculate thrust vector in rocket frame
        thrust_direction = np.array([
            np.sin(pitch_gimbal),
            np.sin(yaw_gimbal),
            np.cos(pitch_gimbal) * np.cos(yaw_gimbal)
        ])
        thrust_vector = self.thrust * thrust_direction
        
        # Calculate forces and moments
        gravity_force = self.mass * self.gravity
        drag_force = self._calculate_drag()
        total_force = thrust_vector + gravity_force + drag_force
        
        # Calculate moments
        thrust_moment = np.cross(np.array([0, 0, -self.length/2]), thrust_vector)
        drag_moment = np.cross(np.array([0, 0, -self.length/2]), drag_force)
        total_moment = thrust_moment + drag_moment
        
        # Update state
        self.state.velocity += (total_force / self.mass) * self.dt
        self.state.position += self.state.velocity * self.dt
        self.state.angular_velocity += (total_moment / self.moment_of_inertia) * self.dt
        self.state.orientation += self.state.angular_velocity * self.dt
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        self.step_counter += 1
        
        # Check if episode is done
        done = self._check_termination()
        
        return self.get_observation(), reward, done, False, {}
    
    def _calculate_drag(self):
        """Calculate drag force based on velocity"""
        velocity_magnitude = np.linalg.norm(self.state.velocity)
        if velocity_magnitude < 1e-6:
            return np.zeros(3)
        
        drag_force = -0.5 * self.air_density * self.drag_coefficient * \
                    np.pi * self.radius**2 * velocity_magnitude * self.state.velocity
        return drag_force
    
    def _calculate_reward(self):
        """Calculate reward based on stability - much softer for training"""
        # Very soft orientation penalties (much more forgiving)
        pitch_penalty = -1.0 * self.state.orientation[0]**2  # Much softer penalty for pitch
        yaw_penalty = -1.0 * self.state.orientation[1]**2    # Much softer penalty for yaw
        
        # Very soft angular rate penalties
        pitch_rate_penalty = -0.2 * self.state.angular_velocity[0]**2
        yaw_rate_penalty = -0.2 * self.state.angular_velocity[1]**2
        
        # Very soft position penalty
        position_penalty = -0.01 * (self.state.position[0]**2 + self.state.position[1]**2)
        
        # More generous stability bonus
        stability_bonus = 0.0
        if abs(self.state.orientation[0]) < 0.3 and abs(self.state.orientation[1]) < 0.3:  # ~17 degrees
            stability_bonus = 1.0
        
        # Larger survival bonus
        survival_bonus = 0.5
        
        # Bonus for being close to vertical
        vertical_bonus = 0.0
        if abs(self.state.orientation[0]) < 0.1 and abs(self.state.orientation[1]) < 0.1:  # ~5.7 degrees
            vertical_bonus = 2.0
        
        total_reward = pitch_penalty + yaw_penalty + pitch_rate_penalty + yaw_rate_penalty + position_penalty + stability_bonus + survival_bonus + vertical_bonus
        return total_reward
    
    def _check_termination(self):
        """Check if episode should terminate"""
        # Terminate if rocket is too far from vertical (more lenient)
        if np.any(np.abs(self.state.orientation[:2]) > np.radians(60)):
            return True
        # Terminate if rocket has fallen too far
        if self.state.position[2] < -15:
            return True
        # Add maximum episode length to prevent infinite episodes
        if self.step_counter > 500:  # Shorter episodes for faster training
            return True
        return False
    
    def render(self):
        """Simple, clear rocket visualization"""
        # Clear and setup
        self.ax.clear()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('🚀 Rocket TVC Simulation', fontsize=14)
        
        # Draw rocket
        rocket_x = self.state.position[0]
        rocket_z = self.state.position[2]
        
        from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
        # Rocket body - much more prominent
        rocket = Rectangle(
            (rocket_x - self.radius, rocket_z - self.length/2),
            2*self.radius, self.length,
            angle=np.degrees(self.state.orientation[0]),
            fill=True, linewidth=3, color='darkblue', alpha=0.8
        )
        self.ax.add_patch(rocket)
        
        # Rocket outline for better definition
        rocket_outline = Rectangle(
            (rocket_x - self.radius, rocket_z - self.length/2),
            2*self.radius, self.length,
            angle=np.degrees(self.state.orientation[0]),
            fill=False, linewidth=2, color='white'
        )
        self.ax.add_patch(rocket_outline)
        
        # Thrust indicator - more prominent
        thrust = Circle(
            (rocket_x, rocket_z - self.length/2),
            self.radius/2, color='red', alpha=0.9, linewidth=2, edgecolor='darkred'
        )
        self.ax.add_patch(thrust)
        
        # Add thrust vector arrow for clarity
        thrust_start = np.array([rocket_x, rocket_z - self.length/2])
        thrust_end = thrust_start + 0.4 * np.array([np.sin(self.last_action[0]), np.cos(self.last_action[0])])
        thrust_arrow = FancyArrowPatch(
            thrust_start, thrust_end,
            arrowstyle='->', color='red', linewidth=3, mutation_scale=25, label='Thrust'
        )
        self.ax.add_patch(thrust_arrow)
        
        # Add velocity arrow (green)
        if np.linalg.norm(self.state.velocity[:2]) > 0.01:
            vel_start = np.array([rocket_x, rocket_z])
            vel_end = vel_start + 0.3 * self.state.velocity[:2]
            velocity_arrow = FancyArrowPatch(
                vel_start, vel_end,
                arrowstyle='->', color='green', linewidth=2, mutation_scale=20, label='Velocity'
            )
            self.ax.add_patch(velocity_arrow)
        
        # Add gravity arrow (purple)
        gravity_start = np.array([rocket_x, rocket_z])
        gravity_end = gravity_start + 0.2 * np.array([0, -1])  # Downward
        gravity_arrow = FancyArrowPatch(
            gravity_start, gravity_end,
            arrowstyle='->', color='purple', linewidth=2, mutation_scale=20, label='Gravity'
        )
        self.ax.add_patch(gravity_arrow)
        
        # Add angular velocity indicator (orange)
        if np.linalg.norm(self.state.angular_velocity[:2]) > 0.01:
            ang_vel_start = np.array([rocket_x + 0.3, rocket_z])
            ang_vel_end = ang_vel_start + 0.2 * np.array([self.state.angular_velocity[0], self.state.angular_velocity[1]])
            ang_vel_arrow = FancyArrowPatch(
                ang_vel_start, ang_vel_end,
                arrowstyle='->', color='orange', linewidth=2, mutation_scale=15, label='Angular Velocity'
            )
            self.ax.add_patch(ang_vel_arrow)
        
        # Add trajectory line (store positions for path)
        if not hasattr(self, 'trajectory_x'):
            self.trajectory_x = []
            self.trajectory_z = []
        
        self.trajectory_x.append(rocket_x)
        self.trajectory_z.append(rocket_z)
        
        # Keep only last 100 points to avoid clutter
        if len(self.trajectory_x) > 100:
            self.trajectory_x = self.trajectory_x[-100:]
            self.trajectory_z = self.trajectory_z[-100:]
        
        # Draw trajectory
        if len(self.trajectory_x) > 1:
            self.ax.plot(self.trajectory_x, self.trajectory_z, 'g-', linewidth=1, alpha=0.6, label='Trajectory')
        
        # Add text displays
        pitch_deg = np.degrees(self.state.orientation[0])
        yaw_deg = np.degrees(self.state.orientation[1])
        
        # Status text
        self.ax.text(1.0, 1.4, 
            f'Pitch: {pitch_deg:6.1f}°\nYaw: {yaw_deg:6.1f}°',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        # Performance text
        self.ax.text(-1.9, -1.8,
            f'Step: {self.step_counter} | Reward: {self.total_reward:.1f}',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        
        # Gimbal text
        self.ax.text(1.0, -1.8,
            f'Pitch Gimbal: {np.degrees(self.last_action[0]):6.1f}°\nYaw Gimbal: {np.degrees(self.last_action[1]):6.1f}°',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
        
        # Add instruction text
        self.ax.text(-1.9, 1.6, 
            'Red = Thrust | Green = Velocity | Purple = Gravity | Orange = Angular Velocity',
            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Add legend
        self.ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.7), fontsize=8)
        
        # Update display
        plt.draw()
        plt.pause(0.3)  # Much slower for better observation 