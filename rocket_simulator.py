import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
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
        self.dt = 0.01  # time step
        self.state = RocketState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3)
        )
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.rocket_patch = None
        self.thrust_patch = None
        
    def reset(self):
        """Reset the rocket to initial state"""
        self.state = RocketState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3)
        )
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
        """Calculate reward based on stability"""
        orientation_penalty = -np.sum(self.state.orientation[:2]**2)  # Penalize pitch and yaw
        angular_penalty = -0.1 * np.sum(self.state.angular_velocity[:2]**2)  # Penalize angular rates
        return orientation_penalty + angular_penalty
    
    def _check_termination(self):
        """Check if episode should terminate"""
        # Terminate if rocket is too far from vertical
        if np.any(np.abs(self.state.orientation[:2]) > np.radians(45)):
            return True
        # Terminate if rocket has fallen too far
        if self.state.position[2] < -10:
            return True
        return False
    
    def render(self):
        """Render the current state of the rocket"""
        if self.rocket_patch is None:
            # Create rocket body
            self.rocket_patch = Rectangle(
                (-self.radius, -self.length/2),
                2*self.radius,
                self.length,
                angle=np.degrees(self.state.orientation[0]),
                fill=False
            )
            self.ax.add_patch(self.rocket_patch)
            
            # Create thrust indicator
            self.thrust_patch = Circle(
                (0, -self.length/2),
                self.radius/2,
                color='red'
            )
            self.ax.add_patch(self.thrust_patch)
        
        # Update rocket position and orientation
        self.rocket_patch.set_xy((self.state.position[0] - self.radius, 
                                 self.state.position[2] - self.length/2))
        self.rocket_patch.set_angle(np.degrees(self.state.orientation[0]))
        
        # Update thrust position
        self.thrust_patch.set_center((
            self.state.position[0],
            self.state.position[2] - self.length/2
        ))
        
        plt.draw()
        plt.pause(0.001) 