import gymnasium
from gymnasium import spaces
import numpy as np
from imu_reader import get_imu_data
from servo_controller import send_to_servos
from rocket_simulator import RocketSimulator

class RocketTVCEnv(gymnasium.Env):
    def __init__(self, use_simulator=True):
        super(RocketTVCEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.use_simulator = use_simulator
        
        if self.use_simulator:
            self.simulator = RocketSimulator()
        else:
            self.alpha = 0.1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.use_simulator:
            return self.simulator.reset(), {}
        else:
            self.pitch, self.yaw, self.pitch_rate, self.yaw_rate = get_imu_data()
            self.t = 0
            return self._get_obs(), {}

    def _get_obs(self):
        if self.use_simulator:
            return self.simulator.get_observation()
        return np.array([self.pitch, self.yaw, self.pitch_rate, self.yaw_rate], dtype=np.float32)

    def step(self, action):
        if self.use_simulator:
            return self.simulator.step(action)
        else:
            pitch_gimbal, yaw_gimbal = action
            send_to_servos(pitch_gimbal, yaw_gimbal)

            self.pitch_rate += pitch_gimbal * 0.01
            self.yaw_rate += yaw_gimbal * 0.01
            self.pitch += self.pitch_rate
            self.yaw += self.yaw_rate

            reward = - (self.pitch**2 + self.yaw**2 + self.alpha * (self.pitch_rate**2 + self.yaw_rate**2))
            self.t += 1
            done = self.t > 200
            return self._get_obs(), reward, done, False, {}

    def render(self, mode="human"):
        if self.use_simulator:
            self.simulator.render()
        else:
            print(f"Pitch: {self.pitch:.2f}, Yaw: {self.yaw:.2f}")
