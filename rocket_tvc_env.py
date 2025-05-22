import gym
from gym import spaces
import numpy as np

class RocketTVCEnv(gym.Env):
    def __init__(self):
        super(RocketTVCEnv, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)

        self.alpha = 0.1
        self.reset()

    def reset(self):
        self.pitch = np.random.uniform(-5, 5)
        self.yaw = np.random.uniform(-5, 5)
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.pitch, self.yaw, self.pitch_rate, self.yaw_rate], dtype=np.float32)

    def step(self, action):
        pitch_gimbal, yaw_gimbal = action

        self.pitch_rate += pitch_gimbal * 0.01
        self.yaw_rate += yaw_gimbal * 0.01

        self.pitch += self.pitch_rate
        self.yaw += self.yaw_rate

        reward = - (self.pitch**2 + self.yaw**2 + self.alpha * (self.pitch_rate**2 + self.yaw_rate**2))

        self.t += 1
        done = self.t > 200
        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        print(f"Pitch: {self.pitch:.2f}, Yaw: {self.yaw:.2f}")
