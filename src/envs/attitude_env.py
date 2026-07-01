import gymnasium as gym 
import torch
import numpy as np
from omegaconf import DictConfig
from src.envs.utils import(
    random_unit_quaternion,
    omega_matrix,
    quaternion_error,
    rotate_vector,
    random_unit_vector,
    rotate_around_axis,
    to_obs
)

class AttitudeEnv(gym.Env):
    """Custom Gynasium ENV for SAC Agent"""
    def __init__(self, config: DictConfig):
        """Define observation space and action space"""
        self.alpha: float = config.alpha
        self.beta: float = config.beta
        self.f_zone_half_angle:float = np.radians(config.f_zone_half_angle)
        self.boresight_axis: np.ndarray = config.boresight_axis
        self.tau_max: torch.tensor = config.tau_max
        self.inertia: np.ndarray = np.diag(config.inertia_tensor)
        self.dt: int = config.dt
        self.max_steps: int = config.max_steps
        self.q_target: torch.tensor = config.q_target
        self.observation_space: gym.spaces.Box = gym.spaces.Box(low = -1.0, 
                                                                high = 1.0, 
                                                                shape = (11,))
        self.action_space = gym.spaces.Box(low = -self.tau_max, 
                                           high = self.tau_max, 
                                           shape = (3,))
        self.success_threshold: float = config.success_threshold
        self.success_bonus: float = config.success_bonus
        self.terminated: bool = False
        self.truncated: bool = False
        self.q: np.ndarray = random_unit_quaternion()
        self.omega: np.ndarray = np.zeros(3)
        self.t: int = 0
        self.sun_vector = random_unit_vector()
        self.orbital_rate = config.orbital_rate
    def reset(self, seed = None, options = {}) -> np.ndarray:
        """Return initial (objects, info)"""
        self.q: np.ndarray = random_unit_quaternion()
        self.omega: np.ndarray = np.zeros(3, dtype=np.float32)
        self.t: int = 0
        self.truncated = False
        self.terminated = False
        self.sun_vector = random_unit_vector()
        return self._get_obs(), self.render()   
    def step(self,action) -> tuple:
        """return (objs, reward, terminated,truncated info)"""
        ang_momentum: np.ndarray = self.inertia @ self.omega
        torque: np.ndarray = action - np.cross(self.omega, ang_momentum)

        dw_dt: np.ndarray = np.linalg.solve(self.inertia, torque) #optimization
        self.omega: np.ndarray = self.omega + dw_dt*self.dt
        dq_dt: np.ndarray = 0.5*omega_matrix(self.omega) @ self.q
        self.q: np.ndarray = self.q + dq_dt*self.dt
        self.q = self.q / np.linalg.norm(self.q)

        self.sun_vector = rotate_around_axis(self.sun_vector, self.orbital_rate*self.dt)
        self.t+=1
        fzone_violated = self._check_fzone_violation()
        if fzone_violated:
            self.terminated = True
        return self._get_obs(), self.reward(fzone_violated), self.terminated, self.truncated, {}
    def reward(self, fzone_violated) -> float:
        """Calculates reward for current episode"""
        q_error = quaternion_error(self.q, self.q_target)
        angle_error = 2 * np.arccos(np.clip(np.abs(q_error[0]), -1.0, 1.0))
        r_attitude = -angle_error
        r_velocity = -self.beta*np.linalg.norm(self.omega)
        correct = angle_error < self.success_threshold
        if correct:
            self.terminated = True
        if self.t >= self.max_steps:
            self.truncated = True
        r_success = self.success_bonus if correct else 0.0
        r_fzone = -self.alpha*20 if fzone_violated else 0.0
        return r_attitude + r_velocity + r_success + r_fzone
    def _check_fzone_violation(self) -> bool:
        """Checks forbidden zone condition"""
        boresight_inertial = rotate_vector(self.q, self.boresight_axis)
        boresight_project = np.dot(boresight_inertial, self.sun_vector)
        return boresight_project > np.cos(self.f_zone_half_angle)

    def _get_obs(self) -> np.ndarray:
        """helper to package state into observation vector"""
        e_1, e_2, e_3, e_4 = quaternion_error(self.q, self.q_target)
        obs_vector = to_obs(np.array([self.q[0], self.q[1], self.q[2],self.q[3],
                      self.omega[0], self.omega[1],self.omega[2], 
                      e_1, e_2, e_3,e_4]))
        return obs_vector
    def render(self) -> dict:
        return {}
    