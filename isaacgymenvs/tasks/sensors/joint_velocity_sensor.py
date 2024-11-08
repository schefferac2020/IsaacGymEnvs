from .sensor import Sensor
import torch

class JointVelocitySensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        return self.env.dof_velocities[:, 2:]
    
    def get_noise_vec(self):
        return torch.zeros(self.env.num_actions, device=self.env.device)
        # return torch.ones(self.env.num_actuated_dof, device=self.env.device) * \
        #     self.env.cfg.noise_scales.dof_pos * \
        #     self.env.cfg.noise.noise_level * \
        #     self.env.cfg.obs_scales.dof_pos
    
    def get_dim(self):
        return self.env.num_actuated_dof