from .sensor import Sensor
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *

import torch
import numpy as np

class PointCameraSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset
        
        print("Initializing the PointCameraSensor")

    def get_observation(self, env_ids = None):
        rb_states_tensor = self.env.gym.acquire_rigid_body_state_tensor(self.env.sim)
        rb_states = gymtorch.wrap_tensor(rb_states_tensor).view(self.env.num_envs, -1, 13)
        
        # Get the camera pose
        cam_idx = 4
        camera_pos, camera_quat = rb_states[...,cam_idx,0:3], rb_states[...,cam_idx,3:7]
        
        # Get the ball pose
        ball_idx = 6
        # ball_pos_world = rb_states[...,ball_idx,0:3]
        ball_pos_world = torch.tensor([5.0, 0.0, 5.0], device=rb_states.device).expand(4096, -1) # TODO: Debug, this just set the "ball position to a hardset value for reward function testing"
        
        # Project ball pose to camera
        camera_quat_inv, camera_pos_inv = tf_inverse(camera_quat, camera_pos)
        point_camera = tf_apply(camera_quat_inv, camera_pos_inv, ball_pos_world)

        #TODO: could do focal length and offset here....
        normalized = point_camera[:] / point_camera[:, 2].unsqueeze(1)
        normalized[point_camera[:, 2] <= 0] = 0 # Set points behind the camera to not be visble # TODO: do we need this? 
        u, v = normalized[:, 0], normalized[:, 1]
        
        # print("Position Camera: ", u[0], v[0])
       
        visibility_flag = (point_camera[:, 2] > 0).float() # 1 if visible
        
        
        return torch.stack([u, v, visibility_flag], dim=-1) #u, v, visibility_flag
    
    def get_noise_vec(self): # TODO: This could be the foviation thing!
        import torch
        return torch.ones(3, device=self.env.device) * self.env.cfg.noise_scales.ball_pos * self.env.cfg.noise.noise_level * self.env.cfg.obs_scales.ball_pos
    
    def get_dim(self):
        return 3