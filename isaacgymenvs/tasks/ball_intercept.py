# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import to_torch, torch_rand_float, tensor_clamp, torch_random_dir_2
from .base.vec_task import VecTask
import cv2
import random as randy




def _indent_xml(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _indent_xml(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


class BallIntercept(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        

        
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.action_speed_scale = self.cfg["env"]["actionSpeedScale"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]
        
        self.fpv_render = (self.cfg["render_fpv_vid"] and not headless) # render a video if it's testing and not headless
        self.fpv_imgs = []
        self.complete_fpv_imgs = []
        self.fpv_env = 0

        actors_per_env = 2
        dofs_per_env = 4
        
        self.a = 0

        self.cfg["env"]["numObservations"] = (3 + 4 + 2 + 2)*2
        self.prev_obs = None
        
       

        # Actions: target velocities for the 3 actuated DOFs
        self.cfg["env"]["numActions"] = 4

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)



        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # self.sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
                
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, -1, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]
        
        

        self.root_states = vec_root_tensor
        # self.base_pos = 
        self.tray_positions = vec_root_tensor[..., 0, 0:3] # TODO: Meaningless
        self.ball_positions = vec_root_tensor[..., 1, 0:3] # T
        self.ball_orientations = vec_root_tensor[..., 1, 3:7]
        self.ball_linvels = vec_root_tensor[..., 1, 7:10]
        self.ball_angvels = vec_root_tensor[..., 1, 10:13]
        
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # self.initial_dof_states = self.dof_states.clone()
        self.initial_root_states = vec_root_tensor.clone()
        print("This is the shape of the initial root_states tensor:",self.root_states.shape )

        self.dof_position_targets = torch.zeros((self.num_envs, dofs_per_env), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(actors_per_env * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, actors_per_env)
        self.all_bbot_indices = actors_per_env * torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # vis
        self.axes_geom = gymutil.AxesGeometry(0.2)
        

    def __del__(self):
        if self.fpv_render:
            fpv_vid_imgs = self.fpv_imgs
            if len(self.complete_fpv_imgs) > 0:
                fpv_vid_imgs = self.complete_fpv_imgs
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20 #int(1.0/self.dt)
            out = cv2.VideoWriter("./videos/fpv.mp4", fourcc, fps, (fpv_vid_imgs[0].shape[0], fpv_vid_imgs[0].shape[1]))
            
            for image in fpv_vid_imgs:
                out.write(image)
            out.release()
            print("example video has been saved to: ./videos/fpv.mp4")
            cv2.destroyAllWindows()

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # self._create_balance_bot_asset()
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole2.urdf"
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        
        # Get jiont limits
        robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        self.bbot_dof_lower_limits = []
        self.bbot_dof_upper_limits = []
        for i in range(self.num_dof):
            self.bbot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.bbot_dof_upper_limits.append(robot_dof_props['upper'][i])

        self.bbot_dof_lower_limits = to_torch(self.bbot_dof_lower_limits, device=self.device)
        self.bbot_dof_upper_limits = to_torch(self.bbot_dof_upper_limits, device=self.device)
        
        
        
        print("This is the number of DOFs:", self.num_dof)
        print("Num bodies:", self.num_bodies)
        print("DOF Names:", self.dof_names)
        
        # asset is rotated z-up by default, no additional rotations needed
        pose = gymapi.Transform()
        pose.p.z = 0.13
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # create ball asset
        self.ball_radius = 0.1
        ball_options = gymapi.AssetOptions()
        ball_options.density = 200
        ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_options)

        self.envs = []
        self.robot_actor_handles = []
        self.robot_rigid_body_idxs = []
        self.robot_actor_idxs = []
        
        self.object_actor_idxs = []
        self.object_rigid_body_idxs = []
        self.object_actor_handles = []
        
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            # Add robots
            # rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            
            robot_handle = self.gym.create_actor(env_handle, self.robot_asset, pose, "robot", i, 0, 0)
            for bi in body_names:
                self.robot_rigid_body_idxs.append(self.gym.find_actor_rigid_body_handle(env_handle, robot_handle, bi))
            dof_props = self.gym.get_actor_dof_properties(env_handle, robot_handle)
            dof_props['driveMode'][np.array([0, 1, 2, 3])] = gymapi.DOF_MODE_POS #TODO: Is this right?
            dof_props['stiffness'][:] = 4000.0
            dof_props['damping'][:] = 100.0
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            # body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)
            
            self.robot_actor_handles.append(robot_handle)
            self.robot_actor_idxs.append(self.gym.get_actor_index(env_handle, robot_handle, gymapi.DOMAIN_SIM))
            

            # For Ball
            ball_pose = gymapi.Transform()
            ball_pose.p.x = 0.2
            ball_pose.p.z = 2.0
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "ball", i, 0, 0)
            ball_idx = self.gym.get_actor_rigid_body_index(env_handle, ball_handle, 0, gymapi.DOMAIN_SIM)
            
            self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.99, 0.66, 0.25))
            ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            ball_body_props[0].mass = 0.318*(np.random.rand()*0.3+0.5)
            self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_body_props, recomputeInertia=True)
            self.object_actor_handles.append(ball_handle)
            self.object_rigid_body_idxs.append(ball_idx)
            self.object_actor_idxs.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))

            self.envs.append(env_handle)
        
        self.robot_actor_idxs = torch.Tensor(self.robot_actor_idxs).to(device=self.device,dtype=torch.long)
        self.object_actor_idxs = torch.Tensor(self.object_actor_idxs).to(device=self.device,dtype=torch.long)
        self.object_rigid_body_idxs = torch.Tensor(self.object_rigid_body_idxs).to(device=self.device,dtype=torch.long)
        self.robot_rigid_body_idxs = torch.Tensor(self.robot_rigid_body_idxs).to(device=self.device,dtype=torch.long)
        
        print("These are the rigid body idxs:", self.robot_rigid_body_idxs)
        print("These are the rigid body idxs:", self.robot_rigid_body_idxs.shape)
        
        self.initialize_sensors()
        
        if self.fpv_render == True: 
            self.initialize_fpv_cameras(env_ids=[self.fpv_env])
        
    def initialize_fpv_cameras(self, env_ids):
        self.cams = {"fpv": []}
        self.camera_sensors = {}
        
        from .sensors.attached_camera_sensor import AttachedCameraSensor
        
        camera_label = "fpv"
        camera_pose = np.array([0, 0, 0])
        camera_rpy = np.array([0, np.pi/2, 0])
        
        self.camera_sensors[camera_label] = AttachedCameraSensor(self)
        self.camera_sensors[camera_label].initialize(camera_label, camera_pose, camera_rpy, env_ids=env_ids)
    
    
    def get_fpv_rgb_images(self, env_ids):
        rgb_images = {}
        for camera_name in ["fpv"]: #TODO: make this a config too
            rgb_images[camera_name] = self.camera_sensors[camera_name].get_rgb_images(env_ids)
        return rgb_images
        
        
        
    def initialize_sensors(self):
        '''Initialize the Sensors
        
        populates teh self.sensors array
        '''
        
        from .sensors import ALL_SENSORS

        self.sensors = []
        
        sensor_names = ["PointCameraSensor", "LastActionSensor", "JointPositionSensor", "JointVelocitySensor"] #TODO: Make this a cfg
        sensor_args = {} #TODO: This likely is not right
        for sensor_name in sensor_names:
            if sensor_name in ALL_SENSORS.keys():
                print("Initializing this sensor: ", sensor_name)
                self.sensors.append(ALL_SENSORS[sensor_name](self, **sensor_args))
            else:
                print(f"ERR: {sensor_name} does not exist")
        
    def compute_observations(self):
        '''These are the observations that are sent into the policy'''
        
        self.pre_obs_buf = []
        for sensor in self.sensors:
            self.pre_obs_buf += [sensor.get_observation()]
            
        curr_timestep_obs = torch.reshape(torch.cat(self.pre_obs_buf, dim=-1), (self.num_envs, -1))
        if self.prev_obs is None:
            self.prev_obs = curr_timestep_obs
        self.pre_obs_buf = torch.reshape(torch.cat([curr_timestep_obs, self.prev_obs], dim=-1), (self.num_envs, -1))
        self.prev_obs = curr_timestep_obs
        
        self.obs_buf[:] = self.pre_obs_buf
        
        print("This is an observation example: ", len(self.obs_buf[0]))
        print("This is an observation example: ", self.obs_buf[0])
        print("1st half: ", self.obs_buf[0][:11])
        print("2nd half: ", self.obs_buf[0][11:])

        return self.obs_buf

    def compute_reward(self):
        image_position = self.obs_buf[..., 0:3]
        
        self.rew_buf[:], self.reset_buf[:] = compute_bbot_reward(
            image_position,
            self.ball_positions,
            self.ball_radius,
            self.reset_buf, 
            self.progress_buf, 
            self.max_episode_length)

    def reset_idx(self, env_ids):
        '''also called at the beginning of the sim'''
        num_resets = len(env_ids)

        # reset bbot and ball root states
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        h_pos_mean = torch.tensor([5.0, 0.0], device=self.device)
        min_d = 0.0  # min horizontal dist from mean
        max_d = 0.5  # max horizontal dist from mean
        min_height = 2
        max_height = 3
        min_horizontal_speed = 5
        max_horizontal_speed = 10
        min_vertical_speed = 5 #TODO: check these
        max_vertical_speed = 6

        dists = torch_rand_float(min_d, max_d, (num_resets, 1), self.device)
        dirs = torch_random_dir_2((num_resets, 1), self.device)
        hpos = h_pos_mean + dists * dirs

        speedscales = (dists - min_d) / (max_d - min_d)
        hspeeds = torch_rand_float(min_horizontal_speed, max_horizontal_speed, (num_resets, 1), self.device)
        
        rand_angle = torch_rand_float(-np.pi/5, np.pi/5, (num_resets, 1), self.device).squeeze(-1)
        rand_horizontal_dir = torch.stack([torch.cos(rand_angle), torch.sin(rand_angle)], dim=-1)
        
        hvels = -speedscales * hspeeds * rand_horizontal_dir
        vspeeds = torch_rand_float(min_vertical_speed, max_vertical_speed, (num_resets, 1), self.device).squeeze()

        self.ball_positions[env_ids, 0] = hpos[..., 0]
        self.ball_positions[env_ids, 2] = torch_rand_float(min_height, max_height, (num_resets, 1), self.device).squeeze()
        self.ball_positions[env_ids, 1] = hpos[..., 1]
        self.ball_orientations[env_ids, 0:3] = 0
        self.ball_orientations[env_ids, 3] = 1
        self.ball_linvels[env_ids, 0] = hvels[..., 0]
        self.ball_linvels[env_ids, 2] = vspeeds
        self.ball_linvels[env_ids, 1] = hvels[..., 1]
        self.ball_angvels[env_ids] = 0

        # reset root state for bbots and balls in selected envs
        actor_indices = self.all_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_indices), len(actor_indices))


        #TODO: set the initial position of the eyes
        # robo_init_state = torch.zeros((self.num_envs, 4), device=self.device)
        # robo_init_state[:, 2] = np.pi / 2
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(robo_init_state)) # TODO: I don't know if this is the right way to do this


        # reset DOF states for bbots in selected envs
        # bbot_indices = self.all_bbot_indices[env_ids].flatten()
        # self.dof_states[env_ids] = self.initial_dof_states[env_ids]
        # self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(bbot_indices), len(bbot_indices))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, _actions):
        '''
        This is where the actions are applied to the robot
        
        
        we are using a type of velocity control now
        '''
        
        self.actions = _actions
        
        
        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            
        # print("This is the size of the actions: ", _actions.shape)
        # _actions[:, (0, 1)] = 0.0 # Make it so the robot can't move
        
        # Use Velocity Control
        actuated_indices = torch.LongTensor([0, 1, 2, 3])
        self.dof_position_targets[..., actuated_indices] += self.dt*self.action_speed_scale*_actions #TODO: what is the action_speed_scale?
        self.dof_position_targets[:] = tensor_clamp(self.dof_position_targets, self.bbot_dof_lower_limits, self.bbot_dof_upper_limits) 

        self.dof_position_targets[reset_env_ids] = 0
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))

    def post_physics_step(self):
        '''
            Description description blah blah blah
        '''
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
                
        # TODO: I don't know why these two lines are required now? 
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.progress_buf += 1

        self.compute_observations()
        self.compute_reward()
        
        # Render images at this timestep if we're in TESTING mode and are not HEADLESS
        if self.fpv_render:
            self.render_fpv_video()
                
    def render_fpv_video(self):
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
            
        rgb_images = self.get_fpv_rgb_images(env_ids=[self.fpv_env])
        
        rgb_image_np = rgb_images["fpv"].cpu().numpy()[0]  # Convert to NumPy
        rgb_image_np = (rgb_image_np[..., :3]*255).astype(np.uint8)
        self.fpv_imgs.append(rgb_image_np)
        
        # If env0 is done, write the images to a movie and output it
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if self.fpv_env in reset_env_ids:
            self.complete_fpv_imgs = self.fpv_imgs
            self.fpv_imgs = []
        
            
            
#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_bbot_reward(ball_pos_in_iamge, ball_positions, ball_radius, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    # calculating the norm for ball distance to desired height above the ground plane (i.e. 0.7)
    u = ball_pos_in_iamge[...,0]
    v = ball_pos_in_iamge[...,1]
    
    dist_from_center_sq = u*u + v*v + 0.01 #offset param
    visibility_flag = ball_pos_in_iamge[...,2]
    
    reward = (1/dist_from_center_sq)*visibility_flag
    # print("This is the reward", reward[0])
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf) # Max Episodes
    reset = torch.where(ball_positions[..., 2] < (ball_radius +.25) , torch.ones_like(reset_buf), reset) # ball hit ground #TODO: IDK if this is right tho

    return reward, reset
