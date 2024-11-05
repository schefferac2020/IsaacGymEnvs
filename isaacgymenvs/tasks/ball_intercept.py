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

        sensors_per_env = 3
        actors_per_env = 1
        dofs_per_env = 4
        bodies_per_env = 7 + 1
        
        self.a = 0

        # Observations:
        # 0:3 - activated DOF positions
        # 3:6 - activated DOF velocities
        # 6:9 - ball position
        # 9:12 - ball linear velocity
        # 12:15 - sensor force (same for each sensor)
        # 15:18 - sensor torque 1
        # 18:21 - sensor torque 2
        # 21:24 - sensor torque 3
        self.cfg["env"]["numObservations"] = 24

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
        # self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # print("This is the dof_state_tensor:", self.dof_state_tensor.shape)
        # self.sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # print("This is the sensor_tensor:", self.sensor_tensor.shape)
        

        
        # vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)
        # vec_sensor_tensor = gymtorch.wrap_tensor(self.sensor_tensor).view(self.num_envs, sensors_per_env, 6)

        self.root_states = vec_root_tensor
        # self.base_pos = 
        self.tray_positions = vec_root_tensor[..., 0, 0:3] # TODO: Meaningless
        self.ball_positions = vec_root_tensor[..., 1, 0:3] # T
        self.ball_orientations = vec_root_tensor[..., 1, 3:7]
        self.ball_linvels = vec_root_tensor[..., 1, 7:10]
        self.ball_angvels = vec_root_tensor[..., 1, 10:13]

        # self.dof_states = vec_dof_tensor
        # self.dof_positions = vec_dof_tensor[..., 0]
        # self.dof_velocities = vec_dof_tensor[..., 1]

        # self.sensor_forces = vec_sensor_tensor[..., 0:3]
        # self.sensor_torques = vec_sensor_tensor[..., 3:6]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # self.initial_dof_states = self.dof_states.clone()
        self.initial_root_states = vec_root_tensor.clone()

        self.dof_position_targets = torch.zeros((self.num_envs, dofs_per_env), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(actors_per_env * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, actors_per_env)
        self.all_bbot_indices = actors_per_env * torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # vis
        self.axes_geom = gymutil.AxesGeometry(0.2)
        

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

    def _create_balance_bot_asset(self):
        # there is an asset balance_bot.xml, here we override some features.

        tray_radius = 0.5
        tray_thickness = 0.02
        leg_radius = 0.02
        leg_outer_offset = tray_radius - 0.1
        leg_length = leg_outer_offset - 2 * leg_radius
        leg_inner_offset = leg_outer_offset - leg_length / math.sqrt(2)

        tray_height = leg_length * math.sqrt(2) + 2 * leg_radius + 0.5 * tray_thickness

        root = ET.Element('mujoco')
        root.attrib["model"] = "BalanceBot"
        compiler = ET.SubElement(root, "compiler")
        compiler.attrib["angle"] = "degree"
        compiler.attrib["coordinate"] = "local"
        compiler.attrib["inertiafromgeom"] = "true"
        worldbody = ET.SubElement(root, "worldbody")

        tray = ET.SubElement(worldbody, "body")
        tray.attrib["name"] = "tray"
        tray.attrib["pos"] = "%g %g %g" % (0, 0, tray_height)
        tray_joint = ET.SubElement(tray, "joint")
        tray_joint.attrib["name"] = "root_joint"
        tray_joint.attrib["type"] = "free"
        tray_geom = ET.SubElement(tray, "geom")
        tray_geom.attrib["type"] = "cylinder"
        tray_geom.attrib["size"] = "%g %g" % (tray_radius, 0.5 * tray_thickness)
        tray_geom.attrib["pos"] = "0 0 0"
        tray_geom.attrib["density"] = "100"

        leg_angles = [0.0, 2.0 / 3.0 * math.pi, 4.0 / 3.0 * math.pi]
        for i in range(len(leg_angles)):
            angle = leg_angles[i]

            upper_leg_from = gymapi.Vec3()
            upper_leg_from.x = leg_outer_offset * math.cos(angle)
            upper_leg_from.y = leg_outer_offset * math.sin(angle)
            upper_leg_from.z = -leg_radius - 0.5 * tray_thickness
            upper_leg_to = gymapi.Vec3()
            upper_leg_to.x = leg_inner_offset * math.cos(angle)
            upper_leg_to.y = leg_inner_offset * math.sin(angle)
            upper_leg_to.z = upper_leg_from.z - leg_length / math.sqrt(2)
            upper_leg_pos = (upper_leg_from + upper_leg_to) * 0.5
            upper_leg_quat = gymapi.Quat.from_euler_zyx(0, -0.75 * math.pi, angle)
            upper_leg = ET.SubElement(tray, "body")
            upper_leg.attrib["name"] = "upper_leg" + str(i)
            upper_leg.attrib["pos"] = "%g %g %g" % (upper_leg_pos.x, upper_leg_pos.y, upper_leg_pos.z)
            upper_leg.attrib["quat"] = "%g %g %g %g" % (upper_leg_quat.w, upper_leg_quat.x, upper_leg_quat.y, upper_leg_quat.z)
            upper_leg_geom = ET.SubElement(upper_leg, "geom")
            upper_leg_geom.attrib["type"] = "capsule"
            upper_leg_geom.attrib["size"] = "%g %g" % (leg_radius, 0.5 * leg_length)
            upper_leg_geom.attrib["density"] = "1000"
            upper_leg_joint = ET.SubElement(upper_leg, "joint")
            upper_leg_joint.attrib["name"] = "upper_leg_joint" + str(i)
            upper_leg_joint.attrib["type"] = "hinge"
            upper_leg_joint.attrib["pos"] = "%g %g %g" % (0, 0, -0.5 * leg_length)
            upper_leg_joint.attrib["axis"] = "0 1 0"
            upper_leg_joint.attrib["limited"] = "true"
            upper_leg_joint.attrib["range"] = "-45 45"

            lower_leg_pos = gymapi.Vec3(-0.5 * leg_length, 0, 0.5 * leg_length)
            lower_leg_quat = gymapi.Quat.from_euler_zyx(0, -0.5 * math.pi, 0)
            lower_leg = ET.SubElement(upper_leg, "body")
            lower_leg.attrib["name"] = "lower_leg" + str(i)
            lower_leg.attrib["pos"] = "%g %g %g" % (lower_leg_pos.x, lower_leg_pos.y, lower_leg_pos.z)
            lower_leg.attrib["quat"] = "%g %g %g %g" % (lower_leg_quat.w, lower_leg_quat.x, lower_leg_quat.y, lower_leg_quat.z)
            lower_leg_geom = ET.SubElement(lower_leg, "geom")
            lower_leg_geom.attrib["type"] = "capsule"
            lower_leg_geom.attrib["size"] = "%g %g" % (leg_radius, 0.5 * leg_length)
            lower_leg_geom.attrib["density"] = "1000"
            lower_leg_joint = ET.SubElement(lower_leg, "joint")
            lower_leg_joint.attrib["name"] = "lower_leg_joint" + str(i)
            lower_leg_joint.attrib["type"] = "hinge"
            lower_leg_joint.attrib["pos"] = "%g %g %g" % (0, 0, -0.5 * leg_length)
            lower_leg_joint.attrib["axis"] = "0 1 0"
            lower_leg_joint.attrib["limited"] = "true"
            lower_leg_joint.attrib["range"] = "-70 90"

        _indent_xml(root)
        ET.ElementTree(root).write("balance_bot.xml")

        # save some useful robot parameters
        self.tray_height = tray_height
        self.leg_radius = leg_radius
        self.leg_length = leg_length
        self.leg_outer_offset = leg_outer_offset
        self.leg_angles = leg_angles

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
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
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

    def compute_observations(self):
        '''These are the observations that are sent into the MLP that is optimized'''
        #print("~!~!~!~! Computing obs")

        actuated_dof_indices = torch.tensor([1, 3, 5], device=self.device)
        #print(self.dof_states[:, actuated_dof_indices, :])
        
        # print("Right before")
        # Pan Angle: self.dof_positions[0][2]
        # Tilt Angle: self.dof_positions[0][3]
        
        # Get Transform from Ball to world
        T_WB = np.eye(4)
        T_WB[:3, 3] = self.ball_positions[0].cpu()
        
        print("This is the position of the bot", self.dof_positions[0][0:2])
        
        print("This is a ball_position:", self.ball_positions[0])

        self.obs_buf[..., 0:3] = 0 #self.dof_positions[..., actuated_dof_indices]
        self.obs_buf[..., 3:6] = 0 #self.dof_velocities[..., actuated_dof_indices]
        self.obs_buf[..., 6:9] = self.ball_positions
        self.obs_buf[..., 9:12] = self.ball_linvels
        self.obs_buf[..., 12:15] = 0 #self.sensor_forces[..., 0] / 20  # !!! lousy normalization
        self.obs_buf[..., 15:18] = 0 #self.sensor_torques[..., 0] / 20  # !!! lousy normalization
        self.obs_buf[..., 18:21] = 0 #self.sensor_torques[..., 1] / 20  # !!! lousy normalization
        self.obs_buf[..., 21:24] = 0 #self.sensor_torques[..., 2] / 20  # !!! lousy normalization

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_bbot_reward(
            self.tray_positions,
            self.ball_positions,
            self.ball_linvels,
            self.ball_radius,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # reset bbot and ball root states
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        min_d = 0.001  # min horizontal dist from origin
        max_d = 0.5  # max horizontal dist from origin
        min_height = 1.0
        max_height = 2.0
        min_horizontal_speed = 10
        max_horizontal_speed = 15

        dists = torch_rand_float(min_d, max_d, (num_resets, 1), self.device)
        dirs = torch_random_dir_2((num_resets, 1), self.device)
        hpos = dists * dirs

        speedscales = (dists - min_d) / (max_d - min_d)
        hspeeds = torch_rand_float(min_horizontal_speed, max_horizontal_speed, (num_resets, 1), self.device)
        hvels = -speedscales * hspeeds * dirs
        vspeeds = -torch_rand_float(5.0, 5.0, (num_resets, 1), self.device).squeeze()

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

        # reset DOF states for bbots in selected envs
        # bbot_indices = self.all_bbot_indices[env_ids].flatten()
        # self.dof_states[env_ids] = self.initial_dof_states[env_ids]
        # self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(bbot_indices), len(bbot_indices))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, _actions):
        '''
        This is where the actions are applied to the robot
        '''
        
        
        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            
        # print("This is the size of the actions: ", _actions.shape)
        _actions[:, (0, 1)] = 0.3 # Make it so the robot can't move
        
        if self.a < 3:
            _actions[:, 2] = 0.0 # Pan
            _actions[:, 3] = 0 # Tilt
        else:
            _actions[:, 2] = 0 # Pan
            _actions[:, 3] = 0 # Tilt
            
        self.a += 1

        # actions = _actions.to(self.device)

        # actuated_indices = torch.LongTensor([1, 3, 5])
        
        # actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        # actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort

        # update position targets from actions
        # self.dof_position_targets[..., actuated_indices] += self.dt * self.action_speed_scale * actions
        # self.dof_position_targets[:] = tensor_clamp(self.dof_position_targets, self.bbot_dof_lower_limits, self.bbot_dof_upper_limits)

        # reset position targets for reset envs
        # self.dof_position_targets[reset_env_ids] = 0

        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)        
        actions_tensor[:] = _actions.to(self.device).view(-1) * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):

        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        
        
        # Prepare quantities
        # self.base_pos[:]
        
        

        self.compute_observations()
        self.compute_reward()
#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_bbot_reward(tray_positions, ball_positions, ball_velocities, ball_radius, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    # calculating the norm for ball distance to desired height above the ground plane (i.e. 0.7)
    ball_dist = torch.sqrt(ball_positions[..., 0] * ball_positions[..., 0] +
                           (ball_positions[..., 2] - 0.7) * (ball_positions[..., 2] - 0.7) +
                           (ball_positions[..., 1]) * ball_positions[..., 1])
    ball_speed = torch.sqrt(ball_velocities[..., 0] * ball_velocities[..., 0] +
                            ball_velocities[..., 1] * ball_velocities[..., 1] +
                            ball_velocities[..., 2] * ball_velocities[..., 2])
    pos_reward = 1.0 / (1.0 + ball_dist)
    speed_reward = 1.0 / (1.0 + ball_speed)
    reward = pos_reward * speed_reward

    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf) # Max Episodes
    reset = torch.where(ball_positions[..., 2] < ball_radius * 1.5, torch.ones_like(reset_buf), reset) # ball hit ground

    return reward, reset
