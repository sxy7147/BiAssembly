"""
    Franka Panda Robot Arm
        support panda.urdf, panda_gripper.urdf
"""

from __future__ import division
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from PIL import Image


class Robot(object):
    def __init__(self, env, urdf, material, open_gripper=False, scale=1.0):
        self.env = env
        self.timestep = env.scene.get_timestep()

        # load robot
        loader = env.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = scale
        self.robot = loader.load(urdf, {"material": material})
        self.robot_model=self.robot.create_pinocchio_model()
        #self.robot = loader.load(urdf, material)
        self.robot.name = "robot"
        self.active_joints = self.robot.get_active_joints()

        # hand (EE), two grippers, the rest arm joints (if any)
        self.end_effector_index, self.end_effector = \
            [(i, l) for i, l in enumerate(self.robot.get_links()) if l.name == 'panda_hand'][0]
        self.hand_actor_id = self.end_effector.get_id()
        self.gripper_joints = [joint for joint in self.robot.get_joints() if 
                joint.get_name().startswith("panda_finger_joint")]
        self.collision_link_ids = [l.get_id() for l in self.robot.get_links() if l.get_name().startswith("panda")]
        self.gripper_actor_ids = [joint.get_child_link().get_id() for joint in self.gripper_joints]
        self.arm_joints = [joint for joint in self.robot.get_joints() if
                joint.get_dof() > 0 and not joint.get_name().startswith("panda_finger")]
        self.arm_actor_ids = [joint.get_child_link().get_id() for joint in self.arm_joints]

        # set drive joint property
        for joint in self.arm_joints:
            joint.set_drive_property(1000, 200)
            # print(joint.get_name())
        for joint in self.gripper_joints:
            joint.set_drive_property(200, 60)

        # open/close the gripper at start
        # if open_gripper:
        #     joint_angles = []
        #     for j in self.robot.get_joints():
        #         if j.get_dof() == 1:
        #             if j.get_name().startswith("panda_finger_joint"):
        #                 joint_angles.append(0.04)
        #             else:
        #                 joint_angles.append(0)
        #     self.robot.set_qpos(joint_angles)
       
        # self.setup_planner()
    
    def compute_fk_sapien_links(self, qpos, link_idx):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i).to_transformation_matrix())
        return link_pose_ls
    def compute_joint_velocity_from_twist(self, twist: np.ndarray) -> np.ndarray:
        """
        This function is a kinematic-level calculation which do not consider dynamics.
        Pay attention to the frame of twist, is it spatial twist or body twist

        Jacobian is provided for your, so no need to compute the velocity kinematics
        ee_jacobian is the geometric Jacobian on account of only the joint of robot arm, not gripper
        Jacobian in SAPIEN is defined as the derivative of spatial twist with respect to joint velocity

        Args:
            twist: (6,) vector to represent the twist

        Returns:
            (7, ) vector for the velocity of arm joints (not include gripper)

        """
        assert twist.size == 6
        # Jacobian define in SAPIEN use twist (v, \omega) which is different from the definition in the slides
        # So we perform the matrix block operation below
        dense_jacobian = self.robot.compute_spatial_twist_jacobian()  # (num_link * 6, dof())
        ee_jacobian = np.zeros([6, self.robot.dof - 2])
        ee_jacobian[:3, :] = dense_jacobian[self.end_effector_index * 6 - 3: self.end_effector_index * 6, :self.robot.dof - 2]
        ee_jacobian[3:6, :] = dense_jacobian[(self.end_effector_index - 1) * 6: self.end_effector_index * 6 - 3, :self.robot.dof - 2]

        #numerical_small_bool = ee_jacobian < 1e-1
        #ee_jacobian[numerical_small_bool] = 0
        #inverse_jacobian = np.linalg.pinv(ee_jacobian)
        inverse_jacobian = np.linalg.pinv(ee_jacobian, rcond=1e-2)
        #inverse_jacobian[np.abs(inverse_jacobian) > 5] = 0
        #print(inverse_jacobian)
        return inverse_jacobian @ twist

    def internal_controller(self, qvel: np.ndarray) -> None:
        """Control the robot dynamically to execute the given twist for one time step

        This method will try to execute the joint velocity using the internal dynamics function in SAPIEN.

        Note that this function is only used for one time step, so you may need to call it multiple times in your code
        Also this controller is not perfect, it will still have some small movement even after you have finishing using
        it. Thus try to wait for some steps using self.wait_n_steps(n) like in the hw2.py after you call it multiple
        time to allow it to reach the target position

        Args:
            qvel: (7,) vector to represent the joint velocity

        """
        assert qvel.size == len(self.arm_joints)
        # print("qvel:", qvel)
        # print("timestep:", self.timestep)
        target_qpos = qvel * self.timestep + self.robot.get_drive_target()[:-2]
        for i, joint in enumerate(self.arm_joints):
            joint.set_drive_velocity_target(qvel[i])
            joint.set_drive_target(target_qpos[i])
        # passive_force = self.robot.compute_passive_force()
        # self.robot.set_qf(passive_force)

    def move_to_target_pose(self, target_ee_pose: np.ndarray, num_steps: int, vis_gif=False, vis_gif_interval=300, cam=None, check_tactile=False, id1=None, id2=None) -> None:
        """
        Move the robot hand dynamically to a given target pose
        Args:
            target_ee_pose: (4, 4) transformation of robot hand in robot base frame (ee2base)
            num_steps:  how much steps to reach to target pose, 
                        each step correspond to self.scene.get_timestep() seconds
                        in physical simulation
        """
        if vis_gif:
            imgs = []

        executed_time = num_steps * self.timestep

        spatial_twist = self.calculate_twist(executed_time, target_ee_pose)
        for i in range(num_steps):
            if i % 100 == 0:
                spatial_twist = self.calculate_twist((num_steps - i) * self.timestep, target_ee_pose)
            qvel = self.compute_joint_velocity_from_twist(spatial_twist)
            # qvel = np.array([(random.random() * 2 - 1) * 0.25 for idx in range(6)])
            self.internal_controller(qvel)
            self.env.step()
            self.env.render()
            if vis_gif and ((i + 1) % vis_gif_interval == 0):
                rgb_pose, _ = cam.get_observation()
                fimg = (rgb_pose*255).astype(np.uint8)
                fimg = Image.fromarray(fimg)
                imgs.append(fimg)
            if check_tactile and i % 100 == 0 and i != 0:   # simulate tactile-based method
                contact = self.env.check_contact_exist(id1, id2)
                if contact:
                    if vis_gif:
                        return imgs 
                    else:
                        return 
        if vis_gif:
            return imgs

    def move_to_target_qvel(self, qvel) -> None:

        """
        Move the robot hand dynamically to a given target pose
        Args:
            target_ee_pose: (4, 4) transformation of robot hand in robot base frame (ee2base)
            num_steps:  how much steps to reach to target pose, 
                        each step correspond to self.scene.get_timestep() seconds
                        in physical simulation
        """
        assert qvel.size == len(self.arm_joints)
        for idx_step in range(100):
            target_qpos = qvel * self.timestep + self.robot.get_drive_target()[:-2]
            for i, joint in enumerate(self.arm_joints):
                # ipdb.set_trace()
                joint.set_drive_velocity_target(qvel[i])
                joint.set_drive_target(target_qpos[i])
            # passive_force = self.robot.compute_passive_force()
            # self.robot.set_qf(passive_force)
            self.env.step()
            self.env.render()
        return

    def open_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.4)
        for i in range(100): 
            self.env.step()
                # self.env.scene.update_render()
                # self.viewer.render()

    def close_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0)
        for i in range(100):  
            self.env.step()

    def clear_velocity_command(self):
        for joint in self.arm_joints:
            joint.set_drive_velocity_target(0)

    def wait_n_steps(self, n: int, vis_gif=False, vis_gif_interval=500, cam=None):
        if vis_gif:
            imgs = []
        self.clear_velocity_command()
        for i in range(n):
            self.env.step()
            if vis_gif and ((i + 1) % vis_gif_interval == 0):
                self.env.render()
                rgb_pose, _ = cam.get_observation()
                fimg = (rgb_pose*255).astype(np.uint8)
                fimg = Image.fromarray(fimg)
                imgs.append(fimg)

        if vis_gif:
            return imgs
        
    
    def add_point_cloud(self, points):
        self.planner.update_point_cloud(points)  

    def plan_pose(self, pose, time_step=1 / 250, use_point_cloud=True, use_attach=False):
        try:
            result = self.planner.plan_screw(pose, self.robot.get_qpos(), time_step=time_step, use_point_cloud=use_point_cloud, use_attach=use_attach)  # self.use_point_cloud
            if result['status'] != "Success":
                result = self.planner.plan(pose, self.robot.get_qpos(), time_step=1 / 250, use_point_cloud=use_point_cloud, use_attach=False)  # self.use_point_cloud
                if result['status'] != "Success":
                    print(result['status'])
                    return -1
            else:
                return result
            # if vis_gif:
            #     imgs = self.follow_path(result, vis_gif, vis_gif_interval, cam)
            # else:
            #     self.follow_path(result, vis_gif, vis_gif_interval, cam)
            # return 0, imgs
        except np.linalg.LinAlgError:
            print('np.linalg.LinAlgError')
            return -1
    
    
    def follow_path(self, result, vis_gif, vis_gif_interval, cam):
        n_step = result['position'].shape[0]
        imgs = []
        for i in range(n_step):
            # qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            # self.robot.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(result['position'][i][j])
                self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            self.env.step()
            self.env.render()
            if vis_gif and ((i + 1) % vis_gif_interval == 0):
                rgb_pose, _ = cam.get_observation()
                fimg = (rgb_pose * 255).astype(np.uint8)
                fimg = Image.fromarray(fimg)
                imgs.append(fimg)
        if vis_gif:
            return imgs