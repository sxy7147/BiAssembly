"""
    Environment with one object at center
        external: one robot, one camera
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, SceneConfig
import numpy as np
# from utils import process_angle_limit, get_random_number
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from PIL import Image
import trimesh
import random


class ContactError(Exception):
    pass


class SVDError(Exception):
    pass


class Env(object):
    def __init__(self, flog=None, show_gui=True, render_rate=20, timestep=1/500,
                 object_position_offset=0.0, succ_ratio=0.1, set_ground=False,
                 static_friction=100.0, dynamic_friction=100.0):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.object_position_offset = object_position_offset

        self.settings = []
        self.objects = []

        # engine and renderer
        self.engine = sapien.Engine(0, 0.001, 0.005)
        self.engine.set_log_level('error')
        # render_config = OptifuserConfig()
        # render_config.shadow_map_size = 8192
        # render_config.shadow_frustum_size = 10
        # render_config.use_shadow = False
        # render_config.use_ao = True
        sapien.render_config.camera_shader_dir="rt"
        sapien.render_config.viewer_shader_dir="rt"
        self.renderer = sapien.SapienRenderer(offscreen_only=True)
        # self.renderer = sapien.VulkanRenderer(offscreen_only=True)
        # self.renderer =sapien.KuafuRenderer()
        #self.renderer.enable_global_axes(False)
        
        self.engine.set_renderer(self.renderer)

        # GUI
        self.window = False
        if show_gui:
            self.renderer_controller = sapien.OptifuserController(self.renderer)
            self.renderer_controller.set_camera_position(-3.0+object_position_offset, 1.0, 3.0)
            self.renderer_controller.set_camera_rotation(-0.4, -0.8)

        # scene
        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0
        scene_config.default_static_friction = static_friction
        scene_config.default_dynamic_friction = dynamic_friction

        self.scene = self.engine.create_scene(config=scene_config)
        if set_ground:
            self.scene.add_ground(altitude=-0.8, render=True)
        if show_gui:
            self.renderer_controller.set_current_scene(self.scene)

        self.scene.set_timestep(timestep)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1+object_position_offset, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1+object_position_offset, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1+object_position_offset, 0, 1], [1, 1, 1])

        # default Nones
        self.object = None
        self.object_target_joint = None

        # check contact
        self.check_contact = False
        self.contact_error = False
        
        self.all_link_ids = []
        self.all_joint_types = []

        '''        
        # load ground
        b = self.scene.create_actor_builder()
        b.add_multiple_convex_shapes_from_file('/media/sim/WD_BLACK/songxy/env/ground/ground.obj')
        b.add_visual_from_file('/media/sim/WD_BLACK/songxy/env/ground/ground.obj')
        self.ground = b.build_static(name="ground")
        self.ground.set_pose(sapien.Pose([-2.5,0.1,-1.0]))  # all the ground

        
    # load desk
    def load_desk(self, desk_urdf_path, desk_material, scale=0.8, desk_position=[0, 0, 0], desk_orientation=[1, 0, 0, 0]):
        loader = self.scene.create_urdf_loader()
        loader.scale = scale
        desk = loader.load(desk_urdf_path, {"material": desk_material})

        # Set the initial pose of the desk
        pose = Pose(desk_position, desk_orientation)
        desk.set_root_pose(pose)
        return desk
        '''
        
        # table top
        # table_thickness = 0.05
        # builder = self.scene.create_actor_builder()
        # builder.add_box_shape( Pose([0, 0, 0.2], [1, 0, 0, 0]), size=[0.4, 0.4, table_thickness])  # Make the top surface's z equal to 0
        # builder.add_box_visual(Pose([0, 0, 0.2], [1, 0, 0, 0]), size=[0.4, 0.4, table_thickness], color=(1.0,  1.0, 1.0))
        
        self.table_length = 0.9
        self.table_height = 0.2
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[self.table_length, self.table_length, self.table_height], material=self.get_material(100, 100, 0.01))
        # builder.add_box_visual(half_size=[self.table_length, self.table_length, self.table_height], color=[0, 0, 1])
        builder.add_box_visual(half_size=[self.table_length, self.table_length, self.table_height])
        # self.table = builder.build_kinematic(name='table')
        self.table = builder.build_static(name='table')
        self.table.set_pose(sapien.Pose([self.table_length, 0, 0]))
        
        # builder.add_box_visual(half_size=[0.4, 0.4, 0.025])
        # self.table = builder.build_kinematic(name='table')
        # self.table.setpose(Pose(p=[0, 0, 1.0]))
                                                                         
        

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.renderer_controller.set_camera_position(x, y, z)
        self.renderer_controller.set_camera_rotation(yaw, pitch)
        self.renderer_controller.render()

    def load_object(self, urdf, material, scale=1.0, density=1.0, given_pose=None, random_pose=False, color=None):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        loader.scale = scale
        object = loader.load(urdf, {"material": material, "density": density})
        #print("material,dir",dir(material),"dynamic_friction",material.dynamic_friction)
        #print("object_dir",dir(object))
        if given_pose:
            pose = given_pose
        elif random_pose:
            obj_x = random.random() * (self.table_length - 0.4) + 0.2   # !!! Notice that  obj's root != obj's center
            obj_y = random.random() * (self.table_length - 0.4) + 0.2
            obj_z = max(random.random(), self.table_height)
            # position = ((np.random.rand(3) - 0.5) * 1.6).tolist()
            # position[-1] = max(np.abs(position[-1]), self.table_height)
            position = np.array([obj_x, obj_y, obj_z])
            quaternion = np.random.rand(4)
            quaternion = (quaternion / np.linalg.norm(quaternion)).tolist()
            pose = Pose(position, quaternion)
        else:
            pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])  # （1,0,0）
        object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids.extend([l.get_id() for l in object.get_links()])
        self.all_joint_types.extend([j.type for j in object.get_joints()])
        movable_link_ids = []
        for j in object.get_joints():
            if j.get_dof() == 1:
                movable_link_ids.append(j.get_child_link().get_id())

        # item = {'urdf': urdf, 'material': material, 'scale': scale, 'density': density, 'pose': pose,
        #         'all_link_ids': self.all_link_ids, 'all_joint_types': self.all_joint_types, 'movable_link_ids': movable_link_ids}
        item = {'urdf': urdf, 'scale': scale, 'density': density, 'pose_p': pose.p.flatten().tolist(), 'pose_q': pose.q.flatten().tolist(),
                'all_link_ids': self.all_link_ids, 'movable_link_ids': movable_link_ids}
        self.settings.append(item)
        self.objects.append(object)

    def get_link(self, link_id):
        for object in self.objects:
            for link in object.get_links():
                if link.get_id() == link_id:
                    return link

    def get_all_qpos_pose(self):
        ret = []
        for obj in self.objects:
            if obj is not None:
                pose = obj.get_root_pose()
                ret += pose.p.flatten().tolist()
                ret += pose.q.flatten().tolist()
                ret += obj.get_qpos().flatten().tolist()
        return np.array(ret, dtype=np.float32)

    def sample_pc(self, v, f, n_points = 10000):
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
        return points

    def wait_for_object_still(self, cam=None, visu=False):
        #print('start wait for still')
        still_timesteps, wait_timesteps = 0, -1
        imgs = []
        qpos_pose = self.get_all_qpos_pose()
        while still_timesteps < 5000 and wait_timesteps < 20000:
            self.step()
            
            new_qpos_pose = self.get_all_qpos_pose()

            if len(qpos_pose) == len(new_qpos_pose):
                dist = np.max(np.abs(qpos_pose - new_qpos_pose))
            else:
                dist = 1
            if dist < 1e-5:
                still_timesteps += 1
            else:
                still_timesteps = 0
            qpos_pose = new_qpos_pose

            if visu and wait_timesteps % 100 == 0 and wait_timesteps != 0:
                # self.render()
                rgb_pose, _ = cam.get_observation()
                fimg = (rgb_pose * 255).astype(np.uint8)
                fimg = Image.fromarray(fimg)
                imgs.append(fimg)

            wait_timesteps += 1

        if visu:
            return still_timesteps, imgs
        else:
            return still_timesteps
        



    def get_target_part_axes(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[1, 0]), float(mat[2, 0]), float(-mat[0, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')
        return joint_axes

    def get_target_part_axes_new(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[0, 0]), float(-mat[1, 0]), float(mat[2, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')

        return joint_axes

    def get_target_part_axes_dir(self, target_part_id):
        joint_axes = self.get_target_part_axes(target_part_id=target_part_id)
        # print("joint_origins:", joint_origins)
        # print("joint_axes", joint_axes)
        axes_dir = -1
        for idx_axes_dim in range(3):
            if abs(joint_axes[idx_axes_dim]) > 0.5:
                axes_dir = idx_axes_dim
        return axes_dir

    def get_target_part_axes_dir_new(self, target_part_id):
        joint_axes = self.get_target_part_axes_new(target_part_id=target_part_id)
        # print("joint_origins:", joint_origins)
        # print("joint_axes", joint_axes)
        axes_dir = -1
        for idx_axes_dim in range(3):
            if abs(joint_axes[idx_axes_dim]) > 0.1:
                axes_dir = idx_axes_dim
        return axes_dir

    def get_target_part_origins_new(self, target_part_id):
        joint_origins = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    joint_origins = pos.p.tolist()
                    # ipdb.set_trace()
                    # mat = pos.to_transformation_matrix()
                    # print("pos:", pos.p)
                    # joint_origins = [float(-mat[1, 3]), float(mat[2, 3]), float(-mat[0, 3])]
        if joint_origins is None:
            raise ValueError('joint origins error!')

        return joint_origins

    def get_target_part_origins(self, target_part_id):
        print("attention!!! origin")
        joint_origins = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    # joint_origins = pos.p.tolist()
                    # ipdb.set_trace()
                    mat = pos.to_transformation_matrix()
                    # print("pos:", pos.p)
                    joint_origins = [float(-mat[1, 3]), float(mat[2, 3]), float(-mat[0, 3])]
        if joint_origins is None:
            raise ValueError('joint origins error!')

        return joint_origins

    def get_target_part_object(self, target_part_id):
        for obj_id in range(len(self.objects)):
            object = self.objects[obj_id]
            for link in object.get_links():
                if int(link.get_id()) == target_part_id:
                    return obj_id, object


    def set_object_joint_angles(self, joint_angles):
        self.object.set_qpos(joint_angles)
        
    

    def set_target_object_part_actor_id(self, actor_id):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))

        # get the link handler
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_actor_link = j.get_child_link()

        # moniter the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_joint_id = idx
                    self.target_object_part_joint_type = j.type
                idx += 1

    def set_target_object_part_actor_id2(self, actor_id):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id     # not movable
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))

        # get the link handler
        for j in self.object.get_joints():
            # if j.get_dof() == 1:
            if j.get_child_link().get_id() == actor_id:     # j应该是fix or undefined
                self.target_object_part_actor_link = j.get_child_link()

        # monitor the target joint
        idx = 0
        for j in self.object.get_joints():
            # if j.get_dof() == 1:
            if j.get_child_link().get_id() == actor_id:
                self.target_object_part_joint_id = idx
                self.target_object_part_joint_type = j.type
            idx += 1

    def set_target_joint_actor_id(self, target_joint_type):
        idx = 0
        actor_id = None
        for j in self.object.get_joints():
            # if j.get_dof() == 1:
            if j.type == target_joint_type:
                self.target_object_part_actor_link = j.get_child_link()
                self.target_object_part_joint_id = idx
                self.target_object_part_joint_type = j.type
                actor_id = j.get_child_link().get_id()
                if self.flog is not None:
                    self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
                self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))
            idx += 1
        return actor_id

    def get_object_qpos(self):
        return self.object.get_qpos()

    def get_object_root_pose(self):
        return self.object.get_root_pose()

    def get_target_part_qpos(self):
        qpos = self.object.get_qpos()
        # ipdb.set_trace()
        return float(qpos[self.target_object_part_joint_id])
    
    def get_target_part_pose(self):
        return self.target_object_part_actor_link.get_pose()

    # def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
    #     self.check_contact = True
    #     self.check_contact_strict = strict
    #     self.first_timestep_check_contact = True
    #     self.robot_hand_actor_id = robot_hand_actor_id
    #     self.robot_gripper_actor_ids = robot_gripper_actor_ids
    #     self.contact_error = False
    #
    # def end_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
    #     self.check_contact = False
    #     self.check_contact_strict = strict
    #     self.first_timestep_check_contact = False
    #     self.robot_hand_actor_id = robot_hand_actor_id
    #     self.robot_gripper_actor_ids = robot_gripper_actor_ids

    def start_checking_contact(self):
        self.check_contact = True
        self.first_timestep_check_contact = True
        self.contact_error = False

    def end_checking_contact(self):
        self.check_contact = False

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def render(self):
        if self.show_gui and (not self.window):
            self.window = True
            self.renderer_controller.show_window()
        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.renderer_controller.render()

    def step(self):
        self.current_step += 1
        self.scene.step()
        if self.check_contact:
            if not self.check_contact_is_valid():
                raise ContactError()

    def check_main_part(self, part_id):
        main_joint_type = [ArticulationJointType.FIX, ArticulationJointType.PRISMATIC]
        for joint in self.object.get_joints():
            if joint.get_child_link().get_id() == part_id:
                if joint.type in main_joint_type:
                    return True
                else:
                    return False

    # check the first contact: only gripper links can touch the target object part link
    # def check_contact_is_valid(self):
    #     self.contacts = self.scene.get_contacts()
    #     contact = False; valid = False
    #     for c in self.contacts:
    #         aid1 = c.actor1.get_id()
    #         aid2 = c.actor2.get_id()
    #         has_impulse = False
    #         for p in c.points:
    #             if abs(p.impulse @ p.impulse) > 1e-4:
    #                 has_impulse = True
    #                 break
    #         if has_impulse:
    #             if (aid1 in self.robot_gripper_actor_ids and aid2 == self.target_object_part_actor_id) or \
    #                (aid2 in self.robot_gripper_actor_ids and aid1 == self.target_object_part_actor_id):
    #                    contact, valid = True, True
    #             if (aid1 in self.robot_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
    #                (aid2 in self.robot_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
    #                 if self.check_contact_strict:
    #                     self.contact_error = True
    #                     return False
    #                 else:
    #                     contact, valid = True, True
    #             if (aid1 == self.robot_hand_actor_id or aid2 == self.robot_hand_actor_id):
    #                 if self.check_contact_strict:
    #                     self.contact_error = True
    #                     return False
    #                 else:
    #                     contact, valid = True, True
    #             # starting pose should have no collision at all
    #             if (aid1 in self.robot_gripper_actor_ids or aid1 == self.robot_hand_actor_id or \
    #                 aid2 in self.robot_gripper_actor_ids or aid2 == self.robot_hand_actor_id) and self.first_timestep_check_contact:
    #                     self.contact_error = True
    #                     return False
    #
    #     self.first_timestep_check_contact = False
    #     if contact and valid:
    #         self.check_contact = False
    #     return True
    
    def check_contact_exist(self, id1, id2,vis=False):
        contacts = self.scene.get_contacts()
        
        if(vis):
            contact_pair=[]
            for contact in contacts:
                aid1 = contact.actor0.get_id()
                aid2 = contact.actor1.get_id()
                contact_pair.append([aid1,aid2])
            print('contact_pair:',contact_pair)
        for contact in contacts:
            aid1 = contact.actor0.get_id()
            aid2 = contact.actor1.get_id()
            if (aid1 == id1 and aid2 == id2) or (aid1 == id2 and aid2 == id1):
                return True 
        return False


    def check_contacts_exist(self, id1s, id2s):
        contacts = self.scene.get_contacts()
        
        for contact in contacts:
            aid1 = contact.actor0.get_id()
            aid2 = contact.actor1.get_id()
            if (aid1 in id1s and aid2 in id2s) or (aid1 in id2s and aid2 in id1s):
                return True 
        return False


    def check_contact_is_valid(self):
        self.contacts = self.scene.get_contacts()
        # print('contacts!!', self.contacts)
        # contact = False; valid = False
        if self.first_timestep_check_contact:
            self.first_timestep_check_contact = False
            for c in self.contacts:
                for p in c.points:
                    if abs(p.impulse @ p.impulse) > 1e-4:
                        self.contact_error = True
                        return False
            # self.contact_error = True
            # return False
        else:
            # if len(self.contacts) > 0:
            #     self.contact_error = True
            #     return False
            # for c in self.contacts:
            #     for p in c.points:
            #         print('!!impluse: ', abs(p.impulse @ p.impulse))
            #         if abs(p.impulse @ p.impulse) > 1e-3:
            #             self.contact_error = True
            #             return False
            pass

        return True
    
    def count_contact(self):
        self.contacts = self.scene.get_contacts()
        num_contacts = len(self.contacts)
        return num_contacts
        

    def close_render(self):
        if self.window:
            self.renderer_controller.hide_window()
        self.window = False
    
    def wait_to_start(self):
        print('press q to start\n')
        while not self.renderer_controller.should_quit:
            self.scene.update_render()
            if self.show_gui:
                self.renderer_controller.render()

    def close(self):
        if self.show_gui:
            self.renderer_controller.set_current_scene(None)
        self.scene = None

    def get_global_mesh(self, obj):
        final_vs = [];
        final_fs = [];
        vid = 0;
       
        for l in obj.get_links():
            vs = []
            for s in l.get_collision_shapes():
                v = np.array(s.geometry.vertices, dtype=np.float32)
                f = np.array(s.geometry.indices, dtype=np.uint32).reshape(-1, 3)
                vscale = s.geometry.scale
                v[:, 0] *= vscale[0];
                v[:, 1] *= vscale[1];
                v[:, 2] *= vscale[2];
                ones = np.ones((v.shape[0], 1), dtype=np.float32)
                v_ones = np.concatenate([v, ones], axis=1)
                transmat = s.get_local_pose().to_transformation_matrix()
                v = (v_ones @ transmat.T)[:, :3]
                vs.append(v)
                final_fs.append(f + vid)
                vid += v.shape[0]
            if len(vs) > 0:
                vs = np.concatenate(vs, axis=0)
                ones = np.ones((vs.shape[0], 1), dtype=np.float32)
                vs_ones = np.concatenate([vs, ones], axis=1)
                transmat = l.get_pose().to_transformation_matrix()
                vs = (vs_ones @ transmat.T)[:, :3]
                final_vs.append(vs)
        final_vs = np.concatenate(final_vs, axis=0)
        final_fs = np.concatenate(final_fs, axis=0)
        return final_vs, final_fs

    def get_part_mesh(self, obj, part_id):
        final_vs = [];
        final_fs = [];
        vid = 0;
        for l in obj.get_links():
            l_id = l.get_id()
            if l_id != part_id:
                continue
            vs = []
            for s in l.get_collision_shapes():
                v = np.array(s.geometry.vertices, dtype=np.float32)
                f = np.array(s.geometry.indices, dtype=np.uint32).reshape(-1, 3)
                vscale = s.geometry.scale
                v[:, 0] *= vscale[0];
                v[:, 1] *= vscale[1];
                v[:, 2] *= vscale[2];
                ones = np.ones((v.shape[0], 1), dtype=np.float32)
                v_ones = np.concatenate([v, ones], axis=1)
                transmat = s.pose.to_transformation_matrix()
                v = (v_ones @ transmat.T)[:, :3]
                vs.append(v)
                final_fs.append(f + vid)
                vid += v.shape[0]
            if len(vs) > 0:
                vs = np.concatenate(vs, axis=0)
                ones = np.ones((vs.shape[0], 1), dtype=np.float32)
                vs_ones = np.concatenate([vs, ones], axis=1)
                transmat = l.get_pose().to_transformation_matrix()
                vs = (vs_ones @ transmat.T)[:, :3]
                final_vs.append(vs)
        final_vs = np.concatenate(final_vs, axis=0)
        final_fs = np.concatenate(final_fs, axis=0)
        return final_vs, final_fs
    ''' make the objects move'''

    def compute_joint_velocity_from_twist(self, twist: np.ndarray, obj_idx) -> np.ndarray:
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
        dense_jacobian = self.objects[obj_idx].compute_spatial_twist_jacobian()  # (num_link * 6, dof())
        # ee_jacobian = np.zeros([6, self.objects[obj_idx].dof - 2])
        # ee_jacobian[:3, :] = dense_jacobian[self.end_effector_index * 6 - 3: self.end_effector_index * 6, :self.objects[obj_idx].dof - 2]
        # ee_jacobian[3:6, :] = dense_jacobian[(self.end_effector_index - 1) * 6: self.end_effector_index * 6 - 3, :self.objects[obj_idx].dof - 2]

        #numerical_small_bool = ee_jacobian < 1e-1
        #ee_jacobian[numerical_small_bool] = 0
        #inverse_jacobian = np.linalg.pinv(ee_jacobian)
        # inverse_jacobian = np.linalg.pinv(ee_jacobian, rcond=1e-2)  # 雅可比矩阵, 类似于多元函数的导数
        inverse_jacobian = np.linalg.pinv(dense_jacobian, rcond=1e-2)  # 雅可比矩阵, 类似于多元函数的导数
        #inverse_jacobian[np.abs(inverse_jacobian) > 5] = 0
        #print(inverse_jacobian)

        print('inverse_jacobian: ', inverse_jacobian.shape)
        print('twist: ', twist.shape)

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
        passive_force = self.robot.compute_passive_force()
        self.robot.set_qf(passive_force)

    def move_objects(self, move_direction, num_steps, veolovity=1e-4, cam=None, vis_gif=False, vis_gif_interval=200):
        imgs = []
        contact_error = False

        if vis_gif:
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            for idx in range(5):
                imgs.append(fimg)

        for i in range(num_steps):
            cur_pose0 = self.objects[0].get_root_pose()
            cur_p0, cur_q0 = cur_pose0.p.flatten().tolist(), cur_pose0.q.flatten().tolist()
            next_p0 = cur_p0 + move_direction * veolovity
            new_pose0 = Pose(next_p0, cur_q0)
            self.objects[0].set_root_pose(new_pose0)

            cur_pose1 = self.objects[1].get_root_pose()
            cur_p1, cur_q1 = cur_pose1.p.flatten().tolist(), cur_pose1.q.flatten().tolist()
            next_p1 = cur_p1 - move_direction * veolovity
            new_pose1 = Pose(next_p1, cur_q1)
            self.objects[1].set_root_pose(new_pose1)

            try:
                self.step()
                self.render()
            except Exception:
                contact_error = True
                if imgs:
                    return imgs, contact_error
                else:
                    return contact_error

            if vis_gif and (i % vis_gif_interval == 0):
                rgb_pose, _ = cam.get_observation()
                fimg = (rgb_pose*255).astype(np.uint8)
                fimg = Image.fromarray(fimg)
                imgs.append(fimg)

        if vis_gif:
            return imgs, contact_error
        else:
            return contact_error
    
    def move_single_object(self, object_id, move_direction, num_steps, veolovity=1e-4, cam=None, vis_gif=False, vis_gif_interval=200):
        imgs = []
        contact_error = False

        if vis_gif:
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            for idx in range(5):
                imgs.append(fimg)

        for i in range(num_steps):
            cur_pose = self.objects[object_id].get_root_pose()
            cur_p, cur_q = cur_pose.p.flatten().tolist(), cur_pose.q.flatten().tolist()
            next_p = cur_p + move_direction * veolovity
            new_pose = Pose(next_p, cur_q)
            self.objects[object_id].set_root_pose(new_pose)

            try:
                self.step()
                self.render()
            except Exception:
                contact_error = True
                if imgs:
                    return imgs, contact_error
                else:
                    return contact_error

            if vis_gif and (i % vis_gif_interval == 0):
                rgb_pose, _ = cam.get_observation()
                fimg = (rgb_pose*255).astype(np.uint8)
                fimg = Image.fromarray(fimg)
                imgs.append(fimg)

        if vis_gif:
            return imgs, contact_error
        else:
            return contact_error
    
    def wait_after_object_move(self, n, vis_gif=False, vis_gif_interval=200, cam=None):
        if vis_gif:
            imgs = []
        for i in range(n):
            self.step()
            self.render()
            if vis_gif and ((i + 1) % vis_gif_interval == 0):
                rgb_pose, _ = cam.get_observation()
                fimg = (rgb_pose*255).astype(np.uint8)
                fimg = Image.fromarray(fimg)
                imgs.append(fimg)
        if vis_gif:
            return imgs
    
    def env_remove_articulation(self, articulation_list,close=True):
        for articulation in articulation_list:
            self.scene.remove_articulation(articulation)
        if(close):
            self.close()
