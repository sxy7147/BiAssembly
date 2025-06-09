import os
import sys
import h5py
# import torch
import numpy as np
from pyquaternion import Quaternion
from PIL import Image
import sapien.core as sapien
import transforms3d
from sapien.core import Pose
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import json
from scipy.spatial.transform import Rotation
import transforms3d.quaternions as tq
import transforms3d.euler as te
from copy import deepcopy



def get_dataset(data_dir="../assets/object/breaking_bad",json_file="data_classify.json",split_name="Train"):
    #support: "Train", "Test1", "Test2"
    with open(json_file) as f:
        split_dict = json.load(f)
    dataset = {}
    total_num =0
    for category in os.listdir(data_dir):
        category_num = 0
        category_dir = os.path.join(data_dir, category)
        if category not in split_dict:
            continue
        category_split = split_dict[category]
        dataset[category] = {}
        for shape_id in os.listdir(category_dir):
            if shape_id not in category_split:
                continue
            if category_split[shape_id] == split_name:
                dataset[category][shape_id] = os.listdir(os.path.join(category_dir, shape_id))
                category_num += len(dataset[category][shape_id])
        total_num += category_num
        if len(dataset[category]) == 0:
            del dataset[category]
            continue
        print(f"Category {category} has {category_num} samples")
    print(f"Total {total_num} samples in {split_name} set")
    return dataset
            


All_list=[]
Need_save_list=[]
def save_need_data(env, cam, trial_id, out_dir_root,out_sub_dir,category,shape_id):
    global All_list
    All_list.append([env, cam, trial_id, out_dir_root,out_sub_dir,category,shape_id])
    env.step()
    env.render()
    rgb, depth = cam.get_observation()
    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
    cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
    cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]
    init_cam_XYZA_list=cam_XYZA_list

    depth_img=(depth*255).astype(np.uint8)
    depth_img = Image.fromarray(depth_img)
    fimg = (rgb * 255).astype(np.uint8)
    fimg = Image.fromarray(fimg)
    global Need_save_list
    Need_save_list.append([deepcopy(init_cam_XYZA_list),deepcopy(depth_img),deepcopy(fimg)])
    return init_cam_XYZA_list

def save_all_data():
    global All_list,Need_save_list
    for all_list,need_save_list in zip(All_list,Need_save_list):
        env, cam, trial_id, out_dir_root,out_sub_dir,category,shape_id=all_list
        cam_XYZA_list,depth_img,fimg=need_save_list
        save_sub_dir=os.path.join(out_dir_root, out_sub_dir+'_data')
        os.makedirs(save_sub_dir, exist_ok=True)
        save_cam_XYZA_list(os.path.join(save_sub_dir, out_sub_dir+'_cam_XYZA_%s_%s_%d.h5' % (category, shape_id[:4], trial_id)), cam_XYZA_list)
        depth_img.save(os.path.join(save_sub_dir, out_sub_dir+'_depth_%s_%s_%d.png' % (category, shape_id[:4], trial_id)))
        fimg.save(os.path.join(save_sub_dir, out_sub_dir+'_rgb_%s_%s_%d.png' % (category, shape_id[:4], trial_id)))
    
    All_list=[]

def clear_all_data():
    global All_list,Need_save_list
    All_list=[]
    Need_save_list=[]

def find_nearest_point_index(point_cloud, target_point):
    # 计算点云中每个点与目标点的距离
    distances = np.linalg.norm(point_cloud - target_point, axis=1)
    # 找到距离最近的点的索引
    nearest_point_index = np.argmin(distances)
    return nearest_point_index

def save_h5(fn, data):
    fout = h5py.File(fn, 'w')
    for d, n, t in data:
        fout.create_dataset(n, data=d, compression='gzip', compression_opts=4, dtype=t)
    fout.close()

def save_cam_XYZA_list(fn, cam_XYZA_list):
    save_h5(fn, [(cam_XYZA_list[0].astype(np.uint64), 'id1', 'uint64'),
                 (cam_XYZA_list[1].astype(np.uint64), 'id2', 'uint64'),
                 (cam_XYZA_list[2].astype(np.float32), 'pc', 'float32')])


''' for motion control '''


from copy import deepcopy
def get_valid_rotmat(cam,env,height=0.5,move_dis=0.5):
    rgb, depth = cam.get_observation()
    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
    cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
    cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]
    mat44 = cam.get_metadata()['mat44']
    
    part1_id = env.objects[0].get_links()[0].get_id()
    part2_id = env.objects[1].get_links()[0].get_id()
    # print('part1_id: ', env.objects[0].get_links())
    # print('part2_id: ', env.objects[1].get_links())
    object_all_link_ids = np.array([part1_id, part2_id])
    gt_all_link_mask = cam.get_movable_link_mask(object_all_link_ids)  # (448, 448)
   # print('gt_all_link_mask: ', gt_all_link_mask.shape)
    xs, ys = np.where(gt_all_link_mask > 0)

    if len(xs) == 0:
        print("No valid points found!")
        return None,None,None,None
    # 获取相机到世界坐标系的变换矩阵
    id=object_all_link_ids[gt_all_link_mask[xs, ys] - 1]
    y1_id=ys[np.where(id==part1_id)]
    y2_id=ys[np.where(id==part2_id)]
    x1_id=xs[np.where(id==part1_id)]
    x2_id=xs[np.where(id==part2_id)]
    sample_id1_array=np.array(cam_XYZA[x1_id,y1_id,:])
    sample_id2_array=np.array(cam_XYZA[x2_id,y2_id,:])
    if(len(sample_id1_array)==0 or len(sample_id2_array)==0):
        return None,None,None,1
    mean_id1_position=np.mean(sample_id1_array,axis=0)
    mean_id2_position=np.mean(sample_id2_array,axis=0)
    #此处开始变成1*3的
    position_world1 = (mat44 @ mean_id1_position.T).T[:3]
    position_world2 = (mat44 @ mean_id2_position.T).T[:3]
    object1_matrix=env.objects[0].get_pose().to_transformation_matrix()
    object2_matrix=env.objects[1].get_pose().to_transformation_matrix()
    
    #提升高度
    position_world1[2]+=height
    position_world2[2]+=height
    object1_matrix[2,3]+=height
    object2_matrix[2,3]+=height
    
    #生成一个随机的旋转矩阵，然后进行旋转
    try_times=100
    while(try_times>0):
        try_times-=1
        axis11=[np.random.uniform(-1,1),np.random.uniform(-1,1),0]
        theta11 = np.pi*(np.random.rand(1)-0.5)/2
        new_rotation_matrix1 = te.axangle2mat(axis11, theta11)
        #print("before rotation",object1_matrix[:3,:3],object2_matrix[:3,:3])
        position_world1 = (new_rotation_matrix1 @ position_world1.T).T
        object1_matrix[:3,:3]=new_rotation_matrix1@object1_matrix[:3,:3]
        position_world2 = (new_rotation_matrix1 @ position_world2.T).T
        object2_matrix[:3,:3]=new_rotation_matrix1@object2_matrix[:3,:3]
        #print("after rotation",object1_matrix[:3,:3],object2_matrix[:3,:3])
        ##计算两个点之间的向量，然后分离一定距离，再更新矩阵
    
        relative_position = position_world1 - position_world2
        relative_position /= np.linalg.norm(relative_position)
        if(relative_position[1]<-0.8):
            break
    object1_matrix[:3,3]+=relative_position*move_dis/2
    position_world1+=relative_position*move_dis
    object2_matrix[:3,3]-=relative_position*move_dis/2
    position_world2-=relative_position*move_dis
    
    return object1_matrix,object2_matrix,part1_id,part2_id
        
def updata_rotmat_pose(i,try_num,pre_start_rotmat,start_rotmat,final_rotmat):
    theta=i*np.pi/try_num
    forward=pre_start_rotmat[:3,0]
    left=pre_start_rotmat[:3,1]
    up=pre_start_rotmat[:3,2]
    left=left*np.cos(theta)+forward*np.sin(theta)
    forward=np.cross(left,up)
    left=np.cross(up,forward)
    start_rotmat[:3,0]=forward
    start_rotmat[:3,1]=left
    start_rotmat[:3,2]=up
    final_rotmat[:3,:3]=start_rotmat[:3,:3]
    start_pose1=Pose().from_transformation_matrix(start_rotmat)
    final_pose1=Pose().from_transformation_matrix(final_rotmat)
    return start_rotmat,final_rotmat,start_pose1,final_pose1

''' get the gripper's start/end rotmat '''
def cal_final_pose_new(cam, x, y, number, out_info, start_dist=0.20, final_dist=0.08):

    rgb, depth = cam.get_observation()
    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
    cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

    # get camera XYZA matrix
    mat44 = cam.get_metadata()['mat44']
    

    position_cam = cam_XYZA[x, y,:]  
    position_cam_xyz1= position_cam
    position_world = (mat44 @ position_cam_xyz1.T).T
    
    # get pixel 3D position (cam/world)
    position_world = position_world[:3]
    out_info['position_cam' + number] = position_cam.tolist()   # contact point at camera c
    out_info['position_world' + number] = position_world.tolist()   # world

    # get pixel 3D pulling direction (cam/world)
    gt_nor = cam.get_normal_map()
    direction_cam = gt_nor[x, y, :3]
    direction_cam /= np.linalg.norm(direction_cam)
    direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam
    out_info['norm_direction_camera' + number] = direction_cam.tolist()
    out_info['norm_direction_world' + number] = direction_world.tolist()

    degree = min(30,np.abs(np.random.normal(loc=0, scale=25, size=[1])))
    threshold_up = (degree+3) * np.pi / 180
    threshold_down = max(degree-3,0) * np.pi / 180
        # sample a random direction in the hemisphere (cam/world)
    action_direction_cam = np.random.randn(3).astype(np.float32)
    action_direction_cam /= np.linalg.norm(action_direction_cam)
    if(action_direction_cam[2]>0):
        action_direction_cam=-action_direction_cam
        # while action_direction_cam @ direction_cam > -np.cos(np.pi / 6):  # up_norm_thresh: 30
    num_trial = 0
    while (action_direction_cam @ direction_cam > -np.cos(threshold_up) or action_direction_cam @ direction_cam < -np.cos(threshold_down))\
            and num_trial < 1000 :  # up_norm_thresh: 30
        action_direction_cam = np.random.randn(3).astype(np.float32)
        action_direction_cam /= np.linalg.norm(action_direction_cam)
        num_trial += 1
        if(action_direction_cam[2]>0):
            action_direction_cam=-action_direction_cam
    action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam

    out_info['gripper_direction_world' + number] = action_direction_world.tolist()

    # compute final pose
    up = np.array(action_direction_world, dtype=np.float32)
    forward = np.random.randn(3).astype(np.float32)
    while abs(up @ forward) > 0.99:
        forward = np.random.randn(3).astype(np.float32)

    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)
    forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
    left_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ left
    out_info['gripper_forward_direction_world' + number] = forward.tolist()
    out_info['gripper_forward_direction_cam' + number] = forward_cam.tolist()
    out_info['gripper_left_direction_world' + number] = left.tolist()
    out_info['gripper_left_direction_cam' + number] = left_cam.tolist()
    rotmat = np.eye(4).astype(np.float32)   # rotmat: world coordinate
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up

    
    final_rotmat = np.array(rotmat, dtype=np.float32)
    final_rotmat[:3, 3] = position_world - action_direction_world * final_dist
    final_pose = Pose().from_transformation_matrix(final_rotmat)
    out_info['target_rotmat_world' + number] = final_rotmat.tolist()

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = position_world - action_direction_world * start_dist
    start_pose = Pose().from_transformation_matrix(start_rotmat)
    out_info['start_rotmat_world' + number] = start_rotmat.tolist()

    return start_pose, start_rotmat, final_pose, final_rotmat, up, forward

def transform_action_from_world_to_robot(action : np.ndarray, pose : sapien.Pose):
    # :param action: (7,) np.ndarray in world frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # :param pose: sapien.Pose of the robot base in world frame
    # :return: (7,) np.ndarray in robot frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # transform action from world to robot frame
    action_mat = np.zeros((4,4))
    action_mat[:3,:3] = transforms3d.euler.euler2mat(action[3], action[4], action[5])
    action_mat[:3,3] = action[:3]
    action_mat[3,3] = 1
    action_mat_in_robot = np.matmul(np.linalg.inv(pose.to_transformation_matrix()),action_mat)
    action_robot = np.zeros(7)
    action_robot[:3] = action_mat_in_robot[:3,3]
    action_robot[3:6] = transforms3d.euler.mat2euler(action_mat_in_robot[:3,:3],axes='sxyz')
    action_robot[6] = action[6]
    return action_robot

def dual_gripper_wait_n_steps(robot1, robot2, n, vis_gif=False, vis_gif_interval=500, cam=None):
    imgs = []

    robot1.clear_velocity_command()
    robot2.clear_velocity_command()
    for i in range(500):
        qf1=robot1.robot.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)
        qf2=robot2.robot.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)
        robot1.robot.set_qf(qf1)
        robot2.robot.set_qf(qf2)
        robot2.env.step()
        robot2.env.render()
        if vis_gif and ((i + 1) % vis_gif_interval == 0):
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            imgs.append(fimg)

    if vis_gif:
        return imgs
    

def single_gripper_wait_n_steps(robot, n, vis_gif=False, vis_gif_interval=500, cam=None):
    imgs = []
    robot.clear_velocity_command()
    for i in range(500):
        qf=robot.robot.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)
        robot.robot.set_qf(qf)
        robot.env.step()
        if vis_gif and ((i + 1) % vis_gif_interval == 0 or i == 0):
            robot.env.render()
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            imgs.append(fimg)
    # robot.robot.set_qf([0] * robot.robot.dof)

    if vis_gif:
        return imgs
    
'''new version'''
def interpolate_pose_batch(start_pose, final_pose, num_steps):
    # receive start_pose and final_pose,return a list of middle poses
    ANS=[]
    if(num_steps==0):
        print("num_steps should be greater than 0")
        return ANS
    start_xyz=start_pose[:3,3]
    final_xyz=final_pose[:3,3]
    start_quat_obj = Quaternion(tq.mat2quat(start_pose[:3,:3]))
    final_quat_obj = Quaternion(tq.mat2quat(final_pose[:3,:3]))
    for i in range(num_steps+1):
        xyz=start_xyz+(final_xyz-start_xyz)*i/num_steps
        quat = Quaternion.slerp(start_quat_obj, final_quat_obj, i/num_steps).elements
        pose=Pose(xyz,quat)
        ANS.append(pose)
    return ANS

def quick_rot_rotmat(original_mat, theta):
    # keep up no change
    forward, left, up = original_mat[:3,0], original_mat[:3,1], original_mat[:3,2]
    left=left*np.cos(theta)+forward*np.sin(theta)
    forward=np.cross(left,up)
    left=np.cross(up,forward)
    return_mat=original_mat.copy()
    return_mat[:3,0], return_mat[:3,1], return_mat[:3,2] = forward, left, up
    return_pose=Pose().from_transformation_matrix(return_mat)
    return return_pose, return_mat

def single_gripper_move_to_target_pose(robot, target_ee_pose, num_steps=1000, robot_id=None,vis_gif=False, cam=None,check_tactile=False,part_id=None,part2_id=None,env=None):   
    imgs = []
    if(type(part_id)==list):
        part2_id=part_id
    else:
        part2_id=[part_id]
    if env is None:
        check_tactile=False
    ## add table check
    part2_id.append(2)
    ##
    num_steps=2000
    start_ee_pos = robot.robot.get_root_pose().to_transformation_matrix().copy()
    mid_pos=interpolate_pose_batch(start_ee_pos, target_ee_pose, num_steps)
    if check_tactile:
        init_pose = env.objects[0].get_root_pose()
    i=0
    for i in range(num_steps):
        #if i % 50 == 0:
        robot.robot.set_root_pose(mid_pos[i])
        qf=robot.robot.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)
        robot.robot.set_qf(qf)
        robot.env.step()
        if vis_gif and ((i + 1) % 250 == 0):
            robot.env.render()
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            imgs.append(fimg) 
        if check_tactile:
            now_pose=env.objects[0].get_root_pose()
            if(np.linalg.norm(now_pose.p-init_pose.p)>0.01 or np.linalg.norm(now_pose.q-init_pose.q)>0.01):
                break     
    if vis_gif:
         return imgs
    return

def dual_gripper_move_to_target_pose(robot1, robot2, target_ee_pose1, target_ee_pose2, num_steps=2000, vis_gif=False,cam=None, check_tactile=False, robot_id1=None, part_id1=None, robot_id2=None, part_id2=None):
    
    imgs = []
    num_steps=2000
    start_ee_pos1 = robot1.robot.get_root_pose().to_transformation_matrix().copy()
    start_ee_pos2 = robot2.robot.get_root_pose().to_transformation_matrix().copy()
    mid_pos1=interpolate_pose_batch(start_ee_pos1,target_ee_pose1,num_steps)
    mid_pos2=interpolate_pose_batch(start_ee_pos2,target_ee_pose2,num_steps)
    i=0

    if (type(part_id1)==list):
        check_list1=[robot_id1]+part_id1
    else:
        check_list1=[robot_id1,part_id1]
    if (type(part_id2)==list):
        check_list2=[robot_id2]+part_id2
    else:
        check_list2=[robot_id2,part_id2]
    
    for i in range(num_steps):
        #if i % 50 == 0:
        robot1.robot.set_root_pose(mid_pos1[i])
        robot2.robot.set_root_pose(mid_pos2[i])
        qf1=robot1.robot.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)
        qf2=robot2.robot.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)
        robot1.robot.set_qf(qf1)
        robot2.robot.set_qf(qf2)
        robot1.env.step()
        if vis_gif and ((i + 1) % 500 == 0):
            robot1.env.render()
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            imgs.append(fimg)
        if check_tactile:
            contact = robot1.env.check_contacts_exist(check_list1, check_list2)
            if contact:
                break
    if vis_gif:
         return imgs


def add_suction(part1, part2, env, use_lock_motion=False):
    from sapien.core import Pose
    relative_pose = part1.get_pose().inv() * part2.get_pose()
    # magic_number = -0.03
    # relative_pos = [0, 0, magic_number]
    # relative_pose.set_p(relative_pos)
    suction_drive = env.scene.create_drive(part1, relative_pose, part2, Pose())
    
    if use_lock_motion:
        suction_drive.lock_motion(1, 1, 1, 1, 1, 1)
    else:
        suction_drive.set_x_properties(1e10, 1e10, 1e10)
        suction_drive.set_y_properties(1e10, 1e10, 1e10)
        suction_drive.set_z_properties(1e10, 1e10, 1e10)
        suction_drive.set_x_twist(1e10, 1e10, 1e10)
        suction_drive.set_y_twist(1e10, 1e10, 1e10)
        suction_drive.set_z_twist(1e10, 1e10, 1e10)
    return suction_drive

def release_suction(suction_drive, use_lock_motion=False):
    if use_lock_motion:
        suction_drive.free_motion(1, 1, 1, 1, 1, 1)
    else:
        suction_drive.set_x_properties(0, 0, 0)
        suction_drive.set_y_properties(0, 0, 0)
        suction_drive.set_z_properties(0, 0, 0)
        suction_drive.set_x_twist(0, 0, 0)
        suction_drive.set_y_twist(0, 0, 0)
        suction_drive.set_z_twist(0, 0, 0)

def get_pose_delta(Pose1:Pose, Pose2:Pose):
    # now_qpos: [x,y,z,quaternion], target_pose: [x,y,z,quaternion]
    p1, q1, p2, q2 = Pose1.p, Pose1.q, Pose2.p, Pose2.q
    delta_p = p2 - p1
    q1_inv = tq.qinverse(q1)
    q_rel = tq.qmult(q2, q1_inv)
    if np.abs(q_rel[0]) > 1:
        q_rel[0] = 1
    angle = 2 * np.arccos(q_rel[0])
    angle_degrees = np.degrees(angle)
    ans = np.zeros(4).astype(np.float16)
    ans[:3], ans[3] = delta_p, angle_degrees
    if np.abs(ans[3]) > 180:
        ans[3] = 360 - ans[3]
    return np.linalg.norm(ans[:3]), ans[3]  # distance, angle