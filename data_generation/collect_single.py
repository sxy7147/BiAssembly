import os
import numpy as np
from PIL import Image
import utils
from argparse import ArgumentParser
from sapien.core import Pose
from env import Env
from camera import Camera
import json
import random
import imageio
import math
import copy
import sys
import time
import sys
import logging
import transforms3d.quaternions as tq

# 设置日志级别为 ERROR
logging.basicConfig(level=logging.ERROR)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'robots'))
sys.path.append(os.path.join(BASE_DIR, '../'))
# from method.inference_data_fast_GT_type3 import simulation
from robots.panda_robot import Robot


#time.sleep(100)

parser = ArgumentParser()
parser.add_argument('--trial_id', type=int)
parser.add_argument('--category', type=str)
parser.add_argument('--shape_id', type=str)
parser.add_argument('--cut_type', type=str)
parser.add_argument('--cnt_id', type=str)
parser.add_argument('--file', type=str, default="None")
parser.add_argument('--out_dir', type=str)
parser.add_argument('--random_seed', type=int, default=None)

parser.add_argument('--damping', type=int, default=10)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--start_dist', type=float, default=0.30)
parser.add_argument('--final_dist', type=float, default=0.10)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--long_move_steps', type=int, default=2000)
parser.add_argument('--short_move_steps', type=int, default=2000)
parser.add_argument('--wait_steps', type=int, default=2000)
parser.add_argument('--threshold', type=int, default=3)

parser.add_argument('--checkType', type=str, default='euler')
parser.add_argument('--save_data', action='store_true', default=True)
parser.add_argument('--no_gui', action='store_true', default=True, help='no_gui [default: False]')

parser.add_argument('--object_dir', type=str)

parser.add_argument("--scale", type=float, default=0.4)
parser.add_argument("--height",type=float,default=0.3)
parser.add_argument('--obj_dir', type=str,default='../assets/object/everyday2pieces_selected')
parser.add_argument("--no_suction",action='store_true',default=False)
args = parser.parse_args()

debug_info=False
t0=time.time()
shape_id = args.shape_id
category = args.category
trial_id = args.trial_id
if args.file == "None":
    args.file = os.path.join(args.obj_dir, args.category, args.shape_id, args.cut_type)

object_urdf_fn_A = os.path.join(args.file, 'piece_0.urdf')
object_urdf_fn_B = os.path.join(args.file, 'piece_1.urdf')

density_dict = {'Cookie': 5.0, 'Plate': 2.0}
density=30
color_list = [[1.0, 0, 0, 1], [0, 0, 1.0, 1], [0, 1.0, 0, 1]]   # rgba
out_dir_root = args.out_dir
out_dir = os.path.join(args.out_dir, category, shape_id)
if not os.path.exists(out_dir):
    os.makedirs(out_dir,exist_ok=True)
if args.random_seed is not None:
    np.random.seed(args.random_seed)

out_info = dict()
fimg = None
success = False

env = Env(show_gui=(not args.no_gui), set_ground=True)
cam = Camera(env, fixed_position=True)

use_lock_motion = True
object_material = env.get_material(1, 1, 0.01)

axis=[0,0,1]
angle=random.uniform(0, 2*math.pi)
size_dict={"BeerBottle":0.4,"Bottle":0.4,"Bowl":0.4,"Cookie":0.4,"Cup":0.4,"DrinkBottle":0.4,"DrinkingUtensil":0.4,"Mirror":0.4,"Mug":0.4,"PillBottle":0.4,
           "Plate":0.4,"Ring":0.4,"Spoon":0.4,"Statue":0.4,"Teacup":0.6,"Teapot":0.4,"ToyFigure":0.4,"Vase":0.4,"WineBottle":0.4,"WineGlass":0.4,
           }

scaled=args.scale
height=args.height
rand_s_Pose=Pose([0.5,0,height+0.1],[1,0,0,0])
env.load_object(object_urdf_fn_A, object_material, scale=scaled, density=density, given_pose=rand_s_Pose, color=color_list[0])    
env.load_object(object_urdf_fn_B, object_material, scale=scaled, density=density, given_pose=rand_s_Pose, color=color_list[1])
suction_pre_objects=utils.add_suction(env.objects[0].get_links()[-1], env.objects[1].get_links()[-1], env, use_lock_motion=use_lock_motion)
''' setup robot '''
robot_urdf_fn = './robots/panda_gripper.urdf'     # gripper
robot_material = env.get_material(1, 1, 0.01)
robot_scale = 3

robot1 = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_scale)
robot2 = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_scale)
robot1_id = robot1.robot.get_links()[-1].get_id()
robot2_id = robot2.robot.get_links()[-1].get_id()
robot1.open_gripper()
robot2.open_gripper()

robot1_actor_ids = [robot1.hand_actor_id] + robot1.gripper_actor_ids
robot2_actor_ids = [robot2.hand_actor_id] + robot2.gripper_actor_ids
robot1_full_ids=robot1_actor_ids + [robot1_id]
robot2_full_ids=robot2_actor_ids + [robot2_id]

robot1_y, robot2_y = -1.1, 1.1
robot1_pose = Pose([1, robot1_y, 1], [0.707, 0, 0, 0.707])
robot2_pose = Pose([1, robot2_y, 1], [0.707, 0, 0, -0.707])
robot1.robot.set_root_pose(robot1_pose)
robot2.robot.set_root_pose(robot2_pose)

env.step()
env.render()
rgb_pose, _ = cam.get_observation()

# env.start_checking_contact()
try:
    still_timesteps, wait_imgs = env.wait_for_object_still(cam=cam, visu=True)
except Exception:
    print(f'{args.trial_id} collision during initializing!!!')
    exit(1)



gif_imgs = []
gif_imgs.extend(wait_imgs)


''' get the init pose of the two parts '''
init_pose1 = env.objects[0].get_root_pose() # world coordinate
init_position1 = init_pose1.p.flatten()
init_rotation1 = init_pose1.q.flatten()

init_pose2 = env.objects[1].get_root_pose()
init_position2 = init_pose2.p.flatten()
init_rotation2 = init_pose2.q.flatten()


mid_obj1_matrix,mid_obj2_matrix,part1_id,part2_id=utils.get_valid_rotmat(cam,env)

utils.release_suction(suction_pre_objects, use_lock_motion=use_lock_motion)
#更新两个物体的初始位置，
axis=np.array([0,0,1])
rand_pose_p_1=np.clip(np.random.randn(3),-0.1,0.1)+np.array([0.5,-0.3,0.3])
rand_pose_p_2=np.clip(np.random.randn(3),-0.1,0.1)+np.array([0.5,0.3,0.3])
rand_pose_p_1[2]=0.3
rand_pose_p_2[2]=0.3
axis1=np.clip(np.random.randn(3),-0.1,0.1)+np.array([0,0,1])
axis2=np.clip(np.random.randn(3),-0.1,0.1)+np.array([0,0,1])
axis1=axis1/np.linalg.norm(axis1)
axis2=axis2/np.linalg.norm(axis2)
angle1=np.random.uniform(0,2*math.pi)
angle2=np.random.uniform(0,2*math.pi)
rand_s_PoseA=Pose(rand_pose_p_1,tq.axangle2quat(axis1,angle1))
rand_s_PoseB=Pose(rand_pose_p_2,tq.axangle2quat(axis2,angle2))
out_info['objA_init_p']=rand_s_PoseA.p.tolist()
out_info['objA_init_q']=rand_s_PoseA.q.tolist()
out_info['objB_init_p']=rand_s_PoseB.p.tolist()
out_info['objB_init_q']=rand_s_PoseB.q.tolist()
env.objects[0].set_root_pose(rand_s_PoseA)
env.objects[1].set_root_pose(rand_s_PoseB)

try:
    still_timesteps, wait_imgs = env.wait_for_object_still(cam=cam, visu=True)
except Exception:
    print(f'{args.trial_id} collision during initializing!!!')
    exit(1)


if(mid_obj1_matrix is None):
    print("No valid rotation matrix found, exit")
    exit(1)





''' get the init pose of the two parts '''
init_pose1 = env.objects[0].get_root_pose() # world coordinate
init_position1 = init_pose1.p.flatten()
init_rotation1 = init_pose1.q.flatten()

init_pose2 = env.objects[1].get_root_pose()
init_position2 = init_pose2.p.flatten()
init_rotation2 = init_pose2.q.flatten()

init_cam_XYZA_list = utils.save_need_data(env=env, cam=cam, out_dir_root=out_dir_root, out_sub_dir="init", trial_id=trial_id, category=category, shape_id=shape_id)
''' sample two points on the two parts '''
object_all_link_ids = [part1_id, part2_id]



gt_all_link_mask = cam.get_movable_link_mask(object_all_link_ids)   # (448, 448)

xs, ys = np.where(gt_all_link_mask > 0)

if len(xs) == 0:
    print('Terrible!!!')
    exit(1)

# 选择两个抓取点
for _ in range(50):
    idx1 = np.random.randint(len(xs))
    x1, y1 = xs[idx1], ys[idx1]
    part_id = object_all_link_ids[gt_all_link_mask[x1, y1] - 1]
    if part_id == part1_id:
        break
for _ in range(50): 
    idx2 = np.random.randint(len(xs))
    x2, y2 = xs[idx2], ys[idx2]
    part_id = object_all_link_ids[gt_all_link_mask[x2, y2] - 1]
    if part_id == part2_id:
        break



env.render()
out_info['random_seed'] = args.random_seed
out_info['camera_metadata'] = cam.get_metadata_json()
out_info['shape_id'] = shape_id
out_info['category'] = category
out_info['load_object_setting'] = env.settings
out_info['init_position1'] = init_position1.tolist()
out_info['init_rotation1'] = init_rotation1.tolist()
out_info['init_position2'] = init_position2.tolist()
out_info['init_rotation2'] = init_rotation2.tolist()
out_info['pixel1_idx'] = int(idx1)
out_info['pixel2_idx'] = int(idx2)
out_info['part1_id'] = int(part1_id)
out_info['part2_id'] = int(part2_id)
out_info['use_arm'] = False
out_info['success'] = 'False'
out_info['result'] = 'VALID'
out_info['start_dist'] = args.start_dist
out_info['final_dist'] = args.final_dist
out_info['robot_scale'] = robot_scale
out_info['long_move_steps'] = args.long_move_steps
out_info['short_move_steps'] = args.short_move_steps

# randomly sample the pick-up pose
start_pose1, start_rotmat1, final_pose1, final_rotmat1, _, _ = utils.cal_final_pose_new(cam, x1, y1, '1', out_info, start_dist=args.start_dist, final_dist=args.final_dist)
start_pose2, start_rotmat2, final_pose2, final_rotmat2, _, _ = utils.cal_final_pose_new(cam, x2, y2, '2', out_info, start_dist=args.start_dist, final_dist=args.final_dist)
### save point cloud
# get the mask of object A & object B
object_link_id1 = env.objects[0].get_links()[-1].get_id()
object_link_id2 = env.objects[1].get_links()[-1].get_id()
gt_object1_mask = cam.get_movable_link_mask([object_link_id1])
gt_object2_mask = cam.get_movable_link_mask([object_link_id2])

num_pts = 8192
final_vs, final_fs = env.get_global_mesh(env.objects[0])
whole_pc_0 = env.sample_pc(final_vs, final_fs, n_points=16384)
final_vs, final_fs = env.get_global_mesh(env.objects[1])
whole_pc_1 = env.sample_pc(final_vs, final_fs, n_points=16384)

print(f"{trial_id} initialize: {time.time() - t0:.2f} seconds")
# 先处理爪子1 
try_num=6
object_id=0
object_id2=1-object_id

object_pose1=env.objects[object_id].get_root_pose()
object_pose2=env.objects[1-object_id].get_root_pose()

pre_start_rotmat1=copy.deepcopy(start_rotmat1)
pre_final_rotmat1=copy.deepcopy(final_rotmat1)
mid_away_gripper_pose1=Pose([1,robot1_y,0.8],[0.707, 0, 0, 0.707])
mid_away_gripper_pose2=Pose([1,robot2_y,0.8],[0.707,0,0,-0.707])

for i in range(try_num):
    ret_trial=0
    robot1.robot.set_root_pose(start_pose1)
    robot1.open_gripper()
    while(ret_trial<10):
        env.objects[object_id].set_root_pose(object_pose1)
        env.objects[object_id2].set_root_pose(object_pose2)
        utils.single_gripper_wait_n_steps(robot1, n=args.wait_steps, cam=cam, vis_gif=False)
        if(np.linalg.norm(env.objects[object_id].pose.p-object_pose1.p)<0.01) and (np.linalg.norm(env.objects[object_id2].pose.p-object_pose2.p)<0.01):
            break
        ret_trial+=1
    imgs=utils.single_gripper_move_to_target_pose(robot1, final_rotmat1, cam=cam, vis_gif=True,check_tactile=True,robot_id=robot1_actor_ids, part_id=part1_id,env=env)
    #print("step",i,"target pose",final_pose1,"current pose",robot1.end_effector.get_pose())
    gif_imgs.extend(imgs)
    
    robot1.close_gripper()
    imgs = utils.single_gripper_wait_n_steps(robot1, n=args.wait_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
    flag_contact = env.check_contacts_exist(robot1_full_ids, [part1_id])
    gripper_open_size=robot1.robot.get_qpos()[-2:]
    if(gripper_open_size[0]+gripper_open_size[1]<0.01):
        flag_contact=False    
    if(np.linalg.norm(env.objects[object_id2].pose.p-object_pose2.p)>0.01):
        flag_contact=False
    if (not flag_contact):
        start_pose1, start_rotmat1 = utils.quick_rot_rotmat(pre_start_rotmat1, (i+1)*np.pi/try_num)
        final_rotmat1[:3,:3]=start_rotmat1[:3,:3]
        final_pose1=Pose().from_transformation_matrix(final_rotmat1)
        continue
    if not args.no_suction:
        suction_arm1 = utils.add_suction(robot1.robot.get_links()[-1], env.get_link(part1_id), env=env, use_lock_motion=use_lock_motion)

    lift_pose1 = env.objects[object_id].get_root_pose() # world coordinate
    lift_position1 = lift_pose1.p.flatten()
    lift_rotation1 = lift_pose1.q.flatten()
    lift_obj1_matrix = lift_pose1.to_transformation_matrix() 
    finger1_pose = robot1.robot.get_root_pose()
    finger1_pose_matrix = robot1.robot.get_root_pose().to_transformation_matrix()

    
    out_info['initial_position1'] = lift_position1.tolist()
    out_info['initial_rotation1'] = lift_rotation1.tolist()
    out_info['initial_finger1_pose_matrix'] = finger1_pose_matrix.tolist()

    mid_rotmat1 =  mid_obj1_matrix @ np.linalg.inv(lift_obj1_matrix) @ finger1_pose_matrix
    mid_away_gripper_rotmat1=mid_away_gripper_pose1.to_transformation_matrix() 
    imgs=utils.single_gripper_move_to_target_pose(robot1, mid_away_gripper_rotmat1, cam=cam, vis_gif=True, robot_id=robot1_actor_ids, part_id=part1_id)
    gif_imgs.extend(imgs)
    robot1.close_gripper()
    imgs = utils.single_gripper_wait_n_steps(robot1, n=args.wait_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
    
    flag_contact = env.check_contacts_exist(robot1_full_ids, [part1_id])
    gripper_open_size=robot1.robot.get_qpos()[-2:]
    if(gripper_open_size[0]+gripper_open_size[1]<0.01):
        flag_contact=False    
    if not flag_contact:
        if not args.no_suction:
            utils.release_suction(suction_arm1, use_lock_motion=use_lock_motion)
        start_pose1, start_rotmat1 = utils.quick_rot_rotmat(pre_start_rotmat1, (i+1)*np.pi/try_num)
        final_rotmat1[:3,:3]=start_rotmat1[:3,:3]
        final_pose1=Pose().from_transformation_matrix(final_rotmat1)
        continue
    else:
        break
saved_dir=os.path.join(out_dir_root, 'collision_files')
os.makedirs(saved_dir, exist_ok=True)
Image.fromarray((gt_object1_mask > 0).astype(np.uint8) * 255).save(
os.path.join(saved_dir, 'interaction_mask_%s_%s_%d_1.png' % (category, shape_id[:4], trial_id)))
Image.fromarray((gt_object2_mask > 0).astype(np.uint8) * 255).save(
os.path.join(saved_dir, 'interaction_mask_%s_%s_%d_2.png' % (category, shape_id[:4], trial_id)))
np.savez(os.path.join(saved_dir, 'collision_visual_shape_%s_%s_%d_1' % (category, shape_id[:4], trial_id)), pts=whole_pc_0)
np.savez(os.path.join(saved_dir, 'collision_visual_shape_%s_%s_%d_2' % (category, shape_id[:4], trial_id)), pts=whole_pc_1)
table_contact = robot1.env.check_contacts_exist(robot1_actor_ids, [2])
if (not flag_contact):
    os.makedirs(os.path.join(out_dir_root, 'first_step_fail_gif'), exist_ok=True)
    os.makedirs(os.path.join(out_dir_root, 'first_step_fail_files'), exist_ok=True)
    if args.save_data:
        imageio.mimsave(os.path.join(out_dir_root, 'first_step_fail_gif', 'fail_%s_%s_%d.gif' % (category, shape_id[:4], trial_id)), gif_imgs)
        with open(os.path.join(out_dir_root, 'first_step_fail_files', 'fail_%s_%s_%d.json' % (category, shape_id[:4], trial_id)), 'w') as fout:
            json.dump(out_info, fout) 
        utils.save_all_data()
    exit(2)

print(f"{trial_id} first step: {time.time() - t0:.2f} seconds")

object_pose2=env.objects[1-object_id].get_root_pose()
pre_start_rotmat2=copy.deepcopy(start_rotmat2)
pre_final_rotmat2=copy.deepcopy(final_rotmat2)
for i in range(try_num):
    ret_trial=0
    robot2.robot.set_root_pose(start_pose2)
    robot2.open_gripper()
    while(ret_trial<10):
        env.objects[object_id2].set_root_pose(object_pose2)
        utils.single_gripper_wait_n_steps(robot2, n=args.wait_steps, cam=cam, vis_gif=True)
        if(np.linalg.norm(env.objects[object_id2].pose.p-object_pose2.p))<0.01:
            break
        ret_trial+=1
    imgs=utils.single_gripper_move_to_target_pose(robot2, final_rotmat2, cam=cam, vis_gif=True,check_tactile=True, robot_id=robot2_actor_ids, part_id=part2_id,env=env)
    gif_imgs.extend(imgs)
    robot2.close_gripper()
    imgs = utils.single_gripper_wait_n_steps(robot2, n=args.wait_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
    flag_contact = env.check_contacts_exist(robot2_full_ids, [part2_id])
    gripper_open_size=robot2.robot.get_qpos()[-2:]
    if(gripper_open_size[0]+gripper_open_size[1]<0.01):
        flag_contact=False
    if (not flag_contact):
        start_pose2, start_rotmat2 = utils.quick_rot_rotmat(pre_start_rotmat2, (i+1)*np.pi/try_num)
        final_rotmat2[:3,:3]=start_rotmat2[:3,:3]
        final_pose2=Pose().from_transformation_matrix(final_rotmat2)
        continue

    if not args.no_suction:
        suction_arm2 = utils.add_suction(robot2.robot.get_links()[-1], env.get_link(part2_id), env=env, use_lock_motion=use_lock_motion)
    
    lift_pose2 = env.objects[object_id2].get_root_pose() # world coordinate
    lift_position2 = lift_pose2.p.flatten()
    lift_rotation2 = lift_pose2.q.flatten()
    lift_obj2_matrix = lift_pose2.to_transformation_matrix()
    finger2_pose = robot2.robot.get_root_pose()
    finger2_pose_matrix = robot2.robot.get_root_pose().to_transformation_matrix()
    
    out_info['initial_position2'] = lift_position2.tolist()
    out_info['initial_rotation2'] = lift_rotation2.tolist()
    out_info['initial_finger2_pose_matrix'] = finger2_pose_matrix.tolist()

    mid_rotmat2 =  mid_obj2_matrix @ np.linalg.inv(lift_obj2_matrix) @ finger2_pose_matrix
    mid_away_gripper_rotmat2=mid_away_gripper_pose2.to_transformation_matrix() 
    imgs=utils.single_gripper_move_to_target_pose(robot2, mid_away_gripper_rotmat2, cam=cam, vis_gif=True, robot_id=robot2_actor_ids, part_id=part2_id)
    gif_imgs.extend(imgs)
    robot2.close_gripper()
    imgs = utils.single_gripper_wait_n_steps(robot2, n=args.wait_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
    
    flag_contact = env.check_contacts_exist(robot2_full_ids, [part2_id])
    gripper_open_size=robot2.robot.get_qpos()[-2:]
    if(gripper_open_size[0]+gripper_open_size[1]<0.01):
        flag_contact=False
    if not flag_contact:
        if not args.no_suction:
            utils.release_suction(suction_arm2, use_lock_motion=use_lock_motion)
        start_pose2, start_rotmat2 = utils.quick_rot_rotmat(pre_start_rotmat2, (i+1)*np.pi/try_num)
        final_rotmat2[:3,:3]=start_rotmat2[:3,:3]
        final_pose2=Pose().from_transformation_matrix(final_rotmat2)
        continue
    else:
        break
table_contact = robot2.env.check_contacts_exist(robot2_actor_ids, [2])   
if (not flag_contact):
    os.makedirs(os.path.join(out_dir_root, 'second_step_fail_gif'), exist_ok=True)
    os.makedirs(os.path.join(out_dir_root, 'second_step_fail_files'), exist_ok=True)
    if args.save_data:
        imageio.mimsave(os.path.join(out_dir_root, 'second_step_fail_gif', 'fail_%s_%s_%d.gif' % (category, shape_id[:4], trial_id)), gif_imgs)
        with open(os.path.join(out_dir_root, 'second_step_fail_files', 'fail_%s_%s_%d.json' % (category, shape_id[:4], trial_id)), 'w') as fout:
            json.dump(out_info, fout)
        utils.save_all_data()
    exit(3)    

out_info['objA_init_p']=object_pose1.p.tolist()
out_info['objA_init_q']=object_pose1.q.tolist()
out_info['objB_init_p']=object_pose2.p.tolist()
out_info['objB_init_q']=object_pose2.q.tolist()

out_info['mid_away_objA_p']=env.objects[object_id].get_root_pose().p.tolist()
out_info['mid_away_objA_q']=env.objects[object_id].get_root_pose().q.tolist()
out_info['mid_away_objB_p']=env.objects[object_id2].get_root_pose().p.tolist()
out_info['mid_away_objB_q']=env.objects[object_id2].get_root_pose().q.tolist()

print(f"{trial_id} second step: {time.time() - t0:.2f} seconds")   
utils.save_need_data(env=env, cam=cam, trial_id=trial_id, out_dir_root=out_dir_root, out_sub_dir="pickup", category=category, shape_id=shape_id)


imgs=utils.single_gripper_move_to_target_pose(robot1, mid_rotmat1, cam=cam, vis_gif=True,robot_id=robot1_actor_ids, part_id=part1_id)
gif_imgs.extend(imgs)
imgs = utils.single_gripper_wait_n_steps(robot1, n=args.wait_steps, cam=cam, vis_gif=True)
gif_imgs.extend(imgs)  

imgs=utils.single_gripper_move_to_target_pose(robot2, mid_rotmat2, cam=cam, vis_gif=True,robot_id=robot2_actor_ids, part_id=part2_id)
gif_imgs.extend(imgs)  
imgs = utils.single_gripper_wait_n_steps(robot2, n=args.wait_steps, cam=cam, vis_gif=True)
gif_imgs.extend(imgs)  


utils.save_need_data(env=env, cam=cam, trial_id=trial_id, out_dir_root=out_dir_root, out_sub_dir="mid", category=category, shape_id=shape_id)
  
out_info['mid_objA_p']=env.objects[object_id].get_root_pose().p.tolist()
out_info['mid_objA_q']=env.objects[object_id].get_root_pose().q.tolist()
out_info['mid_objB_p']=env.objects[object_id2].get_root_pose().p.tolist()
out_info['mid_objB_q']=env.objects[object_id2].get_root_pose().q.tolist()  
  
out_info['start_rotmat_world1'] = start_rotmat1.tolist()
out_info['start_rotmat_world2'] = start_rotmat2.tolist()
out_info['mid_rotmat_world1'] = mid_rotmat1.tolist()
out_info['mid_rotmat_world2'] = mid_rotmat2.tolist()
out_info['target_rotmat_world1'] = final_rotmat1.tolist()
out_info['target_rotmat_world2'] = final_rotmat2.tolist()   
out_info['mid_away_gripper_rotmat1']=mid_away_gripper_rotmat1.tolist()
out_info['mid_away_gripper_rotmat2']=mid_away_gripper_rotmat2.tolist()  

''' try to assemble the parts '''

finger1_pose = robot1.robot.get_root_pose()
finger1_pose_matrix = robot1.robot.get_root_pose().to_transformation_matrix()
finger2_pose = robot2.robot.get_root_pose()
finger2_pose_matrix = robot2.robot.get_root_pose().to_transformation_matrix()
now_obj1_pose = env.objects[object_id].get_root_pose()
now_obj2_pose = env.objects[object_id2].get_root_pose()
move_dir=now_obj1_pose.p-now_obj2_pose.p
# use gripper
assembly_rotmat1_list, assembly_rotmat2_list = [], []
assembly_rotmat1 = finger1_pose_matrix.copy()
assembly_rotmat2 = finger2_pose_matrix.copy()
assembly_rotmat1[:3, 3] -=move_dir/2
assembly_rotmat2[:3, 3] += move_dir/2
assembly_rotmat1_list.append(assembly_rotmat1)
assembly_rotmat2_list.append(assembly_rotmat2)


''''''
lift_pose1 = env.objects[0].get_root_pose() # world coordinate
lift_position1 = lift_pose1.p.flatten()
lift_rotation1 = lift_pose1.q.flatten()
lift_obj1_matrix = lift_pose1.to_transformation_matrix() 
lift_pose2 = env.objects[1].get_root_pose()
lift_position2 = lift_pose2.p.flatten()
lift_rotation2 = lift_pose2.q.flatten()
lift_obj2_matrix = lift_pose2.to_transformation_matrix() 
relative_position = lift_position1 - lift_position2     # if two parts are assembled, then relative_position = 0 
out_info['final_position1'] = lift_position1.tolist()
out_info['final_rotation1'] = lift_rotation1.tolist()
out_info['final_position2'] = lift_position2.tolist()
out_info['final_rotation2'] = lift_rotation2.tolist()

finger1_pose = robot1.robot.get_root_pose()
finger1_pose_matrix = robot1.robot.get_root_pose().to_transformation_matrix()
finger2_pose = robot2.robot.get_root_pose()
finger2_pose_matrix = robot2.robot.get_root_pose().to_transformation_matrix()

out_info['final_finger1_pose_matrix'] = finger1_pose_matrix.tolist()
out_info['final_finger2_pose_matrix'] = finger2_pose_matrix.tolist()
out_info['move_dir'] = (move_dir/np.linalg.norm(move_dir)).tolist()
out_info['movement_dist'] = move_dir.tolist()
out_info['assembly_rotmat1_list_world'] = [arr.tolist() for arr in assembly_rotmat1_list]
out_info['assembly_rotmat2_list_world'] = [arr.tolist() for arr in assembly_rotmat2_list]


try:
    for i in range(len(assembly_rotmat1_list)):
        #print('!!!end step: ', i)
        pose=Pose.from_transformation_matrix(assembly_rotmat1_list[i])
        imgs = utils.dual_gripper_move_to_target_pose(robot1, robot2, assembly_rotmat1_list[i], assembly_rotmat2_list[i], num_steps=args.long_move_steps, cam=cam, vis_gif=True,check_tactile=True,robot_id1=robot1_actor_ids,robot_id2=robot2_actor_ids,part_id1=part1_id,part_id2=part2_id)
        gif_imgs.extend(imgs)
    imgs = utils.dual_gripper_wait_n_steps(robot1, robot2, n=args.wait_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
except Exception as e:
    print('\n\n', e, '\n\n')
    os.makedirs(os.path.join(out_dir_root, 'third_step_fail_gif'), exist_ok=True)
    os.makedirs(os.path.join(out_dir_root, 'third_step_fail_files'), exist_ok=True)
    if args.save_data:
        imageio.mimsave(os.path.join(out_dir_root, 'third_step_fail_gif', 'fail_%s_%s_%d.gif' % (category, shape_id[:4], trial_id)), gif_imgs)
        with open(os.path.join(out_dir_root, 'third_step_fail_files', 'fail_%s_%s_%d.json' % (category, shape_id[:4], trial_id)), 'w') as fout:
            json.dump(out_info, fout)
        utils.save_all_data()
    exit(4)

utils.save_need_data(env=env, cam=cam, trial_id=trial_id, out_dir_root=out_dir_root, out_sub_dir="assembly", category=category, shape_id=shape_id)

assembly_pose1 = env.objects[0].get_root_pose() # world coordinate
assembly_position1 = assembly_pose1.p.flatten()
assembly_rotation1 = assembly_pose1.q.flatten()
assembly_pose2 = env.objects[1].get_root_pose() # world coordinate
assembly_position2 = assembly_pose2.p.flatten()
assembly_rotation2 = assembly_pose2.q.flatten()
final_relative_position = assembly_position1 - assembly_position2
out_info['assembly_position1'] = assembly_position1.tolist()
out_info['assembly_rotation1'] = assembly_rotation1.tolist()
out_info['assembly_position2'] = assembly_position2.tolist()
out_info['assembly_rotation2'] = assembly_rotation2.tolist()


# check success
contact1_to_1=env.check_contacts_exist(robot1_full_ids, [part1_id])
contact2_to_2=env.check_contacts_exist(robot2_full_ids, [part2_id])
bad_contacts = env.check_contacts_exist(robot1_actor_ids, robot2_actor_ids) or env.check_contacts_exist(robot1_actor_ids, [part2_id]) or env.check_contacts_exist(robot2_actor_ids, [part1_id])


delta_p,delta_q=utils.get_pose_delta(assembly_pose1, assembly_pose2)
out_info['delta_p'] = float(delta_p)
out_info['delta_q'] = float(delta_q)
print(f"{trial_id} third step: {time.time() - t0:.2f} seconds delta_p: {delta_p:.2f} delta_q: {delta_q:.2f} bad_contacts: {bad_contacts} contact1_to_1: {contact1_to_1} contact2_to_2: {contact2_to_2}")
if delta_p > 0.1 or delta_q > 20 or bad_contacts:
    os.makedirs(os.path.join(out_dir_root, 'third_step_fail_gif'), exist_ok=True)
    os.makedirs(os.path.join(out_dir_root, 'third_step_fail_files'), exist_ok=True)
    if args.save_data:
        imageio.mimsave(os.path.join(out_dir_root, 'third_step_fail_gif', 'fail_%s_%s_%d.gif' % (category, shape_id[:4], trial_id)), gif_imgs)
        with open(os.path.join(out_dir_root, 'third_step_fail_files', 'fail_%s_%s_%d.json' % (category, shape_id[:4], trial_id)), 'w') as fout:
            json.dump(out_info, fout)
        utils.save_all_data()
    exit(4)
utils.save_all_data()
print(f"{trial_id} success: {time.time() - t0:.2f} seconds")
os.makedirs(os.path.join(out_dir_root, 'succ_gif'), exist_ok=True)
os.makedirs(os.path.join(out_dir_root, 'succ_files'), exist_ok=True)
print("\n\n\nWe do save!\n\n\n")
json_file=os.path.join(out_dir_root, 'succ_files', 'succ_%s_%s_%d_%f_%f.json' % (category, shape_id[:4], trial_id, delta_p, delta_q))
imageio.mimsave(os.path.join(out_dir_root, 'succ_gif', 'succ_%s_%s_%d_%f_%f.gif' % (category, shape_id[:4], trial_id, delta_p, delta_q)), gif_imgs)
with open(json_file, 'w') as fout:
    json.dump(out_info, fout)
exit(0) 
