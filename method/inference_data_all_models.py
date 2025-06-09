import os
import torch
import numpy as np
from argparse import ArgumentParser
import json
import random
import imageio
import copy
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print('BASE_DIR: ', BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from sapien.core import Pose
from data_generation import utils
from data_generation.env import Env, ContactError, SVDError
from data_generation.camera import Camera
from data_generation.robots.panda_robot import Robot

import inference_utils
import method_utils
import datetime


def simulation(json_file, file_id, repeat_id, models, device, args):
    with open(json_file, 'r') as fin:
        result_data = json.load(fin)
    
    shape_id = result_data['shape_id']
    category = result_data['category']
    random_seed = result_data['random_seed']
    robot_scale = result_data['robot_scale']
    start_dist = result_data['start_dist']
    final_dist = result_data['final_dist']
    print('start_dist: ', start_dist)
    print('final_dist: ', final_dist)
    
    batch_size = 1
    info = [category, 1e10, 1e10]   # category, delta_p, delta_q
    use_lock_motion = True
    out_info = dict()
    out_info['origin_json_file'] = json_file

    np.random.seed(random_seed)
    
    # Initialization
    env = Env(show_gui=(not args.no_gui), set_ground=True)
    cam = Camera(env, fixed_position=True)
    mat44 = np.array(cam.get_metadata_json()['mat44'], dtype=np.float32)
    
    objectA_info = result_data['load_object_setting'][0]
    objectB_info = result_data['load_object_setting'][1]
    object_material = env.get_material(100, 100, 0.01)
    color_list = [[1.0, 0, 0, 1], [0, 0, 1.0, 1], [0, 1.0, 0, 1]]   # rgba
    height = 0.3
    rand_s_Pose=Pose([0.5, 0, height + 0.1],[1,0,0,0])
    env.load_object(objectA_info['urdf'], object_material, scale=objectA_info['scale'], density=objectA_info['density'], given_pose=rand_s_Pose, color=color_list[0]) 
    env.load_object(objectB_info['urdf'], object_material, scale=objectB_info['scale'], density=objectB_info['density'], given_pose=rand_s_Pose, color=color_list[1])     

    # Load in Robots
    robot_urdf_fn = '../assets/robot/panda_gripper.urdf'
    robot_material = env.get_material(4, 4, 0.01)
    robot1 = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_scale)
    robot2 = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_scale)
    robot1.open_gripper()
    robot2.open_gripper()
    robot1_actor_ids = [robot1.hand_actor_id] + robot1.gripper_actor_ids
    robot2_actor_ids = [robot2.hand_actor_id] + robot2.gripper_actor_ids
    robot1_id = [robot1.robot.get_links()[-1].get_id(), robot1.robot.get_links()[-2].get_id()] 
    robot2_id = [robot2.robot.get_links()[-1].get_id(), robot2.robot.get_links()[-2].get_id()]

    robot1_y, robot2_y = -1.1, 1.1
    robot1_pose = Pose([1, robot1_y, 1], [0.707, 0, 0, 0.707])
    robot2_pose = Pose([1, robot2_y, 1], [0.707, 0, 0, -0.707])
    robot1.robot.set_root_pose(robot1_pose)
    robot2.robot.set_root_pose(robot2_pose)  
 
    env.step()
    env.render()
    rgb_pose, _ = cam.get_observation()
    
    # Set the initial pose of the objects
    env.objects[0].set_root_pose(Pose(result_data['objA_init_p'], result_data['objA_init_q']))
    env.objects[1].set_root_pose(Pose(result_data['objB_init_p'], result_data['objB_init_q']))
    try:
        still_timesteps = env.wait_for_object_still(cam=cam, visu=False)
    except Exception:
        env.env_remove_articulation([env.objects[0], env.objects[1], robot1.robot, robot2.robot])
        return 'invalid', info
    
    ''' Acquire the visual observation '''
    env.step()
    env.render()
    rgb, depth = cam.get_observation()
    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
    
    object_link_id1 = env.objects[0].get_links()[-1].get_id()
    object_link_id2 = env.objects[1].get_links()[-1].get_id()
    object1_mask = cam.get_movable_link_mask([object_link_id1])
    object2_mask = cam.get_movable_link_mask([object_link_id2])
    
    if args.setting == 'BiAssembly':
        # initial observation (pointcloud)
        pc_world, object1_mask, object2_mask = inference_utils.get_observed_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, mat44, object1_mask, object2_mask, args, file_id, device, pc_dist=cam.dist)
        pc_world, pcid1, pcid2 = method_utils.sample_points_fps(pc_world, args.num_point_per_shape)
        object1_mask = object1_mask[pcid1, pcid2, :].reshape(batch_size, args.num_point_per_shape)
        object2_mask = object2_mask[pcid1, pcid2, :].reshape(batch_size, args.num_point_per_shape)
        
        # imaginary pc (in any pose)
        num_pts = 25000
        imaginary_pcs, imaginary_pc_pose = inference_utils.get_imaginary_pc(objectA_info['urdf'], objectB_info['urdf'], num_pts, device)   # B * 2 * N * 3
        imaginary_pcs, _, _ = method_utils.sample_points_fps(imaginary_pcs.reshape(batch_size * 2, -1, 3).contiguous(), args.num_point_per_shape)
        imaginary_pcs = imaginary_pcs.reshape(batch_size, 2, args.num_point_per_shape, 3).contiguous()
        
        # Disassembly Predictor and Transformation Predictor 
        with torch.no_grad():
            disassembly_dir, transformation = models['disassembly_predictor'].predict_new(imaginary_pcs, pc_world[:, :, :3].clone())      # disassembly_dir是rotated canonical pcs坐标系下的
        transformation_rotmat, assembly_object_rotmat, transformed_disassembly_dir = inference_utils.get_transformation_and_disassembly_direction(transformation, imaginary_pc_pose, disassembly_dir)
        transformation_rotmat = torch.from_numpy(transformation_rotmat).float().to(device)
        transformed_disassembly_dir = torch.from_numpy(transformed_disassembly_dir).float().to(device)
            
        # Bi-Affordance Predictor
        with torch.no_grad():
            imaginary_pcs_ones = torch.ones((batch_size * 2 * args.num_point_per_shape, 4)).to(device)
            imaginary_pcs_ones[:, :3] = imaginary_pcs.reshape(batch_size * 2 * args.num_point_per_shape, 3)
            target_pcs = (transformation_rotmat @ imaginary_pcs_ones.T).T
            target_pcs = target_pcs[:, :3].reshape(batch_size, 2, args.num_point_per_shape, 3).contiguous()
            target_pcs = inference_utils.process_target_pcs(target_pcs, args.num_point_per_shape, batch_size)
            position1, dir1, position2, dir2, aff1_scores, aff2_scores = models['bi_affordance_predictor'].inference(pc_world.clone(), target_pcs, transformed_disassembly_dir, object1_mask, object2_mask, args.aff_topk, args.critic_topk)
        
        ctpt1 = position1.view(3).detach().cpu().numpy()
        ctpt2 = position2.view(3).detach().cpu().numpy()
        dir1 = dir1.view(6).detach().cpu().numpy()
        dir2 = dir2.view(6).detach().cpu().numpy()
            
        if args.draw_aff_map and repeat_id < args.num_draw:
            inference_utils.draw_aff_map(aff1_scores.detach().cpu().numpy().reshape(-1), aff2_scores.detach().cpu().numpy().reshape(-1), \
                                        object1_mask.detach().cpu().numpy().reshape(-1), object2_mask.detach().cpu().numpy().reshape(-1), \
                                        pc_world[0, :, :3].detach().cpu().numpy(), 
                                        fn=os.path.join(args.out_dir, 'affordance', '%s_%s_%d_%d_%s.ply' % (category, shape_id[:4], file_id, repeat_id, 'map')))

        start_pose1, start_rotmat1, pick_pose1, pick_rotmat1 = inference_utils.get_movement_rotmat(ctpt1, dir1, start_dist, final_dist)
        start_pose2, start_rotmat2, pick_pose2, pick_rotmat2 = inference_utils.get_movement_rotmat(ctpt2, dir2, start_dist, final_dist)
        out_info['start_rotmat_world1'] = start_rotmat1.tolist()
        out_info['pick_rotmat_world1'] = pick_rotmat1.tolist()
        out_info['start_rotmat_world2'] = start_rotmat2.tolist()
        out_info['pick_rotmat_world2'] = pick_rotmat2.tolist()

 
    ''' Acquire the visual observation '''
    env.step()
    env.render()
    
    part1_id = env.objects[0].get_links()[0].get_id()
    part2_id = env.objects[1].get_links()[0].get_id()

    intermediate_gripper_pose1=Pose([1,robot1_y,0.8], [0.707, 0, 0, 0.707]) 
    intermediate_gripper_pose2=Pose([1,robot2_y,0.8], [0.707,0,0,-0.707])  
    intermediate_gripper_rotmat1 = intermediate_gripper_pose1.to_transformation_matrix() 
    intermediate_gripper_rotmat2 = intermediate_gripper_pose2.to_transformation_matrix()
    
    gif_imgs = [] 
    

    robot1.open_gripper()
    robot1.robot.set_root_pose(start_pose1)
    robot1.open_gripper()
    
    imgs = utils.single_gripper_move_to_target_pose(robot1, pick_rotmat1, cam=cam, vis_gif=True, check_tactile=True, robot_id=robot1_actor_ids, part_id=part1_id, env=env)
    gif_imgs.extend(imgs)
    utils.single_gripper_wait_n_steps(robot1, cam=cam, vis_gif=False)
    
    robot1.close_gripper()
    imgs = utils.single_gripper_wait_n_steps(robot1, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
    flag_contact = env.check_contact_exist(robot1_id, part1_id)
    gripper_open_size=robot1.robot.get_qpos()[-2:]
    if (gripper_open_size[0] + gripper_open_size[1] < 0.001) or not flag_contact: 
        if args.save_data:
            imageio.mimsave(os.path.join(args.out_dir, 'first_step_fail_gif', 'fail_%s_%s_%d_%d.gif' % (category, shape_id[:4], file_id, repeat_id)), gif_imgs)
            with open(os.path.join(args.out_dir, 'first_step_fail_files', 'fail_%s_%s_%d_%d.json' % (category, shape_id[:4], file_id, repeat_id)), 'w') as fout:
                json.dump(out_info, fout)
            env.env_remove_articulation([env.objects[0], env.objects[1], robot1.robot, robot2.robot])
            return 'fail1', info

    suction_arm1 = utils.add_suction(robot1.robot.get_links()[-1], env.get_link(part1_id), env=env, use_lock_motion=use_lock_motion)
    out_info['gripper_finger1'] = robot1.robot.get_qpos()[-2:].tolist()
    out_info['gripper_finger2'] = robot2.robot.get_qpos()[-2:].tolist()

    
    if args.setting == 'BiAssembly':
        mid_object_pose1 = env.objects[0].get_root_pose().to_transformation_matrix()
        assembly_gripper_rotmat1 = assembly_object_rotmat @ np.linalg.inv(mid_object_pose1) @ robot1.robot.get_root_pose().to_transformation_matrix()
        alignment_gripper_rotmat1 = copy.deepcopy(assembly_gripper_rotmat1)
        alignment_gripper_rotmat1[:3, 3] += transformed_disassembly_dir.reshape(-1).detach().cpu().numpy() * args.assembly_dist
    
    imgs=utils.single_gripper_move_to_target_pose(robot1, intermediate_gripper_rotmat1, cam=cam, vis_gif=True, robot_id=robot1_actor_ids, part_id=part1_id)
    gif_imgs.extend(imgs)
    utils.single_gripper_wait_n_steps(robot1, cam=cam, vis_gif=False)
    

    # Robot2
    robot2.open_gripper()
    robot2.robot.set_root_pose(start_pose2)
    robot2.open_gripper()
    
    imgs=utils.single_gripper_move_to_target_pose(robot2, pick_rotmat2, cam=cam, vis_gif=True,check_tactile=True, robot_id=robot2_actor_ids, part_id=part2_id ,env=env)
    gif_imgs.extend(imgs)
    utils.single_gripper_wait_n_steps(robot2, cam=cam, vis_gif=False)
    
    robot2.close_gripper()
    imgs = utils.single_gripper_wait_n_steps(robot2, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
    flag_contact = env.check_contact_exist(robot2_id, part2_id)
    gripper_open_size=robot2.robot.get_qpos()[-2:]
    if (gripper_open_size[0] + gripper_open_size[1] < 0.001) or not flag_contact:  
        if args.save_data:
            imageio.mimsave(os.path.join(args.out_dir, 'second_step_fail_gif', 'fail_%s_%s_%d_%d.gif' % (category, shape_id[:4], file_id, repeat_id)), gif_imgs)
            with open(os.path.join(args.out_dir, 'second_step_fail_files', 'fail_%s_%s_%d_%d.json' % (category, shape_id[:4], file_id, repeat_id)), 'w') as fout:
                json.dump(out_info, fout)
            env.env_remove_articulation([env.objects[0], env.objects[1], robot1.robot, robot2.robot])
            return 'fail2', info

    suction_arm2 = utils.add_suction(robot2.robot.get_links()[-1], env.get_link(part2_id), env=env, use_lock_motion=use_lock_motion)

    
    if args.setting == 'BiAssembly':
        mid_object_pose2 = env.objects[1].get_root_pose().to_transformation_matrix()
        assembly_gripper_rotmat2 = assembly_object_rotmat @ np.linalg.inv(mid_object_pose2) @ robot2.robot.get_root_pose().to_transformation_matrix()
        alignment_gripper_rotmat2 = copy.deepcopy(assembly_gripper_rotmat2)
        alignment_gripper_rotmat2[:3, 3] -= transformed_disassembly_dir.reshape(-1).detach().cpu().numpy() * args.assembly_dist

        
    imgs=utils.single_gripper_move_to_target_pose(robot2, intermediate_gripper_rotmat2, cam=cam, vis_gif=True, robot_id=robot2_actor_ids, part_id=part2_id)
    gif_imgs.extend(imgs)
    utils.single_gripper_wait_n_steps(robot2, cam=cam, vis_gif=False)
    out_info['objA_pickup_rotmat'] = env.objects[0].get_root_pose().to_transformation_matrix().tolist()
    out_info['objB_pickup_rotmat'] = env.objects[1].get_root_pose().to_transformation_matrix().tolist()

    
    imgs=utils.single_gripper_move_to_target_pose(robot1, alignment_gripper_rotmat1, cam=cam, vis_gif=True,robot_id=robot1_actor_ids, part_id=part1_id)
    gif_imgs.extend(imgs)
    utils.single_gripper_wait_n_steps(robot1, cam=cam, vis_gif=False)
    imgs=utils.single_gripper_move_to_target_pose(robot2, alignment_gripper_rotmat2, cam=cam, vis_gif=True,robot_id=robot2_actor_ids, part_id=part2_id)
    gif_imgs.extend(imgs)  
    utils.single_gripper_wait_n_steps(robot2, cam=cam, vis_gif=False)    
    out_info['alignment_rotmat_world1'] = alignment_gripper_rotmat1.tolist()
    out_info['alignment_rotmat_world2'] = alignment_gripper_rotmat2.tolist()
    out_info['objA_alignment_rotmat'] = env.objects[0].get_root_pose().to_transformation_matrix().tolist()
    out_info['objB_alignment_rotmat'] = env.objects[1].get_root_pose().to_transformation_matrix().tolist()
    
    
    # assembly the two parts
    imgs = utils.dual_gripper_move_to_target_pose(robot1, robot2, assembly_gripper_rotmat1, assembly_gripper_rotmat2, cam=cam, vis_gif=True, check_tactile=True, robot_id1=robot1_actor_ids,robot_id2=robot2_actor_ids,part_id1=part1_id,part_id2=part2_id)
    gif_imgs.extend(imgs)
    utils.dual_gripper_wait_n_steps(robot1, robot2, cam=cam, vis_gif=False)
    out_info['assembly_rotmat_world1'] = assembly_gripper_rotmat1.tolist()
    out_info['assembly_rotmat_world2'] = assembly_gripper_rotmat2.tolist()
    out_info['objA_assembly_rotmat'] = env.objects[0].get_root_pose().to_transformation_matrix().tolist()
    out_info['objB_assembly_rotmat'] = env.objects[1].get_root_pose().to_transformation_matrix().tolist()

    
    # check success
    bad_contacts = env.check_contacts_exist(robot1_actor_ids, [part2_id]) or env.check_contacts_exist(robot2_actor_ids, [part1_id])
    
    assembly_pose1 = env.objects[0].get_root_pose()
    assembly_pose2 = env.objects[1].get_root_pose()
    delta_p, delta_q = utils.get_pose_delta(assembly_pose1, assembly_pose2)
    print('delta_p: ', delta_p, '\tdelta_q: ', delta_q)
    out_info['delta_p'] = delta_p.tolist()
    out_info['delta_q'] = delta_q.tolist()
    info[1] = delta_p
    info[2] = delta_q

    if delta_p > args.delta_p_threshold or delta_q > args.delta_q_threshold or bad_contacts:   
        if args.save_data:
            imageio.mimsave(os.path.join(args.out_dir, 'fail_gif', 'fail_%s_%s_%d_%d_%f_%f.gif' % (category, shape_id[:4], file_id, repeat_id, delta_p, delta_q)), gif_imgs)
            with open(os.path.join(args.out_dir, 'fail_files', 'fail_%s_%s_%d_%d.json' % (category, shape_id[:4], file_id, repeat_id)), 'w') as fout:
                json.dump(out_info, fout)
        env.env_remove_articulation([env.objects[0], env.objects[1], robot1.robot, robot2.robot])
        return 'fail', info

    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    imageio.mimsave(os.path.join(args.out_dir, 'succ_gif', 'succ_%s_%s_%d_%d_%f_%f_%s.gif' % (category, shape_id[:4], file_id, repeat_id, delta_p, delta_q, formatted_time)), gif_imgs)
    with open(os.path.join(args.out_dir, 'succ_files', 'succ_%s_%s_%d_%d_%s.json' % (category, shape_id[:4], file_id, repeat_id, formatted_time)), 'w') as fout:
        json.dump(out_info, fout)
    env.env_remove_articulation([env.objects[0], env.objects[1], robot1.robot, robot2.robot])
    return 'succ', info
        
        
        

