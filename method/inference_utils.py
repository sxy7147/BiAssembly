import os
import sys
import h5py
import torch
import numpy as np
import importlib
import random
from pyquaternion import Quaternion
import shutil
import math
from PIL import Image
# from pyquaternion import Quaternion
# from sapien.core import Pose, ArticulationJointType
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from subprocess import call
import json
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.spatial.transform import Rotation
# from plyfile import PlyData, PlyElement
# import pandas as pd
import open3d as o3d
import trimesh
from sapien.core import Pose
import copy
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from method_utils import get_canonical_imaginary_pc
import datetime
import time



def process_mask(object_mask, valid_mask, in_bounds, indices):
    mask = (object_mask > 0).astype(np.uint8) * 255
    mask = mask.astype(np.float32) > 127
    mask = mask[valid_mask]
    mask = mask[in_bounds]
    mask = mask[indices]
    return mask


def get_observed_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, mat44, object1_mask, object2_mask, args, file_id, device, pc_dist=5.5):
    out = np.zeros((448, 448, 4), dtype=np.float32)
    out[cam_XYZA_id1, cam_XYZA_id2, :3] = cam_XYZA_pts
    out[cam_XYZA_id1, cam_XYZA_id2, 3] = 1
    # out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
    
    # Extract valid 3D points            
    valid_mask = (out[:, :, 3] > 0.5)
    pc = out[valid_mask, :3]
    pc[:, 0] -= pc_dist
    pc_world = (mat44[:3, :3] @ pc.T).T
    
    # Filter out points outside the region of interest
    in_bounds = (pc_world[:, 2] >= 0.205) & (pc_world[:, 1] >= -0.70) & (pc_world[:, 1] <= 0.70) & (pc_world[:, 2] <= 1.20)
    pc_world = pc_world[in_bounds]   
    
    # randomly sample points
    idx = np.arange(pc_world.shape[0])
    np.random.shuffle(idx)
    while len(idx) < 30000:
        idx = np.concatenate([idx, idx])
    idx = idx[:30000]
    pc_world = pc_world[idx, :]
    # print('pc_world_x: ', np.max(pc_world[:, 0]), np.min(pc_world[:, 0]))
    # print('pc_world_y: ', np.max(pc_world[:, 1]), np.min(pc_world[:, 1]))
    # print('pc_world_z: ', np.max(pc_world[:, 2]), np.min(pc_world[:, 2]))
    
    # save the pointcloud for visualization
    visu_pc = False
    if visu_pc:    
        import open3d as o3d
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc_world)
        o3d.io.write_point_cloud(os.path.join(args.out_dir, "point_cloud_%d.ply" % file_id), point_cloud)
    
    pc_world = torch.from_numpy(pc_world).unsqueeze(0).float().to(device)
    
    object1_mask = process_mask(object1_mask, valid_mask, in_bounds, idx)
    object2_mask = process_mask(object2_mask, valid_mask, in_bounds, idx)
    object1_mask = torch.tensor(object1_mask, dtype=torch.float32).reshape(1, 30000, 1).to(device)
    object2_mask = torch.tensor(object2_mask, dtype=torch.float32).reshape(1, 30000, 1).to(device)
    
    return pc_world, object1_mask, object2_mask


def get_imaginary_pc(objectA_urdf, objectB_urdf, num_pts, device):    
    canonical_pcs = get_canonical_imaginary_pc([objectA_urdf, objectB_urdf], num_pts=num_pts)

    # Apply a random rotation on the canonical GT pc
    rotation = Rotation.random()
    rotation_matrix = rotation.as_matrix()
    canonical_pcs = canonical_pcs.reshape(2 * num_pts, 3)
    imaginary_pcs = (rotation_matrix @ canonical_pcs.T).T
    imaginary_pcs = imaginary_pcs.reshape(2, num_pts, 3)
    imaginary_pcs = torch.from_numpy(imaginary_pcs).unsqueeze(0).float().to(device)
    
    return imaginary_pcs, rotation_matrix



def process_target_pcs(target_pcs, num_point_per_shape, batch_size):
    aggregated_target_pcs = torch.zeros((1, 2, num_point_per_shape, 4)).to(target_pcs.device)
    aggregated_target_pcs[:, :, :, :3] = target_pcs
    aggregated_target_pcs[:, 1, :, 3] = 1
    aggregated_target_pcs = aggregated_target_pcs.contiguous().reshape(1, 2 * num_point_per_shape, 4)
    pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, num_point_per_shape).long().reshape(-1)  # BN
    target_pcid2 = furthest_point_sample(aggregated_target_pcs, num_point_per_shape).long().reshape(-1)  # BN
    aggregated_target_pcs = aggregated_target_pcs[pcid1, target_pcid2, :].reshape(batch_size, num_point_per_shape, 4).contiguous()
    return aggregated_target_pcs


def select_topk_indices_randomly(scores, topk_ratio, num_selected):
    bs = scores.shape[0]
    length = scores.shape[1]
    sorted_idx = torch.argsort(scores, dim=1, descending=True).view(bs, length)
    batch_idx = torch.tensor(range(bs)).view(bs, 1)
    random_topk_indices = torch.randint(0, int(length * topk_ratio), size=(bs, num_selected))
    selected_idx = sorted_idx[batch_idx, random_topk_indices]
    return batch_idx, selected_idx
    
    
def get_rotmat(up, forward):
    up = up / np.linalg.norm(up)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)

    rotmat = np.eye(4).astype(np.float32)   # rotmat: world coordinate
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    return rotmat
    

def get_transformation_and_disassembly_direction(transformation, imaginary_pc_pose, disassembly_dir):
    up = transformation[0, 0:3].detach().cpu().numpy()
    forward = transformation[0, 3:6].detach().cpu().numpy()
    transformation_rotmat = get_rotmat(up, forward)

    trans = transformation[:, 6:9]
    transformation_rotmat[:3, 3] = trans[0].detach().cpu().numpy()  # rotated canonical -> world coordinate system
    
    imaginary_pc_rotmat = np.eye(4)
    imaginary_pc_rotmat[:3, :3] = imaginary_pc_pose
    assembly_object_rotmat = transformation_rotmat @ imaginary_pc_rotmat 
    transformed_disassembly_dir = (transformation_rotmat[:3, :3] @ disassembly_dir[0].detach().cpu().numpy().T).T
    
    return transformation_rotmat, assembly_object_rotmat, transformed_disassembly_dir

    
def get_movement_rotmat(ctpt, dir, start_dist, final_dist):
    up = dir[0:3]
    forward = dir[3:6]
    rotmat = get_rotmat(up, forward)

    final_rotmat = np.array(rotmat, dtype=np.float32)
    final_rotmat[:3, 3] = ctpt - up * final_dist
    final_pose = Pose().from_transformation_matrix(final_rotmat)

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = ctpt - up * start_dist
    start_pose = Pose().from_transformation_matrix(start_rotmat)

    return start_pose, start_rotmat, final_pose, final_rotmat


def convert_to_rgb_i(val):
    EPSILON = sys.float_info.epsilon
    # print(val)
    #"colors" is a series of RGB colors delineating a series of
    # adjacent linear color gradients between each pair.
    # Determine where the given value falls proportionality within
    # the range from minval->maxval and scale that fractional value
    # by the total number in the"colors" pallette.
    minval = 0
    maxval = 1
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    # Determine the lower index of the pair of color indices this
    # value corresponds and its fractional distance between the lower
    # and the upper colors.
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    if f < EPSILON:
        return colors[i]
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

def convert_to_rgb(vals):
    visu_rgbs = []
    for ii in range(len(vals)):
        visu_rgbs.append(convert_to_rgb_i(vals[ii]))
    visu_rgbs = np.array(visu_rgbs) / 255
    return visu_rgbs


def render_pc(pc, pc_score_rgb, fn):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(pc_score_rgb)
    # res = pcd.remove_statistical_outlier(20, 0.5)
    # pcd = res[0]
    o3d.io.write_point_cloud(fn, pcd)
    # render_png(pcd, path, name)
    
    
def normalize_nonzero_scores(aff_scores, ):
    # print('aff: ', np.max(aff_scores), np.min(aff_scores))
    non_zero_elements = copy.deepcopy(aff_scores[aff_scores != 0])
    normalized_elements = (non_zero_elements - non_zero_elements.min()) / (non_zero_elements.max() - non_zero_elements.min())
    aff_scores[aff_scores != 0] = normalized_elements
    return aff_scores
    
    
def draw_aff_map(aff1_scores, aff2_scores, object1_mask, object2_mask, init_pcs, fn):
    try:
        aff1_scores = normalize_nonzero_scores(aff1_scores)
        aff2_scores = normalize_nonzero_scores(aff2_scores)
    except Exception as e:
        print("Error in normalizing scores: ", e)
    
    visu_seperate = True 
    if visu_seperate:
        aff1_rgbs = convert_to_rgb(aff1_scores) 
        aff1_fn = fn.replace('map', 'map1')
        render_pc(init_pcs, aff1_rgbs, aff1_fn)
        
        aff2_rgbs = convert_to_rgb(aff2_scores)
        aff2_fn = fn.replace('map', 'map2')
        render_pc(init_pcs, aff2_rgbs, aff2_fn)
        
    aff_scores = aff1_scores * object1_mask + aff2_scores * object2_mask
    try:
        aff_rgbs = convert_to_rgb(aff_scores)
        render_pc(init_pcs, aff_rgbs, fn)
    except Exception as e:
        print("Error in converting scores to RGB: ", e)
    


def print_stats(category, stats_dict):
    A = stats_dict['succ']
    B = stats_dict['fail1']
    C = stats_dict['fail2']
    D = stats_dict['fail']
    E = stats_dict['invalid']
    F = stats_dict['total']

    line = (
        f"Category: {category} | Episode: {F} | "
        f"Succ: {A/F:.4f} | fail1: {B/F:.4f} | fail2: {C/F:.4f} | fail: {D/F:.4f} | Invalid: {E/F:.4f} |"
    )
    return line

def print_time(episode_start_time, total_start_time):
    line = f"Running Time: {time.time() - episode_start_time:.4f} | Total Time: {datetime.datetime.now() - total_start_time}"
    return line

def print_info(category, stats_dict, episode_start_time, total_start_time, flog, print_time_flag=True):
    print_line = print_stats(category, stats_dict)
    if print_time_flag:
        print_line += print_time(episode_start_time, total_start_time)
    print(print_line)
    flog.write(str(print_line) + '\n')

