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
import trimesh

from pointnet2_ops.pointnet2_utils import furthest_point_sample


def printout(flog, strout):
    print(strout)
    if flog is not None:
        flog.write(strout + '\n')


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
 
 
def get_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module('models.' + model_version)


def collate_feats(b):
    return list(zip(*b))


def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


def load_data(file_dir, cat_shape_dict):
    with open(file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            cat_shape_dict[cat].append(shape_id)
    return cat_shape_dict


def save_h5(fn, data):
    fout = h5py.File(fn, 'w')
    for d, n, t in data:
        fout.create_dataset(n, data=d, compression='gzip', compression_opts=4, dtype=t)
    fout.close()

def save_cam_XYZA_list(fn, cam_XYZA_list):
    save_h5(fn, [(cam_XYZA_list[0].astype(np.uint64), 'id1', 'uint64'),
                 (cam_XYZA_list[1].astype(np.uint64), 'id2', 'uint64'),
                 (cam_XYZA_list[2].astype(np.float32), 'pc', 'float32')])



def quaternion_to_rotation_matrix(q):
    """
    Convert a unit quaternion to a 3x3 rotation matrix.

    Parameters:
    q (numpy.ndarray): Unit quaternion in the form [w, x, y, z].

    Returns:
    numpy.ndarray: 3x3 rotation matrix.
    """
    w, x, y, z = q
    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = y * z
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotation_matrix = np.array([[1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]])
    return rotation_matrix


def posquat_to_SE3(position, quaternion):
    # rot = Rotation.from_quat(quaternion)
    # rotation_matrix = rot.as_matrix()
    # print('rotation_matrix1: ', rotation_matrix)
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)

    SE3_matrix = np.eye(4)
    SE3_matrix[:3, :3] = rotation_matrix
    SE3_matrix[:3, 3] = position
    return SE3_matrix



def quaternion_angle(q1, q2):
    dot_product = np.dot(q1, q2)
    angle = 2 * np.arccos(np.abs(dot_product))
    return angle


def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])


def conjugate_quaternion(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def relative_position(q1, q2):
    relative_q = multiply_quaternions(conjugate_quaternion(q1), q2)
    return relative_q


def random_quaternion():
    rand_quat = np.random.rand(4)
    rand_quat /= np.linalg.norm(rand_quat)
    return rand_quat


def sample_points_fps(pc, num_point_per_shape):  
    batch_size = pc.shape[0]
    pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, num_point_per_shape).long().reshape(-1)  # BN
    pcid2 = furthest_point_sample(pc, num_point_per_shape).long().reshape(-1)  # BN
    pc = pc[pcid1, pcid2, :].reshape(batch_size, num_point_per_shape, -1)
    return pc, pcid1, pcid2


def get_canonical_imaginary_pc(urdf_list, num_pts=3000, object_scale=0.4):
    canonical_pcs = []
    for urdf_file in urdf_list:
        obj_file = urdf_file[: -4] + 'obj'
        mesh = trimesh.load_mesh(obj_file)
        canonical_pcs.append(trimesh.sample.sample_surface(mesh, num_pts)[0]) 
    canonical_pcs = np.array(canonical_pcs)
    canonical_pcs = canonical_pcs * object_scale
    return canonical_pcs


def get_data_list(data_dir, flog, name):
    data_list = [
        data_file
        for data_dir in data_dir
        for data_file in [
            os.path.join(data_dir, 'succ_files'),
            os.path.join(data_dir, 'first_step_fail_files'),
            os.path.join(data_dir, 'second_step_fail_files'),
            os.path.join(data_dir, 'third_step_fail_files'),
            os.path.join(data_dir, 'fail_files'),
        ]
    ]
    printout(flog, 'len(%s_data_list): %d' % (name, len(data_list)))
    printout(flog, ' '.join(map(str, data_list)))
    return data_list


def get_urdf_path(obj_dir, category, shape_id, cut_id):
    obj_dir = os.path.join(obj_dir, category, shape_id, cut_id)
    object_urdf_fn_A = os.path.join(obj_dir,'piece_0.urdf')
    object_urdf_fn_B = os.path.join(obj_dir,'piece_1.urdf')
    return [object_urdf_fn_A, object_urdf_fn_B]