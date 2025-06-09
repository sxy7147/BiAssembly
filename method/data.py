import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
import random
import method_utils as utils
import copy
import ipdb
from scipy.spatial.transform import Rotation
from tqdm import tqdm



class SAPIENVisionDataset(data.Dataset):

    def __init__(self, data_features, train_test_type, buffer_max_num=1e5, 
                 succ_proportion=[0.5], fail1_proportion=[0.4], fail2_proportion=[0], fail3_proportion=[0.0], fail4_proportion=[0.0],
                 assigned_category=None, obj_asset_dir=None, num_points_imaginary_pc=3000,
                 ):
        self.train_test_type = train_test_type
        self.obj_asset_dir = obj_asset_dir
        self.data_features = data_features
        self.num_points_imaginary_pc = num_points_imaginary_pc
        self.buffer_max_num = buffer_max_num

        self.dir_type_to_key = {
            'succ_files': ('pos', succ_proportion),
            'first_step_fail_files': ('neg1', fail1_proportion),
            'second_step_fail_files': ('neg2', fail2_proportion),
            'third_step_fail_files': ('neg3', fail3_proportion),
            'fail_files': ('neg4', fail4_proportion),
        } 
        
        self.num_dict = {'pos': 0, 'neg1': 0, 'neg2': 0, 'neg3': 0, 'neg4': 0}
        self.dataset = []
        
        self.flag_assigned_category = False
        self.category_cnt_dict = dict()
        if assigned_category:
            self.flag_assigned_category = True
            self.assigned_category = assigned_category.split(',')
            for category in self.assigned_category:
                self.category_cnt_dict[category] = 0


    def load_data(self, data_list):

        data_split_file = '../data_generation/data_classify.json'
        with open(data_split_file) as f:
            data_split_dict = json.load(f)
            
        saved_disassembly_info_dict = {}
        
        for dir_id, cur_dir in tqdm(enumerate(data_list)):
            if not os.path.exists(cur_dir):
                continue
            
            dir_type = cur_dir.split('/')[-1]            
            
            json_files = [file for file in os.listdir(cur_dir) if file.endswith('.json')]
            json_files.sort(key=lambda file_name: int(file_name.split('.')[0].split('_')[3]))     # sorted by file id
            
            for file in json_files:
                if dir_type not in self.dir_type_to_key.keys():
                    continue
                result_type, proportion_list = self.dir_type_to_key[dir_type]
                if self.num_dict[result_type] >= self.buffer_max_num * proportion_list[dir_id // len(self.dir_type_to_key.keys())]:
                    continue
                
                file_idx = int(file.split('.')[0].split('_')[3])
                try:
                    with open(os.path.join(cur_dir, file), 'r') as fin:
                        result_data = json.load(fin)
                except Exception:
                    print('error: ', os.path.join(cur_dir, file))
                    continue
                
                category = result_data['category']
                shape_id = result_data['shape_id']
                if self.flag_assigned_category:
                    if category not in self.assigned_category:
                        print('Wrong category')
                        continue
                else:
                    if category not in self.category_cnt_dict.keys():
                        self.category_cnt_dict[category] = 0
                if data_split_dict[category][shape_id] != self.train_test_type:
                    # print('Wrong shape id')
                    continue
                
                objects_setting = result_data['load_object_setting']
                urdf_list = [object['urdf'] for object in objects_setting]
                if not os.path.exists(os.path.join(cur_dir, '../init_data', 'init_cam_XYZA_%s_%s_%d.h5' % (category, shape_id[:4], file_idx))):
                    print('not exist: ', os.path.join(cur_dir, '../init_data', 'init_cam_XYZA_%s_%s_%d.h5' % (category, shape_id[:4], file_idx)))
                    continue
                
                  
                success = True if result_type == 'pos' else False 
                self.num_dict[result_type] += 1
                self.category_cnt_dict[category] += 1
            
                
                contact_point1 = np.array(result_data['position_world1'], dtype=np.float32)
                contact_point2 = np.array(result_data['position_world2'], dtype=np.float32)
                grasp_rotmat1 = np.array(result_data['start_rotmat_world1'], dtype=np.float32)
                grasp_rotmat2 = np.array(result_data['start_rotmat_world2'], dtype=np.float32)
                if 'assembly_position1' in result_data.keys():
                    assembled_position = np.array(result_data['assembly_position1'], dtype=np.float32)
                    assembled_rotation = np.array(result_data['assembly_rotation1'], dtype=np.float32)
                    assembled_rotation = utils.quaternion_to_rotation_matrix(assembled_rotation)
                    disassembly_dir = np.array(result_data['move_dir'], dtype=np.float32)
                    
                    if shape_id not in saved_disassembly_info_dict.keys():
                        saved_disassembly_info_dict[shape_id] = [(assembled_position, assembled_rotation, disassembly_dir)]
                    else:
                        saved_disassembly_info_dict[shape_id].append((assembled_position, assembled_rotation, disassembly_dir))
                
                else:   # neg1, neg2
                    if shape_id in saved_disassembly_info_dict.keys():
                        assembled_position, assembled_rotation, disassembly_dir = random.choice(saved_disassembly_info_dict[shape_id])
                    else:
                        assembled_position = np.array([0, 0, 0], dtype=np.float32)
                        assembled_rotation = np.eye(3, dtype=np.float32)
                        disassembly_dir = np.random.randn(3)
                        disassembly_dir = disassembly_dir / np.linalg.norm(disassembly_dir)

                camera_metadata = result_data['camera_metadata']
                mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)  

                cur_data = (cur_dir, shape_id, category, file_idx, urdf_list,
                            contact_point1, grasp_rotmat1, contact_point2, grasp_rotmat2, 
                            disassembly_dir, assembled_position, assembled_rotation, 
                            success, mat44, 
                            )
                self.dataset.append(cur_data)


        for key, value in self.num_dict.items():
            print('num of %s: %d' % (key, value))
        print('\n')
        for category_name in self.category_cnt_dict.keys():
            if self.category_cnt_dict[category_name] != 0:
                print(category_name, self.category_cnt_dict[category_name])
        print('\n')


    def __str__(self):
        return self.train_test_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        cur_dir, shape_id, category, file_idx, urdf_list, \
        contact_point1, grasp_rotmat1, contact_point2, grasp_rotmat2, \
        disassembly_dir, assembled_position, assembled_rotation, \
        success, mat44, = self.dataset[index]
        
        # acquire imaginary point cloud in canonical pose
        canonical_pcs = utils.get_canonical_imaginary_pc(urdf_list, num_pts=self.num_points_imaginary_pc, object_scale=0.4)
                
        # apply random rotation on the pcs
        rotation = Rotation.random()
        rotation_matrix = rotation.as_matrix()
        imaginary_pcs = (rotation_matrix @ canonical_pcs.reshape(2 * self.num_points_imaginary_pc, 3).T).T
        

        data_feats = ()
        for feat in self.data_features:
            if feat == 'imaginary_pc':
                imaginary_pcs = imaginary_pcs.reshape(2, self.num_points_imaginary_pc, 3)
                imaginary_pcs = torch.from_numpy(imaginary_pcs).unsqueeze(0)
                data_feats = data_feats + (imaginary_pcs,)
                
            elif feat == 'init_pc':
                init_cam_XYZA_file = os.path.join(cur_dir, '../init_data', 'init_cam_XYZA_%s_%s_%d.h5' % (category, shape_id[:4], file_idx))
                with h5py.File(init_cam_XYZA_file, 'r') as fin:
                    cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                    cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                    cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                
                cam_dist = 5.5
                out = np.zeros((448, 448, 4), dtype=np.float32)
                out[cam_XYZA_id1, cam_XYZA_id2, :3] = cam_XYZA_pts
                out[cam_XYZA_id1, cam_XYZA_id2, 3] = 1
                mask = (out[:, :, 3] > 0.5)
                pc = out[mask, :3]
                pc[:, 0] -= cam_dist
                
                pc_world = (mat44[:3, :3] @ pc.T).T     # camera coordinate -> world coordinate
                save_idx = (pc_world[:, 2] >= 0.205) & (pc_world[:, 1] >= -0.70) & (pc_world[:, 1] <= 0.70)
                pc_world = pc_world[save_idx]   

                idx = np.arange(pc_world.shape[0])
                np.random.shuffle(idx)
                while len(idx) < 30000:
                    idx = np.concatenate([idx, idx])
                idx = idx[:30000]
                pc_world = pc_world[idx, :]
                pc_world = torch.from_numpy(pc_world).unsqueeze(0)
                
                data_feats = data_feats + (pc_world,)
            
                
            elif feat == 'target_pc':
                canonical_pcs = canonical_pcs.reshape(2 * self.num_points_imaginary_pc, 3)
                target_pcs = (assembled_rotation @ canonical_pcs.T).T + assembled_position
                target_pcs = target_pcs.reshape(2, self.num_points_imaginary_pc, 3)
                aggregated_target_pcs = np.zeros((2, self.num_points_imaginary_pc, 4))
                aggregated_target_pcs[:, :, :3] = target_pcs
                aggregated_target_pcs[1, :, 3] = 1
                aggregated_target_pcs = torch.from_numpy(aggregated_target_pcs.reshape(-1, 4)).unsqueeze(0)
                data_feats = data_feats + (aggregated_target_pcs,)
        

            elif feat == 'target_transformation':
                target_rot = assembled_rotation @ np.linalg.inv(rotation_matrix)
                target_rot = np.concatenate([target_rot[:3, 2], target_rot[:3, 0]])
                target_tran = assembled_position
                target_transformation = np.concatenate([target_rot, target_tran])
                data_feats = data_feats + (target_transformation,)
            
            elif feat == 'ctpt1':
                ctpt1 = contact_point1
                data_feats = data_feats + (ctpt1,)

            elif feat == 'ctpt2':
                ctpt2 = contact_point2
                data_feats = data_feats + (ctpt2,)

            elif feat == 'pickup_dir1':
                dir1 = np.concatenate([grasp_rotmat1[:3, 2], grasp_rotmat1[:3, 0]])
                data_feats = data_feats + (dir1,)

            elif feat == 'pickup_dir2':
                dir2 = np.concatenate([grasp_rotmat2[:3, 2], grasp_rotmat2[:3, 0]])
                data_feats = data_feats + (dir2,)
            
            elif feat == 'transformed_disassembly_dir':
                data_feats = data_feats + (disassembly_dir,)
            
            elif feat == 'imaginary_disassembly_dir':    
                imaginary_disassembly_dir = (rotation_matrix @ (np.linalg.inv(assembled_rotation) @ disassembly_dir.T)).T
                data_feats = data_feats + (imaginary_disassembly_dir,)

            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)
            
            elif feat == 'category':
                data_feats = data_feats + (category,)

            elif feat == 'success':
                data_feats = data_feats + (success,)

            elif feat == 'file_idx':
                data_feats = data_feats + (file_idx,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

