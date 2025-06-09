import os
import sys

import torch
import numpy as np
import method_utils 
import inference_utils
from argparse import ArgumentParser
import time
import random
import multiprocessing as mp
from subprocess import call
import math
import datetime
import time
from inference_data_all_models import simulation


parser = ArgumentParser()
parser.add_argument('--device', type=str, nargs='+', default='cuda:0')
parser.add_argument('--out_dir', type=str)
parser.add_argument('--test_data_dir', type=str, nargs='+', help='data directory')
parser.add_argument('--test_data_num', type=int)
parser.add_argument('--repeat_num', type=int, default=3)
parser.add_argument('--assigned_category', type=str, default='BeerBottle')
parser.add_argument('--object_asset_dir', type=str, default='../assets/object/everyday2pieces_selected')

parser.add_argument('--delta_p_threshold', type=float, default=0.2)
parser.add_argument('--delta_q_threshold', type=float, default=30)
parser.add_argument('--assembly_dist', type=float, default=0.50)

parser.add_argument('--setting', type=str, default='BiAssembly')
parser.add_argument('--disassembly_predictor_path', type=str, default='')
parser.add_argument('--disassembly_predictor_epoch', type=str, default='100')   
parser.add_argument('--pickup_predictor_path', type=str, default='') 
parser.add_argument('--pickup_predictor_epoch', type=str, default='100')

parser.add_argument('--feat_dim', type=int, default=128)
parser.add_argument('--cp_feat_dim', type=int, default=32)
parser.add_argument('--dir_feat_dim', type=int, default=32)
parser.add_argument('--z_dim', type=int, default=32)
parser.add_argument('--dir_dim', type=int, default=6)
parser.add_argument('--num_point_per_shape', type=int, default=1024)
parser.add_argument('--aff_topk', type=float, default=0.1)
parser.add_argument('--critic_topk', type=float, default=0.1)

parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--save_data', action='store_true', default=False)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--draw_aff_map', action='store_true', default=False)
parser.add_argument('--num_draw', type=int, default=1)
args = parser.parse_args()



def run_jobs(idx_process, device, args, transition_Q, cur_file_list):
    
    print('origin device: ', device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('idx_process %d, device: ' % idx_process, device, 'num_files: %d' % len(cur_file_list))
    # random.shuffle(cur_file_list)
    
    models = {}
    
    if args.setting == 'BiAssembly':
        disassembly_predictor_def = method_utils.get_model_module('model_disassembly')
        disassembly_predictor = disassembly_predictor_def.Network(args.dir_feat_dim, z_dim=args.z_dim)
        disassembly_predictor.load_state_dict(torch.load(os.path.join(args.disassembly_predictor_path, 'ckpts', '%s-network.pth' % args.disassembly_predictor_epoch)))
        disassembly_predictor.to(device).eval()
        models['disassembly_predictor'] = disassembly_predictor
        
        pickup_predictor_def = method_utils.get_model_module('model_bi_affordance')
        pickup_predictor = pickup_predictor_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, z_dim=args.z_dim, pts_channel=3, target_pts_channel=4)
        pickup_predictor.load_state_dict(torch.load(os.path.join(args.pickup_predictor_path, 'ckpts', '%s-network.pth' % args.pickup_predictor_epoch)))
        pickup_predictor.to(device).eval()
        models['bi_affordance_predictor'] = pickup_predictor
    
    for file in cur_file_list:
        # print('file: ', file)
        file_id = int(file.split('/')[-1].split('.')[0].split('_')[3])
        category = file.split('/')[-1].split('.')[0].split('_')[1]
        if category not in args.assigned_category:
            continue

        for repeat_id in range(args.repeat_num):
            torch.cuda.empty_cache()
            result, info = simulation(file, file_id, repeat_id, models, device, args)
            transition_Q.put([result, info, file, repeat_id])
            
        

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    out_dir = args.out_dir
    print('out_dir: ', out_dir)
    
    args.assigned_category = args.assigned_category.split(',')
    print('categories: ', args.assigned_category)
    category_num_dict = {}
    for cate in args.assigned_category:
        category_num_dict[cate] = 0
    
    # get json files
    # json_files = [file for file in os.listdir(args.test_data_dir) if file.endswith('.json')]
    json_files = []
    for data_dir in args.test_data_dir:
        for file in os.listdir(data_dir):
            if file.endswith('.json'):
                json_files.append(os.path.join(data_dir, file))
    json_files.sort(key=lambda file_name: int(file_name.split('/')[-1].split('.')[0].split('_')[3]))     # sorted by file id
    print('total_num: ', len(json_files))
    
    filtered_json_files = []
    for json_file in json_files:
        # print('json_file: ', json_file)
        for cate in args.assigned_category:
            if ('_%s_' % cate) in json_file:
                if category_num_dict[cate] < args.test_data_num:
                    category_num_dict[cate] += 1
                    filtered_json_files.append(json_file)
                break 
    json_files = filtered_json_files
    print('json_files: ', json_files, len(json_files))
    NUM = len(json_files) * args.repeat_num
    
    # create folders
    subdirs = [
        'succ_files', 'first_step_fail_files', 'second_step_fail_files', 'fail_files',
        'succ_gif', 'first_step_fail_gif', 'second_step_fail_gif', 'fail_gif',
        'affordance'
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)
    
    flog = open(os.path.join(out_dir, 'results_log.txt'), 'a')
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    flog.write(f"Current time: {current_time}\n")

    
    mp.set_start_method('spawn', force=True)
    process_list = []
    trans_q = mp.Queue()
    num_file_per_process = (len(json_files) // args.num_processes) + 1
    for idx_process in range(args.num_processes):
        cur_file_list = json_files[idx_process * num_file_per_process: min((idx_process+1) * num_file_per_process, len(json_files))]
        cur_device = device
        p = mp.Process(target=run_jobs, args=(idx_process, cur_device, args, trans_q, cur_file_list))
        p.start()
        process_list.append(p)



    cnt_dict = {}
    for category in args.assigned_category:
        cnt_dict[category] = {'succ': 0, 'fail1': 0, 'fail2': 0, 'fail': 0, 'invalid': 0, 'total': 0}
    cnt_dict['All'] = {'succ': 0, 'fail1': 0, 'fail2': 0, 'fail': 0, 'invalid': 0, 'total': 0}
    
    episode_start_time = time.time()
    total_start_time = datetime.datetime.now()
    while True:
        if not trans_q.empty():
            result, info, file_id, repeat_id = trans_q.get()
            category, delta_p, delta_q = info
            
            cnt_dict['All'][result] += 1
            cnt_dict['All']['total'] += 1
            cnt_dict[category][result] += 1
            cnt_dict[category]['total'] += 1
            inference_utils.print_info('All', cnt_dict['All'], episode_start_time, total_start_time, flog, print_time_flag=True)
            
            # Print success rate for each category
            if cnt_dict['All']['total'] % 100 == 0 or (cnt_dict['All']['total'] >= NUM - 50):
                for category in sorted(cnt_dict.keys()):
                    if cnt_dict[category]['total'] == 0:
                        continue
                    inference_utils.print_info(category, cnt_dict[category], episode_start_time, total_start_time, flog, print_time_flag=False)

            flog.write('\n')
            flog.flush()
            episode_start_time = time.time()
                
                
        if cnt_dict['All']['total'] == NUM:
            for p in process_list:
                p.join()
            flog.close()
            break

