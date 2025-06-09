import os
import sys
import multiprocessing as mp
import subprocess
from utils import get_dataset
import argparse
import numpy as np
np.random.seed(1945)

def call_cmd(cmd_arg):
    traid_id, category, shape_id, cut_id, out_dir,cuda_id= cmd_arg
    try:
        command= f"CUDA_VISIBLE_DEVICES={cuda_id} python collect_single.py --trial_id {traid_id} --category {category} --shape_id {shape_id} --cut_type {cut_id} --out_dir {out_dir}"
        process = subprocess.Popen(
            command,
            shell=True,
            start_new_session=True  
        )
        
        timeout_seconds = 300  
        process.wait(timeout=timeout_seconds)
        return process.returncode, [traid_id, category, shape_id, cut_id]
    except subprocess.TimeoutExpired:
        print(f"🕒 Trial {traid_id} outtime")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()  
        return -1, [traid_id, category, shape_id, cut_id]
    except Exception as e:
        print(f"Error in Trial {traid_id}: {e}")
        return -1, [traid_id, category, shape_id, cut_id]

def process_result(exit_code_lst):
    exit_code, traid_id, category = exit_code_lst[0], exit_code_lst[1][0], exit_code_lst[1][1]
    global collect_dict, total_dict
    map_dict = {0: "success", 1: "invalid", 2: "grasp1_fail", 3: "grasp2_fail", 4: "assem_fail"}
    if exit_code not in map_dict:
        print(f"Unknown exit code: {exit_code}")
        value="invalid"
    else:
        value = map_dict[exit_code]
    collect_dict[category][value] += 1
    total_dict[value] += 1
    if traid_id % 10 == 0:
        print(f"Trial {traid_id}: {category}: {collect_dict[category]}, total: {total_dict}")
        # print(f"Trial {traid_id}: {category}: {collect_dict[category]}, total: {total_dict}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_name", type=str, default="Train", help="split name", choices=["Test1", "Test2", "Train"])
    parser.add_argument("--exp_name", type=str, default="ICML_v2", help="experiment name")
    parser.add_argument("--num_process", type=int, default=5, help="number of processes per GPU")
    parser.add_argument("--gpu_list", type=list, default=[4,5,6], help="list of GPUs to use")
    parser.add_argument("--require_num", type=int, default=100, help="number of required samples per category")
    parser.add_argument("--require_type", type=str, default="per", help="all or per")
    parser.add_argument("--per_epoch_num", type=int, default=100, help="number of samples per epoch")

    args = parser.parse_args()

    dataset = get_dataset(split_name=args.split_name)
    output_dir = f"../output/data_generation/{args.exp_name}/{args.split_name}"

    collect_dict={}
    for category in dataset:
        collect_dict[category] = {"invalid": 0, "grasp1_fail": 0, "grasp2_fail": 0, "assem_fail": 0, "success": 0}
    total_dict={"invalid": 0, "grasp1_fail": 0, "grasp2_fail": 0, "assem_fail": 0, "success": 0}

    Trail_id=0
    category_list = list(dataset.keys())
    GPU_list = args.gpu_list
    GPU_num = len(GPU_list)
    while(True):
        cmd_arg_lst=[]
        if args.require_type == "all":
            if total_dict["success"] >= args.require_num:
                break
            for i in range(args.per_epoch_num):
                category = np.random.choice(category_list)
                shape_id = np.random.choice(list(dataset[category].keys()))
                cut_id = np.random.choice(dataset[category][shape_id])
                cmd_arg_lst.append([Trail_id, category, shape_id, cut_id, output_dir, GPU_list[Trail_id%GPU_num]])
                Trail_id += 1
        elif args.require_type == "per":
            probability = [max(0,args.require_num-collect_dict[category]["success"]) for category in category_list]
            if sum(probability) == 0:
                break
            probability = np.array(probability)/sum(probability)
            for i in range(args.per_epoch_num):
                category = np.random.choice(category_list, p=probability)
                shape_id = np.random.choice(list(dataset[category].keys()))
                cut_id = np.random.choice(dataset[category][shape_id])
                cmd_arg_lst.append([Trail_id, category, shape_id, cut_id, output_dir, GPU_list[Trail_id%GPU_num]])
                Trail_id += 1
        else:
            raise ValueError("require_type must be all or per")
        print("Beginning to collect data...")
        Pool_dict={gpu_id: mp.Pool(processes=args.num_process) for gpu_id in GPU_list}
        for i in range(len(cmd_arg_lst)):
            Pool_dict[cmd_arg_lst[i][-1]].apply_async(call_cmd, args=(cmd_arg_lst[i],), callback=process_result)
        for key in Pool_dict:
            Pool_dict[key].close()
            Pool_dict[key].join()
        print("Data collection finished.")

    