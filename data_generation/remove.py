import subprocess
import os
import re
from pathlib import Path

import psutil
import datetime

def get_process_runtime(pid):
    try:
        proc = psutil.Process(pid)
        create_time = proc.create_time()  
        start_time = datetime.datetime.fromtimestamp(create_time)
        now = datetime.datetime.now()
        runtime = now - start_time
        return runtime.total_seconds()  
    except psutil.NoSuchProcess:
        return f"No process with PID {pid}."
    except Exception as e:
        return str(e)

def get_process_command(pid):
    try:
        cmdline = Path(f'/proc/{pid}/cmdline').read_text().replace('\x00', ' ')
        return cmdline.strip()
    except IOError:
        return ""

def quick_split(line):
    line_parts = line.split(' ')
    new_line_parts = []
    for i in range(len(line_parts)):
        if line_parts[i] == '':
            continue
        new_line_parts.append(line_parts[i])
    return new_line_parts

def get_gpu_processes():
    processes = []
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi'],
            encoding='utf-8'
        )
        
        for line in smi_output.strip().split('\n'):
            if not line:
                continue
            if line[5] not in ['0','1','2','3','4','5','6','7','8']:
                continue
            new_line_parts = quick_split(line) 
            print(new_line_parts)
            gpu_idx=int(new_line_parts[1])
            pid = int(new_line_parts[4])
            memory=int(new_line_parts[-2][:-3])

            cmd = get_process_command(pid)
            cmd_part=quick_split(cmd)
            run_time = get_process_runtime(pid)
            trial_id = -1
            for i in range(len(cmd_part)):
                if cmd_part[i] == '--trial_id':
                    trial_id=int(cmd_part[i+1])
                    break
            if trial_id == -1:
                continue
            processes.append({
                'pid': pid,
                'gpu': gpu_idx,
                'cmd': cmd,
                'trial_id': trial_id,
                'memory': memory,
                'run_time': run_time
            })
    except Exception as e:
        raise e
    return processes

def should_kill(process, target_gpus, cmd_pattern):
    if process['gpu'] not in target_gpus:
        return False
    
    if not re.search(cmd_pattern, process['cmd']):
        return False
    
    if re.search(r'(tmux|screen)', process['cmd']):
        return False
    
    return True

def safe_killer(process):
    
    if process['run_time'] > 1800:
        try:
            print(f"STOPPING PROCESS{process['pid']} (GPU {process['gpu']})(内存 {process['memory']}MB)(trail_id {process['trial_id']})(运行时间 {process['run_time']}秒)...")
            os.system(f'kill -9 {process["pid"]}')
        except Exception as e:
            print(f"终止进程失败: {e}")

if __name__ == "__main__":
    TARGET_GPUS = [2, 3, 4, 5]          
    CMD_PATTERN = r'collect_single\.py' 

    while True:
        try:
            processes = get_gpu_processes()
            
            for p in processes:
                status = '👽' if not re.search(CMD_PATTERN, p['cmd']) else '🟢'
                
                if status == '🟢':
                    # print(f"GPU {p['gpu']} PID {p['pid']} - {status}\n  命令: {p['cmd'][:120]}...")
                    safe_killer(p)
        except Exception as e:
            print(f"error: {e}")
        import time
        time.sleep(1800)
            
    