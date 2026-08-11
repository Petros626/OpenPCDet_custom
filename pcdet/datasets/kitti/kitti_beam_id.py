# source: https://github.com/WoodwindHu/DTS/blob/main/pcdet/datasets/kitti/kitti_beam_id.py
import numpy as np
import os
import argparse
from functools import partial

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default=None, help='kitti data path')

    args = parser.parse_args()

    return args

def get_beam_id(pc_index, path):
    filename = os.path.join(path, 'velodyne', "%06d.bin" % int(pc_index))
    sample_name = os.path.basename(filename)
    print(f'Processing {pc_index}, sample {sample_name}')

    pc_kitti = np.fromfile(os.path.join(path, 'velodyne', "%06d.bin"%(pc_index)), dtype=np.float32).reshape(-1,4)
    beam_id = np.zeros(pc_kitti.shape[0], dtype=np.float32)

    selected = True
    beam_count=0
    cloud_size = pc_kitti.shape[0]
    selectcount = 0
    index = [0] * 65

    for i in range(cloud_size):
        beam_id[i] = beam_count
        angle = np.arctan2(pc_kitti[i,1], -pc_kitti[i,0]) / np.pi * 180
        if i==0:
            last_point_angle = angle
            back_point_angle = angle
        if selected and selectcount<10: 
            selectcount += 1
        else: 
            selected = False
        if ((angle-last_point_angle)>=90) and (not selected) and ((angle-back_point_angle)>=90):
            beam_count += 1
            index[beam_count] = i
            selected = True
            selectcount = 0
        
        back_point_angle = last_point_angle
        last_point_angle = angle
        
    if beam_count<63:
        for i in range(beam_count,64):
            index[i] = index[beam_count]
        print("Scans less than 64")
    elif beam_count>=64:
        print("gg, sort failed")
        return 
    beam_id.tofile(os.path.join(path, "beam_labels", "%06d_beam_label.bin"%(pc_index)))
    
if __name__=='__main__':
    args = parse_config()
    os.makedirs(os.path.join(args.data_path, 'training/beam_labels'), exist_ok=True)

    sample_id_list = range(7481) # all KITTI
    num_workers = os.cpu_count() // 2 # depends on available CPU cores

    import concurrent.futures as futures
    import time

    get_beam_id = partial(get_beam_id, path=os.path.join(args.data_path, "training"))

    start_time = time.time()

    with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(get_beam_id, sample_id_list)

    end_time = time.time()
    print("Total time for loading infos: ", end_time - start_time, "s")
    print("Loading speed for infos: ", len(sample_id_list) / (end_time - start_time), "sample/s")
    