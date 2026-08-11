import numpy as np
import os
import argparse
from functools import partial

def parse_config():
    parser = argparse.ArgumentParser(description='Calculate average/median calibration from KITTI training dataset')
    parser.add_argument('--data_path', type=str, default=None, help='kitti calibration data path')
    parser.add_argument('--method', type=str, default='mean', choices=['mean', 'median'],
                        help='Aggregation method: mean or median')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'testing', 'all'],
                        help='Which split to use: training (7481), testing (7518), or all (14999)')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2,
                        help='Number of parallel workers')

    args = parser.parse_args()

    return args

def get_calib(calib_info, path):
    calib_index, path = calib_info
    filename = os.path.join(path, 'calib', "%06d.txt" % int(calib_index))
    sample_name = os.path.basename(filename)
    print(f'Processing {calib_index}, sample {sample_name}')
    
    calib_data = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            key, value = line.split(':', 1)
            calib_data[key.strip()] = np.array([float(x) for x in value.strip().split()], dtype=np.float32)
    
    return np.concatenate([calib_data['P2'], calib_data['R0_rect'], calib_data['Tr_velo_to_cam']])
    
def save_calib_avg(all_calibs, output_path, method, num_samples):
    all_calibs = np.array(list(all_calibs))

    if method == 'mean':
        calib_avg = np.mean(all_calibs, axis=0)
    else:
        calib_avg = np.median(all_calibs, axis=0)
    
    P2_avg = calib_avg[0:12].reshape(3, 4)
    R0_avg = calib_avg[12:21].reshape(3, 3)
    Tr_avg = calib_avg[21:33].reshape(3, 4)

    Tr_velo_to_cam = np.vstack([Tr_avg, [0, 0, 0, 1]]).astype(np.float32)
    R0 = np.eye(4)
    R0[:3, :3] = R0_avg
    P2 = np.vstack([P2_avg, [0, 0, 0, 0]])

    R0_inv = np.linalg.inv(R0)
    Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam).astype(np.float32)
    P2_inv = np.linalg.pinv(P2)

    # with open(os.path.join(output_path, f'calib_{method}.txt'), 'w') as f:
    #     f.write(f"# KITTI {method.upper()} Calibration ({num_samples} samples)\n\n")

    #     for name, mat in [('Tr_velo_to_cam', Tr_velo_to_cam), ('Tr_velo_to_cam_inv', Tr_velo_to_cam_inv)]:
    #         f.write(f"{name} = np.array([\n")
    #         for row in mat:
    #             f.write("    [" + ", ".join([f"{v:.8e}" for v in row]) + "],\n")
    #         f.write("])\n\n")

    #     for name, mat in [('R0', R0), ('P2', P2), ('R0_inv', R0_inv), ('P2_inv', P2_inv)]:
    #         f.write(f"{name} = np.array([\n")
    #         for row in mat:
    #             f.write("    [" + ", ".join([f"{v:.8f}" for v in row]) + "],\n")
    #         f.write("])\n\n")

    def mat_to_line(name, mat):
        vals = mat.flatten() if mat.shape[0] > 1 else mat
        vals_str = " ".join([f"{v:.12e}" for v in vals])
        return f"{name}: {vals_str}\n"

    with open(os.path.join(output_path, f'calib_{method}.txt'), 'w') as f:
        f.write(mat_to_line("P2", P2[:3, :]))  # 3x4
        f.write(mat_to_line("R0_rect", R0[:3, :3]))  # 3x3
        f.write(mat_to_line("Tr_velo_to_cam", Tr_velo_to_cam[:3, :]))  # 3x4
        f.write(mat_to_line("P2_inv", P2_inv))  # 4x4
        f.write(mat_to_line("R0_inv", R0_inv))  # 4x4
        f.write(mat_to_line("Tr_velo_to_cam_inv", Tr_velo_to_cam_inv))  # 4x4
    
    print(f"\nSaved to: {output_path}")

if __name__=='__main__':
    args = parse_config()
    
    # Build sample list based on split
    sample_id_list = []
    if args.split == 'training' or args.split == 'all':
        training_path = os.path.join(args.data_path, "training")
        sample_id_list.extend([(i, training_path) for i in range(7481)])
    
    if args.split == 'testing' or args.split == 'all':
        testing_path = os.path.join(args.data_path, "testing")
        sample_id_list.extend([(i, testing_path) for i in range(7518)])
    
    print(f"Using {len(sample_id_list)} samples from {args.split} split")
    
    output_path = os.path.join(args.data_path, f'{args.split}/calib_average')
    if args.split == 'all':
        output_path = os.path.join(args.data_path, 'calib_average')
    os.makedirs(output_path, exist_ok=True)
    
    num_workers = args.num_workers

    import concurrent.futures as futures
    import time

    get_calib = partial(get_calib, path=os.path.join(args.data_path, "training"))

    start_time = time.time()

    with futures.ThreadPoolExecutor(num_workers) as executor:
        infos = executor.map(get_calib, sample_id_list)
    
    save_calib_avg(all_calibs=infos, output_path=output_path, method=args.method, num_samples=len(sample_id_list))

    end_time = time.time()
    print("Total time for loading infos: ", end_time - start_time, "s")
    print("Loading speed for infos: ", len(sample_id_list) / (end_time - start_time), "sample/s")
    