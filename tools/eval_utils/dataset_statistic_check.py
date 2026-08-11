import pickle
import numpy as np

def load_pkl_local(pkl_path):
    with open(pkl_path, 'rb') as f:
        infos = pickle.load(f)
    return infos

def split_bbox(list_bbox):
    # should mainly calculate the z statistics
    # for kitti: x-y-z-l-w-h
    bbox_z = []
    bbox_l = []
    bbox_w = []
    bbox_h = []
    for bbox in list_bbox:
        bbox_z.append(bbox[2])
        bbox_l.append(bbox[3])
        bbox_w.append(bbox[4])
        bbox_h.append(bbox[5])
    bbox_z_np = np.array(bbox_z)
    bbox_l_np = np.array(bbox_l)
    bbox_w_np = np.array(bbox_w)
    bbox_h_np = np.array(bbox_h)
    return bbox_z_np, bbox_l_np, bbox_h_np, bbox_w_np

def get_statistic(arr, get_abnorm=False):
    mean_arr = round(float(np.mean(arr)), 2)
    median_arr = round(float(np.median(arr)), 2)
    std_arr = round(float(np.std(arr)), 2)
    min_arr = round(float(np.min(arr)), 2)
    max_arr = round(float(np.max(arr)), 2)
    statis = {"mean":mean_arr, "std": std_arr, "min": min_arr, "max": max_arr, "median": median_arr}
    if get_abnorm:
        abnorm_min = mean_arr - 3 * std_arr
        abnorm_max = mean_arr + 3 * std_arr
        abnorm_max_index = np.where(arr > abnorm_max)
        abnorm_min_index = np.where(arr < abnorm_min)
        abnorm_idx_list = list(abnorm_max_index) + list(abnorm_min_index)
        statis["abnorm_obj_idx"] = abnorm_idx_list
        return statis
    return statis


def process_object_info(class_info, cls_name=None, get_abnorm_idx=False):
    info_order = ["z", "l", "h", "w"]
    statis = {}
    for idx, element in enumerate(split_bbox(class_info)):
        statis[info_order[idx]] = get_statistic(element, get_abnorm=get_abnorm_idx)
    return statis

def process_truncation_info(truncation_list):
    trunc_arr = np.array(truncation_list)

    return get_statistic(trunc_arr, get_abnorm=False)

def kitti_process(pkl_path, get_abnorm_idx=True, analyze_truncation=True):
    kitti_infos = load_pkl_local(pkl_path)
    kitti_car_info = []
    kitti_ped_info = []
    kitti_cyc_info = []

    kitti_car_trunc = []
    kitti_ped_trunc = []
    kitti_cyc_trunc = []
    all_truncation = []

    if isinstance(kitti_infos[0], list):
        for frame_list in kitti_infos:
            for sample in frame_list:
                gt_names = sample.get('gt_names', [])
                gt_boxes = sample.get("gt_boxes", [])
                
                num_boxes = gt_boxes.shape[0] if gt_boxes.ndim > 1 else (1 if gt_boxes.ndim == 1 and len(gt_boxes) > 0 else 0)
                num_names = len(gt_names)
                num_objects = min(num_names, num_boxes)

                for i in range(num_objects):
                    name = gt_names[i]
                    if name == 'Pedestrian':
                        kitti_ped_info.append(gt_boxes[i])
                    elif name == 'Car':
                        kitti_car_info.append(gt_boxes[i])
                    elif name == 'Cyclist':
                        kitti_cyc_info.append(gt_boxes[i])
    else:
        for info in kitti_infos:
            anno_info = info["annos"]
            obj_number = anno_info["name"].shape[0]
            for i in range(obj_number):
                trunc_val = anno_info["truncated"][i]
                all_truncation.append(trunc_val)

                if anno_info["name"][i] == "Pedestrian":
                    kitti_ped_info.append(anno_info["gt_boxes_lidar"][i])
                    kitti_ped_trunc.append(trunc_val)
                elif anno_info["name"][i] == "Car":
                    kitti_car_info.append(anno_info["gt_boxes_lidar"][i])
                    kitti_car_trunc.append(trunc_val)
                elif anno_info["name"][i] == "Cyclist":
                    kitti_cyc_info.append(anno_info["gt_boxes_lidar"][i])
                    kitti_cyc_trunc.append(trunc_val)

    print(f"Car Counts: {len(kitti_car_info)}, Pedestrian: {len(kitti_ped_info)}, Cyclist: {len(kitti_cyc_info)}\n")
    print("Statistics for Cars:")
    print(process_object_info(kitti_car_info, cls_name="car", get_abnorm_idx=get_abnorm_idx))
    print()
    print("Statistics for Pedestrians:")
    print(process_object_info(kitti_ped_info, cls_name="pedestrian", get_abnorm_idx=get_abnorm_idx))
    print()
    print("Statistics for Cyclists:")
    print(process_object_info(kitti_cyc_info, cls_name="cyclist", get_abnorm_idx=get_abnorm_idx))

    if analyze_truncation:
        print("\n")
        print("="*80)
        print("TRUNCATION ANALYSIS")
        print("="*80)
        print("\nOverall Truncation Statistics (all classes):")
        print(process_truncation_info(all_truncation))
        print()
        print("Car Truncation:")
        print(process_truncation_info(kitti_car_trunc))
        print()
        print("Pedestrian Truncation:")
        print(process_truncation_info(kitti_ped_trunc))
        print()
        print("Cyclist Truncation:")
        print(process_truncation_info(kitti_cyc_trunc))


if __name__ == "__main__":
    #kitti_process("/home/rlab10/OpenPCDet/data/kitti/kitti_val_dataset.pkl", get_abnorm_idx=False, analyze_truncation=False)
    # 7481 frames, has the most meaningful values for Statistical Normalization (SN)
    # Source: Train in Germany, Test in The USA: Making 3D Object Detectors Generalize.
    #kitti_process("/home/rlab10/OpenPCDet/data/kitti/kitti_train_dataset.pkl", get_abnorm_idx=False, analyze_truncation=False)

    # ZOD
    #kitti_process("/home/rlab10/OpenPCDet/data/zod/zod_val_dataset.pkl", get_abnorm_idx=False, analyze_truncation=False)
    kitti_process("/home/rlab10/OpenPCDet/data/zod/zod_train_dataset.pkl", get_abnorm_idx=False, analyze_truncation=False)