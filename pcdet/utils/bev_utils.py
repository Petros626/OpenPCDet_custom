from functools import lru_cache
from pathlib import Path
from numpy import float32, ndarray, array
from typing import Union, List, Tuple, DefaultDict
from numpy.typing import NDArray
from collections import defaultdict
import numpy as np
import cv2
import os
from math import log

AUGMENT_SHORT = [
    "org",  # 0: Original
    "gs",    # 1: gt_sampling
    "wf",    # 2: random_world_flip
    "wr",    # 3: random_world_rotation
    "lr",    # 4: random_local_rotation
    "ls",    # 5: random_local_scaling
    "wt",    # 6: random_world_translation
]


# ==============================================================================
# LOAD KITTI DATA AS SINGLE FILE, SEVERAL FILES AND .PKL FILE
# ==============================================================================
@lru_cache(maxsize=None)
def load_kitti_pointcloud(bin_path: str = None, pkl_path: str = None, sample_idx: int = 0, augment_idx: int = 0) -> Union[ndarray, List[ndarray]]:
    from os import listdir
    from numpy import float32, fromfile, load

    if bin_path:
        bin_path = Path(bin_path)

        if bin_path.is_dir(): # several .bin files
            point_clouds = []
            bin_files = sorted([f for f in listdir(bin_path) if f.endswith('.bin')])

            for filename in bin_files:
                file_path = bin_path / filename
                if file_path.is_file():
                    try:
                        # N x 4 array (x, y, z, intensity)
                        point_cloud = fromfile(file_path, dtype=float32).reshape(-1, 4)
                        point_clouds.append(point_cloud)
                    except Exception as e:
                        print(f'Error during loading {filename}: {e}')
            return point_clouds
        
        elif bin_path.is_file(): # single .bin file
            return fromfile(bin_path, dtype=float32).reshape(-1, 4)
        else:
            raise FileNotFoundError(f'.bin file or directory not found: {bin_path}')
    
    elif pkl_path:
        pkl_path = Path(pkl_path)

        if not pkl_path.is_file():
            raise FileNotFoundError(f".pkl file not found: {pkl_path}")
        
        # Loading training entries (slow)
        #with open(pkl_path, 'rb') as f:
        #    data = load(f)

        # Loading training entries (faster)
        data = load(pkl_path, allow_pickle=True)
        num_samples = len(data)
            
        try:
            #if isinstance(data, list): # old
            # Access the sample at sample_idx
            sample = data[sample_idx]

            if 'frame_id' in str(sample): # train .pkl
                # Train data for BEV
                selected_data = sample[augment_idx]
                point_cloud = selected_data['points']
                curr_frame_id = selected_data['frame_id']
                # Labels for BEV train
                gt_boxes = selected_data['gt_boxes']
    
                return point_cloud, gt_boxes, num_samples, curr_frame_id
            
            elif 'point_cloud' in sample: # val .pkl
                # Val data for BEV
                sample = data[sample_idx]
                point_cloud = sample['points']
                lidar_idx = sample['point_cloud'].get('lidar_idx', 'unknown')
                # Labels for BEV val
                gt_boxes_lidar = sample['annos']['gt_boxes_lidar']
                truncated = sample['annos']['truncated']
                occluded = sample['annos']['occluded']
                box_2d = sample['annos']['bbox']

                # Pack additional information for validation
                val_info = {
                    'bbox': box_2d,
                    'truncated': truncated,
                    'occluded': occluded
                }

                return point_cloud, gt_boxes_lidar, val_info, num_samples, lidar_idx
            
        except IndexError:
            raise ValueError(f"Invalid sample_idx {sample_idx} or augment_idx {augment_idx} for .pkl dataset.")
    else:
        raise ValueError("Either bin_path or pkl_path must be provided.")

# ==============================================================================
# LOAD LAST PROCESSED SAMPLE
# ==============================================================================
@lru_cache(maxsize=None)
def load_progress(progress_file: str) -> Tuple[int, int, DefaultDict[str, int]]:

    if Path(progress_file).exists():
        from json import load

        with open(progress_file, "r") as f:
            progress = load(f)
            
        frame_id_dict = defaultdict(int)
        frame_id_dict.update(progress.get("processed_frames", {}))
        
        return (
            progress.get("sample_idx", 0),
            progress.get("augment_idx", 0),
            frame_id_dict
        )
    return 0, 0, defaultdict(int)   

# ==============================================================================
# SAVE RECENT PROCESSED SAMPLE
# ==============================================================================
def save_progress(sample_idx: int, augment_idx: int, frame_id_dict: DefaultDict[str, int], progress_file: str) -> None:
    from json import dump

    progress = {
        "sample_idx": sample_idx,
        "augment_idx": augment_idx,
        "processed_frames": dict(frame_id_dict)
    }
    with open(progress_file, "w") as f:
        dump(progress, f, indent=2)

# ==============================================================================
# CREATES UNIQUE FRAME ID
# ==============================================================================
def get_unique_frame_id(frame_id: str, augment_idx: int, frame_id_dict: DefaultDict[str, int]) -> Tuple[str, DefaultDict[str, int]]:
    needs_suffix = frame_id_dict[frame_id] > 0
    base_id = f"{frame_id}_r" if needs_suffix else frame_id
    
    if augment_idx == 4:
        frame_id_dict[frame_id] += 1
        
    return base_id, frame_id_dict

# ==============================================================================
# MAP HEIGHT (M) TO 255 (PX)
# ==============================================================================
@lru_cache(maxsize=None)
def map_height2channel(val_m: float, OFFSET_LIDAR: float, Z_MIN_HEIGHT: float, Z_MAX_HEIGHT: float, OUT_MIN: int, OUT_MAX: int) -> int:
    # Shift z-coordinate by LiDAR offset
    val_m += OFFSET_LIDAR
    # Normalize the height value to the range 0...1
    normalized_value = (val_m - Z_MIN_HEIGHT) / (Z_MAX_HEIGHT - Z_MIN_HEIGHT)
    # Alternative method to ensure the value stays within [0.0, 1.0]
    # This is commented out but can be used to limit the range of normalized values
    #normalized_value = min(1.0, max(0.0, normalized_value))
    # Scaling to the target range [OUT_MIN, OUT_MAX]
    scaled_value = normalized_value * (OUT_MAX - OUT_MIN) + OUT_MIN
    # see above
    #scaled_value = normalized_value * 255.0
    # Limitation of the value to the target range [OUT_MIN, OUT_MAX]
    clamped_value = min(OUT_MAX, max(OUT_MIN, scaled_value))
    # Rounding and conversion to int
    final_value = round(clamped_value)

    return int(final_value)

# ==============================================================================
# MAP INTENSITY (0...1) TO 255 (PX)
# ==============================================================================
@lru_cache(maxsize=None)
def map_intensity2channel(val_int: float, MAX_INTENSITY: float) -> int:
    # Scaling the intensity (use this when your val_int. values are 0...255)
    #scaled_intensity = (val_int * 255.0) / self.MAX_INTENSITY
    # Unmap 0->1 to 0->255 (use this when you val_int is alreasy in 0...1)
    scaled_intensity = val_int * 255.0
    # Limit the intensity to the range 0-255
    clamped_intensity = min(255, scaled_intensity) if scaled_intensity > 0 else 0

    return int(clamped_intensity)

# ==============================================================================
# MAP DENSITY TO 255 (PX)
# ==============================================================================
@lru_cache(maxsize=None)
def map_density2channel(val_den: int, MAX_DENSITY: float) -> int:
    # Scaling the density
    scaled_density = (val_den * 255.0) / MAX_DENSITY
    # Limit the density to the range 0-255
    clamped_density = min(255, scaled_density) if scaled_density > 0 else 0

    return int(clamped_density)

# ==============================================================================
# CONVERT POINT CLOUD (COORDS.) TO IMAGE ROW/COLUMN (COORDS.)
# ==============================================================================
def map_pc2rc(x: float, y: float, row: int, column: int, IMAGE_HEIGHT: int, IMAGE_WIDTH: int, CELL_SIZE: float) -> int:
    # 3D point cloud value -> row 2D Mapping
    row[0] = int(round(((IMAGE_HEIGHT * CELL_SIZE) / 1.0 - x) / (IMAGE_HEIGHT * CELL_SIZE) * IMAGE_HEIGHT))

    # 3D point cloud value -> column 2D Mapping
    column[0] = int(round(((IMAGE_WIDTH * CELL_SIZE) / 2.0 - y) / (IMAGE_WIDTH * CELL_SIZE) * IMAGE_WIDTH))
   
    return 1 # Return success,if in range and row/column are set

# ==============================================================================
# CONVERT IMAGE ROW/COLUMN (COORDS.) TO POINT CLOUD (COORDS.)
# ==============================================================================
def map_rc2pc(x: float, y: float, row: int, column: int, IMAGE_HEIGHT: int, IMAGE_WIDTH: int, CELL_SIZE: float) -> int:
    if 0 <= row < IMAGE_HEIGHT and 0 <= column < IMAGE_WIDTH:
        # row pixel value -> x Mapping (2D)
        x[0] = float(CELL_SIZE *-1.0 * (row - (IMAGE_HEIGHT / 1.0))) # coordinate system: 1.0 bottom center, 2.0 middle center

        # column pixel value -> y Mapping (2D)
        y[0] = float(CELL_SIZE *-1.0 * (column - (IMAGE_WIDTH / 2.0)))
      
        return 1 # Return success, if in range and x/y are set

    return 0

# ==============================================================================
# NORMALIZE BEV BOUNDING BOX COORDINATES [0...1]
# ==============================================================================
@lru_cache(maxsize=None)
def normalize_coordinates(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float, img_width: int, img_height: int
                          ) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    x_n = x / image_width
    y_n = y / image_height
    """
    x1 /= img_width
    x2 /= img_width
    x3 /= img_width
    x4 /= img_width
    y1 /= img_height
    y2 /= img_height
    y3 /= img_height
    y4 /= img_height

    return x1, y1, x2, y2, x3, y3, x4, y4

# ==============================================================================
# COMPUTES THE DENSITY DISTRIBUTION AND MAX DENSITY
# ==============================================================================
def max_density_distribution(density_array: NDArray[float32], image_height: int, image_width: int, distribution_bins: int = 10) -> None:
    max_density = 0
    distribution = [0] * distribution_bins  # Predefine distribution array with `distribution_bins` elements

    for i in range(image_height):
        for j in range(image_width):
            # Find max density
            if density_array[i][j] > max_density:
                max_density = density_array[i][j]

            # Find distribution of points
            density_value = density_array[i][j]
            
            # Make sure the distribution array can accommodate the density value
            if density_value >= len(distribution):
                distribution.extend([0] * (density_value - len(distribution) + 1))

            distribution[int(density_array[i][j])] += 1

    print(f"Max density: {max_density}")

    print("Max Distribution:")
    for i in range(min(distribution_bins, len(distribution))):
        print(f"Cells with {i} points: {distribution[i]}")

# max_density = 0
#         distribution = [0] * 10 # higher values had no influence
#         for i in range(self.config.IMAGE_HEIGHT):
#             for j in range(self.config.IMAGE_WIDTH):

#                 # find max density
#                 if self.densityArray[i][j] > max_density:
#                     max_density = self.densityArray[i][j]

#                 # find distribution of points
#                 # Note: this prevents list index put of range mode='pkl'
#                 density_value = self.densityArray[i][j]
#                 if density_value >= len(distribution):
#                     distribution.extend([0] * (density_value - len(distribution) + 1))

#                 distribution[int(self.densityArray[i][j])] += 1
            
#             print(f"Max density: {max_density}")

#             print("Max Distribution:")
#             for i in range(10):
#                 print(f"Cells with {i} points: {distribution[i]}")

def encode_height_mv3d(z_values, z_min, z_max, slice_idx, num_slices=5, output_range=(0, 255)):
    """
    Encodes height value to pixel value according to MV3D paper.
    
    The point cloud is divided equally into M slices. For each slice,
    the height feature is computed as the maximum height of points in each cell.
    
    Paper: "the point cloud is divided equally into M slices. 
            A height map is computed for each slice"
    
    Args:
        z_value: The z-coordinate of the point (in meters)
        z_min: Minimum z boundary (e.g., -2.73m for KITTI)
        z_max: Maximum z boundary (e.g., 1.27m for KITTI)
        slice_idx: Which height slice (0 to num_slices-1)
        num_slices: Total number of height slices M
        output_range: (min_pixel, max_pixel) for output, default (0, 255)
    
    Returns:
        Encoded height value as integer pixel value [0, 255]
    pass
    """
    total_height = z_max - z_min
    slice_height = total_height / num_slices
    
    z_shifted = z_values - z_min
    height_in_slice = z_shifted - (slice_idx * slice_height)
    
    normalized_height = np.clip(height_in_slice / slice_height, 0.0, 1.0)
    pixel_value = normalized_height * (output_range[1] - output_range[0]) + output_range[0]
     
    return pixel_value.astype(np.uint8)

def encode_intensity_mv3d(intensities, intensity_min=0.0, intensity_max=1.0, output_range=(0, 255)):
    """
    Encodes intensity value to pixel value according to MV3D paper.
    
    The intensity feature is the reflectance value of the point which has
    the maximum height in each cell.
    
    Paper: "The intensity feature is the reflectance value of the point 
            which has the maximum height in each cell"
    
    Args:
        intensity: The intensity/reflectance value of the point
                   KITTI: typically normalized to [0, 1]
                   Some datasets: [0, 255]
        intensity_min: Minimum intensity value (for normalization)
        intensity_max: Maximum intensity value (for normalization)
        output_range: (min_pixel, max_pixel) for output, default (0, 255)
    
    Returns:
        Encoded intensity value as integer pixel value [0, 255]
    """
    normalized_intensity = np.zeros_like(intensities) if intensity_max == intensity_min else (intensities - intensity_min) / (intensity_max - intensity_min)
    normalized_intensity = np.clip(normalized_intensity, 0.0, 1.0)
    pixel_value = normalized_intensity * (output_range[1] - output_range[0]) + output_range[0]
    
    return pixel_value.astype(np.uint8)

def encode_density_mv3d(point_count, calc_log_base, output_range=(0, 255)):
    """
    Encodes density (point count) to pixel value according to MV3D paper.
    
    The density indicates the number of points in each cell, normalized
    using logarithmic scaling.
    
    Paper: "it is computed as min(1.0, log(N+1)/log(64)), 
            where N is the number of points in the cell"
    
    Args:
        point_count: Number of points N in the cell
        log_base: Base for logarithmic normalization (default 64 as per paper)
        output_range: (min_pixel, max_pixel) for output, default (0, 255)
    
    Returns:
        Encoded density value as integer pixel value [0, 255]
    """    
    # Paper formula: min(1.0, log(N+1)/log(64))
    normalized_density = np.minimum(1.0, np.log(point_count + 1) / calc_log_base)
    pixel_value = normalized_density * (output_range[1] - output_range[0]) + output_range[0]
 
    return pixel_value.astype(np.uint8)

def get_sample(data, sample_idx, augment_idx=0, mode='train', class_names=None):
    
    if mode == 'train':
        sample = data[sample_idx][augment_idx]
        
        return {
            'points': sample['points'],
            'gt_boxes': sample['gt_boxes'],
            'gt_names': sample['gt_names'], # currently not used, bc gt_boxes contains cls_idx
            'frame_id': sample['frame_id'],
        }
    else: # val
        sample = data[sample_idx]
        gt_boxes_lidar = sample['annos']['gt_boxes_lidar']
        cls_idx = np.array([class_names.index(name) + 1 for name in sample['annos']['name']])
        gt_boxes_lidar = np.concatenate([gt_boxes_lidar, cls_idx[:, None]], axis=1)

        return {
            'frame_id': sample['point_cloud']['lidar_idx'],
            'image_shape': sample['image']['image_shape'],
            'calib': sample['calib'],
            'points': sample['points'],
            'gt_boxes': gt_boxes_lidar,
            'gt_names': sample['annos']['name'],
            'truncated': sample['annos']['truncated'],
            'occluded': sample['annos']['occluded'],
            'alpha': sample['annos']['alpha'],
            'bbox': sample['annos']['bbox'],
            'dimensions': sample['annos']['dimensions'],
            'location': sample['annos']['location'],
            'rotation_y': sample['annos']['rotation_y'],
            'score': sample['annos']['score'],
        }

def remove_points_outside_range(points, boundary_cond):

    if isinstance(boundary_cond, (list, tuple)):
        x_min, y_min, z_min, x_max, y_max, z_max = boundary_cond
    else:
        # Boundary condition
        x_min = boundary_cond['minX']
        x_max = boundary_cond['maxX']
        y_min = boundary_cond['minY']
        y_max = boundary_cond['maxY']
        z_min = boundary_cond['minZ']
        z_max = boundary_cond['maxZ']
    
    # Remove the point out of range x,y,z
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )

    return points[mask]
    
def pointcloud3d_to_bevimage2d(points, cfg, num_slices=5, filter_points=False):
    """
    Fast, vectorized BEV generation according to MV3D Paper, Complex-YOLO.
    
    Uses the encoding functions:
    - encode_density_mv3d(): min(1.0, log(N+1)/log(64))
    - encode_height_mv3d(): Max z-value per cell, normalized
    - encode_intensity_mv3d(): Reflectance of the highest point
    
    Coordinate mapping:
        LiDAR x (front)  → BEV row (vertical, top=far)
        LiDAR y (left)  → BEV col (horizontal, left=left)
    
    RGB channel assignment: R=Density, G=Height, B=Intensity
    
    Args:
        points: (N, 4) Array [x, y, z, intensity]
        cfg: EasyDict with POINT_CLOUD_RANGE, BEV_IMAGE_HEIGHT/WIDTH
        filter_points: If True, points outside are filtered
    
    Returns:
        bev_image: (H, W, 3) uint8 RGB
    """
    from pcdet.utils.bev_utils import remove_points_outside_range

    pc_range = cfg.POINT_CLOUD_RANGE
    z_min, z_max = pc_range[2], pc_range[5]

    bev_img_h = cfg.BEV_IMAGE_HEIGHT # 1024 px
    bev_img_w = cfg.BEV_IMAGE_WIDTH # 1024 px
    cell_size = cfg.get('CELL_SIZE', 0.06) # m

    data_variant = cfg.get('DATA_VARIANT', 'default')
    log_base = cfg.get('LIDAR_LAYERS_RBRS', 127) if data_variant == 'upsampled' else cfg.get('LIDAR_LAYERS', 64)

    if filter_points:
        points = remove_points_outside_range(points, pc_range)

    # Coordinate mapping LiDAR -> BEV
    rows = np.clip(((bev_img_h * cell_size - points[:, 0]) / (bev_img_h * cell_size) * bev_img_h).astype(np.int32), 0, bev_img_h - 1)
    cols = np.clip(((bev_img_w * cell_size / 2.0 - points[:, 1]) / (bev_img_w * cell_size) * bev_img_w).astype(np.int32), 0, bev_img_w - 1)
                     
    # Density channel
    cell_idx = rows * bev_img_w + cols
    unique_cells, counts = np.unique(cell_idx, return_counts=True)
    log_base_value = np.log(log_base)

    density_values = encode_density_mv3d(point_count=counts, calc_log_base=log_base_value)

    density_map = np.zeros((bev_img_h, bev_img_w), dtype=np.uint8)
    rows_unique = unique_cells // bev_img_w # explain
    cols_unique = unique_cells % bev_img_w # explain
    density_map[rows_unique, cols_unique] = density_values

    # Pre calculations
    total_height = z_max - z_min
    slice_height = total_height / num_slices
    z_shifted = points[:, 2] - z_min
    slice_indices = np.clip((z_shifted / slice_height).astype(np.int32), 0, num_slices - 1)
    
    # Height channel
    height_slices = []
    for slice_idx in range(num_slices):
        slice_map = np.zeros((bev_img_h, bev_img_w), dtype=np.float32)
        
        slice_mask = slice_indices == slice_idx

        if np.any(slice_mask):
            slice_rows = rows[slice_mask]
            slice_cols = cols[slice_mask]
            slice_z = points[:, 2][slice_mask]

            # Keep maximum z-value per cell
            np.maximum.at(slice_map, (slice_rows, slice_cols), slice_z)
            
            # Encode only valid cells
            valid_mask = slice_map > 0
            if np.any(valid_mask):
                z_values = slice_map[valid_mask]

                pixel_values = encode_height_mv3d(
                        z_values=z_values,
                        z_min=z_min,
                        z_max=z_max,
                        slice_idx=slice_idx,
                        num_slices=num_slices
                )
                slice_map_encoded = np.zeros((bev_img_h, bev_img_w), dtype=np.uint8)
                slice_map_encoded[valid_mask] = pixel_values
                height_slices.append(slice_map_encoded)
            
            else:
                height_slices.append(np.zeros((bev_img_h, bev_img_w), dtype=np.uint8))
        else:
            height_slices.append(np.zeros((bev_img_h, bev_img_w), dtype=np.uint8))
    
    height_combined = np.maximum.reduce(height_slices)

    # Intensity channel
    z_temp = np.full((bev_img_h, bev_img_w), -np.inf, dtype=np.float32)
    intensity_temp = np.zeros((bev_img_h, bev_img_w), dtype=np.float32)

    # Sort by z to ensure highest points overwrite
    sort_idx = np.argsort(points[:, 2])
    rows_sorted = rows[sort_idx]
    cols_sorted = cols[sort_idx]
    z_sorted = points[:, 2][sort_idx]
    intensity_sorted = points[:, 3][sort_idx]

    # Update with highest points (last write wins)
    z_temp[rows_sorted, cols_sorted] = z_sorted
    intensity_temp[rows_sorted, cols_sorted] = intensity_sorted
    
    # Encode only valid cells
    valid_mask = z_temp > -np.inf
    intensity_map = np.zeros((bev_img_h, bev_img_w), dtype=np.uint8)

    if np.any(valid_mask):
        intensities = intensity_temp[valid_mask]
        pixel_values = encode_intensity_mv3d(
            intensities=intensities,
            intensity_min=0.0,
            intensity_max=1.0
        )

        intensity_map[valid_mask] = pixel_values

    # BGR BEV Image 
    bev_image = np.zeros((bev_img_h, bev_img_w, 3), dtype=np.uint8)
    bev_image[:, :, 0] = density_map # B(0) density
    bev_image[:, :, 1] = intensity_map # G(1) intensity
    bev_image[:, :, 2] = height_combined # R(2) height

    return bev_image, height_slices

def boxes3d_lidar_to_rotated_bev_boxes(bev_img, bev_image_height, bev_image_width, bev_res, boxes3d, min_points=5):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, l, w, h, heading, cls_idx] in lidar coordinate

    Returns:
        rotated_bev_boxes: (8,) [cls_idx, x1, y1, x2, y2, x3, y3, x4, y4] in image coordinate (YOLO format)
    """
    x, y, z, l, w, h, heading, cls_idx = boxes3d
    centroid = [x, y]

    # source: https://github.com/maudzung/Complex-YOLOv4-Pytorch/blob/master/src/data_process/kitti_bev_utils.py
    if cls_idx == 1: # Car
        l = l + 0.4
        w = w + 0.4
    elif cls_idx == 2: # Pedestrian
        l = l + 0.3
        w = w + 0.3
    elif cls_idx == 3: # Cyclist
        l = l + 0.3
        w = w + 0.3

    yaw_bev = -heading # Invert yaw from LiDAR frame (CW) to Image frame (CCW), bc zero angle (0°) aligned differently in both systems

    corners = np.array([
        [centroid[0] + l/2, centroid[1] + w/2],  # 0: top-left
        [centroid[0] + l/2, centroid[1] - w/2],  # 1: top-right
        [centroid[0] - l/2, centroid[1] - w/2],  # 2: bottom-right
        [centroid[0] - l/2, centroid[1] + w/2],  # 3: bottom-left
    ])
    """ 
            Λ           
    (x1,y1)---(x2,y2)     
       |    |    |            
       |    x    |      
       |         |             
    (x4,y4)---(x3,y3)        
    """

    R = np.array([[np.cos(yaw_bev), -np.sin(yaw_bev)],[np.sin(yaw_bev),  np.cos(yaw_bev)]])
    """
    2D-Rotation matrix of CW
    [x'] = [cos(yaw) -sin(yaw)] * [x]
    [y']   [sin(yaw)  cos(yaw)]   [y]
    """

    rotated_corners = np.dot(corners - centroid, R) + centroid      

    x_img = bev_image_width / 2 - rotated_corners[:, 1] / bev_res # lidar world -y -> image x (u)
    y_img = bev_image_height - rotated_corners[:, 0] / bev_res # lidar world x -> image y (v)

    # Policy
    # NOTE: KITTI readme: "...to avoid false positives - detections not visible on the image plane should be filtered."

    # Check if all 2D BEV label corners are in image area
    is_fully_visible = np.all(
        (x_img >= 0) & (x_img < bev_image_width) &
        (y_img >= 0) & (y_img < bev_image_height)
    )
    if not is_fully_visible:
        return None  # skip box, if truncated (outside image borders)

    # Determine ROI
    x_min = max(0, int(np.floor(np.min(x_img))))
    x_max = min(bev_image_width - 1, int(np.ceil(np.max(x_img))))
    y_min = max(0, int(np.floor(np.min(y_img))))
    y_max = min(bev_image_height - 1, int(np.ceil(np.max(y_img))))

    roi = bev_img[y_min:y_max+1, x_min:x_max+1]
    nonzero = np.count_nonzero(np.sum(roi, axis=2))
    if nonzero < min_points:
        return None  # skip box, if less than 5 points
    
    rotated_bev_boxes = np.stack([x_img, y_img], axis=1).reshape(-1) # x1, y1, x2, y2, x3, y3, x4, y4, cls_idx

    return np.concatenate((rotated_bev_boxes, [cls_idx]))

def show_bev_image_preview(bev_image, window_name="BEV Preview", BGR2RGB=True, win_size=(800, 800)):
    if BGR2RGB:
        bev_image = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
    else:
        bev_image = bev_image
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *win_size)
    cv2.startWindowThread()
    cv2.imshow(window_name, bev_image)
    key = cv2.waitKey(1)

    return key

def draw_bev_boxes(bev_image, bev_box, cls_idx, box_colormap=None, thickness=1, obj_direction_color=(255,0,0)):

    rgb = box_colormap[(cls_idx)]
    color = tuple(int(255 * c) for c in rgb[::-1])  # RGB -> BGR

    pts = np.array(bev_box, dtype=np.int32).reshape(4, 2)
    for i in range(4):
        pt1 = tuple(pts[i])
        pt2 = tuple(pts[(i + 1) % 4])
        if i == 0:
            # front side of the object
            cv2.line(bev_image, pt1, pt2, obj_direction_color, thickness=2)
        else:
            cv2.line(bev_image, pt1, pt2, color, thickness=thickness)
    return bev_image

def save_bev_images_and_boxes(save_path, frame_id, bev_image_bgr, valid_bev_boxes, augment_idx=0, 
                              compression=[cv2.IMWRITE_PNG_COMPRESSION, 1], normalize_coords=False, 
                              data_variant='default', mode="train"):

    bev_img_dir = os.path.join(save_path, f'bev_images_{data_variant}')
    bev_annos_dir = os.path.join(save_path, 'bev_annos')
    os.makedirs(bev_img_dir, exist_ok=True)
    os.makedirs(bev_annos_dir, exist_ok=True)

    if mode == "train":
        aug_tag = AUGMENT_SHORT[augment_idx] if augment_idx < len(AUGMENT_SHORT) else f"aug{augment_idx}"
        filename_img = f"{frame_id}_{aug_tag}.png"
        filename_anno = f"{frame_id}_{aug_tag}.txt"
    else:
        filename_img = f"{frame_id}.png"
        filename_anno = f"{frame_id}.txt"

    filepath_img = os.path.join(bev_img_dir, filename_img)
    filepath_anno = os.path.join(bev_annos_dir, filename_anno)

    bev_image_rgb = cv2.cvtColor(bev_image_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename=filepath_img, img=bev_image_rgb , params=compression)
    
    if normalize_coords:
        valid_bev_boxes = normalize_pixels_to_range(np.array(valid_bev_boxes), bev_image_bgr.shape[0], bev_image_bgr.shape[1])

    if data_variant == 'default': # save annos just once
        with open(filepath_anno, "w") as f:
            for box in valid_bev_boxes:
                cls_idx = box[-1]
                coords = box[:-1]
                f.write(f"{int(cls_idx)} " + " ".join([str(coord) for coord in coords]) + "\n") # class_index x1 y1 x2 y2 x3 y3 x4 y4

def normalize_pixels_to_range(bev_boxes, img_height, img_width):
    norm_boxes = bev_boxes.copy()

    norm_boxes[:, :8:2] /= img_width # x
    norm_boxes[:, 1:8:2] /= img_height # y

    return norm_boxes

def generate_kitti_groundtruth_txts(sample, valid_indices, output_path=None, frame_id=None, offsets=True):
 
    gt_dict = {
        'name': np.array(sample['gt_names'])[valid_indices],
        'truncated': np.array(sample['truncated'])[valid_indices],
        'occluded': np.array(sample['occluded'])[valid_indices],
        'alpha': np.array(sample['alpha'])[valid_indices],
        'bbox': np.array(sample['bbox'])[valid_indices],
        'dimensions': np.array(sample['dimensions'])[valid_indices].copy(),
        'location': np.array(sample['location'])[valid_indices],
        'rotation_y': np.array(sample['rotation_y'])[valid_indices],
        'boxes_lidar': np.array(sample['gt_boxes'])[valid_indices]
    }

    if output_path is not None and frame_id is not None:
        os.makedirs(output_path, exist_ok=True)
        label_file = os.path.join(output_path, f"{frame_id}.txt")
        bbox = gt_dict['bbox']
        loc = gt_dict['location']
        dims = gt_dict['dimensions']  # l, w, h (LiDAR)

        if offsets:
            l_offsets = np.where(gt_dict['name'] == 'Car', 0.4, np.where(np.isin(gt_dict['name'], ['Pedestrian', 'Cyclist']), 0.3, 0.0))
            w_offsets = np.where(gt_dict['name'] == 'Car', 0.4, np.where(np.isin(gt_dict['name'], ['Pedestrian', 'Cyclist']), 0.3, 0.0))

            dims[:, 0] += l_offsets # l
            dims[:, 1] += w_offsets # w
        else:
            print("[WARNING] generate_groundtruth_txts: offsets=False. "
                  "Training labels use offsets (Car: +0.4, Ped/Cyc: +0.3). "
                  "Mismatch may reduce AP scores!")

        with open(label_file, 'w') as f:
            for idx in range(len(bbox)):
                print('%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'
                      % (gt_dict['name'][idx],
                         gt_dict['truncated'][idx],
                         gt_dict['occluded'][idx],
                         gt_dict['alpha'][idx],
                         bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3], # xmin, ymin, xmax, ymax
                         dims[idx][2], dims[idx][1], dims[idx][0],  # stored as h, w, l (Camera)
                         loc[idx][0], loc[idx][1], loc[idx][2], # x, y, z
                         gt_dict['rotation_y'][idx]),
                      file=f)

    return gt_dict

def generate_zod_groundtruth_txts(sample, valid_indices, output_path=None, frame_id=None, offsets=True):
    gt_dict = {
        'name': np.array(sample['gt_names'])[valid_indices],
        'truncated': np.array(sample['truncated'])[valid_indices],
        'occluded': np.array(sample['occluded'])[valid_indices],
        'alpha': np.array(sample['alpha'])[valid_indices],
        'bbox': np.array(sample['bbox'])[valid_indices],
        #'dimensions': np.array(sample['dimensions'])[valid_indices].copy(),
        #'location': np.array(sample['location'])[valid_indices],
        #'rotation_y': np.array(sample['rotation_y'])[valid_indices],
        'boxes_lidar': np.array(sample['gt_boxes'])[valid_indices]
    }

    if output_path is not None and frame_id is not None:
        os.makedirs(output_path, exist_ok=True)
        label_file = os.path.join(output_path, f"{frame_id}.txt")
        bbox = gt_dict['bbox']
        loc = gt_dict['boxes_lidar'][:, 0:3] # x, y, z (LiDAR)
        dims = gt_dict['boxes_lidar'][:, 3:6]  # l, w, h (LiDAR)
        heading = gt_dict['boxes_lidar'][:, 6] # heading

        if offsets:
            l_offsets = np.where(gt_dict['name'] == 'Car', 0.4, np.where(np.isin(gt_dict['name'], ['Pedestrian', 'Cyclist']), 0.3, 0.0))
            w_offsets = np.where(gt_dict['name'] == 'Car', 0.4, np.where(np.isin(gt_dict['name'], ['Pedestrian', 'Cyclist']), 0.3, 0.0))

            dims[:, 0] += l_offsets # l
            dims[:, 1] += w_offsets # w
        else:
            print("[WARNING] generate_zod_groundtruth_txts(): offsets=False. "
                  "Training labels use offsets (Car: +0.4, Ped/Cyc: +0.3). "
                  "Mismatch may reduce AP scores!")

        with open(label_file, 'w') as f:
            for idx in range(len(bbox)):
                print('%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'
                      % (gt_dict['name'][idx],
                         gt_dict['truncated'][idx],
                         gt_dict['occluded'][idx],
                         gt_dict['alpha'][idx],
                         bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3], # xmin, ymin, xmax, ymax
                         dims[idx][0], dims[idx][1], dims[idx][2],  # stored as l, w ,h (LiDAR)
                         loc[idx][0], loc[idx][1], loc[idx][2], # x, y, z
                         heading[idx]), # rotation_z
                      file=f)

    return gt_dict

def draw_fov_bev(bev_image, calib, fov_range=245.0, color=(255, 255, 255), thickness=2):
    bev_h, bev_w = bev_image.shape[:2]
    cell_size = 0.06  

    horizontal_fov = calib['cam_field_of_view'][0] # ~120°
    cx = bev_w // 2
    cy = bev_h

    num_points = 200
    angles = np.linspace(-horizontal_fov/2, horizontal_fov/2, num_points) * np.pi / 180
    xs = fov_range * np.sin(angles)
    ys = fov_range * np.cos(angles)

    x_img = cx - xs / cell_size
    y_img = cy - ys / cell_size

    pts = np.vstack([np.append(cx, x_img), np.append(cy, y_img)]).T.astype(np.int32)

    cv2.polylines(bev_image, [pts], isClosed=False, color=color, thickness=thickness)

    pt_left = (int(x_img[0]), int(y_img[0]))
    pt_right = (int(x_img[-1]), int(y_img[-1]))
    center = (int(cx), int(cy))
    cv2.line(bev_image, center, pt_left, color, thickness)
    cv2.line(bev_image, center, pt_right, color, thickness)

    return bev_image

def get_frame_ids(directory, ext):
    import re

    return set(
        re.match(r"(\d{6})", f).group(1)
        for f in os.listdir(directory)
        if f.endswith(ext) and re.match(r"(\d{6})", f)
    )


