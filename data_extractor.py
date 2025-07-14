import os
import numpy as np
import cv2
import sys

# Add nuscenes-devkit to path
sys.path.append('/home/mostafa21314/NuScenes work/nuscenes-devkit/python-sdk')

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from pyquaternion import Quaternion
from tqdm import tqdm

# === CONFIG ===
NUSCENES_VERSION = 'v1.0-mini'
NUSCENES_DATA_ROOT = '/home/mostafa21314/NuScenes work/v1.0-mini'
OUTPUT_DIR = '/home/mostafa21314/Extractor/output_yolo_format'
CAMERA = 'CAM_FRONT'

IMAGE_OUTPUT = os.path.join(OUTPUT_DIR, 'images')
LABEL_OUTPUT = os.path.join(OUTPUT_DIR, 'labels')

os.makedirs(IMAGE_OUTPUT, exist_ok=True)
os.makedirs(LABEL_OUTPUT, exist_ok=True)

# === CLASS MAPPING ===
NUSCENES_CLASSES = sorted([
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck'
])
CATEGORY_TO_ID = {cls: idx for idx, cls in enumerate(NUSCENES_CLASSES)}

def compute_ego_speed(nusc: NuScenes, sample_token: str) -> float:
    """
    Estimate ego speed using translation delta over time between current and previous ego poses.
    """
    sample = nusc.get('sample', sample_token)
    
    # Try LIDAR_TOP first
    sensor_keys = ['LIDAR_TOP', 'CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    
    for sensor in sensor_keys:
        if sensor not in sample['data']:
            continue
            
        curr_sd = nusc.get('sample_data', sample['data'][sensor])
        if not curr_sd['prev']:
            continue  # Try next sensor
            
        prev_sd = nusc.get('sample_data', curr_sd['prev'])
        
        curr_pose = nusc.get('ego_pose', curr_sd['ego_pose_token'])
        prev_pose = nusc.get('ego_pose', prev_sd['ego_pose_token'])
        
        dt = (curr_sd['timestamp'] - prev_sd['timestamp']) / 1e6  # seconds
        dx = np.array(curr_pose['translation']) - np.array(prev_pose['translation'])
        
        if dt > 0:
            speed = np.linalg.norm(dx) / dt
            return speed
    
    # If none of the sensors had a valid previous frame
    return 0.0

def compute_annotation_velocity(nusc: NuScenes, instance_token: str, curr_ann_token: str):
    """
    Estimate object velocity using current and previous annotations for the same instance.
    """
    try:
        # Current annotation and its sample timestamp
        curr_ann = nusc.get('sample_annotation', curr_ann_token)
        curr_sample = nusc.get('sample', curr_ann['sample_token'])
        curr_time = curr_sample['timestamp']
        
        # Previous annotation and its sample timestamp
        if not curr_ann['prev']:
            return 0.0
        prev_ann = nusc.get('sample_annotation', curr_ann['prev'])
        prev_sample = nusc.get('sample', prev_ann['sample_token'])
        prev_time = prev_sample['timestamp']
        
        # Position delta
        dx = np.array(curr_ann['translation']) - np.array(prev_ann['translation'])
        dt = (curr_time - prev_time) / 1e6  # convert from microseconds to seconds
        
        speed = np.linalg.norm(dx) / dt if dt > 0 else 0.0
        return speed
        
    except Exception as e:
        print(f"[WARN] Failed to compute velocity for instance {instance_token}: {e}")
        return 0.0

# === INIT ===
print(f"Loading NuScenes dataset from: {NUSCENES_DATA_ROOT}")
nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATA_ROOT, verbose=True)

# === PROCESS SCENES ===
print(f"Processing {len(nusc.sample)} samples...")
for sample in tqdm(nusc.sample, desc="Processing samples"):
    try:
        cam_data = nusc.get('sample_data', sample['data'][CAMERA])
        anns = sample['anns']
        cam_token = sample['data'][CAMERA]

        im_path = os.path.join(NUSCENES_DATA_ROOT, cam_data['filename'])
        
        # Check if image exists
        if not os.path.exists(im_path):
            print(f"Warning: Image not found at {im_path}")
            continue
            
        image = cv2.imread(im_path)
        if image is None:
            print(f"Warning: Could not read image at {im_path}")
            continue
            
        height, width = image.shape[:2]

        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(cs_record['camera_intrinsic'])

        label_lines = []
        for ann_token in anns:
            ann = nusc.get('sample_annotation', ann_token)
            cat = ann['category_name']
            if cat not in CATEGORY_TO_ID:
                continue

            box = nusc.get_box(ann_token)

            # Transform box from global to ego
            box.translate(-ego_translation)
            box.rotate(ego_rotation.inverse)

            # Transform box from ego to camera
            cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            sensor_translation = np.array(cs_record['translation'])
            sensor_rotation = Quaternion(cs_record['rotation'])

            box.translate(-sensor_translation)
            box.rotate(sensor_rotation.inverse)

            # Now project to 2D using camera intrinsics
            corners = view_points(box.corners(), camera_intrinsic, True)

            x_min = max(0, np.min(corners[0]))
            y_min = max(0, np.min(corners[1]))
            x_max = min(width, np.max(corners[0]))
            y_max = min(height, np.max(corners[1]))

            if x_max <= x_min or y_max <= y_min:
                continue

            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            box_w = (x_max - x_min) / width
            box_h = (y_max - y_min) / height

            pos = np.array(ann['translation'])
            rel = pos - ego_translation
            rel = np.dot(ego_rotation.inverse.rotation_matrix, rel)

            distance = np.linalg.norm(rel[:2])
            angle = np.arctan2(rel[1], rel[0])

            class_id = CATEGORY_TO_ID[cat]
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f} {distance:.3f} {angle:.3f}")

        if not label_lines:
            continue

        # Save image
        img_filename = f"{cam_token}.jpg"
        label_filename = f"{cam_token}.txt"
        cv2.imwrite(os.path.join(IMAGE_OUTPUT, img_filename), image)

        # Save label
        with open(os.path.join(LABEL_OUTPUT, label_filename), 'w') as f:
            f.write("\n".join(label_lines))
            
    except Exception as e:
        print(f"Error processing sample {sample['token']}: {e}")
        continue

print(f"Extraction complete! Images saved to: {IMAGE_OUTPUT}")
print(f"Labels saved to: {LABEL_OUTPUT}") 