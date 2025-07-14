import os
import json
import numpy as np
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm
import sys

# === CONFIG ===
DATA_ROOT = '/home/mostafa21314/Extractor'
V1_TRAINVAL_DIR = os.path.join(DATA_ROOT, 'v1.0-trainval')
SAMPLES_DIR = os.path.join(DATA_ROOT, 'samples')
SWEEPS_DIR = os.path.join(DATA_ROOT, 'sweeps')
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

def load_json_file(filepath):
    """Load and parse a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def view_points(points, view, normalize=False):
    """
    This is a helper class that maps 3d points to a 2d plane. It can be thought of as a batch camera
    with transformation matrix (view) and can be used to realize both forward and backward projection.
    Args:
        points: (3, N) 3d points in (x, y, z) in reference 1 coordinate system
        view: (3, 4) transformation matrix
        normalize: Whether to normalize the image coordinate system around the center of image. By default,
            the image coordinate system is around the top-left corner of the image.
    Returns:
        points: (>=2, N) projected 3d points in image. The coordinate system is defined by the view.
            (2, N) for small segments (x, y) and (3, N) for full 3d coordinates (x, y, z)
    """
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogeneous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, axis=0).reshape(3, nbr_points)

    return points

def transform_matrix(translation, rotation, inverse=False):
    """
    Convert pose to transformation matrix.
    Args:
        translation: <np.float32: 3>. Translation in x, y, z.
        rotation: <np.float32: 4>. Quaternion in x, y, z, w.
        inverse: Whether to compute inverse transform.
    Returns:
        <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.inverse
        trans = np.transpose(-np.array(translation))
        rot_inv = rot_inv.rotation_matrix
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.array(translation)

    return tm

def project_3d_to_2d(box, ego_pose, calibrated_sensor):
    """
    Project 3D box to 2D image coordinates.
    Args:
        box: 3D box with corners() method
        ego_pose: ego pose dict with translation and rotation
        calibrated_sensor: calibrated sensor dict with translation, rotation, and camera_intrinsic
    Returns:
        corners_2d: 2D corners in image coordinates
    """
    # Transform box from global to ego
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    # Transform box from ego to camera
    sensor_translation = np.array(calibrated_sensor['translation'])
    sensor_rotation = Quaternion(calibrated_sensor['rotation'])
    
    # Get camera intrinsics
    camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
    
    # Get box corners in global coordinates
    corners = box.corners()  # (3, 8)
    
    # Transform to ego coordinates
    corners = corners - ego_translation.reshape(3, 1)
    corners = np.dot(ego_rotation.inverse.rotation_matrix, corners)
    
    # Transform to camera coordinates
    corners = corners - sensor_translation.reshape(3, 1)
    corners = np.dot(sensor_rotation.inverse.rotation_matrix, corners)
    
    # Project to 2D
    corners_2d = view_points(corners, camera_intrinsic, normalize=True)
    
    return corners_2d

class Box:
    """Simple 3D box class."""
    def __init__(self, center, size, orientation):
        self.center = np.array(center)
        self.size = np.array(size)
        self.orientation = Quaternion(orientation)
    
    def corners(self):
        """Get 8 corners of the box."""
        # Create unit cube
        x_corners = [1, 1, 1, 1, -1, -1, -1, -1]
        y_corners = [1, -1, -1, 1, 1, -1, -1, 1]
        z_corners = [1, 1, -1, -1, 1, 1, -1, -1]
        corners = np.vstack([x_corners, y_corners, z_corners])
        
        # Scale by size
        corners = corners * self.size.reshape(3, 1) / 2
        
        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)
        
        # Translate
        corners = corners + self.center.reshape(3, 1)
        
        return corners

# === LOAD DATA ===
print("Loading JSON files...")
samples = load_json_file(os.path.join(V1_TRAINVAL_DIR, 'sample.json'))
sample_data = load_json_file(os.path.join(V1_TRAINVAL_DIR, 'sample_data.json'))
sample_annotations = load_json_file(os.path.join(V1_TRAINVAL_DIR, 'sample_annotation.json'))
ego_poses = load_json_file(os.path.join(V1_TRAINVAL_DIR, 'ego_pose.json'))
calibrated_sensors = load_json_file(os.path.join(V1_TRAINVAL_DIR, 'calibrated_sensor.json'))
instances = load_json_file(os.path.join(V1_TRAINVAL_DIR, 'instance.json'))
categories = load_json_file(os.path.join(V1_TRAINVAL_DIR, 'category.json'))

# Get list of available images
print("Scanning for available images...")
available_images = set()
cam_front_dir = os.path.join(SAMPLES_DIR, CAMERA)
if os.path.exists(cam_front_dir):
    for filename in os.listdir(cam_front_dir):
        if filename.endswith('.jpg'):
            available_images.add(filename)
print(f"Found {len(available_images)} available images in {cam_front_dir}")

# Create lookup dictionaries for faster access
print("Creating lookup dictionaries...")
ego_pose_dict = {pose['token']: pose for pose in ego_poses}
calibrated_sensor_dict = {sensor['token']: sensor for sensor in calibrated_sensors}

# Create category lookup dictionaries
category_dict = {cat['token']: cat['name'] for cat in categories}
instance_dict = {inst['token']: inst['category_token'] for inst in instances}

# Filter sample_data to only include entries with available images
filtered_sample_data = []
for data in sample_data:
    if CAMERA in data['filename']:
        # Extract filename from the path
        filename = os.path.basename(data['filename'])
        if filename in available_images:
            filtered_sample_data.append(data)

print(f"Filtered to {len(filtered_sample_data)} sample_data entries with available images")

# Group annotations by sample_token
annotations_by_sample = {}
for ann in sample_annotations:
    sample_token = ann['sample_token']
    if sample_token not in annotations_by_sample:
        annotations_by_sample[sample_token] = []
    annotations_by_sample[sample_token].append(ann)

# Create set of samples that have available camera data
available_samples = set()
for data in filtered_sample_data:
    available_samples.add(data['sample_token'])

# Filter samples to only those with available images
filtered_samples = [s for s in samples if s['token'] in available_samples]
print(f"Filtered to {len(filtered_samples)} samples with available images")

# === PROCESS SAMPLES ===
print(f"Processing {len(filtered_samples)} samples...")
processed_count = 0

for sample in tqdm(filtered_samples, desc="Processing samples"):
    try:
        sample_token = sample['token']
        
        # Find camera data for this sample
        cam_data = None
        for data in filtered_sample_data:
            if data['sample_token'] == sample_token and CAMERA in data['filename']:
                cam_data = data
                break
        
        if cam_data is None:
            continue
            
        # Get image path
        im_path = os.path.join(DATA_ROOT, cam_data['filename'])
        
        # Check if image exists
        if not os.path.exists(im_path):
            print(f"Warning: Image not found at {im_path}")
            continue
            
        image = cv2.imread(im_path)
        if image is None:
            print(f"Warning: Could not read image at {im_path}")
            continue
            
        height, width = image.shape[:2]

        # Get ego pose and calibrated sensor
        ego_pose = ego_pose_dict[cam_data['ego_pose_token']]
        calibrated_sensor = calibrated_sensor_dict[cam_data['calibrated_sensor_token']]
        
        # Check if camera_intrinsic is available
        if not calibrated_sensor.get('camera_intrinsic'):
            print(f"Warning: No camera intrinsics for sensor {cam_data['calibrated_sensor_token']}")
            continue

        # Get annotations for this sample
        annotations = annotations_by_sample.get(sample_token, [])
        
        label_lines = []
        for ann in annotations:
            # Get category name through instance and category tokens
            instance_token = ann['instance_token']
            if instance_token not in instance_dict:
                continue
                
            category_token = instance_dict[instance_token]
            if category_token not in category_dict:
                continue
                
            cat = category_dict[category_token]
            if cat not in CATEGORY_TO_ID:
                continue

            # Create 3D box
            box = Box(
                center=ann['translation'],
                size=ann['size'],
                orientation=ann['rotation']
            )

            # Project to 2D
            corners_2d = project_3d_to_2d(box, ego_pose, calibrated_sensor)
            
            if corners_2d is None or corners_2d.shape[1] == 0:
                continue

            # Get 2D bounding box
            x_coords = corners_2d[0, :]
            y_coords = corners_2d[1, :]
            
            # Filter out points outside image
            valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
            if not np.any(valid_mask):
                continue
                
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            
            if len(x_coords) == 0:
                continue

            x_min = max(0, np.min(x_coords))
            y_min = max(0, np.min(y_coords))
            x_max = min(width, np.max(x_coords))
            y_max = min(height, np.max(y_coords))

            if x_max <= x_min or y_max <= y_min:
                continue

            # Convert to YOLO format (normalized)
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            box_w = (x_max - x_min) / width
            box_h = (y_max - y_min) / height

            # Calculate distance and angle
            ego_translation = np.array(ego_pose['translation'])
            ego_rotation = Quaternion(ego_pose['rotation'])
            
            obj_pos = np.array(ann['translation'])
            rel = obj_pos - ego_translation
            rel = np.dot(ego_rotation.inverse.rotation_matrix, rel)

            distance = np.linalg.norm(rel[:2])
            angle = np.arctan2(rel[1], rel[0])

            class_id = CATEGORY_TO_ID[cat]
            x, y = obj_pos[0], obj_pos[1]
            ego_x, ego_y = ego_translation[0], ego_translation[1]
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f} {distance:.3f} {angle:.3f} {x:.3f} {y:.3f} {ego_x:.3f} {ego_y:.3f}")

        if not label_lines:
            continue

        # Save image
        img_filename = f"{cam_data['token']}.jpg"
        label_filename = f"{cam_data['token']}.txt"
        cv2.imwrite(os.path.join(IMAGE_OUTPUT, img_filename), image)

        # Save label
        with open(os.path.join(LABEL_OUTPUT, label_filename), 'w') as f:
            f.write("\n".join(label_lines))
            
        processed_count += 1
            
    except Exception as e:
        print(f"Error processing sample {sample['token']}: {e}")
        continue

print(f"Extraction complete! Processed {processed_count} samples.")
print(f"Images saved to: {IMAGE_OUTPUT}")
print(f"Labels saved to: {LABEL_OUTPUT}") 