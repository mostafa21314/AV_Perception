import os
import torch
import torch.nn as nn
import cv2
import numpy as np

# === CONFIG ===
LABEL_FILE = '/home/mostafa21314/Extractor/output_yolo_format/labels/0d1ffa042f004c67b9e12413cf2acbb7.txt'  # Update this to the label file for the image
IMAGE_FILE = '/home/mostafa21314/Extractor/output_yolo_format/images/0d1ffa042f004c67b9e12413cf2acbb7.jpg'  # Update this to the image file
MODEL_PATH = 'distance_angle_regressor.pth'
CATEGORY_TO_ID = {
    'animal': 0,
    'human.pedestrian.adult': 1,
    'human.pedestrian.child': 2,
    'human.pedestrian.construction_worker': 3,
    'human.pedestrian.personal_mobility': 4,
    'human.pedestrian.police_officer': 5,
    'human.pedestrian.stroller': 6,
    'human.pedestrian.wheelchair': 7,
    'movable_object.barrier': 8,
    'movable_object.debris': 9,
    'movable_object.pushable_pullable': 10,
    'movable_object.trafficcone': 11,
    'static_object.bicycle_rack': 12,
    'vehicle.bicycle': 13,
    'vehicle.bus.bendy': 14,
    'vehicle.bus.rigid': 15,
    'vehicle.car': 16,
    'vehicle.construction': 17,
    'vehicle.emergency.ambulance': 18,
    'vehicle.emergency.police': 19,
    'vehicle.motorcycle': 20,
    'vehicle.trailer': 21,
    'vehicle.truck': 22
}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}
NUM_CLASSES = len(CATEGORY_TO_ID)

# === MODEL ===
class DistanceAngleRegressor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, 8)
        self.fc = nn.Sequential(
            nn.Linear(8 + 4 + 2, 128),  # FIXED: include ego_xy (2)
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, class_id, bbox, ego_xy):
        class_feat = self.class_embed(class_id)
        x = torch.cat([class_feat, bbox, ego_xy], dim=1)
        return self.fc(x)

# === LOAD MODEL ===
model = DistanceAngleRegressor(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# === LOAD DATA ===
def load_label_file(label_file):
    objects = []
    with open(label_file) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 11:
                class_id = int(vals[0])
                bbox = [float(x) for x in vals[1:5]]
                x = float(vals[7])
                y = float(vals[8])
                ego_x = float(vals[9])
                ego_y = float(vals[10])
                objects.append((class_id, bbox, x, y, ego_x, ego_y))
    return objects

objects = load_label_file(LABEL_FILE)



# === NORMALIZATION PARAMS (copy these from your training output) ===
X_MEAN = 917.0  # <-- Replace with actual mean of x
X_STD = 475.0    # <-- Replace with actual std of x
Y_MEAN = 1339.0  # <-- Replace with actual mean of y
Y_STD = 270.0    # <-- Replace with actual std of y

# === PREDICT ===
class_ids = torch.tensor([obj[0] for obj in objects], dtype=torch.long)
bboxes = torch.tensor([obj[1] for obj in objects], dtype=torch.float32)
ego_xys = torch.tensor([
    [
        (obj[4] - X_MEAN) / X_STD,
        (obj[5] - Y_MEAN) / Y_STD
    ] for obj in objects
], dtype=torch.float32)

with torch.no_grad():
    preds = model(class_ids, bboxes, ego_xys)
    # preds: (N, 5) -- distance, sin(angle), cos(angle), x, y (normalized)


for i, (class_id, bbox, x_gt, y_gt, ego_x, ego_y) in enumerate(objects):
    distance, sin_angle, cos_angle, x_norm, y_norm = preds[i].tolist()
    angle = np.arctan2(sin_angle, cos_angle)
    # Denormalize x and y
    x = x_norm * X_STD + X_MEAN
    y = y_norm * Y_STD + Y_MEAN
    print(f"Object {i}: class={ID_TO_CATEGORY[class_id]}, bbox={bbox}, Predicted distance={distance:.2f}m, angle={angle:.2f}rad, x={x:.2f}, y={y:.2f}, GT x={x_gt:.2f}, GT y={y_gt:.2f}, ego_x={ego_x:.2f}, ego_y={ego_y:.2f}")