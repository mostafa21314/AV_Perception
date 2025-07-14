import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np

# === CONFIG ===
LABEL_DIR = '/home/mostafa21314/Extractor/output_yolo_format/labels'  # Update if needed
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
NUM_CLASSES = len(CATEGORY_TO_ID)

# === COMPUTE NORMALIZATION PARAMS (x/y mean and std) ===
all_x, all_y = [], []
for fname in os.listdir(LABEL_DIR):
    if fname.endswith('.txt'):
        with open(os.path.join(LABEL_DIR, fname)) as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) == 11:
                    all_x.append(float(vals[7]))
                    all_y.append(float(vals[8]))
X_MEAN = float(np.mean(all_x))
X_STD = float(np.std(all_x))
Y_MEAN = float(np.mean(all_y))
Y_STD = float(np.std(all_y))
print(f"[INFO] X_MEAN={X_MEAN}, X_STD={X_STD}, Y_MEAN={Y_MEAN}, Y_STD={Y_STD}")

# === DATASET ===
class BBoxDistanceAngleDataset(Dataset):
    def __init__(self, label_dir):
        self.samples = []
        for fname in os.listdir(label_dir):
            if fname.endswith('.txt'):
                with open(os.path.join(label_dir, fname)) as f:
                    for line in f:
                        vals = line.strip().split()
                        # Now expecting 11 values: class_id, bbox(4), distance, angle, obj_x, obj_y, ego_x, ego_y
                        if len(vals) == 11:
                            class_id = int(vals[0])
                            bbox = [float(x) for x in vals[1:5]]
                            distance = float(vals[5])
                            angle = float(vals[6])
                            x = float(vals[7])
                            y = float(vals[8])
                            ego_x = float(vals[9])
                            ego_y = float(vals[10])
                            # Normalize x, y, ego_x, ego_y
                            x_norm = (x - X_MEAN) / X_STD
                            y_norm = (y - Y_MEAN) / Y_STD
                            ego_x_norm = (ego_x - X_MEAN) / X_STD
                            ego_y_norm = (ego_y - Y_MEAN) / Y_STD
                            # Use sin/cos representation for angle
                            target = [distance, np.sin(angle), np.cos(angle), x_norm, y_norm]
                            self.samples.append((class_id, bbox, ego_x_norm, ego_y_norm, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_id, bbox, ego_x_norm, ego_y_norm, target = self.samples[idx]
        return (
            torch.tensor(class_id, dtype=torch.long),
            torch.tensor(bbox, dtype=torch.float32),
            torch.tensor([ego_x_norm, ego_y_norm], dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )

# === MODEL ===
class DistanceAngleRegressor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, 8)  # Embedding for class_id
        self.fc = nn.Sequential(
            nn.Linear(8 + 4 + 2, 128),  # +2 for ego_x, ego_y
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # distance, sin(angle), cos(angle), x, y
        )

    def forward(self, class_id, bbox, ego_xy):
        # class_id: (batch,)
        # bbox: (batch, 4)
        # ego_xy: (batch, 2)
        class_feat = self.class_embed(class_id)
        x = torch.cat([class_feat, bbox, ego_xy], dim=1)
        return self.fc(x)

# === TRAINING ===
def train():
    dataset = BBoxDistanceAngleDataset(LABEL_DIR)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = DistanceAngleRegressor(NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for class_id, bbox, ego_xy, target in dataloader:
            pred = model(class_id, bbox, ego_xy)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * class_id.size(0)
        avg_loss = running_loss / len(dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    # Save model
    torch.save(model.state_dict(), 'distance_angle_regressor.pth')
    print('Training complete. Model saved as distance_angle_regressor.pth')
    print('Note: At inference, recover angle as atan2(sin, cos)')

if __name__ == '__main__':
    train() 