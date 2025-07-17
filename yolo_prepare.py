import os
import shutil
import random
from pathlib import Path

def split_and_clean_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    """Split dataset into train/val/test sets and clean labels for YOLOv8"""
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)

    # Calculate split indices
    train_end = int(len(image_files) * train_ratio)
    val_end = train_end + int(len(image_files) * val_ratio)

    # Split files
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    # Copy files to respective directories and clean labels simultaneously
    for split, files in splits.items():
        for img_file in files:
            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, 'images', split, img_file)
            shutil.copy2(src_img, dst_img)

            # Process corresponding label file
            label_file = img_file.replace('.jpg', '.txt')
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(output_dir, 'labels', split, label_file)

            if os.path.exists(src_label):
                # Clean label directly during copy (extract only first 5 values for YOLOv8)
                with open(src_label, 'r') as infile, open(dst_label, 'w') as outfile:
                    for line in infile:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Keep only class_id and bbox coordinates (first 5 values)
                            yolo_line = ' '.join(parts[:5])
                            outfile.write(yolo_line + '\n')

    print(f"Dataset split and label cleaning complete:")
    print(f"Train: {len(splits['train'])} images")
    print(f"Val: {len(splits['val'])} images")
    print(f"Test: {len(splits['test'])} images")

# Usage
if __name__ == '__main__':
    split_and_clean_dataset(
        images_dir='/home/aly/Desktop/Mostafa/AV_Perception/output_yolo_format/images',
        labels_dir='/home/aly/Desktop/Mostafa/AV_Perception/output_yolo_format/labels',
        output_dir='/home/aly/Desktop/Mostafa/AV_Perception/yolo_dataset'
    )
