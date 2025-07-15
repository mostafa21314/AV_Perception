from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_yolo_model(data_yaml='dataset.yaml', model_size='yolov8n', epochs=100, batch_size=16):
    """Train YOLOv8 model and save only .pt files"""
    
    # Create output directory
    output_dir = Path('trained_models')
    output_dir.mkdir(exist_ok=True)
    
    # Load pre-trained model
    model = YOLO(f'{model_size}.pt')
    
    print(f"Starting training with {model_size} model...")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=0,
        workers=8,
        project='runs/detect',
        name='nuscenes_yolo',
        save=True,
        save_period=10,
        cache=False,
        augment=True,
        lr0=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=50,
        close_mosaic=10,
        resume=False,
        amp=True,
        val=True,
    )
    
    # Validate the model
    metrics = model.val()
    
    # Save only the best model for inference
    best_model_path = Path(f'runs/detect/nuscenes_yolo/weights/best.pt')
    if best_model_path.exists():
        final_model_path = output_dir / f'nuscenes_{model_size}_best.pt'
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        print(f"Best model saved to: {final_model_path}")
    
    print(f"Training completed!")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return model, results, metrics

if __name__ == '__main__':
    model, results, metrics = train_yolo_model(
        data_yaml='dataset.yaml',
        model_size='yolov8n',
        epochs=100,
        batch_size=16
    )
