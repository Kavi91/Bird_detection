#!/usr/bin/env python3
"""
Quick test script for bird detection setup
"""

from ultralytics import YOLO
import torch
import numpy as np

def main():
    print("🐦 Bird Detection Setup Test")
    print("=" * 40)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  Using CPU")
    
    # Load model
    print("\n📥 Loading model...")
    model = YOLO("yolo11n.pt")
    print("✅ Model loaded successfully")
    
    # Quick inference test
    print("\n🔍 Testing inference...")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(test_image, verbose=False)
    
    detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"✅ Inference completed: {detections} detections")
    
    print("\n🎉 Setup test passed!")
    print("\nReady for:")
    print("1. python scripts/download_dataset.py --sample")
    print("2. python train.py --epochs 10")
    print("3. python inference.py --source 0 --model yolo11n.pt")

if __name__ == "__main__":
    main()
