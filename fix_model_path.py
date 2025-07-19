#!/usr/bin/env python3
"""
Fix model path configuration to use manually downloaded model
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def check_downloaded_model():
    """Check if the model exists and what's available"""
    print("🔍 Checking downloaded models...")
    
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pt"))
    
    if not model_files:
        print("❌ No .pt files found in models/ directory")
        return None
    
    print("✅ Found models:")
    for model_file in model_files:
        print(f"   📁 {model_file}")
    
    # Use the first available model (should be yolo11n.pt)
    return model_files[0]


def copy_model_to_root():
    """Copy model to root directory for easier access"""
    print("📋 Setting up model for easy access...")
    
    model_path = check_downloaded_model()
    if not model_path:
        return None
    
    # Copy to root directory
    root_model_path = Path(model_path.name)
    
    if not root_model_path.exists():
        shutil.copy2(model_path, root_model_path)
        print(f"✅ Copied {model_path} to {root_model_path}")
    else:
        print(f"✅ Model already exists at {root_model_path}")
    
    return root_model_path


def test_model(model_path):
    """Test the downloaded model"""
    print(f"🧪 Testing model: {model_path}")
    
    try:
        # Load model
        model = YOLO(str(model_path))
        print("✅ Model loaded successfully")
        
        # Get model info
        print(f"   Model type: {model_path.name}")
        
        # Test with dummy inference
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        results = model(dummy_image, verbose=False)
        print("✅ Inference test successful")
        
        # Check GPU
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  GPU not available, using CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def update_config_files(model_name):
    """Update configuration files to use the correct model"""
    print("⚙️  Updating configuration files...")
    
    # Get model base name (without .pt extension)
    model_base_name = model_name.replace('.pt', '')
    
    # Update train.py default model
    train_file = Path("train.py")
    if train_file.exists():
        try:
            content = train_file.read_text()
            
            # Replace default model
            content = content.replace("default='yolov11n'", f"default='{model_base_name}'")
            content = content.replace("default='yolov8n'", f"default='{model_base_name}'")
            
            train_file.write_text(content)
            print(f"✅ Updated train.py to use {model_name}")
            
        except Exception as e:
            print(f"⚠️  Could not update train.py: {e}")
    
    # Create a model config file
    config_content = f"""# Bird Detection Model Configuration

# Working Model Information
MODEL_NAME = "{model_name}"
MODEL_PATH = "{model_name}"

# For use in scripts:
# python train.py --model {model_base_name}
# python inference.py --model {model_name}
"""
    
    with open("model_config.py", "w") as f:
        f.write(config_content)
    
    print("✅ Created model_config.py")


def create_test_script(model_name):
    """Create a simple test script"""
    test_script_content = f'''#!/usr/bin/env python3
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
        print(f"✅ GPU: {{torch.cuda.get_device_name(0)}}")
        print(f"   Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")
    else:
        print("⚠️  Using CPU")
    
    # Load model
    print("\\n📥 Loading model...")
    model = YOLO("{model_name}")
    print("✅ Model loaded successfully")
    
    # Quick inference test
    print("\\n🔍 Testing inference...")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(test_image, verbose=False)
    
    detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"✅ Inference completed: {{detections}} detections")
    
    print("\\n🎉 Setup test passed!")
    print("\\nReady for:")
    print("1. python scripts/download_dataset.py --sample")
    print("2. python train.py --epochs 10")
    print("3. python inference.py --source 0 --model {model_name}")

if __name__ == "__main__":
    main()
'''
    
    with open("test_setup.py", "w") as f:
        f.write(test_script_content)
    
    print("✅ Created test_setup.py")


def print_usage_instructions(model_name):
    """Print usage instructions"""
    print("\n" + "="*60)
    print("🎉 MODEL SETUP COMPLETED!")
    print("="*60)
    
    print(f"\n✅ Working model: {model_name}")
    print(f"📁 Location: {Path(model_name).absolute()}")
    
    print("\n📋 USAGE COMMANDS:")
    print(f"   Test setup:       python test_setup.py")
    print(f"   Download data:    python scripts/download_dataset.py --sample")
    print(f"   Train model:      python train.py --model {model_name.replace('.pt', '')} --epochs 100")
    print(f"   Evaluate:         python evaluate.py --model-path runs/train/exp/weights/best.pt")
    print(f"   Inference:        python inference.py --source 0 --model {model_name}")
    
    print("\n💡 TIPS:")
    print("   • Always activate virtual environment first: source bird_detection_env/bin/activate")
    print("   • Your RTX 4070 Ti Super is perfect for large batch sizes (32-64)")
    print("   • Use --batch-size 32 for faster training with your 80GB RAM")
    
    print("\n🎯 RECOMMENDED FIRST STEPS:")
    print("1. python test_setup.py                    # Verify everything works")
    print("2. python scripts/download_dataset.py --sample  # Get sample data")
    print("3. wandb login                             # Setup experiment tracking")
    print("4. python train.py --epochs 10 --batch-size 16  # Quick test training")


def main():
    """Main function to fix model path configuration"""
    print("🔧 Fixing Model Path Configuration")
    print("=" * 50)
    
    # Copy model to root for easier access
    model_path = copy_model_to_root()
    
    if not model_path:
        print("❌ No model found in models/ directory")
        print("Please ensure you have a .pt file in the models/ directory")
        return False
    
    # Test the model
    if not test_model(model_path):
        print("❌ Model test failed")
        return False
    
    # Update configuration
    update_config_files(model_path.name)
    
    # Create test script
    create_test_script(model_path.name)
    
    # Print instructions
    print_usage_instructions(model_path.name)
    
    return True


if __name__ == "__main__":
    main()
