#!/usr/bin/env python3
"""
Complete Bird Detection Setup Verification Script
Checks all components: virtual environment, dependencies, project structure,
model files, and training scripts
"""

import os
import sys
import subprocess
from pathlib import Path
import importlib


class SetupVerifier:
    """Comprehensive setup verification"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
        
    def print_header(self, text):
        """Print section header"""
        print(f"\n{'='*60}")
        print(f"🔍 {text}")
        print('='*60)
        
    def check_item(self, description, check_func, critical=True):
        """Check an individual item"""
        self.total_checks += 1
        print(f"\n🔧 Checking: {description}")
        
        try:
            result = check_func()
            if result:
                print(f"✅ {description} - OK")
                self.success_count += 1
                return True
            else:
                if critical:
                    print(f"❌ {description} - FAILED")
                    self.errors.append(description)
                else:
                    print(f"⚠️  {description} - Missing (optional)")
                    self.warnings.append(description)
                return False
        except Exception as e:
            if critical:
                print(f"❌ {description} - ERROR: {e}")
                self.errors.append(f"{description}: {e}")
            else:
                print(f"⚠️  {description} - WARNING: {e}")
                self.warnings.append(f"{description}: {e}")
            return False
    
    def check_virtual_environment(self):
        """Check if virtual environment is active and properly set up"""
        # Check if we're in a virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if in_venv:
            venv_path = sys.prefix
            print(f"   Virtual environment: {venv_path}")
            
            # Check if it's the expected bird_detection_env
            if "bird_detection_env" in venv_path:
                print(f"   ✅ Correct virtual environment: bird_detection_env")
                return True
            else:
                print(f"   ⚠️  Different virtual environment detected")
                return True  # Still valid, just different name
        else:
            print(f"   ❌ Not in virtual environment")
            print(f"   Current Python: {sys.executable}")
            return False
    
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major == 3 and version.minor >= 8:
            return True
        else:
            print(f"   ❌ Python 3.8+ required, found {version.major}.{version.minor}")
            return False
    
    def check_project_structure(self):
        """Check if project directory structure exists"""
        required_dirs = [
            "data/train/images",
            "data/train/labels",
            "data/val/images", 
            "data/val/labels",
            "data/test/images",
            "data/test/labels",
            "scripts",
            "configs",
            "runs",
            "models",
            "notebooks",
            "evaluation_results",
            "visualization_results"
        ]
        
        missing_dirs = []
        for directory in required_dirs:
            if not Path(directory).exists():
                missing_dirs.append(directory)
            else:
                print(f"   ✅ {directory}")
        
        if missing_dirs:
            print(f"   ❌ Missing directories: {missing_dirs}")
            return False
        
        print(f"   ✅ All {len(required_dirs)} directories present")
        return True
    
    def check_core_dependencies(self):
        """Check if core Python packages are installed"""
        required_packages = [
            ('torch', 'PyTorch'),
            ('torchvision', 'TorchVision'),
            ('ultralytics', 'Ultralytics YOLO'),
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('matplotlib', 'Matplotlib'),
            ('sklearn', 'Scikit-learn'),
            ('PIL', 'Pillow'),
            ('yaml', 'PyYAML'),
            ('tqdm', 'TQDM'),
        ]
        
        missing_packages = []
        installed_packages = []
        
        for package, name in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                    version = cv2.__version__
                elif package == 'PIL':
                    from PIL import Image
                    version = Image.__version__ if hasattr(Image, '__version__') else 'Unknown'
                elif package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                elif package == 'yaml':
                    import yaml
                    version = getattr(yaml, '__version__', 'Unknown')
                else:
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'Unknown')
                
                installed_packages.append((name, version))
                print(f"   ✅ {name}: {version}")
                
            except ImportError:
                missing_packages.append(name)
                print(f"   ❌ {name}: Not installed")
        
        if missing_packages:
            print(f"   ❌ Missing packages: {missing_packages}")
            return False
        
        print(f"   ✅ All {len(required_packages)} core packages installed")
        return True
    
    def check_optional_dependencies(self):
        """Check optional packages"""
        optional_packages = [
            ('wandb', 'Weights & Biases'),
            ('seaborn', 'Seaborn'),
            ('jupyter', 'Jupyter'),
            ('roboflow', 'Roboflow'),
        ]
        
        installed = []
        missing = []
        
        for package, name in optional_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
                installed.append((name, version))
                print(f"   ✅ {name}: {version}")
            except ImportError:
                missing.append(name)
                print(f"   ⚠️  {name}: Not installed (optional)")
        
        print(f"   ✅ {len(installed)} optional packages installed")
        if missing:
            print(f"   ⚠️  {len(missing)} optional packages missing: {missing}")
        
        return True  # Optional packages don't cause failure
    
    def check_gpu_availability(self):
        """Check GPU and CUDA availability"""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_gpu = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_gpu)
                gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1e9
                
                print(f"   ✅ CUDA available")
                print(f"   ✅ GPU count: {gpu_count}")
                print(f"   ✅ Current GPU: {gpu_name}")
                print(f"   ✅ GPU Memory: {gpu_memory:.1f} GB")
                
                # Check CUDA version
                cuda_version = torch.version.cuda
                print(f"   ✅ CUDA version: {cuda_version}")
                
                return True
            else:
                print(f"   ⚠️  CUDA not available, will use CPU")
                return True  # Not critical, can work with CPU
                
        except Exception as e:
            print(f"   ❌ GPU check failed: {e}")
            return False
    
    def check_yolo_model(self):
        """Check if YOLO model is available and working"""
        model_locations = [
            "yolo11n.pt",          # Root directory
            "models/yolo11n.pt",   # Models directory
            "yolov11n.pt",         # Alternative naming
            "models/yolov11n.pt",  # Alternative in models
            "yolov8n.pt",          # YOLOv8 fallback
            "models/yolov8n.pt",   # YOLOv8 in models
        ]
        
        found_models = []
        for model_path in model_locations:
            if Path(model_path).exists():
                size_mb = Path(model_path).stat().st_size / (1024 * 1024)
                found_models.append((model_path, size_mb))
                print(f"   ✅ Found: {model_path} ({size_mb:.1f} MB)")
        
        if not found_models:
            print(f"   ❌ No YOLO models found in expected locations")
            print(f"   Expected locations: {model_locations}")
            return False
        
        # Test loading the first model
        try:
            from ultralytics import YOLO
            
            model_path = found_models[0][0]
            print(f"   🧪 Testing model: {model_path}")
            
            model = YOLO(model_path)
            print(f"   ✅ Model loaded successfully")
            
            # Quick inference test
            import numpy as np
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = model(test_image, verbose=False)
            
            print(f"   ✅ Inference test passed")
            return True
            
        except Exception as e:
            print(f"   ❌ Model test failed: {e}")
            return False
    
    def check_config_files(self):
        """Check if configuration files exist"""
        config_files = [
            ("requirements.txt", "Dependencies list"),
            ("configs/yolov11_birds.yaml", "Dataset configuration"),
            (".gitignore", "Git ignore file"),
            ("activate_env.sh", "Environment activation script"),
        ]
        
        missing_configs = []
        
        for file_path, description in config_files:
            if Path(file_path).exists():
                size_kb = Path(file_path).stat().st_size / 1024
                print(f"   ✅ {description}: {file_path} ({size_kb:.1f} KB)")
            else:
                missing_configs.append((file_path, description))
                print(f"   ❌ Missing: {description} ({file_path})")
        
        if missing_configs:
            print(f"   ❌ {len(missing_configs)} config files missing")
            return False
        
        print(f"   ✅ All {len(config_files)} config files present")
        return True
    
    def check_training_scripts(self):
        """Check if training scripts are present"""
        script_files = [
            ("train.py", "Main training script"),
            ("evaluate.py", "Model evaluation script"),
            ("inference.py", "Inference script"),
            ("scripts/download_dataset.py", "Dataset download script"),
            ("scripts/visualize_results.py", "Results visualization script"),
        ]
        
        missing_scripts = []
        present_scripts = []
        
        for file_path, description in script_files:
            if Path(file_path).exists():
                size_kb = Path(file_path).stat().st_size / 1024
                present_scripts.append((file_path, description))
                print(f"   ✅ {description}: {file_path} ({size_kb:.1f} KB)")
            else:
                missing_scripts.append((file_path, description))
                print(f"   ❌ Missing: {description} ({file_path})")
        
        print(f"   📊 Scripts present: {len(present_scripts)}/{len(script_files)}")
        
        if missing_scripts:
            print(f"   ⚠️  Missing scripts: {len(missing_scripts)}")
            for file_path, description in missing_scripts:
                print(f"      - {description} ({file_path})")
        
        return len(present_scripts) > 0  # At least some scripts should be present
    
    def check_dataset_structure(self):
        """Check dataset structure (expected to be empty initially)"""
        data_dirs = ["data/train/images", "data/val/images", "data/test/images"]
        
        total_images = 0
        for data_dir in data_dirs:
            if Path(data_dir).exists():
                image_files = list(Path(data_dir).glob("*.jpg")) + list(Path(data_dir).glob("*.png"))
                total_images += len(image_files)
                print(f"   📁 {data_dir}: {len(image_files)} images")
        
        if total_images == 0:
            print(f"   ℹ️  No dataset images found (expected before download)")
        else:
            print(f"   ✅ Total images found: {total_images}")
        
        return True  # Always return True, dataset is optional at this stage
    
    def run_verification(self):
        """Run complete verification"""
        print("🐦 Bird Detection Setup Verification")
        print("=" * 60)
        
        # Core system checks
        self.print_header("SYSTEM ENVIRONMENT")
        self.check_item("Python Version", self.check_python_version, critical=True)
        self.check_item("Virtual Environment", self.check_virtual_environment, critical=True)
        self.check_item("GPU/CUDA Availability", self.check_gpu_availability, critical=False)
        
        # Project structure
        self.print_header("PROJECT STRUCTURE")
        self.check_item("Directory Structure", self.check_project_structure, critical=True)
        self.check_item("Configuration Files", self.check_config_files, critical=False)
        
        # Dependencies
        self.print_header("DEPENDENCIES")
        self.check_item("Core Python Packages", self.check_core_dependencies, critical=True)
        self.check_item("Optional Packages", self.check_optional_dependencies, critical=False)
        
        # Models and scripts
        self.print_header("MODELS AND SCRIPTS")
        self.check_item("YOLO Model", self.check_yolo_model, critical=True)
        self.check_item("Training Scripts", self.check_training_scripts, critical=False)
        
        # Data
        self.print_header("DATASET")
        self.check_item("Dataset Structure", self.check_dataset_structure, critical=False)
        
        # Final report
        self.print_final_report()
    
    def print_final_report(self):
        """Print final verification report"""
        print("\n" + "="*60)
        print("📊 VERIFICATION REPORT")
        print("="*60)
        
        success_rate = (self.success_count / self.total_checks) * 100
        
        print(f"✅ Successful checks: {self.success_count}/{self.total_checks} ({success_rate:.1f}%)")
        
        if self.errors:
            print(f"\n❌ CRITICAL ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # Overall status
        print(f"\n🎯 OVERALL STATUS:")
        if len(self.errors) == 0:
            print("✅ SETUP IS READY!")
            print("   Core components are working properly")
            
            if len(self.warnings) == 0:
                print("   Perfect setup - everything is in place")
            else:
                print(f"   Minor issues: {len(self.warnings)} optional components missing")
            
            print(f"\n📋 NEXT STEPS:")
            if "Training Scripts" in [w.split(':')[0] for w in self.warnings]:
                print("   1. Copy training scripts (train.py, evaluate.py, inference.py)")
            print("   2. Download dataset: python scripts/download_dataset.py --sample")
            print("   3. Setup WandB: wandb login")
            print("   4. Start training: python train.py --epochs 10")
            
        else:
            print("❌ SETUP HAS ISSUES")
            print(f"   {len(self.errors)} critical errors need to be fixed")
            print("\n🔧 REQUIRED FIXES:")
            for error in self.errors:
                print(f"   • Fix: {error}")
        
        print(f"\n💡 HARDWARE DETECTED:")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   🔥 {gpu_name} ({gpu_memory:.1f} GB)")
                print(f"   💪 Recommended batch size: 32-64")
            else:
                print(f"   🖥️  CPU-only setup")
        except:
            pass


def main():
    """Main verification function"""
    verifier = SetupVerifier()
    verifier.run_verification()


if __name__ == "__main__":
    main()
