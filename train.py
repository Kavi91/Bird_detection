#!/usr/bin/env python3
"""
Bird Detection Training Script with YOLOv11 and WandB Integration
University Project - Complete Training Pipeline
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import torch
import wandb
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class BirdDetectionTrainer:
    """
    Advanced trainer class for bird detection using YOLOv11
    with comprehensive metrics tracking and WandB integration
    """
    
    def __init__(self, args):
        self.args = args
        self.setup_environment()
        self.setup_wandb()
        
    def setup_environment(self):
        """Setup training environment and verify GPU availability"""
        # Set device
        if torch.cuda.is_available():
            self.device = f'cuda:{self.args.device}'
            print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = 'cpu'
            print("⚠️  CUDA not available, using CPU")
            
        # Create directories
        os.makedirs('runs', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        try:
            wandb.init(
                project="bird-detection-yolov11",
                name=f"yolov11_birds_{int(time.time())}",
                config={
                    "model": self.args.model,
                    "epochs": self.args.epochs,
                    "batch_size": self.args.batch_size,
                    "img_size": self.args.img_size,
                    "learning_rate": self.args.lr0,
                    "device": self.device,
                    "optimizer": "AdamW",
                    "architecture": "YOLOv11"
                },
                tags=["bird-detection", "yolov11", "computer-vision"],
            )
            print("✅ WandB initialized successfully")
            self.use_wandb = True
            
        except Exception as e:
            print(f"⚠️  WandB initialization failed: {e}")
            print("   Continuing without WandB logging...")
            self.use_wandb = False
    
    def verify_dataset(self):
        """Verify dataset structure and configuration"""
        config_path = Path(self.args.data)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
            
        with open(config_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Check required keys
        required_keys = ['train', 'val', 'names']
        for key in required_keys:
            if key not in data_config:
                raise ValueError(f"Missing '{key}' in dataset config")
        
        # Get base path (can be relative or absolute)
        base_path = Path(data_config.get('path', '.'))
        if not base_path.is_absolute():
            base_path = Path.cwd() / base_path
            
        # Verify paths exist
        train_path = base_path / data_config['train']
        val_path = base_path / data_config['val']
        
        print(f"   Checking training path: {train_path}")
        print(f"   Checking validation path: {val_path}")
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation data not found: {val_path}")
        
        # Count images in directories
        train_images = len(list(train_path.glob("*.jpg")) + list(train_path.glob("*.png")))
        val_images = len(list(val_path.glob("*.jpg")) + list(val_path.glob("*.png")))
        
        print(f"   Training images: {train_images}")
        print(f"   Validation images: {val_images}")
            
        num_classes = len(data_config['names'])
        print(f"✅ Dataset verified: {num_classes} classes")
        
        return data_config
    
    def setup_model(self):
        """Initialize and configure the YOLO model"""
        print(f"🔧 Loading model: {self.args.model}")
        
        # Load model
        if self.args.model.endswith('.pt'):
            # Pre-trained model
            model = YOLO(self.args.model)
        else:
            # Model architecture (will download weights)
            model = YOLO(f"{self.args.model}.pt")
            
        # Log model info
        print(f"   Model: {model.model}")
        print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        
        return model
    
    def train(self):
        """Main training function with comprehensive logging"""
        print("🚀 Starting Bird Detection Training")
        print("=" * 50)
        
        # Verify dataset
        data_config = self.verify_dataset()
        
        # Setup model
        model = self.setup_model()
        
        # Configure training arguments
        train_args = {
            'data': self.args.data,
            'epochs': self.args.epochs,
            'batch': self.args.batch_size,
            'imgsz': self.args.img_size,
            'device': self.args.device,
            'workers': self.args.workers,
            'project': self.args.project,
            'name': self.args.name,
            'exist_ok': True,
            
            # Optimization settings
            'optimizer': 'AdamW',
            'lr0': self.args.lr0,
            'lrf': self.args.lrf,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Augmentation settings
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            
            # Validation settings
            'val': True,
            'save': True,
            'save_period': self.args.save_period,
            'patience': self.args.patience,
            'plots': True,
            'verbose': True,
        }
        
        # Add WandB integration if available
        if self.use_wandb:
            # YOLO automatically integrates with WandB if it's initialized
            train_args['plots'] = True
        
        print("📊 Training Configuration:")
        for key, value in train_args.items():
            if key in ['data', 'epochs', 'batch', 'imgsz', 'lr0', 'optimizer']:
                print(f"   {key}: {value}")
        print()
        
        # Start training
        start_time = time.time()
        
        try:
            results = model.train(**train_args)
            
            training_time = time.time() - start_time
            print(f"\n✅ Training completed in {training_time/3600:.2f} hours")
            
            # Log final results to WandB
            if self.use_wandb:
                self.log_final_results(results, training_time)
                
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            if self.use_wandb:
                wandb.finish(exit_code=1)
            raise
    
    def log_final_results(self, results, training_time):
        """Log final training results to WandB"""
        try:
            # Extract metrics from results
            metrics = results.results_dict if hasattr(results, 'results_dict') else {}
            
            final_metrics = {
                "training_time_hours": training_time / 3600,
                "final_map50": metrics.get('metrics/mAP50(B)', 0),
                "final_map50_95": metrics.get('metrics/mAP50-95(B)', 0),
                "final_precision": metrics.get('metrics/precision(B)', 0),
                "final_recall": metrics.get('metrics/recall(B)', 0),
                "total_epochs": self.args.epochs,
            }
            
            wandb.log(final_metrics)
            print("📊 Final metrics logged to WandB")
            
        except Exception as e:
            print(f"⚠️  Failed to log final results: {e}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bird Detection Training with YOLOv11')
    
    # Model configuration
    parser.add_argument('--model', default='yolov11n', type=str,
                       help='Model variant (yolov11n/s/m/l/x) or path to .pt file')
    parser.add_argument('--data', default='configs/yolov11_birds.yaml', type=str,
                       help='Dataset configuration file')
    
    # Training parameters
    parser.add_argument('--epochs', default=100, type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', default=16, type=int,
                       help='Batch size for training')
    parser.add_argument('--img-size', default=640, type=int,
                       help='Input image size')
    parser.add_argument('--device', default=0, type=int,
                       help='GPU device index (0, 1, 2, ...) or cpu')
    parser.add_argument('--workers', default=8, type=int,
                       help='Number of data loading workers')
    
    # Optimization parameters
    parser.add_argument('--lr0', default=0.01, type=float,
                       help='Initial learning rate')
    parser.add_argument('--lrf', default=0.01, type=float,
                       help='Final learning rate factor')
    
    # Logging and saving
    parser.add_argument('--project', default='runs/train', type=str,
                       help='Project directory for saving results')
    parser.add_argument('--name', default='exp', type=str,
                       help='Experiment name')
    parser.add_argument('--save-period', default=-1, type=int,
                       help='Save checkpoint every n epochs (-1 to disable)')
    parser.add_argument('--patience', default=50, type=int,
                       help='Early stopping patience (epochs)')
    
    # Additional options
    parser.add_argument('--resume', default=False, action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--amp', default=True, action='store_true',
                       help='Use Automatic Mixed Precision training')
    
    return parser.parse_args()


def main():
    """Main function to run training"""
    args = parse_arguments()
    
    print("🐦 Bird Detection Training with YOLOv11")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print("=" * 50)
    
    # Initialize trainer
    trainer = BirdDetectionTrainer(args)
    
    try:
        # Start training
        results = trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print(f"Results saved to: {args.project}/{args.name}")
        print("Check WandB dashboard for detailed metrics and visualizations")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if trainer.use_wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if trainer.use_wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)
        
    finally:
        if trainer.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
