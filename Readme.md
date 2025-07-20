# Bird Detection with YOLOv11 - CUB-200-2011 Dataset

A comprehensive bird species detection system using Ultralytics YOLOv11 trained on the CUB-200-2011 dataset with Weights & Biases (WandB) integration for monitoring and evaluation.

## 🎯 Project Overview

This university project implements a state-of-the-art bird detection system capable of identifying 200 different bird species using the renowned CUB-200-2011 (Caltech-UCSD Birds 200-2011) dataset. The system uses YOLOv11 for real-time object detection with comprehensive evaluation metrics.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Kavi91/Bird_detection.git
cd Bird_detection

# Run automated setup (recommended)
chmod +x setup.sh
./setup.sh

# OR Manual setup:
# 1. Create virtual environment
python3 -m venv bird_detection_env
source bird_detection_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup WandB
wandb login

# 4. Download and prepare CUB-200-2011 dataset
python scripts/download_cub_dataset.py

# 5. Train the model
python train.py --data configs/cub_birds.yaml --epochs 100 --batch-size 32

# 6. Evaluate model
python test.py --model-path runs/train/exp/weights/best.pt

# 7. Run inference
python inference.py --source data/test/images --model runs/train/exp/weights/best.pt
```

## 📋 Project Structure

```
Bird_detection/
├── bird_detection_env/          # Virtual environment
├── data/                        # CUB-200-2011 Dataset
│   ├── CUB_200_2011/           # Original dataset
│   ├── train/
│   │   ├── images/             # Training images (4,816 images)
│   │   └── labels/             # YOLO format labels
│   ├── val/
│   │   ├── images/             # Validation images (1,206 images)
│   │   └── labels/             # YOLO format labels
│   └── test/
│       ├── images/             # Test images
│       └── labels/             # YOLO format labels
├── scripts/
│   ├── download_cub_dataset.py # CUB dataset downloader
│   ├── convert_cub_to_yolo.py  # Dataset conversion script
│   └── visualize_results.py    # Results visualization
├── configs/
│   ├── cub_birds.yaml          # Full 200-class config
│   └── cub_birds_simple.yaml   # 10-class subset config
├── models/
│   └── yolo11n.pt             # Pre-trained model weights
├── runs/                       # Training results
├── evaluation_results/         # Evaluation outputs
├── visualization_results/      # Visualization outputs
├── wandb/                      # WandB logs
├── train.py                    # Main training script
├── test.py                     # Evaluation script
├── inference.py                # Inference script
├── verify_setup.py             # Setup verification
├── requirements.txt            # Dependencies
├── activate_env.sh            # Environment activation
└── README.md                  # This file
```

## 🔧 Requirements

### Hardware Recommendations
- **GPU**: NVIDIA RTX 4070 Ti Super or better (16GB+ VRAM)
- **RAM**: 32GB+ (80GB is excellent for large batch sizes)
- **Storage**: 20GB+ free space
- **CPU**: Modern multi-core processor

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10/11
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8+ or 12.0+ (for GPU training)
- **Git**: For version control

### Minimum Requirements (CPU-only)
- **RAM**: 16GB
- **Storage**: 10GB
- Training will be significantly slower without GPU

## 📦 Installation

### Option 1: Automated Setup (Recommended)
```bash
# Make setup script executable
chmod +x setup.sh

# Run automated setup
./setup.sh

# Activate environment
./activate_env.sh
```

### Option 2: Manual Installation

1. **Create and activate virtual environment:**
```bash
python3 -m venv bird_detection_env
source bird_detection_env/bin/activate  # Linux/Mac
# OR
bird_detection_env\Scripts\activate     # Windows
```

2. **Install PyTorch (GPU version):**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. **Install other dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python verify_setup.py
```

## 📊 CUB-200-2011 Dataset Setup

The CUB-200-2011 dataset contains 11,788 images of 200 bird species with detailed annotations.

### Automatic Download (Recommended)
```bash
# Download and convert CUB dataset automatically
python scripts/download_cub_dataset.py

# This will:
# 1. Download CUB_200_2011.tgz (1.1GB)
# 2. Extract the dataset
# 3. Convert annotations to YOLO format
# 4. Split into train/val/test sets
```

### Manual Dataset Setup
If automatic download fails:

1. **Download CUB-200-2011 dataset:**
   - Visit: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
   - Download: `CUB_200_2011.tgz` (1.1GB)
   - Place in `data/` directory

2. **Extract and convert:**
```bash
cd data
tar -xzf CUB_200_2011.tgz
cd ..
python convert_cub_to_yolo.py
```

### Dataset Statistics
- **Total Images**: 11,788
- **Species**: 200 North American bird species
- **Training Set**: 4,816 images
- **Validation Set**: 1,206 images
- **Test Set**: ~5,766 images
- **Annotations**: Bounding boxes, species labels, attributes

## 🎯 Training

### Basic Training (Recommended Start)
```bash
# Quick test with 1 epoch
python train.py --data configs/cub_birds.yaml --epochs 1 --batch-size 8

# Full training
python train.py --data configs/cub_birds.yaml --epochs 100 --batch-size 32
```

### Advanced Training Options
```bash
python train.py \
    --model models/yolo11n.pt \
    --data configs/cub_birds.yaml \
    --epochs 150 \
    --batch-size 32 \
    --img-size 640 \
    --device 0 \
    --workers 8 \
    --patience 50 \
    --lr0 0.01 \
    --save-period 10
```

### Training Parameters Guide

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--model` | `yolo11n.pt` | Model size (n/s/m/l/x) |
| `--epochs` | `100-200` | Training iterations |
| `--batch-size` | `32-64` | Adjust based on GPU memory |
| `--img-size` | `640` | Input image resolution |
| `--workers` | `8-16` | Data loading processes |
| `--patience` | `50` | Early stopping patience |

### Hardware-Specific Recommendations

**RTX 4070 Ti Super (16GB VRAM):**
```bash
python train.py --data configs/cub_birds.yaml --epochs 150 --batch-size 64 --workers 16
```

**RTX 3060 (12GB VRAM):**
```bash
python train.py --data configs/cub_birds.yaml --epochs 100 --batch-size 32 --workers 8
```

**GTX 1660 (6GB VRAM):**
```bash
python train.py --data configs/cub_birds.yaml --epochs 100 --batch-size 16 --workers 4
```

**CPU Only:**
```bash
python train.py --data configs/cub_birds.yaml --epochs 50 --batch-size 8 --device cpu
```

## 📈 Model Evaluation

### Comprehensive Evaluation
```bash
# Evaluate best model
python test.py --model-path runs/train/exp/weights/best.pt --data configs/cub_birds.yaml

# Generate detailed visualizations
python visulaize_results.py --model runs/train/exp/weights/best.pt --data configs/cub_birds.yaml
```

### Evaluation Metrics

The system tracks comprehensive metrics for academic rigor:

#### Detection Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall

#### Per-Class Analysis
- Individual species performance
- Confusion matrix (200x200)
- Top/bottom performing species
- Class imbalance analysis

#### Performance Metrics
- **Inference Speed**: FPS and ms per image
- **Model Efficiency**: Parameters and model size
- **Resource Usage**: GPU memory and CPU utilization

### Expected Results (YOLOv11n baseline)

| Metric | Expected Range | Excellent |
|--------|----------------|-----------|
| mAP@0.5 | 0.75 - 0.85 | > 0.85 |
| mAP@0.5:0.95 | 0.55 - 0.70 | > 0.70 |
| Precision | 0.80 - 0.90 | > 0.90 |
| Recall | 0.70 - 0.85 | > 0.85 |
| Inference Speed | 2-5ms | < 2ms |

## 🔍 Inference and Testing

### Single Image Inference
```bash
python inference.py --source data/test/images/bird1.jpg --model runs/train/exp/weights/best.pt
```

### Batch Processing
```bash
python inference.py --source data/test/images/ --model runs/train/exp/weights/best.pt --output results/
```

### Real-time Webcam Detection
```bash
python inference.py --source 0 --model runs/train/exp/weights/best.pt
```

### Video Processing
```bash
python inference.py --source bird_video.mp4 --model runs/train/exp/weights/best.pt --output bird_detections.mp4
```

## 📊 Weights & Biases Integration

### Setup WandB
```bash
# Create account at https://wandb.ai/
wandb login
# Enter your API key when prompted
```

### WandB Features
- **Real-time Training Metrics**: Loss curves, mAP progression
- **System Monitoring**: GPU usage, memory consumption
- **Model Comparisons**: Different architectures and hyperparameters
- **Sample Predictions**: Visual validation during training
- **Experiment Organization**: Track multiple training runs

### Accessing Results
1. Visit your WandB dashboard: https://wandb.ai/
2. Navigate to project: `bird-detection-yolov11`
3. Compare experiments and analyze performance

## 🛠️ Advanced Configuration

### Dataset Configuration (configs/cub_birds.yaml)
```yaml
path: data
train: train/images
val: val/images
test: test/images

nc: 200  # Number of classes

names:
  0: '001.Black footed Albatross'
  1: '002.Laysan Albatross'
  # ... (all 200 species)
  199: '200.Common Yellowthroat'
```

### Hyperparameter Tuning
Key parameters to experiment with:

1. **Learning Rate Schedule**:
   - `--lr0`: Initial learning rate (0.001 - 0.01)
   - `--lrf`: Final learning rate factor (0.01 - 0.1)

2. **Data Augmentation**:
   - Modify in training script for specific augmentations
   - HSV adjustments, rotation, scaling

3. **Model Architecture**:
   - YOLOv11n: Fastest, smallest
   - YOLOv11s: Balanced speed/accuracy
   - YOLOv11m: Better accuracy
   - YOLOv11l/x: Best accuracy, slower

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train.py --batch-size 16
   
   # Use gradient accumulation
   python train.py --batch-size 8 --accumulate 2
   ```

2. **Dataset Path Errors**
   ```bash
   # Verify dataset structure
   ls data/train/images | head -5
   ls data/train/labels | head -5
   
   # Clear label cache
   rm -f data/train/labels.cache data/val/labels.cache
   ```

3. **Low Performance Issues**
   ```bash
   # Check for corrupt labels
   python verify_setup.py
   
   # Increase training epochs
   python train.py --epochs 200
   
   # Try larger model
   python train.py --model yolo11s.pt
   ```

4. **WandB Login Issues**
   ```bash
   wandb login --relogin
   # Or train without WandB:
   wandb offline
   python train.py
   ```

### Performance Optimization Tips

1. **Maximize GPU Usage**:
   ```bash
   # Monitor GPU utilization
   nvidia-smi -l 1
   
   # Increase batch size until GPU memory is ~90% used
   python train.py --batch-size 64
   ```

2. **Speed up Data Loading**:
   ```bash
   # Increase workers (1 per CPU core)
   python train.py --workers 16
   ```

3. **Mixed Precision Training**:
   ```bash
   # Automatic Mixed Precision (enabled by default)
   python train.py --amp
   ```

## 📚 Academic Usage

### Citation
If using this project for academic work, please cite:

```bibtex
@misc{bird_detection_cub_yolov11,
  title={CUB-200-2011 Bird Species Detection using YOLOv11},
  author={[Your Name]},
  year={2025},
  institution={[Your University]},
  note={University Project - YOLOv11 implementation for bird species detection}
}

@techreport{WahCUB_200_2011,
  Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
  Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
  Year = {2011},
  Institution = {California Institute of Technology},
  Number = {CNS-TR-2011-001}
}
```

### Research Applications
This project can be extended for:
- **Fine-grained Classification**: Species-level bird identification
- **Ecological Monitoring**: Automated bird population studies
- **Conservation Research**: Tracking endangered species
- **Computer Vision Research**: Object detection methodology

## 🤝 Contributing

For university collaboration:

1. **Fork the Repository**
   ```bash
   git clone https://github.com/[username]/Bird_detection
   cd Bird_detection
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/improvement-name
   ```

3. **Make Changes and Test**
   ```bash
   python verify_setup.py
   python train.py --epochs 1  # Quick test
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: [description of improvement]"
   git push origin feature/improvement-name
   ```

5. **Create Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Test changes with small epoch runs
- Update documentation for new features

## 📞 Support and Resources

### Getting Help
1. **Setup Issues**: Run `python verify_setup.py`
2. **Training Problems**: Check WandB dashboard for metrics
3. **Dataset Issues**: Verify with `ls data/train/images | wc -l`
4. **Performance Questions**: Review hardware recommendations

### Useful Resources
- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [CUB-200-2011 Dataset Paper](http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

### Community
- **GitHub Issues**: For bug reports and feature requests
- **University Forums**: For academic collaboration
- **WandB Community**: For experiment tracking help

---

## 🎓 Project Checklist for Students

### Setup Phase ✅
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] GPU drivers and CUDA installed (if using GPU)
- [ ] WandB account created and logged in

### Dataset Phase ✅
- [ ] CUB-200-2011 dataset downloaded
- [ ] Dataset converted to YOLO format
- [ ] Training/validation/test splits verified
- [ ] Sample images and labels inspected

### Training Phase ✅
- [ ] Quick 1-epoch test successful
- [ ] Full training started with appropriate batch size
- [ ] WandB logging configured and working
- [ ] Training progress monitored
- [ ] Best model weights saved

### Evaluation Phase ✅
- [ ] Model evaluation completed
- [ ] Metrics documented (mAP, precision, recall)
- [ ] Confusion matrix generated
- [ ] Per-class performance analyzed
- [ ] Inference speed measured

### Documentation Phase ✅
- [ ] Results documented with screenshots
- [ ] Code commented and clean
- [ ] README updated with findings
- [ ] Academic citation prepared
- [ ] Project presentation ready

---

**Happy Bird Detecting! 🐦✨**

*This project serves as a comprehensive example of modern computer vision techniques applied to biological classification, suitable for university coursework, research projects, and practical applications in ornithology and conservation.*
