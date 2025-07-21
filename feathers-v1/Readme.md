# FeathersV1 Dataset to YOLO Format Conversion

This guide will help you convert the FeathersV1 bird feather classification dataset into YOLO object detection format for use with YOLOv8/YOLOv11.

## 📋 Prerequisites

- Python 3.7+
- Git
- Basic command line knowledge

## 🚀 Quick Start

### Step 1: Download the Dataset

```bash
# Create a working directory
mkdir ~/feathersv1-classification
cd ~/feathersv1-classification

# Create data directory and download script
mkdir data
cd data
```

Create a download script:
```bash
nano download.sh
```

Add this content:
```bash
#!/bin/bash
git clone https://github.com/feathers-dataset/feathersv1-dataset.git
```

Make it executable and run:
```bash
chmod +x download.sh
./download.sh
```

### Step 2: Install Required Dependencies

```bash
pip install pandas pillow pyyaml ultralytics
```

### Step 3: Download Conversion Scripts

Navigate to the dataset directory:
```bash
cd feathersv1-dataset
```

Create three Python files:

#### A. Dataset Examination Script (`examine_dataset.py`)
```bash
wget -O examine_dataset.py https://raw.githubusercontent.com/your-repo/examine_dataset.py
# Or copy the script content manually
```

#### B. Conversion Script (`feathers_to_yolo.py`)
```bash
# Copy the conversion script content from the artifacts
nano feathers_to_yolo.py
# Paste the complete conversion script
```

#### C. Verification Script (`verify_yolo_format.py`)
```bash
# Copy the verification script content from the artifacts  
nano verify_yolo_format.py
# Paste the complete verification script
```

### Step 4: Examine Your Dataset

```bash
python examine_dataset.py
```

This will show you:
- Available CSV files and their contents
- Number of images per category
- Dataset statistics
- Recommendations for which files to use

### Step 5: Convert to YOLO Format

Edit the conversion script to choose your preferred option:

```bash
nano feathers_to_yolo.py
```

**Available Options:**

1. **Top-50 Species (Recommended for testing)**
   ```python
   converter.convert(
       use_presplit=True,
       train_csv="train_top_50_species.csv",
       test_csv="test_top_50_species.csv"
   )
   ```
   - 10,314 images across 50 species
   - Fastest conversion and training

2. **Top-100 Species**
   ```python
   converter.convert(
       use_presplit=True,
       train_csv="train_top_100_species.csv",
       test_csv="test_top_100_species.csv"
   )
   ```
   - 14,941 images across 100 species

3. **All Species (Complete Dataset)**
   ```python
   converter.convert(csv_file="feathers_data.csv")
   ```
   - 28,272 images across 595 species
   - Most comprehensive but slowest

Run the conversion:
```bash
python feathers_to_yolo.py
```

### Step 6: Verify the Conversion

```bash
python verify_yolo_format.py
```

This will check:
- ✅ Directory structure
- ✅ YAML configuration file
- ✅ Image-label pairs
- ✅ YOLO format compliance
- ✅ YOLOv8/v11 compatibility

## 📁 Output Structure

After conversion, you'll have:

```
yolo_feathers_dataset/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── labels/
│   ├── train/          # Training annotations (.txt)
│   ├── val/            # Validation annotations (.txt)
│   └── test/           # Test annotations (.txt)
├── dataset.yaml        # YOLOv8/v11 configuration
└── class_mapping.txt   # Species to class ID mapping
```

## 🎯 Training with YOLOv8/YOLOv11

Once conversion is complete:

```bash
# Install ultralytics if not already installed
pip install ultralytics

# Train YOLOv8
yolo detect train data=~/feathersv1-classification/yolo_feathers_dataset/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640

# Train YOLOv11
yolo detect train data=~/feathersv1-classification/yolo_feathers_dataset/dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
```

## 📊 Dataset Options Comparison

| Option | Images | Species | Training Time | Use Case |
|--------|--------|---------|---------------|----------|
| Top-50 | 10,314 | 50 | Fast | Testing, prototyping |
| Top-100 | 14,941 | 100 | Medium | Balanced experiments |
| Complete | 28,272 | 595 | Slow | Full research, production |

## 🔧 Troubleshooting

### Common Issues:

**1. "CSV file not found"**
```bash
# Check if you're in the right directory
ls data/
# Should show the CSV files
```

**2. "Image not found"**
```bash
# Verify images directory exists
ls images/
# Should show order directories (accipitriformes, etc.)
```

**3. "Permission denied"**
```bash
# Make scripts executable
chmod +x *.py
```

**4. "Module not found"**
```bash
# Install missing dependencies
pip install pandas pillow pyyaml
```

### Memory Issues:
If you encounter memory issues with the complete dataset:
1. Start with Top-50 species
2. Use a machine with more RAM
3. Process in smaller batches

## 📝 Dataset Information

- **Source**: [FeathersV1 Dataset](https://github.com/feathers-dataset/feathersv1-dataset)
- **Paper**: "Feathers dataset for Fine-Grained Visual Categorization"
- **Total Images**: 28,272 feather images
- **Species**: 595 bird species
- **Organization**: Taxonomic order (Order → Species)

## 🤝 Contributing

If you find issues or improvements:
1. Check existing issues
2. Create detailed bug reports
3. Submit pull requests with fixes

## 📄 License

This conversion tool respects the original FeathersV1 dataset license. Please check the [AUTHORS](https://github.com/feathers-dataset/feathersv1-dataset/blob/master/AUTHORS) and [LICENSE](https://github.com/feathers-dataset/feathersv1-dataset/blob/master/LICENSE) files in the original dataset.

## 🔗 Useful Links

- [FeathersV1 Dataset Repository](https://github.com/feathers-dataset/feathersv1-dataset)
- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

---

**Happy Training! 🚀**

For questions or issues, please open an issue in this repository or refer to the original dataset documentation.
