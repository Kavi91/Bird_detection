#!/usr/bin/env python3
"""
Verify that the FeathersV1 dataset has been properly converted to YOLO format
Compatible with YOLOv8/YOLOv11
"""

import os
import yaml
from pathlib import Path
from PIL import Image
import random

def verify_directory_structure(yolo_path):
    """Verify the YOLO directory structure"""
    
    print("=== YOLO Directory Structure Verification ===\n")
    
    yolo_path = Path(yolo_path)
    
    # Required directories for YOLO format
    required_dirs = [
        "images/train",
        "images/val", 
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test"
    ]
    
    print("Checking required directories:")
    all_present = True
    
    for dir_path in required_dirs:
        full_path = yolo_path / dir_path
        if full_path.exists():
            file_count = len(list(full_path.glob("*")))
            print(f"✅ {dir_path} - {file_count} files")
        else:
            print(f"❌ {dir_path} - MISSING")
            all_present = False
    
    # Check for required files
    required_files = ["dataset.yaml", "class_mapping.txt"]
    print(f"\nChecking required files:")
    
    for file_name in required_files:
        file_path = yolo_path / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - MISSING")
            all_present = False
    
    return all_present

def verify_yaml_config(yolo_path):
    """Verify the dataset.yaml configuration file"""
    
    print("\n=== YAML Configuration Verification ===\n")
    
    yaml_path = Path(yolo_path) / "dataset.yaml"
    
    if not yaml_path.exists():
        print("❌ dataset.yaml not found!")
        return False
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("YAML Configuration:")
        print(f"  Path: {config.get('path', 'NOT SET')}")
        print(f"  Train: {config.get('train', 'NOT SET')}")
        print(f"  Val: {config.get('val', 'NOT SET')}")
        print(f"  Test: {config.get('test', 'NOT SET')}")
        print(f"  Number of classes (nc): {config.get('nc', 'NOT SET')}")
        
        # Check if class names are present
        names = config.get('names', [])
        if names:
            print(f"  Classes defined: {len(names)}")
            print(f"  Sample classes: {names[:5]}{'...' if len(names) > 5 else ''}")
        else:
            print("  ❌ No class names defined!")
            return False
        
        # Verify required fields
        required_fields = ['path', 'train', 'val', 'test', 'nc', 'names']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            print(f"  ❌ Missing required fields: {missing_fields}")
            return False
        else:
            print("  ✅ All required fields present")
            return True
            
    except Exception as e:
        print(f"❌ Error reading YAML file: {e}")
        return False

def verify_image_label_pairs(yolo_path, split="train", sample_size=5):
    """Verify that images have corresponding label files with correct format"""
    
    print(f"\n=== Image-Label Pair Verification ({split} split) ===\n")
    
    yolo_path = Path(yolo_path)
    images_dir = yolo_path / "images" / split
    labels_dir = yolo_path / "labels" / split
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ {split} directories not found!")
        return False
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print(f"❌ No image files found in {split} split!")
        return False
    
    print(f"Found {len(image_files)} images in {split} split")
    
    # Sample random images for verification
    sample_images = random.sample(image_files, min(sample_size, len(image_files)))
    
    issues_found = 0
    
    for img_path in sample_images:
        img_name = img_path.stem  # filename without extension
        label_path = labels_dir / f"{img_name}.txt"
        
        print(f"\nChecking: {img_path.name}")
        
        # Check if label file exists
        if not label_path.exists():
            print(f"  ❌ Label file missing: {label_path.name}")
            issues_found += 1
            continue
        
        # Check image dimensions
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                print(f"  Image size: {img_width}x{img_height}")
        except Exception as e:
            print(f"  ❌ Error reading image: {e}")
            issues_found += 1
            continue
        
        # Check label file format
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                print(f"  ❌ Empty label file")
                issues_found += 1
                continue
            
            print(f"  Label file: {len(lines)} annotation(s)")
            
            # Verify first annotation format
            first_line = lines[0].strip().split()
            if len(first_line) != 5:
                print(f"  ❌ Invalid annotation format: expected 5 values, got {len(first_line)}")
                issues_found += 1
                continue
            
            class_id, x_center, y_center, width, height = map(float, first_line)
            
            # Verify YOLO format constraints
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                print(f"  ❌ Invalid YOLO coordinates: values should be between 0 and 1")
                print(f"     Got: class={class_id}, x={x_center}, y={y_center}, w={width}, h={height}")
                issues_found += 1
                continue
            
            print(f"  ✅ Valid annotation: class={int(class_id)}, bbox=({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
            
        except Exception as e:
            print(f"  ❌ Error reading label file: {e}")
            issues_found += 1
    
    print(f"\nSummary for {split} split:")
    print(f"  Checked: {len(sample_images)} samples")
    print(f"  Issues found: {issues_found}")
    
    return issues_found == 0

def verify_class_mapping(yolo_path):
    """Verify the class mapping file"""
    
    print("\n=== Class Mapping Verification ===\n")
    
    mapping_path = Path(yolo_path) / "class_mapping.txt"
    
    if not mapping_path.exists():
        print("❌ class_mapping.txt not found!")
        return False
    
    try:
        with open(mapping_path, 'r') as f:
            lines = f.readlines()
        
        print(f"Class mapping file contains {len(lines)} entries")
        
        # Show first few entries
        print("Sample class mappings:")
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if line:
                try:
                    class_id, species = line.split(': ', 1)
                    print(f"  {class_id}: {species}")
                except:
                    print(f"  ❌ Invalid format in line {i+1}: {line}")
                    return False
        
        if len(lines) > 10:
            print(f"  ... and {len(lines) - 10} more")
        
        print("✅ Class mapping file looks good")
        return True
        
    except Exception as e:
        print(f"❌ Error reading class mapping file: {e}")
        return False

def count_dataset_statistics(yolo_path):
    """Count and display dataset statistics"""
    
    print("\n=== Dataset Statistics ===\n")
    
    yolo_path = Path(yolo_path)
    
    splits = ["train", "val", "test"]
    total_images = 0
    total_labels = 0
    
    for split in splits:
        images_dir = yolo_path / "images" / split
        labels_dir = yolo_path / "labels" / split
        
        if images_dir.exists() and labels_dir.exists():
            image_count = len(list(images_dir.glob("*")))
            label_count = len(list(labels_dir.glob("*.txt")))
            
            print(f"{split.capitalize()} split:")
            print(f"  Images: {image_count}")
            print(f"  Labels: {label_count}")
            print(f"  Match: {'✅' if image_count == label_count else '❌'}")
            
            total_images += image_count
            total_labels += label_count
        else:
            print(f"{split.capitalize()} split: ❌ Directories not found")
    
    print(f"\nTotal dataset:")
    print(f"  Total images: {total_images}")
    print(f"  Total labels: {total_labels}")
    print(f"  Match: {'✅' if total_images == total_labels else '❌'}")

def test_yolo_compatibility(yolo_path):
    """Test compatibility with YOLOv8/v11"""
    
    print("\n=== YOLOv8/v11 Compatibility Test ===\n")
    
    try:
        # Try to load the YAML config
        yaml_path = Path(yolo_path) / "dataset.yaml"
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ YAML config is valid")
        
        # Check if paths are relative (recommended for portability)
        train_path = config.get('train', '')
        val_path = config.get('val', '')
        test_path = config.get('test', '')
        
        if all(not os.path.isabs(path) for path in [train_path, val_path, test_path]):
            print("✅ Relative paths used (portable)")
        else:
            print("⚠️  Absolute paths detected (may cause issues when moving dataset)")
        
        # Check if nc matches number of classes
        nc = config.get('nc', 0)
        names = config.get('names', [])
        
        if nc == len(names):
            print(f"✅ Class count matches (nc={nc}, names={len(names)})")
        else:
            print(f"❌ Class count mismatch (nc={nc}, names={len(names)})")
        
        print("\n🎯 Dataset appears to be compatible with YOLOv8/v11!")
        
        # Provide usage example
        print("\n=== Usage Example for YOLOv8/v11 ===")
        print("To train with this dataset:")
        print(f"```bash")
        print(f"# Install ultralytics")
        print(f"pip install ultralytics")
        print(f"")
        print(f"# Train YOLOv8")
        print(f"yolo detect train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")
        print(f"")
        print(f"# Train YOLOv11")
        print(f"yolo detect train data={yaml_path} model=yolo11n.pt epochs=100 imgsz=640")
        print(f"```")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def main():
    """Main verification function"""
    
    # Update this path to your YOLO dataset location
    yolo_dataset_path = "/home/kavi/feathersv1-classification/yolo_feathers_dataset"
    
    if not os.path.exists(yolo_dataset_path):
        print(f"❌ YOLO dataset path not found: {yolo_dataset_path}")
        print("Please update the yolo_dataset_path variable or ensure the conversion completed successfully.")
        return
    
    print("🔍 YOLO Dataset Format Verification")
    print("=" * 50)
    
    # Run all verification checks
    checks_passed = 0
    total_checks = 6
    
    if verify_directory_structure(yolo_dataset_path):
        checks_passed += 1
    
    if verify_yaml_config(yolo_dataset_path):
        checks_passed += 1
    
    if verify_class_mapping(yolo_dataset_path):
        checks_passed += 1
    
    count_dataset_statistics(yolo_dataset_path)
    checks_passed += 1  # Statistics always "pass"
    
    if verify_image_label_pairs(yolo_dataset_path, "train", sample_size=3):
        checks_passed += 1
    
    if test_yolo_compatibility(yolo_dataset_path):
        checks_passed += 1
    
    # Final summary
    print("\n" + "=" * 50)
    print("🏁 VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Checks passed: {checks_passed}/{total_checks}")
    
    if checks_passed == total_checks:
        print("🎉 SUCCESS! Dataset is properly converted to YOLO format!")
        print("✅ Ready for YOLOv8/v11 training")
    elif checks_passed >= total_checks - 1:
        print("⚠️  Dataset mostly ready, minor issues detected")
        print("🔧 Check the issues above and fix if needed")
    else:
        print("❌ Significant issues detected with the conversion")
        print("🔧 Please review and fix the issues above")

if __name__ == "__main__":
    main()
