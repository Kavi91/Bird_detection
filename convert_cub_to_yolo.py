#!/usr/bin/env python3
"""
Convert CUB-200-2011 Dataset to YOLO Format
Converts the CUB dataset with bounding boxes to YOLO detection format
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import random


class CUBToYOLOConverter:
    """Convert CUB-200-2011 dataset to YOLO format"""
    
    def __init__(self, cub_root, output_dir="data"):
        self.cub_root = Path(cub_root)
        self.output_dir = Path(output_dir)
        
        # CUB dataset structure
        self.images_dir = self.cub_root / "images"
        self.annotations_dir = self.cub_root
        
        # Output directories
        self.train_images = self.output_dir / "train" / "images"
        self.train_labels = self.output_dir / "train" / "labels"
        self.val_images = self.output_dir / "val" / "images"
        self.val_labels = self.output_dir / "val" / "labels"
        self.test_images = self.output_dir / "test" / "images"
        self.test_labels = self.output_dir / "test" / "labels"
        
    def create_directories(self):
        """Create output directory structure"""
        print("📁 Creating output directories...")
        
        for directory in [self.train_images, self.train_labels, 
                         self.val_images, self.val_labels,
                         self.test_images, self.test_labels]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ Directories created")
    
    def verify_cub_structure(self):
        """Verify CUB dataset structure"""
        print("🔍 Verifying CUB dataset structure...")
        
        required_files = [
            "images.txt",
            "image_class_labels.txt", 
            "bounding_boxes.txt",
            "classes.txt",
            "train_test_split.txt"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.cub_root / file_name
            if file_path.exists():
                print(f"   ✅ {file_name}")
            else:
                print(f"   ❌ {file_name} - Missing")
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
        
        # Check images directory
        if self.images_dir.exists():
            image_count = len(list(self.images_dir.rglob("*.jpg")))
            print(f"   ✅ Images directory: {image_count} images found")
        else:
            print(f"   ❌ Images directory not found: {self.images_dir}")
            return False
        
        print("✅ CUB dataset structure verified")
        return True
    
    def load_cub_data(self):
        """Load CUB dataset annotations"""
        print("📊 Loading CUB annotations...")
        
        # Load images list
        images_df = pd.read_csv(
            self.cub_root / "images.txt", 
            sep=" ", 
            header=None, 
            names=["image_id", "image_path"]
        )
        
        # Load class labels
        labels_df = pd.read_csv(
            self.cub_root / "image_class_labels.txt",
            sep=" ",
            header=None,
            names=["image_id", "class_id"]
        )
        
        # Load bounding boxes
        bbox_df = pd.read_csv(
            self.cub_root / "bounding_boxes.txt",
            sep=" ",
            header=None,
            names=["image_id", "x", "y", "width", "height"]
        )
        
        # Load train/test split
        split_df = pd.read_csv(
            self.cub_root / "train_test_split.txt",
            sep=" ",
            header=None,
            names=["image_id", "is_training_image"]
        )
        
        # Load class names
        classes_df = pd.read_csv(
            self.cub_root / "classes.txt",
            sep=" ",
            header=None,
            names=["class_id", "class_name"]
        )
        
        # Merge all data
        data = images_df.merge(labels_df, on="image_id")
        data = data.merge(bbox_df, on="image_id")
        data = data.merge(split_df, on="image_id")
        
        print(f"✅ Loaded {len(data)} images with annotations")
        print(f"   Training images: {len(data[data.is_training_image == 1])}")
        print(f"   Test images: {len(data[data.is_training_image == 0])}")
        print(f"   Classes: {len(classes_df)}")
        
        return data, classes_df
    
    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """Convert CUB bounding box to YOLO format"""
        x, y, width, height = bbox
        
        # CUB format: x, y, width, height (absolute coordinates)
        # YOLO format: center_x, center_y, width, height (normalized)
        
        center_x = x + width / 2
        center_y = y + height / 2
        
        # Normalize
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height
        
        # Clamp to [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return center_x, center_y, width, height
    
    def process_image(self, row, output_images_dir, output_labels_dir):
        """Process a single image and create YOLO annotation"""
        image_path = self.images_dir / row["image_path"]
        
        if not image_path.exists():
            print(f"   ⚠️  Image not found: {image_path}")
            return False
        
        try:
            # Load image to get dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            # Convert bounding box to YOLO format
            bbox = [row["x"], row["y"], row["width"], row["height"]]
            yolo_bbox = self.convert_bbox_to_yolo(bbox, img_width, img_height)
            
            # Class ID (CUB uses 1-based indexing, YOLO uses 0-based)
            class_id = row["class_id"] - 1
            
            # Copy image
            output_image_path = output_images_dir / f"{row['image_id']:06d}.jpg"
            shutil.copy2(image_path, output_image_path)
            
            # Create YOLO label file
            output_label_path = output_labels_dir / f"{row['image_id']:06d}.txt"
            with open(output_label_path, 'w') as f:
                f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path}: {e}")
            return False
    
    def convert_dataset(self):
        """Convert entire CUB dataset to YOLO format"""
        print("🔄 Converting CUB dataset to YOLO format...")
        
        # Load CUB data
        data, classes_df = self.load_cub_data()
        
        # Split data
        train_data = data[data.is_training_image == 1].copy()
        test_data = data[data.is_training_image == 0].copy()
        
        # Further split training data into train/val (80/20)
        train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        val_split = int(0.2 * len(train_data))
        val_data = train_data[:val_split].copy()
        train_data = train_data[val_split:].copy()
        
        print(f"📊 Dataset splits:")
        print(f"   Train: {len(train_data)} images")
        print(f"   Validation: {len(val_data)} images") 
        print(f"   Test: {len(test_data)} images")
        
        # Process each split
        splits = [
            (train_data, self.train_images, self.train_labels, "training"),
            (val_data, self.val_images, self.val_labels, "validation"),
            (test_data, self.test_images, self.test_labels, "test")
        ]
        
        total_processed = 0
        total_failed = 0
        
        for split_data, images_dir, labels_dir, split_name in splits:
            print(f"\n🔄 Processing {split_name} set...")
            
            processed = 0
            failed = 0
            
            for idx, row in split_data.iterrows():
                if self.process_image(row, images_dir, labels_dir):
                    processed += 1
                else:
                    failed += 1
                
                if (processed + failed) % 1000 == 0:
                    print(f"   Progress: {processed + failed}/{len(split_data)}")
            
            print(f"   ✅ {split_name}: {processed} processed, {failed} failed")
            total_processed += processed
            total_failed += failed
        
        print(f"\n🎉 Conversion completed!")
        print(f"   Total processed: {total_processed}")
        print(f"   Total failed: {total_failed}")
        
        return classes_df
    
    def create_yaml_config(self, classes_df):
        """Create YOLO dataset configuration file"""
        print("📝 Creating YOLO configuration file...")
        
        config_content = f"""# CUB-200-2011 Bird Dataset Configuration
path: {self.output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {len(classes_df)}

# Class names
names:
"""
        
        for idx, row in classes_df.iterrows():
            class_id = row["class_id"] - 1  # Convert to 0-based
            class_name = row["class_name"].replace("_", " ")
            config_content += f"  {class_id}: '{class_name}'\n"
        
        config_path = Path("configs/cub_birds.yaml")
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Configuration saved: {config_path}")
        
        # Also create a simplified version with fewer classes for testing
        self.create_simplified_config(classes_df)
        
        return config_path
    
    def create_simplified_config(self, classes_df, num_classes=10):
        """Create simplified config with fewer classes for testing"""
        print(f"📝 Creating simplified configuration ({num_classes} classes)...")
        
        # Select most common classes
        selected_classes = classes_df.head(num_classes)
        
        config_content = f"""# CUB-200-2011 Bird Dataset Configuration (Simplified)
path: {self.output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {num_classes}

# Class names (top {num_classes} classes)
names:
"""
        
        for idx, row in selected_classes.iterrows():
            class_id = row["class_id"] - 1  # Convert to 0-based
            class_name = row["class_name"].replace("_", " ")
            config_content += f"  {class_id}: '{class_name}'\n"
        
        config_path = Path("configs/cub_birds_simple.yaml")
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Simplified configuration saved: {config_path}")
        return config_path
    
    def print_summary(self):
        """Print conversion summary"""
        print("\n" + "="*60)
        print("🎉 CUB-200-2011 TO YOLO CONVERSION COMPLETED!")
        print("="*60)
        
        # Count files
        for split in ["train", "val", "test"]:
            images_dir = self.output_dir / split / "images"
            labels_dir = self.output_dir / split / "labels"
            
            if images_dir.exists():
                num_images = len(list(images_dir.glob("*.jpg")))
                num_labels = len(list(labels_dir.glob("*.txt")))
                print(f"📊 {split.title()}: {num_images} images, {num_labels} labels")
        
        print(f"\n📁 Dataset location: {self.output_dir.absolute()}")
        print(f"⚙️  Config files:")
        print(f"   - configs/cub_birds.yaml (all 200 classes)")
        print(f"   - configs/cub_birds_simple.yaml (10 classes)")
        
        print(f"\n🚀 Ready for training:")
        print(f"   python train.py --data configs/cub_birds_simple.yaml --epochs 10")
        print(f"   python train.py --data configs/cub_birds.yaml --epochs 50")


def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CUB-200-2011 to YOLO format')
    parser.add_argument('--cub-root', required=True, type=str,
                       help='Path to CUB-200-2011 dataset root directory')
    parser.add_argument('--output-dir', default='data', type=str,
                       help='Output directory for YOLO dataset')
    
    args = parser.parse_args()
    
    print("🐦 CUB-200-2011 to YOLO Converter")
    print("=" * 50)
    
    converter = CUBToYOLOConverter(args.cub_root, args.output_dir)
    
    try:
        # Verify input dataset
        if not converter.verify_cub_structure():
            print("❌ CUB dataset verification failed")
            return
        
        # Create output directories
        converter.create_directories()
        
        # Convert dataset
        classes_df = converter.convert_dataset()
        
        # Create configuration files
        converter.create_yaml_config(classes_df)
        
        # Print summary
        converter.print_summary()
        
        print("\n🎉 Conversion successful! Ready for training.")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()