#!/usr/bin/env python3
"""
CUB-200-2011 Dataset Download and Setup Script
Complete automation for downloading and preparing the CUB dataset for YOLO training
"""

import os
import sys
import requests
import tarfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
from PIL import Image
import random

class CUBDatasetDownloader:
    """
    Complete CUB-200-2011 dataset downloader and converter
    """
    
    def __init__(self, data_dir="data", force_download=False):
        self.data_dir = Path(data_dir)
        self.downloads_dir = self.data_dir / "downloads"
        self.cub_dir = self.data_dir / "CUB_200_2011"
        self.force_download = force_download
        
        # Dataset URLs (multiple mirrors for reliability)
        self.dataset_urls = [
            "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
            "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
        ]
        self.dataset_file = "CUB_200_2011.tgz"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.downloads_dir.mkdir(exist_ok=True)
        
    def download_dataset(self):
        """Download CUB-200-2011 dataset with multiple fallback URLs"""
        print("📥 Downloading CUB-200-2011 dataset...")
        print("   Dataset: Caltech-UCSD Birds 200-2011")
        print("   Size: ~1.1GB")
        print("   Images: 11,788 images of 200 bird species")
        
        dataset_path = self.downloads_dir / self.dataset_file
        
        # Check if already downloaded
        if dataset_path.exists() and not self.force_download:
            file_size = dataset_path.stat().st_size / (1024 * 1024)
            print(f"✅ Dataset already downloaded: {dataset_path} ({file_size:.1f}MB)")
            return dataset_path
        
        # Try each URL until successful
        for i, url in enumerate(self.dataset_urls):
            try:
                print(f"🔗 Attempting download from source {i+1}/{len(self.dataset_urls)}")
                print(f"   URL: {url}")
                
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(dataset_path, 'wb') as file:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc="Downloading CUB-200-2011") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                                pbar.update(len(chunk))
                
                # Verify download
                if dataset_path.stat().st_size > 1000000:  # At least 1MB
                    print(f"✅ Download completed: {dataset_path}")
                    print(f"   File size: {dataset_path.stat().st_size / (1024*1024):.1f}MB")
                    return dataset_path
                else:
                    print("⚠️  Downloaded file seems too small, trying next source...")
                    dataset_path.unlink()
                    continue
                    
            except Exception as e:
                print(f"❌ Download from source {i+1} failed: {e}")
                if dataset_path.exists():
                    dataset_path.unlink()
                
                if i < len(self.dataset_urls) - 1:
                    print("   Trying next source...")
                    continue
                else:
                    print(f"\n❌ All download sources failed!")
                    print("Manual download instructions:")
                    print("1. Visit: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html")
                    print("2. Download: CUB_200_2011.tgz")
                    print(f"3. Place in: {self.downloads_dir}/")
                    raise
    
    def extract_dataset(self, dataset_path):
        """Extract CUB dataset"""
        print("📂 Extracting CUB-200-2011 dataset...")
        
        if self.cub_dir.exists() and not self.force_download:
            print(f"✅ Dataset already extracted: {self.cub_dir}")
            return self.cub_dir
        
        try:
            # Remove existing directory if force download
            if self.cub_dir.exists() and self.force_download:
                shutil.rmtree(self.cub_dir)
            
            with tarfile.open(dataset_path, 'r:gz') as tar:
                # Extract with progress bar
                members = tar.getmembers()
                with tqdm(total=len(members), desc="Extracting files") as pbar:
                    for member in members:
                        tar.extract(member, self.data_dir)
                        pbar.update(1)
            
            print(f"✅ Extraction completed: {self.cub_dir}")
            
            # Verify extraction
            if (self.cub_dir / "images").exists():
                image_count = len(list((self.cub_dir / "images").rglob("*.jpg")))
                print(f"   Found {image_count} images")
            
            return self.cub_dir
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            raise
    
    def load_cub_metadata(self):
        """Load CUB dataset metadata"""
        print("📊 Loading CUB dataset metadata...")
        
        # Load class names
        classes_file = self.cub_dir / "classes.txt"
        if not classes_file.exists():
            raise FileNotFoundError(f"Classes file not found: {classes_file}")
        
        classes = {}
        with open(classes_file, 'r') as f:
            for line in f:
                class_id, class_name = line.strip().split(' ', 1)
                classes[int(class_id)] = class_name
        
        # Load image metadata
        images_file = self.cub_dir / "images.txt"
        if not images_file.exists():
            raise FileNotFoundError(f"Images file not found: {images_file}")
        
        images = {}
        with open(images_file, 'r') as f:
            for line in f:
                img_id, img_path = line.strip().split(' ', 1)
                images[int(img_id)] = img_path
        
        # Load bounding boxes
        bboxes_file = self.cub_dir / "bounding_boxes.txt"
        bboxes = {}
        if bboxes_file.exists():
            with open(bboxes_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    img_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]  # x, y, width, height
                    bboxes[img_id] = bbox
        
        # Load train/test split
        train_test_file = self.cub_dir / "train_test_split.txt"
        train_test = {}
        if train_test_file.exists():
            with open(train_test_file, 'r') as f:
                for line in f:
                    img_id, is_train = line.strip().split()
                    train_test[int(img_id)] = int(is_train) == 1
        
        print(f"✅ Loaded metadata for {len(images)} images, {len(classes)} classes")
        return classes, images, bboxes, train_test
    
    def convert_to_yolo_format(self):
        """Convert CUB dataset to YOLO format"""
        print("🔄 Converting CUB dataset to YOLO format...")
        
        # Load metadata
        classes, images, bboxes, train_test = self.load_cub_metadata()
        
        # Create output directories
        train_img_dir = self.data_dir / "train" / "images"
        train_lbl_dir = self.data_dir / "train" / "labels"
        val_img_dir = self.data_dir / "val" / "images"
        val_lbl_dir = self.data_dir / "val" / "labels"
        test_img_dir = self.data_dir / "test" / "images"
        test_lbl_dir = self.data_dir / "test" / "labels"
        
        for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, test_img_dir, test_lbl_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Conversion statistics
        stats = {
            'train': {'images': 0, 'labels': 0},
            'val': {'images': 0, 'labels': 0},
            'test': {'images': 0, 'labels': 0}
        }
        
        # Split training data into train/val (80/20)
        train_images = [img_id for img_id, is_train in train_test.items() if is_train]
        test_images = [img_id for img_id, is_train in train_test.items() if not is_train]
        
        # Further split training into train/val
        random.shuffle(train_images)
        split_idx = int(0.8 * len(train_images))
        actual_train = train_images[:split_idx]
        val_images = train_images[split_idx:]
        
        print(f"📊 Dataset splits:")
        print(f"   Training: {len(actual_train)} images")
        print(f"   Validation: {len(val_images)} images")
        print(f"   Test: {len(test_images)} images")
        
        # Convert images and labels
        for img_id, img_path in tqdm(images.items(), desc="Converting images"):
            src_img_path = self.cub_dir / "images" / img_path
            
            if not src_img_path.exists():
                continue
            
            # Determine split
            if img_id in actual_train:
                split = 'train'
                dst_img_dir = train_img_dir
                dst_lbl_dir = train_lbl_dir
            elif img_id in val_images:
                split = 'val'
                dst_img_dir = val_img_dir
                dst_lbl_dir = val_lbl_dir
            else:
                split = 'test'
                dst_img_dir = test_img_dir
                dst_lbl_dir = test_lbl_dir
            
            # Copy image with new name
            img_name = f"{img_id:06d}.jpg"
            dst_img_path = dst_img_dir / img_name
            
            try:
                # Copy and verify image
                shutil.copy2(src_img_path, dst_img_path)
                
                # Verify image can be opened
                with Image.open(dst_img_path) as img:
                    width, height = img.size
                
                stats[split]['images'] += 1
                
                # Create YOLO label if bounding box exists
                if img_id in bboxes:
                    bbox = bboxes[img_id]
                    x, y, w, h = bbox
                    
                    # Convert to YOLO format (normalized center coordinates)
                    center_x = (x + w/2) / width
                    center_y = (y + h/2) / height
                    norm_w = w / width
                    norm_h = h / height
                    
                    # Get class ID (CUB classes are 1-indexed, YOLO needs 0-indexed)
                    class_name = img_path.split('/')[0]
                    class_id = None
                    for cid, cname in classes.items():
                        if cname == class_name:
                            class_id = cid - 1  # Convert to 0-indexed
                            break
                    
                    if class_id is not None:
                        # Write YOLO label
                        lbl_path = dst_lbl_dir / f"{img_id:06d}.txt"
                        with open(lbl_path, 'w') as f:
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                        
                        stats[split]['labels'] += 1
                
            except Exception as e:
                print(f"⚠️  Error processing {img_path}: {e}")
                continue
        
        # Print conversion statistics
        print("\n✅ Conversion completed!")
        print("📊 Conversion statistics:")
        for split, counts in stats.items():
            print(f"   {split.title()}: {counts['images']} images, {counts['labels']} labels")
        
        total_images = sum(s['images'] for s in stats.values())
        total_labels = sum(s['labels'] for s in stats.values())
        print(f"   Total: {total_images} images, {total_labels} labels")
        
        return stats
    
    def verify_dataset(self):
        """Verify converted dataset"""
        print("🔍 Verifying converted dataset...")
        
        # Check directory structure
        required_dirs = [
            "data/train/images", "data/train/labels",
            "data/val/images", "data/val/labels", 
            "data/test/images", "data/test/labels"
        ]
        
        verification_results = {}
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                if 'images' in dir_path:
                    count = len(list(path.glob("*.jpg")))
                else:
                    count = len(list(path.glob("*.txt")))
                verification_results[dir_path] = count
                print(f"✅ {dir_path}: {count} files")
            else:
                verification_results[dir_path] = 0
                print(f"❌ Missing: {dir_path}")
        
        # Verify label format
        sample_label = Path("data/train/labels").glob("*.txt")
        try:
            sample_file = next(sample_label)
            with open(sample_file, 'r') as f:
                line = f.readline().strip()
                parts = line.split()
                if len(parts) == 5:
                    class_id, x, y, w, h = parts
                    class_id = int(class_id)
                    coords = [float(c) for c in [x, y, w, h]]
                    if 0 <= class_id <= 199 and all(0 <= c <= 1 for c in coords):
                        print("✅ Label format verification passed")
                    else:
                        print("⚠️  Label format might have issues")
                else:
                    print("⚠️  Unexpected label format")
        except Exception as e:
            print(f"⚠️  Could not verify label format: {e}")
        
        # Summary
        train_images = verification_results.get("data/train/images", 0)
        val_images = verification_results.get("data/val/images", 0)
        test_images = verification_results.get("data/test/images", 0)
        total_images = train_images + val_images + test_images
        
        print(f"\n📊 Final Dataset Summary:")
        print(f"   Training images: {train_images}")
        print(f"   Validation images: {val_images}")
        print(f"   Test images: {test_images}")
        print(f"   Total images: {total_images}")
        print(f"   Classes: 200 (CUB-200-2011 bird species)")
        
        if total_images > 10000:
            print("✅ Dataset appears to be properly converted!")
            return True
        else:
            print("⚠️  Dataset might be incomplete. Expected ~11,788 images.")
            return False
    
    def create_sample_visualization(self):
        """Create sample visualization of the dataset"""
        print("🎨 Creating sample visualization...")
        
        try:
            import matplotlib.pyplot as plt
            
            # Find some sample images
            train_img_dir = Path("data/train/images")
            sample_images = list(train_img_dir.glob("*.jpg"))[:9]
            
            if len(sample_images) >= 9:
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                fig.suptitle('CUB-200-2011 Dataset Samples', fontsize=16)
                
                for i, img_path in enumerate(sample_images):
                    row, col = i // 3, i % 3
                    
                    # Load image
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Load corresponding label
                    lbl_path = Path("data/train/labels") / f"{img_path.stem}.txt"
                    if lbl_path.exists():
                        with open(lbl_path, 'r') as f:
                            line = f.readline().strip()
                            if line:
                                parts = line.split()
                                class_id = int(parts[0])
                                center_x, center_y, w, h = [float(x) for x in parts[1:5]]
                                
                                # Convert back to pixel coordinates
                                img_h, img_w = img_rgb.shape[:2]
                                x1 = int((center_x - w/2) * img_w)
                                y1 = int((center_y - h/2) * img_h)
                                x2 = int((center_x + w/2) * img_w)
                                y2 = int((center_y + h/2) * img_h)
                                
                                # Draw bounding box
                                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    axes[row, col].imshow(img_rgb)
                    axes[row, col].set_title(f"Image {img_path.stem}")
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.savefig("data/dataset_samples.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                print("✅ Sample visualization saved to: data/dataset_samples.png")
            
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download and setup CUB-200-2011 dataset for YOLO training')
    parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')
    parser.add_argument('--force-download', action='store_true', help='Force re-download even if files exist')
    parser.add_argument('--skip-download', action='store_true', help='Skip download, only convert existing data')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing dataset')
    
    args = parser.parse_args()
    
    print("🐦 CUB-200-2011 Dataset Setup for YOLO Training")
    print("=" * 60)
    print("Dataset: Caltech-UCSD Birds 200-2011")
    print("Classes: 200 North American bird species")
    print("Images: 11,788 total images")
    print("Task: Object detection (bounding box prediction)")
    print("=" * 60)
    
    downloader = CUBDatasetDownloader(args.data_dir, args.force_download)
    
    try:
        if args.verify_only:
            # Only verify existing dataset
            success = downloader.verify_dataset()
            if success:
                print("\n🎉 Dataset verification completed successfully!")
            else:
                print("\n⚠️  Dataset verification found issues.")
            return
        
        # Download dataset (unless skipped)
        if not args.skip_download:
            dataset_path = downloader.download_dataset()
            
            # Extract dataset
            cub_path = downloader.extract_dataset(dataset_path)
        
        # Convert to YOLO format
        print("\n" + "="*50)
        conversion_stats = downloader.convert_to_yolo_format()
        
        # Verify converted dataset
        print("\n" + "="*50)
        success = downloader.verify_dataset()
        
        # Create sample visualization
        downloader.create_sample_visualization()
        
        if success:
            print("\n🎉 CUB-200-2011 dataset setup completed successfully!")
            print("\n📋 Next steps:")
            print("1. Activate your environment: ./activate_env.sh")
            print("2. Setup WandB: wandb login")
            print("3. Start training: python train.py --data configs/cub_birds.yaml --epochs 100 --batch-size 32")
            print("4. Monitor training: Check WandB dashboard")
            print("5. Evaluate model: python test.py --model-path runs/train/exp/weights/best.pt")
            
            print(f"\n📊 Dataset ready for training:")
            print(f"   Configuration: configs/cub_birds.yaml")
            print(f"   Training command: python train.py --data configs/cub_birds.yaml")
            print(f"   Expected training time: 2-4 hours (RTX 4070 Ti Super)")
            print(f"   Expected mAP@0.5: 0.75-0.90")
        else:
            print("\n⚠️  Dataset setup completed with warnings.")
            print("Please check the verification results above.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\n🔧 Manual setup instructions:")
        print("1. Download CUB_200_2011.tgz from: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html")
        print("2. Place in: data/downloads/")
        print("3. Run: python scripts/download_cub_dataset.py --skip-download")
        sys.exit(1)

if __name__ == "__main__":
    main()