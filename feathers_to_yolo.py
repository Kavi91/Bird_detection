#!/usr/bin/env python3
"""
Convert FeathersV1 Dataset to YOLO format

This script converts the FeathersV1 classification dataset to YOLO object detection format.
Since the original dataset contains pre-cropped single feather images, we create 
full-image bounding boxes for each feather.

Requirements:
- pandas
- PIL (Pillow)
- os, shutil

Dataset structure expected:
- data/ folder with CSV files
- images/ folder with subdirectories organized by Order/Species
"""

import os
import pandas as pd
import shutil
from PIL import Image
from pathlib import Path
import yaml

class FeathersToYOLOConverter:
    def __init__(self, dataset_path, output_path):
        """
        Initialize the converter
        
        Args:
            dataset_path (str): Path to the FeathersV1 dataset root
            output_path (str): Path where YOLO dataset will be created
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.images_path = self.dataset_path / "images"
        self.data_path = self.dataset_path / "data"
        
        # Create output directories
        self.create_output_structure()
        
    def create_output_structure(self):
        """Create YOLO dataset directory structure"""
        directories = [
            self.output_path / "images" / "train",
            self.output_path / "images" / "val", 
            self.output_path / "images" / "test",
            self.output_path / "labels" / "train",
            self.output_path / "labels" / "val",
            self.output_path / "labels" / "test"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def load_csv_data(self, csv_file="all_species.csv"):
        """
        Load the CSV data containing image filenames and labels
        
        Args:
            csv_file (str): Name of the CSV file to load
            
        Returns:
            pd.DataFrame: DataFrame with image data
        """
        csv_path = self.data_path / csv_file
        if not csv_path.exists():
            # Try to find any CSV file in the data directory
            csv_files = list(self.data_path.glob("*.csv"))
            if csv_files:
                csv_path = csv_files[0]
                print(f"Using CSV file: {csv_path}")
            else:
                raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        df = pd.read_csv(csv_path)
        return df
    
    def create_class_mapping(self, df):
        """
        Create mapping from species names to class IDs
        
        Args:
            df (pd.DataFrame): DataFrame with species information
            
        Returns:
            dict: Mapping from species to class ID
            list: List of class names
        """
        unique_species = sorted(df['species'].unique())
        class_mapping = {species: idx for idx, species in enumerate(unique_species)}
        return class_mapping, unique_species
    
    def get_image_dimensions(self, image_path):
        """
        Get image dimensions
        
        Args:
            image_path (Path): Path to image file
            
        Returns:
            tuple: (width, height)
        """
        try:
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None, None
    
    def create_yolo_annotation(self, class_id, img_width, img_height, margin=0.05):
        """
        Create YOLO annotation for full image bounding box
        
        Args:
            class_id (int): Class ID for the object
            img_width (int): Image width
            img_height (int): Image height
            margin (float): Margin to leave around the edges (0.05 = 5%)
            
        Returns:
            str: YOLO format annotation line
        """
        # Create bounding box that covers most of the image with small margin
        x_center = 0.5
        y_center = 0.5
        bbox_width = 1.0 - (2 * margin)
        bbox_height = 1.0 - (2 * margin)
        
        return f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
    
    def find_image_file(self, filename):
        """
        Find the actual image file in the images directory structure
        
        Args:
            filename (str): Filename from CSV
            
        Returns:
            Path or None: Full path to image file if found
        """
        # The images are organized in Order/Species subdirectories
        for image_file in self.images_path.rglob("*.jpg"):
            if image_file.name == filename:
                return image_file
        
        # Also try different extensions
        for image_file in self.images_path.rglob("*.jpeg"):
            if image_file.stem == Path(filename).stem:
                return image_file
                
        for image_file in self.images_path.rglob("*.png"):
            if image_file.stem == Path(filename).stem:
                return image_file
                
        return None
    
    def split_dataset(self, df, train_ratio=0.8, val_ratio=0.1):
        """
        Split dataset into train/val/test sets
        
        Args:
            df (pd.DataFrame): DataFrame with image data
            train_ratio (float): Ratio for training set
            val_ratio (float): Ratio for validation set
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Shuffle the dataframe
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_total = len(df_shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = df_shuffled[:n_train]
        val_df = df_shuffled[n_train:n_train + n_val]
        test_df = df_shuffled[n_train + n_val:]
        
        return train_df, val_df, test_df
    
    def process_split(self, df, split_name, class_mapping):
        """
        Process one data split (train/val/test)
        
        Args:
            df (pd.DataFrame): DataFrame for this split
            split_name (str): Name of the split ('train', 'val', 'test')
            class_mapping (dict): Mapping from species to class ID
        """
        processed_count = 0
        error_count = 0
        
        for idx, row in df.iterrows():
            filename = row['filename']
            species = row['species']
            class_id = class_mapping[species]
            
            # Find the actual image file
            image_path = self.find_image_file(filename)
            if image_path is None:
                print(f"Image not found: {filename}")
                error_count += 1
                continue
            
            # Get image dimensions
            img_width, img_height = self.get_image_dimensions(image_path)
            if img_width is None:
                error_count += 1
                continue
            
            # Copy image to YOLO structure
            new_image_name = f"{Path(filename).stem}.jpg"
            new_image_path = self.output_path / "images" / split_name / new_image_name
            
            try:
                shutil.copy2(image_path, new_image_path)
            except Exception as e:
                print(f"Error copying image {filename}: {e}")
                error_count += 1
                continue
            
            # Create YOLO annotation
            annotation = self.create_yolo_annotation(class_id, img_width, img_height)
            
            # Save annotation file
            annotation_path = self.output_path / "labels" / split_name / f"{Path(filename).stem}.txt"
            with open(annotation_path, 'w') as f:
                f.write(annotation)
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images for {split_name} split")
        
        print(f"Completed {split_name} split: {processed_count} images processed, {error_count} errors")
    
    def create_yaml_config(self, class_names):
        """
        Create YOLO dataset configuration file
        
        Args:
            class_names (list): List of class names
        """
        config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = self.output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created dataset config: {yaml_path}")
    
    def convert(self, csv_file="feathers_data.csv", use_presplit=False, train_csv=None, test_csv=None):
        """
        Main conversion function
        
        Args:
            csv_file (str): Name of the CSV file to process (if not using pre-split)
            use_presplit (bool): Whether to use pre-split train/test CSVs
            train_csv (str): Training CSV filename (if using pre-split)
            test_csv (str): Test CSV filename (if using pre-split)
        """
        if use_presplit and train_csv and test_csv:
            print("Using pre-split dataset...")
            print(f"Loading training data from {train_csv}...")
            train_df = self.load_csv_data(train_csv)
            print(f"Loaded {len(train_df)} training records")
            
            print(f"Loading test data from {test_csv}...")
            test_df = self.load_csv_data(test_csv)
            print(f"Loaded {len(test_df)} test records")
            
            # Combine for class mapping
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            print("Creating class mapping...")
            class_mapping, class_names = self.create_class_mapping(combined_df)
            print(f"Found {len(class_names)} unique species")
            
            # Split training data into train/val (80/20)
            train_size = int(len(train_df) * 0.8)
            train_df_shuffled = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            final_train_df = train_df_shuffled[:train_size]
            val_df = train_df_shuffled[train_size:]
            
            print(f"Split sizes - Train: {len(final_train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
        else:
            print("Loading CSV data...")
            df = self.load_csv_data(csv_file)
            print(f"Loaded {len(df)} image records")
            
            print("Creating class mapping...")
            class_mapping, class_names = self.create_class_mapping(df)
            print(f"Found {len(class_names)} unique species")
            
            print("Splitting dataset...")
            final_train_df, val_df, test_df = self.split_dataset(df)
            print(f"Split sizes - Train: {len(final_train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        print("Processing training set...")
        self.process_split(final_train_df, "train", class_mapping)
        
        print("Processing validation set...")
        self.process_split(val_df, "val", class_mapping)
        
        print("Processing test set...")
        self.process_split(test_df, "test", class_mapping)
        
        print("Creating YAML configuration...")
        self.create_yaml_config(class_names)
        
        # Save class mapping for reference
        mapping_path = self.output_path / "class_mapping.txt"
        with open(mapping_path, 'w') as f:
            for species, class_id in class_mapping.items():
                f.write(f"{class_id}: {species}\n")
        
        print(f"Conversion completed! YOLO dataset saved to: {self.output_path}")
        print(f"Class mapping saved to: {mapping_path}")
        
        # Print summary statistics
        print(f"\n=== Dataset Summary ===")
        print(f"Total species: {len(class_names)}")
        print(f"Training images: {len(final_train_df)}")
        print(f"Validation images: {len(val_df)}")
        print(f"Test images: {len(test_df)}")
        print(f"Total images: {len(final_train_df) + len(val_df) + len(test_df)}")

def main():
    """
    Example usage of the converter with different options
    """
    # Update these paths to match your setup
    dataset_path = "/home/kavi/feathersv1-classification/data/feathersv1-dataset"
    output_path = "/home/kavi/feathersv1-classification/yolo_feathers_dataset"
    
    # Create converter
    converter = FeathersToYOLOConverter(dataset_path, output_path)
    
    # Option 1: Use complete dataset (595 species, 28,272 images)
    print("=== OPTION 1: Complete Dataset ===")
    # converter.convert(csv_file="feathers_data.csv")
    
    # Option 2: Use pre-split Top-50 species (recommended for testing)
    print("=== OPTION 2: Top-50 Species (Pre-split) ===")
    converter.convert(
        use_presplit=True,
        train_csv="train_top_50_species.csv",
        test_csv="test_top_50_species.csv"
    )
    
    # Option 3: Use pre-split Top-100 species
    # print("=== OPTION 3: Top-100 Species (Pre-split) ===")
    # converter.convert(
    #     use_presplit=True,
    #     train_csv="train_top_100_species.csv",
    #     test_csv="test_top_100_species.csv"
    # )
    
    # Option 4: Use pre-split All species
    # print("=== OPTION 4: All Species (Pre-split) ===")
    # converter.convert(
    #     use_presplit=True,
    #     train_csv="train_all_species.csv",
    #     test_csv="test_all_species.csv"
    # )

if __name__ == "__main__":
    main()
