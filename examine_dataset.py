#!/usr/bin/env python3
"""
Script to examine the FeathersV1 dataset CSV structure
Run this first to understand your data before conversion
"""

import pandas as pd
import os
from pathlib import Path

def examine_csv_files(dataset_path):
    """Examine all CSV files in the dataset"""
    
    data_path = Path(dataset_path) / "data"
    
    print("=== FeathersV1 Dataset CSV Analysis ===\n")
    
    # List all CSV files
    csv_files = list(data_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    print("\n" + "="*50 + "\n")
    
    # Examine each CSV file
    for csv_file in csv_files:
        print(f"Analyzing: {csv_file.name}")
        print("-" * 30)
        
        try:
            df = pd.read_csv(csv_file)
            
            print(f"Shape: {df.shape} (rows, columns)")
            print(f"Columns: {list(df.columns)}")
            
            # Show first few rows
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Show data types
            print(f"\nData types:")
            print(df.dtypes)
            
            # Show unique values for categorical columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_count = df[col].nunique()
                    print(f"\n{col}: {unique_count} unique values")
                    if unique_count <= 10:
                        print(f"  Values: {sorted(df[col].unique())}")
                    else:
                        print(f"  Sample values: {sorted(df[col].unique())[:10]}...")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            print("\n" + "="*50 + "\n")

def check_images_structure(dataset_path):
    """Check the images directory structure"""
    
    images_path = Path(dataset_path) / "images"
    
    print("=== Images Directory Structure ===\n")
    
    # List order directories
    order_dirs = [d for d in images_path.iterdir() if d.is_dir()]
    print(f"Found {len(order_dirs)} order directories:")
    
    total_images = 0
    
    for order_dir in sorted(order_dirs):
        print(f"\n{order_dir.name}:")
        
        # List species directories within this order
        species_dirs = [d for d in order_dir.iterdir() if d.is_dir()]
        print(f"  {len(species_dirs)} species directories")
        
        order_images = 0
        for species_dir in species_dirs:
            # Count images in this species directory
            images = list(species_dir.glob("*.jpg")) + list(species_dir.glob("*.jpeg")) + list(species_dir.glob("*.png"))
            order_images += len(images)
            
        print(f"  ~{order_images} total images")
        total_images += order_images
    
    print(f"\nTotal estimated images: {total_images}")

def main():
    # Update this path to your dataset location
    dataset_path = "/home/kavi/feathersv1-classification/data/feathersv1-dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        print("Please update the dataset_path variable to point to your feathersv1-dataset directory")
        return
    
    examine_csv_files(dataset_path)
    check_images_structure(dataset_path)
    
    print("\n=== Recommendations ===")
    print("1. Use 'feathers_data.csv' for the complete dataset")
    print("2. Use 'train_top_50_species.csv'/'test_top_50_species.csv' for faster testing")
    print("3. Use 'train_all_species.csv'/'test_all_species.csv' for pre-split data")

if __name__ == "__main__":
    main()
