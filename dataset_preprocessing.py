"""
RISC Dataset Preprocessing Script
================================

This script preprocesses the RISC (Remote Sensing Image Captioning) dataset
for training PaliGemma-TSN model.

Key Functions:
- Loads original captions.csv file
- Selects one random caption per image (from 5 available)
- Creates train/validation/test splits
- Handles missing images and data validation
- Saves processed data to CSV files

Input:
- dataset/captions.csv: Original captions file
- dataset/resized/: Directory with 224x224 resized images

Output:
- processed_dataset/train.csv: Training data
- processed_dataset/val.csv: Validation data
- processed_dataset/test.csv: Test data
"""

import os
import pandas as pd
import random
from tqdm import tqdm

# Configuration paths
CAPTIONS_PATH = "dataset/captions.csv"  # Original captions file
IMAGES_PATH = "dataset/resized"         # Directory with resized images
OUTPUT_DIR = "processed_dataset"        # Output directory for processed CSV files

# Create output directory for CSV files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load captions data
print("Loading captions...")
captions_df = pd.read_csv(CAPTIONS_PATH)

# Data validation: Check for missing images
print("Checking for missing images...")
all_images = set(os.listdir(IMAGES_PATH))
missing_images = []

# Validate that all referenced images exist in the dataset
for image_name in tqdm(captions_df['image'].unique()):
    if image_name not in all_images:
        missing_images.append(image_name)

print(f"Found {len(missing_images)} missing images")

# Remove rows with missing images to ensure data integrity
if missing_images:
    captions_df = captions_df[~captions_df['image'].isin(missing_images)]
    print(f"Removed {len(missing_images)} rows with missing images")

# Caption processing: Select one random caption per image
print("Processing dataframe to select one random caption per image...")
processed_rows = []

# Process each row to create final dataset
for _, row in tqdm(captions_df.iterrows(), total=len(captions_df)):
    image_name = row['image']
    source = row['source']
    split = row['split']

    # Create relative path for cross-platform compatibility
    image_path = os.path.join(IMAGES_PATH, image_name)

    # Collect all valid captions for this image (up to 5 captions available)
    valid_captions = []
    for i in range(1, 6):  # caption_1 through caption_5
        caption_key = f'caption_{i}'
        if caption_key in row and pd.notna(row[caption_key]) and row[caption_key].strip():
            valid_captions.append(row[caption_key].strip())

    # Process only images with valid captions
    if valid_captions:
        # Randomly select one caption from available options
        selected_caption = random.choice(valid_captions)

        # Add processed row to final dataset
        processed_rows.append({
            'source': source,
            'split': split,
            'image': image_name,
            'image_path': image_path,  # Full path for model loading
            'caption': selected_caption
        })

# Create final processed dataframe
processed_df = pd.DataFrame(processed_rows)
print(f"Processed {len(processed_df)} images, each with one randomly selected caption")

# Data splitting: Create train/validation/test splits based on original splits
train_df = processed_df[processed_df['split'] == 'train']
val_df = processed_df[processed_df['split'] == 'val']
test_df = processed_df[processed_df['split'] == 'test']

# Data shuffling: Randomize order within each split for better training
print("Shuffling the data...")
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Train: {len(train_df)} rows")
print(f"Validation: {len(val_df)} rows")
print(f"Test: {len(test_df)} rows")

# Save processed datasets to CSV files
print("Saving CSV files with image paths...")
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

# Dataset statistics and summary
print("\nDataset Statistics:")
print(f"Total images: {len(processed_df['image'].unique())}")
print(f"Total captions: {len(processed_df)}")
print(f"Train split: {len(train_df)} captions, {len(train_df['image'].unique())} images")
print(f"Validation split: {len(val_df)} captions, {len(val_df['image'].unique())} images")
print(f"Test split: {len(test_df)} captions, {len(test_df['image'].unique())} images")

print("\nData preprocessing complete!")
print(f"CSV files saved to {OUTPUT_DIR}")
print("Ready for training!")