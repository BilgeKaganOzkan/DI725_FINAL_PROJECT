#!/usr/bin/env python3

"""
Dataset Preprocessing Script for Remote Sensing Image Captioning

This script processes raw remote sensing datasets and prepares them for training.
Main functionalities:
- Loads caption files and validates image paths
- Handles missing images and data quality issues
- Splits data into train/validation/test sets
- Generates processed CSV files for model training
- Performs data statistics and quality reporting

Supported datasets:
- NWPU-RESISC45: Northwestern Polytechnical University dataset
- UCM Land Use: University of California Merced dataset
- AID: Aerial Image Dataset
"""

# Core imports for data processing
import os
import pandas as pd
import numpy as np
from PIL import Image
import random

def load_and_process_captions(caption_file, image_dir):
    """
    Load caption data from CSV file and validate corresponding images.
    
    This function:
    - Reads caption CSV file with image_path and caption columns
    - Validates that corresponding image files exist
    - Filters out samples with missing images
    - Handles path normalization for cross-platform compatibility
    
    Args:
        caption_file (str): Path to CSV file containing captions
        image_dir (str): Directory containing image files
        
    Returns:
        pd.DataFrame: Processed dataframe with validated image-caption pairs
    """
    print(f"[LOADING] Reading captions from: {caption_file}")
    
    # Load caption data from CSV
    try:
        df = pd.read_csv(caption_file)
        print(f"[SUCCESS] Loaded {len(df)} caption entries")
    except Exception as e:
        raise FileNotFoundError(f"Could not load caption file: {e}")
    
    # Validate required columns
    required_columns = ['image_path', 'caption']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"[DATA] Processing image paths and validating files...")
    
    # Track data quality metrics
    initial_count = len(df)
    missing_images = 0
    valid_samples = []
    
    # Process each row and validate image existence
    for idx, row in df.iterrows():
        # Normalize image path for cross-platform compatibility
        img_path = os.path.normpath(row['image_path']).replace('\\', '/')
        
        # Construct full image path
        if not os.path.isabs(img_path):
            full_img_path = os.path.join(image_dir, img_path)
        else:
            full_img_path = img_path
        
        # Check if image file exists
        if os.path.exists(full_img_path):
            # Validate image can be opened
            try:
                with Image.open(full_img_path) as img:
                    # Verify image is valid and has reasonable dimensions
                    if img.size[0] > 32 and img.size[1] > 32:
                        valid_samples.append({
                            'image_path': full_img_path,
                            'caption': row['caption'].strip(),
                            'original_index': idx
                        })
                    else:
                        print(f"[WARNING] Image too small: {full_img_path}")
                        missing_images += 1
            except Exception as e:
                print(f"[WARNING] Cannot open image {full_img_path}: {e}")
                missing_images += 1
        else:
            print(f"[WARNING] Missing image: {full_img_path}")
            missing_images += 1
    
    # Create processed dataframe
    processed_df = pd.DataFrame(valid_samples)
    
    # Print processing statistics
    print(f"[STATISTICS] Data processing complete:")
    print(f"   Initial samples: {initial_count}")
    print(f"   Valid samples: {len(processed_df)}")
    print(f"   Missing/invalid images: {missing_images}")
    print(f"   Success rate: {len(processed_df)/initial_count*100:.1f}%")
    
    return processed_df

def quality_filter_captions(df, min_caption_length=10, max_caption_length=200):
    """
    Filter captions based on quality criteria.
    
    Args:
        df (pd.DataFrame): Input dataframe with captions
        min_caption_length (int): Minimum caption length in characters
        max_caption_length (int): Maximum caption length in characters
        
    Returns:
        pd.DataFrame: Filtered dataframe with quality captions
    """
    print(f"[FILTER] Applying caption quality filters...")
    
    initial_count = len(df)
    
    # Filter by caption length
    df_filtered = df[
        (df['caption'].str.len() >= min_caption_length) & 
        (df['caption'].str.len() <= max_caption_length)
    ].copy()
    
    # Remove captions with problematic content
    # Filter out very short or generic captions
    generic_patterns = ['image', 'picture', 'photo', 'unknown', 'error', 'loading']
    for pattern in generic_patterns:
        df_filtered = df_filtered[
            ~df_filtered['caption'].str.lower().str.contains(pattern, na=False)
        ]
    
    # Remove captions with excessive punctuation or special characters
    df_filtered = df_filtered[
        ~df_filtered['caption'].str.contains(r'[^\w\s.,!?-]', regex=True, na=False)
    ]
    
    filtered_count = len(df_filtered)
    removed_count = initial_count - filtered_count
    
    print(f"[FILTER] Quality filtering complete:")
    print(f"   Initial samples: {initial_count}")
    print(f"   After filtering: {filtered_count}")
    print(f"   Removed samples: {removed_count}")
    print(f"   Retention rate: {filtered_count/initial_count*100:.1f}%")
    
    return df_filtered

def split_dataset(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe to split
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set  
        test_ratio (float): Proportion for test set
        random_seed (int): Random seed for reproducible splits
        
    Returns:
        tuple: (train_df, val_df, test_df) - Split dataframes
    """
    print(f"[SPLIT] Splitting dataset with ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Validate split ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle dataset
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    total_samples = len(df_shuffled)
    
    # Calculate split indices
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # Split the dataset
    train_df = df_shuffled[:train_end].copy()
    val_df = df_shuffled[train_end:val_end].copy()
    test_df = df_shuffled[val_end:].copy()
    
    # Print split statistics
    print(f"[SPLIT] Dataset split complete:")
    print(f"   Total samples: {total_samples}")
    print(f"   Training set: {len(train_df)} samples ({len(train_df)/total_samples*100:.1f}%)")
    print(f"   Validation set: {len(val_df)} samples ({len(val_df)/total_samples*100:.1f}%)")
    print(f"   Test set: {len(test_df)} samples ({len(test_df)/total_samples*100:.1f}%)")
    
    return train_df, val_df, test_df

def analyze_dataset_statistics(df, dataset_name="Dataset"):
    """
    Analyze and print dataset statistics.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        dataset_name (str): Name for logging purposes
    """
    print(f"\n[STATISTICS] Analysis for {dataset_name}:")
    print(f"   Total samples: {len(df)}")
    
    if 'caption' in df.columns:
        # Caption length statistics
        caption_lengths = df['caption'].str.len()
        print(f"   Caption length statistics:")
        print(f"     Mean: {caption_lengths.mean():.1f} characters")
        print(f"     Median: {caption_lengths.median():.1f} characters")
        print(f"     Min: {caption_lengths.min()} characters")
        print(f"     Max: {caption_lengths.max()} characters")
        print(f"     Std: {caption_lengths.std():.1f} characters")
        
        # Word count statistics
        word_counts = df['caption'].str.split().str.len()
        print(f"   Word count statistics:")
        print(f"     Mean: {word_counts.mean():.1f} words")
        print(f"     Median: {word_counts.median():.1f} words")
        print(f"     Min: {word_counts.min()} words")
        print(f"     Max: {word_counts.max()} words")
    
    print()

def save_processed_datasets(train_df, val_df, test_df, output_dir="processed_dataset"):
    """
    Save processed datasets to CSV files.
    
    Args:
        train_df (pd.DataFrame): Training dataset
        val_df (pd.DataFrame): Validation dataset
        test_df (pd.DataFrame): Test dataset
        output_dir (str): Output directory for CSV files
    """
    print(f"[SAVING] Saving processed datasets to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    # Save datasets
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[SUCCESS] Datasets saved:")
    print(f"   Training: {train_path} ({len(train_df)} samples)")
    print(f"   Validation: {val_path} ({len(val_df)} samples)")
    print(f"   Test: {test_path} ({len(test_df)} samples)")
    
    # Create summary file
    summary_path = os.path.join(output_dir, "dataset_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Dataset Processing Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Total samples processed: {len(train_df) + len(val_df) + len(test_df)}\n")
        f.write(f"Training samples: {len(train_df)}\n")
        f.write(f"Validation samples: {len(val_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n\n")
        f.write("File paths:\n")
        f.write(f"- train.csv: {len(train_df)} samples\n")
        f.write(f"- val.csv: {len(val_df)} samples\n")
        f.write(f"- test.csv: {len(test_df)} samples\n")
    
    print(f"[SUCCESS] Summary saved to: {summary_path}")

def main():
    """
    Main preprocessing pipeline.
    
    This function orchestrates the complete data preprocessing workflow:
    1. Load and validate data
    2. Apply quality filtering
    3. Split into train/val/test sets
    4. Generate statistics
    5. Save processed datasets
    """
    print("="*60)
    print("REMOTE SENSING DATASET PREPROCESSING")
    print("="*60)
    
    # Configuration - modify these paths for your dataset
    caption_file = "dataset/captions.csv"  # Path to caption CSV file
    image_dir = "dataset/resized"          # Directory containing images
    output_dir = "processed_dataset"       # Output directory for processed files
    
    # Preprocessing parameters
    min_caption_length = 10    # Minimum caption length (characters)
    max_caption_length = 200   # Maximum caption length (characters)
    random_seed = 42           # Random seed for reproducible results
    
    # Split ratios (must sum to 1.0)
    train_ratio = 0.8  # 80% for training
    val_ratio = 0.1    # 10% for validation  
    test_ratio = 0.1   # 10% for testing
    
    try:
        # Step 1: Load and validate data
        print("\nSTEP 1: LOADING AND VALIDATING DATA")
        print("-" * 40)
        df = load_and_process_captions(caption_file, image_dir)
        
        if len(df) == 0:
            raise ValueError("No valid samples found after processing")
        
        # Step 2: Apply quality filtering
        print("\nSTEP 2: QUALITY FILTERING")
        print("-" * 40)
        df_filtered = quality_filter_captions(
            df, 
            min_caption_length=min_caption_length,
            max_caption_length=max_caption_length
        )
        
        if len(df_filtered) == 0:
            raise ValueError("No samples remaining after quality filtering")
        
        # Step 3: Split dataset
        print("\nSTEP 3: DATASET SPLITTING")
        print("-" * 40)
        train_df, val_df, test_df = split_dataset(
            df_filtered,
            train_ratio=train_ratio,
            val_ratio=val_ratio, 
            test_ratio=test_ratio,
            random_seed=random_seed
        )
        
        # Step 4: Generate statistics
        print("\nSTEP 4: DATASET ANALYSIS")
        print("-" * 40)
        analyze_dataset_statistics(train_df, "Training Set")
        analyze_dataset_statistics(val_df, "Validation Set")
        analyze_dataset_statistics(test_df, "Test Set")
        
        # Step 5: Save processed datasets
        print("\nSTEP 5: SAVING PROCESSED DATA")
        print("-" * 40)
        save_processed_datasets(train_df, val_df, test_df, output_dir)
        
        # Final summary
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total processed samples: {len(train_df) + len(val_df) + len(test_df)}")
        print(f"Files saved to: {output_dir}/")
        print(f"Ready for model training!")
        
    except Exception as e:
        print(f"\n[ERROR] Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    """
    Entry point for dataset preprocessing.
    
    Run this script to process your remote sensing dataset:
    python dataset_preprocessing.py
    """
    success = main()
    if not success:
        exit(1)