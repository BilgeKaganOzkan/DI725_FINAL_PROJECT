import os
import pandas as pd
import random
from tqdm import tqdm

# Paths
CAPTIONS_PATH = "dataset/captions.csv"
IMAGES_PATH = "dataset/resized"
OUTPUT_DIR = "processed_dataset"

# Create output directory for CSV files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load captions
print("Loading captions...")
captions_df = pd.read_csv(CAPTIONS_PATH)

# Check for missing images
print("Checking for missing images...")
all_images = set(os.listdir(IMAGES_PATH))
missing_images = []

for image_name in tqdm(captions_df['image'].unique()):
    if image_name not in all_images:
        missing_images.append(image_name)

print(f"Found {len(missing_images)} missing images")

# Remove rows with missing images
if missing_images:
    captions_df = captions_df[~captions_df['image'].isin(missing_images)]
    print(f"Removed {len(missing_images)} rows with missing images")

# Process the dataframe to select one random caption per image
print("Processing dataframe to select one random caption per image...")
processed_rows = []

for _, row in tqdm(captions_df.iterrows(), total=len(captions_df)):
    image_name = row['image']
    source = row['source']
    split = row['split']

    # Use relative path, don't create static absolute paths
    image_path = os.path.join(IMAGES_PATH, image_name)

    # Collect all valid captions for this image
    valid_captions = []
    for i in range(1, 6):
        caption_key = f'caption_{i}'
        if caption_key in row and pd.notna(row[caption_key]) and row[caption_key].strip():
            valid_captions.append(row[caption_key].strip())

    # If there are valid captions, randomly select one
    if valid_captions:
        # Randomly select one caption
        selected_caption = random.choice(valid_captions)

        # Add the row with the selected caption
        processed_rows.append({
            'source': source,
            'split': split,
            'image': image_name,
            'image_path': image_path,  # Add full image path
            'caption': selected_caption
        })

processed_df = pd.DataFrame(processed_rows)
print(f"Processed {len(processed_df)} images, each with one randomly selected caption")

# Split the data
train_df = processed_df[processed_df['split'] == 'train']
val_df = processed_df[processed_df['split'] == 'val']
test_df = processed_df[processed_df['split'] == 'test']

# Shuffle the data
print("Shuffling the data...")
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Train: {len(train_df)} rows")
print(f"Validation: {len(val_df)} rows")
print(f"Test: {len(test_df)} rows")

# Save the split dataframes with image paths
print("Saving CSV files with image paths...")
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

# Note: We're not saving the combined CSV file anymore

# Print statistics
print("\nDataset Statistics:")
print(f"Total images: {len(processed_df['image'].unique())}")
print(f"Total captions: {len(processed_df)}")
print(f"Train split: {len(train_df)} captions, {len(train_df['image'].unique())} images")
print(f"Validation split: {len(val_df)} captions, {len(val_df['image'].unique())} images")
print(f"Test split: {len(test_df)} captions, {len(test_df['image'].unique())} images")

print("\nData preprocessing complete!")
print(f"CSV files saved to {OUTPUT_DIR}")