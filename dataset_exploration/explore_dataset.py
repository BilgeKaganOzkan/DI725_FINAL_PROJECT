"""
RISC Dataset Exploration Script

This script analyzes the RISC (Remote Sensing Image Captioning) dataset,
exploring both the images and captions to provide insights about the dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter
import re
import nltk
from nltk.tokenize import word_tokenize
import seaborn as sns
from wordcloud import WordCloud
import random

# Download NLTK resources
nltk.download('punkt', quiet=True)

# Set paths
DATASET_DIR = os.path.join('..', 'dataset')
CAPTIONS_PATH = os.path.join(DATASET_DIR, 'captions.csv')
IMAGES_DIR = os.path.join(DATASET_DIR, 'resized')
OUTPUT_DIR = os.path.join('.', 'output')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_captions_data():
    """Load and return the captions data."""
    return pd.read_csv(CAPTIONS_PATH)

def analyze_dataset_structure(df):
    """Analyze the basic structure of the dataset."""
    print("Dataset Structure Analysis")
    print("=" * 50)

    # Basic dataset info
    print(f"Total number of entries: {len(df)}")
    print(f"Number of unique images: {df['image'].nunique()}")

    # Split distribution
    split_counts = df['split'].value_counts()
    print("\nSplit Distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} ({count/len(df)*100:.2f}%)")

    # Source distribution
    source_counts = df['source'].value_counts()
    print("\nSource Distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} ({count/len(df)*100:.2f}%)")

    # Create visualizations
    plt.figure(figsize=(12, 5))

    # Split distribution plot
    plt.subplot(1, 2, 1)
    split_counts.plot(kind='bar', color='skyblue')
    plt.title('Dataset Split Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    # Source distribution plot
    plt.subplot(1, 2, 2)
    source_counts.plot(kind='bar', color='lightgreen')
    plt.title('Dataset Source Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dataset_distribution.png'))

    # Cross-tabulation of source and split
    cross_tab = pd.crosstab(df['source'], df['split'])
    print("\nCross-tabulation of Source and Split:")
    print(cross_tab)

    # Visualize cross-tabulation
    plt.figure(figsize=(10, 6))
    cross_tab.plot(kind='bar', stacked=True)
    plt.title('Distribution of Sources across Splits')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Split')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'source_split_distribution.png'))

def analyze_image_properties():
    """Analyze properties of the images in the dataset."""
    print("\nImage Properties Analysis")
    print("=" * 50)

    # Get list of image files
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]

    if not image_files:
        print("No image files found in the specified directory.")
        return

    # Sample a subset of images for analysis (to save time)
    sample_size = min(1000, len(image_files))
    sampled_images = random.sample(image_files, sample_size)

    # Collect image dimensions and sizes
    dimensions = []
    file_sizes = []

    for img_file in sampled_images:
        img_path = os.path.join(IMAGES_DIR, img_file)
        try:
            # Get file size in KB
            file_size = os.path.getsize(img_path) / 1024
            file_sizes.append(file_size)

            # Get image dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                dimensions.append((width, height))
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    # Extract width and height lists
    widths, heights = zip(*dimensions)

    # Print statistics
    print(f"Analyzed {len(dimensions)} images")
    print(f"Image dimensions: All images are {widths[0]}x{heights[0]} pixels")
    print(f"Average file size: {np.mean(file_sizes):.2f} KB")
    print(f"Min file size: {min(file_sizes):.2f} KB")
    print(f"Max file size: {max(file_sizes):.2f} KB")

    # Create histogram of file sizes
    plt.figure(figsize=(10, 6))
    plt.hist(file_sizes, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Image File Sizes')
    plt.xlabel('File Size (KB)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'image_size_distribution.png'))

def analyze_captions(df):
    """Analyze the captions in the dataset."""
    print("\nCaption Analysis")
    print("=" * 50)

    # Combine all captions for analysis
    all_captions = []
    for i in range(1, 6):
        caption_col = f'caption_{i}'
        all_captions.extend(df[caption_col].dropna().tolist())

    # Calculate caption lengths
    caption_lengths = [len(caption.split()) for caption in all_captions]

    # Print statistics
    print(f"Total number of captions: {len(all_captions)}")
    print(f"Average caption length: {np.mean(caption_lengths):.2f} words")
    print(f"Min caption length: {min(caption_lengths)} words")
    print(f"Max caption length: {max(caption_lengths)} words")

    # Create histogram of caption lengths
    plt.figure(figsize=(10, 6))
    plt.hist(caption_lengths, bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Caption Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'caption_length_distribution.png'))

    # Tokenize captions and find most common words
    all_words = []
    for caption in all_captions:
        # Remove punctuation and convert to lowercase
        clean_caption = re.sub(r'[^\w\s]', '', caption.lower())
        words = word_tokenize(clean_caption)
        all_words.extend(words)

    # Remove common stopwords
    stopwords = {'a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'to', 'by', '.', ','}
    filtered_words = [word for word in all_words if word not in stopwords]

    # Count word frequencies
    word_counts = Counter(filtered_words)
    most_common = word_counts.most_common(30)

    print("\nMost common words in captions:")
    for word, count in most_common[:10]:
        print(f"  {word}: {count}")

    # Create bar chart of most common words
    plt.figure(figsize=(12, 6))
    words, counts = zip(*most_common)
    plt.bar(words, counts, color='lightcoral')
    plt.title('30 Most Common Words in Captions')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'common_words.png'))

    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(' '.join(filtered_words))
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Caption Content')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'caption_wordcloud.png'))

    # Analyze caption similarity across the 5 captions for each image
    similarity_scores = []

    # Sample a subset of images for analysis
    sample_size = min(1000, len(df))
    sampled_df = df.sample(sample_size)

    for _, row in sampled_df.iterrows():
        captions = [row[f'caption_{i}'] for i in range(1, 6)]
        captions = [c for c in captions if isinstance(c, str)]  # Filter out any NaN values

        if len(captions) < 2:
            continue

        # Calculate Jaccard similarity between each pair of captions
        caption_words = [set(re.sub(r'[^\w\s]', '', c.lower()).split()) for c in captions]
        pair_similarities = []

        for i in range(len(caption_words)):
            for j in range(i+1, len(caption_words)):
                intersection = len(caption_words[i].intersection(caption_words[j]))
                union = len(caption_words[i].union(caption_words[j]))
                if union > 0:
                    similarity = intersection / union
                    pair_similarities.append(similarity)

        if pair_similarities:
            avg_similarity = np.mean(pair_similarities)
            similarity_scores.append(avg_similarity)

    # Print caption similarity statistics
    print(f"\nAverage Jaccard similarity between captions for the same image: {np.mean(similarity_scores):.4f}")

    # Create histogram of caption similarities
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_scores, bins=20, color='mediumpurple', edgecolor='black')
    plt.title('Distribution of Caption Similarities for the Same Image')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'caption_similarity_distribution.png'))

def analyze_caption_categories(df):
    """Analyze the categories/topics of captions."""
    print("\nCaption Categories Analysis")
    print("=" * 50)

    # Define category keywords
    categories = {
        'airport': ['airport', 'runway', 'terminal'],
        'urban': ['building', 'city', 'urban', 'residential', 'house'],
        'vegetation': ['tree', 'forest', 'vegetation', 'green', 'grass', 'lawn'],
        'water': ['water', 'river', 'lake', 'sea', 'ocean'],
        'transportation': ['plane', 'aircraft', 'road', 'highway', 'car'],
        'landscape': ['mountain', 'hill', 'land', 'field', 'farmland'],
        'infrastructure': ['bridge', 'port', 'dock', 'facility']
    }

    # Function to categorize a caption
    def categorize_caption(caption):
        caption_lower = caption.lower()
        caption_categories = []

        for category, keywords in categories.items():
            if any(keyword in caption_lower for keyword in keywords):
                caption_categories.append(category)

        return caption_categories

    # Analyze categories in captions
    category_counts = {category: 0 for category in categories}
    multi_category_count = 0

    # Sample a subset of captions for analysis
    sample_size = min(5000, len(df))
    sampled_df = df.sample(sample_size)

    for _, row in sampled_df.iterrows():
        caption = row['caption_1']  # Use the first caption for each image
        if not isinstance(caption, str):
            continue

        caption_cats = categorize_caption(caption)

        if len(caption_cats) > 1:
            multi_category_count += 1

        for cat in caption_cats:
            category_counts[cat] += 1

    # Print category statistics
    print(f"Analyzed {sample_size} captions for categories")
    print(f"Captions with multiple categories: {multi_category_count} ({multi_category_count/sample_size*100:.2f}%)")

    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} ({count/sample_size*100:.2f}%)")

    # Create bar chart of categories
    plt.figure(figsize=(12, 6))
    category_names, counts = zip(*sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
    plt.bar(category_names, counts, color='teal')
    plt.title('Distribution of Caption Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'caption_categories.png'))

    # Create a co-occurrence matrix for categories
    category_list = list(category_counts.keys())
    cooccurrence = np.zeros((len(category_list), len(category_list)))

    for _, row in sampled_df.iterrows():
        caption = row['caption_1']
        if not isinstance(caption, str):
            continue

        caption_cats = categorize_caption(caption)

        for i, cat1 in enumerate(category_list):
            for j, cat2 in enumerate(category_list):
                if cat1 in caption_cats and cat2 in caption_cats:
                    cooccurrence[i, j] += 1

    # Create heatmap of category co-occurrences
    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence, annot=True, fmt='g', cmap='Blues',
                xticklabels=sorted(category_counts.keys()), yticklabels=sorted(category_counts.keys()))
    plt.title('Co-occurrence of Categories in Captions')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'category_cooccurrence.png'))

def main():
    """Main function to run the analysis."""
    print("RISC Dataset Exploration")
    print("=" * 50)

    # Load captions data
    df = load_captions_data()

    # Analyze dataset structure
    analyze_dataset_structure(df)

    # Analyze image properties
    analyze_image_properties()

    # Analyze captions
    analyze_captions(df)

    # Analyze caption categories
    analyze_caption_categories(df)

    print("\nAnalysis complete. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
