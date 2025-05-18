#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import os
import pandas as pd
import time
from tqdm import tqdm
import warnings
import logging
import yaml
import wandb  # Import Weights & Biases for experiment tracking
from torch.amp import autocast, GradScaler  # Import for mixed precision training

# Kategori tespiti için yardımcı fonksiyon
def detect_category(caption):
    """
    Açıklamadaki anahtar kelimelere göre kategori tespit eder
    """
    # Basitleştirilmiş kategori tespiti
    caption = caption.lower()

    # Temel kategoriler ve anahtar kelimeler
    if any(word in caption for word in ['airport', 'runway', 'airplane']):
        return 'airport'
    elif any(word in caption for word in ['water', 'river', 'lake', 'ocean']):
        return 'water'
    elif any(word in caption for word in ['tree', 'forest', 'green', 'vegetation']):
        return 'vegetation'
    elif any(word in caption for word in ['city', 'building', 'urban']):
        return 'urban'
    else:
        return 'other'

# Import TSN modules
from models.tsn_paligemma_model import create_tsn_paligemma_model

# Suppress all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="You are passing both `text` and `images` to `PaliGemmaProcessor`")
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Load Configuration from YAML ---
def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        print("Using default configuration instead")
        return {
            "model": {"paligemma_model_id": "google/paligemma-3b-mix-224"},
            "dataset": {
                "captions_path": "dataset/captions.csv",
                "risc_dataset_path": "dataset/resized",
                "max_train_samples": 100,
                "max_val_samples": 20
            },
            "training": {
                "num_epochs": 5,
                "batch_size": 2,
                "learning_rate": 1e-4,
                "gradient_accumulation_steps": 8,
                "weight_decay": 0.01
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            },
            "output": {"output_dir": "paligemma_tsn_lora"},
            "evaluation": {
                "eval_during_training": True,
                "generate_max_length": 64,
                "num_beams": 4
            },
            "prompt": {"input_prompt": "Caption this image:"},
            "tsn": {
                "backbone": "resnet50",
                "pretrained": True,
                "feature_dim": 2048,
                "projection_dim": 1408
            }
        }

# Load configuration from file
CONFIG = load_config('config/config.yaml')

# --- Extract Configuration Variables ---
# Model and data paths
PALIGEMMA_MODEL_ID = CONFIG["model"]["paligemma_model_id"]
PROCESSED_DATASET_DIR = CONFIG["dataset"]["processed_dataset_dir"]
TRAIN_CSV = CONFIG["dataset"]["train_csv"]
VAL_CSV = CONFIG["dataset"]["val_csv"]
TEST_CSV = CONFIG["dataset"]["test_csv"]
RISC_DATASET_PATH = CONFIG["dataset"]["risc_dataset_path"]
OUTPUT_DIR = CONFIG["output"]["output_dir"]

# Training parameters
try:
    NUM_EPOCHS = int(CONFIG["training"]["num_epochs"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert training.num_epochs '{CONFIG['training']['num_epochs']}' to int. Using default 5.")
    NUM_EPOCHS = 5

try:
    BATCH_SIZE = int(CONFIG["training"]["batch_size"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert training.batch_size '{CONFIG['training']['batch_size']}' to int. Using default 2.")
    BATCH_SIZE = 2

# Ensure learning_rate is a float
try:
    LEARNING_RATE = float(CONFIG["training"]["learning_rate"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert learning_rate '{CONFIG['training']['learning_rate']}' to float. Using default 5e-5.")
    LEARNING_RATE = 5e-5

try:
    GRADIENT_ACCUMULATION_STEPS = int(CONFIG["training"]["gradient_accumulation_steps"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert training.gradient_accumulation_steps '{CONFIG['training']['gradient_accumulation_steps']}' to int. Using default 4.")
    GRADIENT_ACCUMULATION_STEPS = 4

# Ensure weight_decay is a float
try:
    WEIGHT_DECAY = float(CONFIG["training"]["weight_decay"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert weight_decay '{CONFIG['training']['weight_decay']}' to float. Using default 0.01.")
    WEIGHT_DECAY = 0.01

# LoRA parameters
try:
    LORA_R = int(CONFIG["lora"]["r"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert lora.r '{CONFIG['lora']['r']}' to int. Using default 16.")
    LORA_R = 16

try:
    LORA_ALPHA = int(CONFIG["lora"]["lora_alpha"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert lora.lora_alpha '{CONFIG['lora']['lora_alpha']}' to int. Using default 32.")
    LORA_ALPHA = 32

try:
    LORA_DROPOUT = float(CONFIG["lora"]["lora_dropout"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert lora.lora_dropout '{CONFIG['lora']['lora_dropout']}' to float. Using default 0.1.")
    LORA_DROPOUT = 0.1

LORA_TARGET_MODULES = CONFIG["lora"]["target_modules"]

# Evaluation parameters
EVAL_DURING_TRAINING = bool(CONFIG["evaluation"]["eval_during_training"])
try:
    GENERATE_MAX_LENGTH = int(CONFIG["evaluation"]["generate_max_length"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert evaluation.generate_max_length '{CONFIG['evaluation']['generate_max_length']}' to int. Using default 32.")
    GENERATE_MAX_LENGTH = 32

try:
    NUM_BEAMS = int(CONFIG["evaluation"]["num_beams"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert evaluation.num_beams '{CONFIG['evaluation']['num_beams']}' to int. Using default 4.")
    NUM_BEAMS = 4

# Dataset parameters
try:
    MAX_TRAIN_SAMPLES = int(CONFIG["dataset"]["max_train_samples"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert dataset.max_train_samples '{CONFIG['dataset']['max_train_samples']}' to int. Using default 100.")
    MAX_TRAIN_SAMPLES = 100

try:
    MAX_VAL_SAMPLES = int(CONFIG["dataset"]["max_val_samples"])
except (ValueError, TypeError):
    print(f"Warning: Could not convert dataset.max_val_samples '{CONFIG['dataset']['max_val_samples']}' to int. Using default 20.")
    MAX_VAL_SAMPLES = 20

# Prompt parameters
INPUT_PROMPT = CONFIG.get("prompt", {}).get("input_prompt", "<image>Describe this image in detail.")

# Dengeleme parametreleri
BALANCE_DATASET = CONFIG.get("training", {}).get("balance_dataset", True)
USE_WEIGHTED_SAMPLING = CONFIG.get("training", {}).get("use_weighted_sampling", True)
USE_LOSS_WEIGHTING = CONFIG.get("training", {}).get("use_loss_weighting", True)

# Kategori ağırlıklandırma parametreleri
RARE_CATEGORIES = CONFIG.get("training", {}).get("rare_categories", ["airport", "infrastructure", "water"])
COMMON_CATEGORIES = CONFIG.get("training", {}).get("common_categories", ["vegetation", "urban", "landscape"])
try:
    RARE_CATEGORY_WEIGHT = float(CONFIG.get("training", {}).get("rare_category_weight", 2.0))
except (ValueError, TypeError):
    print(f"Warning: Could not convert rare_category_weight to float. Using default 2.0.")
    RARE_CATEGORY_WEIGHT = 2.0

# Label smoothing parameter
try:
    # First check if it's in model config
    LABEL_SMOOTHING = float(CONFIG.get("model", {}).get("label_smoothing", 0.1))
    # If not, check in training config
    if "label_smoothing" not in CONFIG.get("model", {}):
        LABEL_SMOOTHING = float(CONFIG.get("training", {}).get("label_smoothing", 0.1))
    print(f"Using label smoothing with value: {LABEL_SMOOTHING}")
except (ValueError, TypeError):
    print(f"Warning: Could not convert label_smoothing to float. Using default 0.1.")
    LABEL_SMOOTHING = 0.1

# Kaynak ağırlıklandırma parametreleri
RARE_SOURCES = CONFIG.get("training", {}).get("rare_sources", ["UCM"])
try:
    SOURCE_WEIGHT = float(CONFIG.get("training", {}).get("source_weight", 1.5))
except (ValueError, TypeError):
    print(f"Warning: Could not convert source_weight to float. Using default 1.5.")
    SOURCE_WEIGHT = 1.5

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Device Setup ---
if not torch.cuda.is_available():
    raise RuntimeError("GPU (CUDA) is required for this project but not available. Please ensure you have a GPU and CUDA installed.")

device = torch.device("cuda")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")

# Global processor variable
processor = None

# --- Simple Dataset Class with TSN ---
class RISCTSNDataset(Dataset):
    def __init__(self, csv_path, img_dir, processor, split='train', max_samples=None):
        """
        Dataset class for RISC dataset with TSN feature extraction.

        Args:
            csv_path: Path to the CSV file containing captions for the specific split
            img_dir: Path to the directory containing images
            processor: PaliGemma processor
            split: Dataset split ('train', 'val', or 'test')
            max_samples: Maximum number of samples to use (for debugging)
        """
        try:
            # Load the CSV file for the specific split
            self.annotations = pd.read_csv(csv_path)
            print(f"Loaded {split} dataset from {csv_path} with {len(self.annotations)} samples")
        except FileNotFoundError:
            print(f"ERROR: CSV file not found at {csv_path}")
            self.annotations = pd.DataFrame(columns=['source', 'split', 'image', 'caption']) # Empty dataframe

        # Limit samples if needed
        if max_samples and max_samples > 0 and max_samples < len(self.annotations):
             self.annotations = self.annotations.head(max_samples)
             print(f"Limited to {max_samples} samples for debugging")
        else:
             print(f"Using all {len(self.annotations)} samples for {split} dataset")

        # Set the image directory
        self.img_dir = img_dir
        self.processor = processor
        self.split = split

        # Detect categories for each sample
        if 'category' not in self.annotations.columns:
            print("Detecting categories for each sample...")
            self.annotations['category'] = self.annotations['caption'].apply(detect_category)

        # Calculate caption lengths
        if 'caption_length' not in self.annotations.columns:
            self.annotations['caption_length'] = self.annotations['caption'].apply(lambda x: len(str(x).split()))

        # Calculate sample weights for balanced training
        self.sample_weights = None
        if split == 'train':
            self._calculate_sample_weights()

        # Analyze dataset categories for robust training
        self._analyze_dataset()

    def _calculate_sample_weights(self):
        """
        Calculate sample weights for balanced training
        """
        # Eğer dengeleme devre dışı bırakılmışsa, ağırlıkları hesaplama
        if not BALANCE_DATASET or not USE_WEIGHTED_SAMPLING:
            print("Dataset balancing or weighted sampling is disabled in config.")
            return

        # Kategori bazlı ağırlıklandırma
        if 'category' in self.annotations.columns:
            category_counts = self.annotations['category'].value_counts()
            total_samples = len(self.annotations)

            # Her kategori için ağırlık hesapla (az olan kategorilere daha yüksek ağırlık)
            category_weights = {}

            for category, count in category_counts.items():
                # Nadir kategorilere daha yüksek ağırlık ver
                if category in RARE_CATEGORIES:
                    # Config'den alınan ağırlık faktörünü kullan
                    category_weights[category] = (total_samples / (count * len(category_counts))) * (RARE_CATEGORY_WEIGHT / 2.0)
                else:
                    category_weights[category] = total_samples / (count * len(category_counts))

            # Her örnek için ağırlık ata
            self.sample_weights = [category_weights[category] for category in self.annotations['category']]

            # Ağırlıkları normalize et
            weight_sum = sum(self.sample_weights)
            self.sample_weights = [w / weight_sum * total_samples for w in self.sample_weights]

            print(f"\nCalculated sample weights for balanced training:")
            for category, weight in category_weights.items():
                print(f"  - {category}: {weight:.4f}")

        # Kaynak bazlı ağırlıklandırma (eğer source bilgisi varsa)
        if 'source' in self.annotations.columns:
            source_counts = self.annotations['source'].value_counts()
            total_samples = len(self.annotations)

            # Her kaynak için ağırlık hesapla
            source_weights = {}

            for source, count in source_counts.items():
                # Nadir kaynaklara daha yüksek ağırlık ver
                if source in RARE_SOURCES:
                    # Config'den alınan ağırlık faktörünü kullan
                    source_weights[source] = (total_samples / (count * len(source_counts))) * (SOURCE_WEIGHT / 1.5)
                else:
                    source_weights[source] = total_samples / (count * len(source_counts))

            # Kaynak ağırlıklarını mevcut ağırlıklara ekle
            if self.sample_weights:
                source_sample_weights = [source_weights[source] for source in self.annotations['source']]
                # Kategori ve kaynak ağırlıklarını birleştir (çarparak)
                self.sample_weights = [w1 * w2 for w1, w2 in zip(self.sample_weights, source_sample_weights)]
            else:
                self.sample_weights = [source_weights[source] for source in self.annotations['source']]

            # Ağırlıkları normalize et
            weight_sum = sum(self.sample_weights)
            self.sample_weights = [w / weight_sum * total_samples for w in self.sample_weights]

            print(f"\nAdjusted sample weights for source balance:")
            for source, weight in source_weights.items():
                print(f"  - {source}: {weight:.4f}")

    def _analyze_dataset(self):
        """
        Analyze dataset categories and distribution for robust training
        """
        # Check if category information is available
        if 'category' in self.annotations.columns:
            categories = self.annotations['category'].value_counts()
            print(f"\nCategory distribution in {self.split} dataset:")
            for category, count in categories.items():
                print(f"  - {category}: {count} samples ({count/len(self.annotations)*100:.1f}%)")

        # Check source distribution if available
        if 'source' in self.annotations.columns:
            sources = self.annotations['source'].value_counts()
            print(f"\nSource distribution in {self.split} dataset:")
            for source, count in sources.items():
                print(f"  - {source}: {count} samples ({count/len(self.annotations)*100:.1f}%)")

        # Check caption length distribution
        if 'caption_length' in self.annotations.columns:
            avg_length = self.annotations['caption_length'].mean()
            print(f"Average caption length: {avg_length:.1f} words")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing processed inputs for the model
        """
        row = self.annotations.iloc[idx]

        # Get image_id and caption from CSV
        image_id = row['image']

        # Make sure we have a valid caption
        if 'caption' in row and isinstance(row['caption'], str) and len(row['caption'].strip()) > 0:
            caption = row['caption'].strip()
        else:
            # Try to find any caption-related column
            caption_cols = [col for col in row.index if 'caption' in col.lower()]
            if caption_cols:
                caption = str(row[caption_cols[0]]).strip()
            else:
                caption = "No caption available."

        # Ensure caption is not just a number
        if caption.isdigit():
            caption = f"Image {image_id} shows a remote sensing scene."
            print(f"WARNING: Numeric-only caption '{row['caption']}' detected for {image_id}. Using generic caption.")

        # Check if the image exists in the dataset directory
        image_path = os.path.join(self.img_dir, image_id)
        if not os.path.exists(image_path):
            # If the image doesn't exist, log a warning and try to find an alternative
            if idx < 5:  # Only print for the first few items
                print(f"WARNING: Image {image_id} not found at {image_path}")

            # Try to use the image_path field if it exists in the CSV
            if 'image_path' in row and os.path.exists(row['image_path']):
                image_path = row['image_path']
                if idx < 5:  # Only print for the first few items
                    print(f"Using alternative path from CSV: {image_path}")
            else:
                # As a last resort, use an available image from the directory
                image_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
                if image_files:
                    # Use a consistent image for each sample to ensure reproducibility
                    alt_image_id = image_files[idx % len(image_files)]
                    image_path = os.path.join(self.img_dir, alt_image_id)
                    if idx < 5:  # Only print for the first few items
                        print(f"Using alternative image: {alt_image_id} instead of {image_id}")
                else:
                    # If no images are available, raise an error
                    raise FileNotFoundError(f"No images found in {self.img_dir}")

        # Update image_id to match the actual file being used
        image_id = os.path.basename(image_path)

        # Print for debugging
        if idx < 5:  # Only print for the first few items
            print(f"Loading image: {image_path}")
            print(f"Caption: {caption}")

        # Get caption from the dataset
        # Check for caption columns in the format 'caption_1', 'caption_2', etc.
        caption_columns = [col for col in row.index if col.startswith('caption_')]

        if caption_columns:
            # Use the first caption column (caption_1)
            alt_caption = row[caption_columns[0]]
            # Ensure caption is a string and not just a number
            if isinstance(alt_caption, str) and len(alt_caption.strip()) > 0 and not alt_caption.isdigit():
                caption = alt_caption.strip()
        elif 'caption' in row and not str(row['caption']).isdigit():
            caption = str(row['caption']).strip()
        elif 'captions' in row and not str(row['captions']).isdigit():
            caption = str(row['captions']).strip()
        elif 'text' in row and not str(row['text']).isdigit():
            caption = str(row['text']).strip()

        # Final check to ensure we don't have a numeric-only caption
        if caption.isdigit():
            caption = f"Image {image_id} shows a remote sensing scene."
            print(f"WARNING: Numeric-only caption detected for {image_id}. Using generic caption.")

        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")

            # Ensure image is properly sized for the model (224x224 for PaliGemma)
            if image.size != (224, 224):
                image = image.resize((224, 224), Image.LANCZOS)

            # Create a copy of the image to avoid potential issues
            image_copy = image.copy()

            # Process input (image + prompt text)
            # Using the prompt from configuration
            try:
                inputs = self.processor(
                    images=image_copy,
                    text=INPUT_PROMPT,  # Use prompt from config
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
            except Exception as e:
                print(f"Error processing input for item {idx}: {e}")
                # Try with default prompt
                inputs = self.processor(
                    images=image_copy,
                    text="<image>Describe this image in detail.",
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )

            # Create another copy for label processing
            image_copy2 = image.copy()

            # Process target text (caption)
            try:
                # Ensure caption is a string
                caption_str = str(caption)

                # For PaliGemma, we need to include the <image> token in the caption
                if not caption_str.startswith("<image>"):
                    caption_str = "<image>" + caption_str

                # Ensure caption is properly formatted for PaliGemma
                if not caption_str.endswith("."):
                    caption_str = caption_str + "."

                label_inputs = self.processor(
                    images=image_copy2,
                    text=caption_str,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
            except Exception as e:
                print(f"Error processing label for item {idx}: {e}")
                # Use empty caption as fallback
                label_inputs = self.processor(
                    images=image_copy2,
                    text="<image>Empty caption.",
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )

            # Create labels, ignoring padding tokens in loss calculation
            labels = label_inputs.input_ids.squeeze(0)
            labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Ignore padding in loss

            # Prepare the sample with all metadata
            sample = {
                "pixel_values": inputs.pixel_values.squeeze(0),
                "input_ids": inputs.input_ids.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0),
                "labels": labels,
                "image_id": image_id,
                "caption": caption,
                "original_caption": caption  # Store the original caption for reference
            }

            # Add category information if available
            if 'category' in self.annotations.columns:
                # Get the category for this sample
                category = row.get('category', 'unknown')
                sample['category'] = category

            # Add source information if available
            if 'source' in self.annotations.columns:
                # Get the source for this sample
                source = row.get('source', 'unknown')
                sample['source'] = source

            return sample

        except Exception as e:
            print(f"Error processing item {idx} (image: {image_id}): {e}. Skipping.")
            # Return a default item instead of recursively calling __getitem__
            # This prevents infinite recursion
            default_item = {
                "pixel_values": torch.zeros((3, 224, 224)),
                "input_ids": torch.zeros(512, dtype=torch.long),
                "attention_mask": torch.zeros(512, dtype=torch.long),
                "labels": torch.ones(512, dtype=torch.long) * -100,
                "image_id": "error_image",
                "caption": "Error loading image",
                "original_caption": "Error loading image"
            }

            # Add category and source if they're in the dataset
            if 'category' in self.annotations.columns:
                default_item['category'] = 'unknown'

            if 'source' in self.annotations.columns:
                default_item['source'] = 'unknown'

            return default_item

# Custom Data Collator
def custom_data_collator(batch):
    """
    Custom data collator for batching samples.

    Args:
        batch: List of samples from the dataset

    Returns:
        Dictionary containing batched tensors
    """
    # Filter out None items resulting from errors
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Return None if batch is empty

    # Stack pixel values
    pixel_values = torch.stack([item['pixel_values'] for item in batch])

    # Pad sequences to max length in batch
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        [item['labels'] for item in batch],
        batch_first=True,
        padding_value=-100  # Ignore padding in loss calculation
    )

    # Collect metadata for evaluation
    image_ids = [item['image_id'] for item in batch]
    captions = [item['caption'] for item in batch]
    original_captions = [item.get('original_caption', item['caption']) for item in batch]

    # Collect category and source information if available (for loss weighting)
    categories = None
    sources = None

    # Check if category information is available in any item
    if any('category' in item for item in batch):
        categories = [item.get('category', 'unknown') for item in batch]

    # Check if source information is available in any item
    if any('source' in item for item in batch):
        sources = [item.get('source', 'unknown') for item in batch]

    result = {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'image_id': image_ids,
        'caption': captions,
        'original_caption': original_captions
    }

    # Add category and source information if available
    if categories:
        result['category'] = categories

    if sources:
        result['source'] = sources

    return result

# --- Evaluation Function ---
def evaluate_model(model, dataloader, processor, eval_device=None):
    """
    Evaluate the model on the validation set.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for validation data
        processor: PaliGemma processor
        eval_device: Device to run evaluation on (optional, uses global device if None)

    Returns:
        avg_loss: Average loss on validation set
        all_preds: List of generated captions
        all_labels: List of reference captions
    """
    # Access the global device variable
    global device

    # Use provided device or fall back to global device
    if eval_device is not None:
        device = eval_device

    # Ensure model is on the correct device
    model = model.to(device)

    # Ensure model is in evaluation mode
    model.eval()

    # Ensure TSN module is in evaluation mode if it exists
    if hasattr(model, 'tsn'):
        model.tsn.eval()
        print("TSN module set to evaluation mode")

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for step, batch in enumerate(progress_bar):
            if batch is None:  # Skip empty batches
                continue

            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass with mixed precision (same as in training)
            with autocast(device_type='cuda'):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            # Accumulate loss - use EXACTLY the same calculation as in training
            loss = outputs.loss
            if loss is not None:
                # Apply the same loss weighting as in training
                loss_weight = 1.0

                # Get batch metadata for category-based loss weighting
                batch_categories = batch.get('category', None)
                batch_sources = batch.get('source', None)

                # Apply category-based loss weighting if enabled in config
                if BALANCE_DATASET and USE_LOSS_WEIGHTING and batch_categories is not None:
                    # Count categories in this batch
                    category_counts = {}
                    for cat in batch_categories:
                        if cat in category_counts:
                            category_counts[cat] += 1
                        else:
                            category_counts[cat] = 1

                    # Calculate inverse frequency weight for this batch
                    if len(category_counts) > 0:
                        # Use rare categories from config
                        # Count rare and common categories in batch
                        rare_count = sum(category_counts.get(cat, 0) for cat in RARE_CATEGORIES)
                        common_count = sum(category_counts.get(cat, 0) for cat in COMMON_CATEGORIES)

                        # Apply higher weight if batch has rare categories
                        if rare_count > 0:
                            rare_ratio = rare_count / (rare_count + common_count + 1e-10)
                            # Scale weight based on config parameter
                            loss_weight = 1.0 + (RARE_CATEGORY_WEIGHT - 1.0) * rare_ratio

                # Apply source-based loss weighting if enabled in config
                if BALANCE_DATASET and USE_LOSS_WEIGHTING and batch_sources is not None:
                    # Count sources in this batch
                    source_counts = {}
                    for src in batch_sources:
                        if src in source_counts:
                            source_counts[src] += 1
                        else:
                            source_counts[src] = 1

                    # Apply higher weight to underrepresented sources from config
                    rare_source_count = sum(source_counts.get(src, 0) for src in RARE_SOURCES)
                    if rare_source_count > 0 and len(source_counts) > 1:
                        rare_source_ratio = rare_source_count / len(batch_sources)
                        # Scale weight based on config parameter
                        source_weight = 1.0 + (SOURCE_WEIGHT - 1.0) * rare_source_ratio
                        loss_weight *= source_weight

                # Apply the weight to the loss
                if loss_weight != 1.0:
                    loss = loss * loss_weight

                # Get original loss value for reporting
                original_loss = loss.item()

                # Scale loss for gradient accumulation
                # Note: We don't actually accumulate gradients in validation, but we scale the loss the same way
                loss = loss / GRADIENT_ACCUMULATION_STEPS

                # Get scaled loss for reporting (this matches training loss scale)
                current_loss = original_loss / GRADIENT_ACCUMULATION_STEPS

                # Add to total loss for epoch average calculation (using scaled loss)
                total_loss += current_loss

                # Print loss occasionally for debugging
                if step % 50 == 0:
                    print(f"Validation step {step}, Loss: {current_loss:.6f} (Original: {original_loss:.6f})")

            # Generate predictions for validation samples
            # Get a single image from the batch for generation
            single_image = pixel_values[0].unsqueeze(0).to(device)

            # Use prompt from config
            prompt_text = INPUT_PROMPT

            # Prepare the image
            image_cpu = single_image.cpu().clone()

            # Normalize the image to [0, 1] range if needed
            if image_cpu.min() < 0:
                image_cpu = (image_cpu + 1) / 2
            image_cpu = torch.clamp(image_cpu, 0, 1)

            # Initialize prediction
            prediction = ""

            # Generate with the model
            with torch.no_grad():
                # Process with the processor
                inputs = processor(
                    images=image_cpu,
                    text=prompt_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512  # Use longer context length for better results
                ).to(device)

                # Set up generation parameters
                generation_inputs = inputs.copy()
                generation_inputs.update({
                    'max_new_tokens': GENERATE_MAX_LENGTH,
                    'num_beams': NUM_BEAMS
                })

                # Generate with the model - make sure we're in eval mode
                model.eval()
                with torch.inference_mode():
                    generated_ids = model.generate(**generation_inputs)

                # Decode the generated caption
                generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

                # Get the raw output directly from the model without any modifications
                prediction = generated_captions[0]
                print(f"Raw model output: {prediction}")

                # If we still don't have a good prediction, try one more time with a different approach
                if not prediction or len(prediction.split()) < 3:
                    # Use prompt from config
                    final_prompt = INPUT_PROMPT

                    # Process with the processor
                    final_inputs = processor(
                        images=image_cpu,
                        text=final_prompt,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512  # Use longer context length for better results
                    ).to(device)

                    # Generate with different settings
                    final_generation_inputs = final_inputs.copy()
                    final_generation_inputs.update({
                        'max_new_tokens': GENERATE_MAX_LENGTH,
                        'num_beams': NUM_BEAMS  # Use beams from config
                    })

                    # Generate with the model
                    try:
                        # Make sure we're in eval mode for generation
                        model.eval()
                        with torch.inference_mode():
                            final_generated_ids = model.generate(**final_generation_inputs)

                        # Decode the generated caption - show exactly what the model generates
                        final_captions = processor.batch_decode(final_generated_ids, skip_special_tokens=True)
                        prediction = final_captions[0]
                        print(f"Second attempt raw model output: {prediction}")
                    except Exception as e:
                        print(f"Error during generation: {e}")
                        # Don't use any fallback, just show the error

                # Don't modify the prediction at all - show exactly what the model generates

                # Get reference caption for the first image
                label_caption = "No caption available"
                if 'caption' in batch and len(batch['caption']) > 0:
                    label_caption = str(batch['caption'][0])

                # Store prediction and reference
                all_preds.append(prediction)
                all_labels.append(label_caption)

                # Print for debugging
                print(f"\nGenerated caption: {prediction}")
                print(f"Reference caption: {label_caption}")

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)

    # Print validation examples
    print("\n=== Validation Examples ===")
    if len(all_preds) == 0:
        print("WARNING: No predictions generated during validation!")
    else:
        print(f"Generated {len(all_preds)} predictions during validation.")

        # Print a few examples
        for i in range(min(3, len(all_preds))):
            print(f"\nExample {i+1}:")
            print(f"  Prediction: \"{all_preds[i]}\"")
            print(f"  Reference: \"{all_labels[i]}\"")
            print()

    # Return to training mode
    model.train()

    return avg_loss, all_preds, all_labels

# --- Main Training Script ---
def main():
    # Access the global device variable
    global device, processor

    # Initialize Weights & Biases if enabled
    USE_WANDB = CONFIG.get("training", {}).get("use_wandb", False)
    if USE_WANDB:
        try:
            # Check if wandb is already initialized (for hyperparameter optimization)
            if wandb.run is None:
                wandb.init(
                    project="2697134-tsn-paligemma-project",
                    config={
                        "model": CONFIG.get("model", {}),
                        "dataset": CONFIG.get("dataset", {}),
                        "training": CONFIG.get("training", {}),
                        "lora": CONFIG.get("lora", {}),
                        "tsn": CONFIG.get("tsn", {}),
                        "evaluation": CONFIG.get("evaluation", {})
                    },
                    name=f"tsn-{CONFIG.get('tsn', {}).get('backbone', 'resnet50')}-lora-r{LORA_R}"
                )
                print("Weights & Biases initialized for experiment tracking")
            else:
                print("Using existing Weights & Biases run (for hyperparameter optimization)")
        except Exception as e:
            print(f"Warning: Failed to initialize Weights & Biases: {e}")
            USE_WANDB = False

    print("Initializing components...")

    # Initialize processor and model
    print(f"Loading PaliGemma processor from {PALIGEMMA_MODEL_ID}...")
    try:
        # Try loading with token if available
        import os
        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token:
            print("Using Hugging Face token for processor")
            processor = PaliGemmaProcessor.from_pretrained(
                PALIGEMMA_MODEL_ID,
                token=hf_token
            )
        else:
            # Try without token
            print("No Hugging Face token found, trying to load processor without token")
            processor = PaliGemmaProcessor.from_pretrained(PALIGEMMA_MODEL_ID)
    except Exception as e:
        print(f"Error loading processor: {e}")
        print("If this is an authentication error, please set the HF_TOKEN environment variable with your Hugging Face token")
        print("You can get a token from https://huggingface.co/settings/tokens")
        raise

    # Update model configuration with label smoothing
    from transformers import PaliGemmaConfig
    model_config = PaliGemmaConfig.from_pretrained(PALIGEMMA_MODEL_ID)

    # Try to set label_smoothing in config
    try:
        # Some models support this directly
        model_config.label_smoothing = LABEL_SMOOTHING
        print(f"Set label_smoothing={LABEL_SMOOTHING} in model config")
    except:
        print(f"Note: Could not set label_smoothing directly in config")

    # Load base model with bfloat16 precision to save memory and directly to GPU
    print(f"Loading PaliGemma model from {PALIGEMMA_MODEL_ID}...")
    try:
        # Try loading with token if available
        import os
        hf_token = os.environ.get("HF_TOKEN", None)

        # Set up loading parameters
        model_kwargs = {
            "torch_dtype": torch.bfloat16,  # Use bfloat16 to save memory
            "device_map": "auto"
        }

        # Add token if available
        if hf_token:
            print("Using Hugging Face token from environment variable")
            model_kwargs["token"] = hf_token
        else:
            print("No Hugging Face token found, trying to load model without token")

        # Load the model with QLoRA-compatible settings
        from transformers import BitsAndBytesConfig

        # Set up quantization config for QLoRA based on the notebook example
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        # Add quantization config to model kwargs
        model_kwargs["quantization_config"] = quantization_config

        # Set device_map for better GPU utilization
        model_kwargs["device_map"] = {"": 0}

        # Load the model
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            PALIGEMMA_MODEL_ID,
            **model_kwargs
        )

        # Freeze the vision tower and multimodal projector to focus fine-tuning on the language model
        print("Freezing vision tower and multimodal projector...")
        for param in base_model.vision_tower.parameters():
            param.requires_grad = False
        for param in base_model.multi_modal_projector.parameters():
            param.requires_grad = False

        print("Model loaded with 4-bit quantization for QLoRA fine-tuning")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If this is an authentication error, please set the HF_TOKEN environment variable with your Hugging Face token")
        print("You can get a token from https://huggingface.co/settings/tokens")
        raise

    print(f"Model loaded with custom config (label_smoothing={LABEL_SMOOTHING})")

    # Ensure model is on GPU
    if next(base_model.parameters()).device.type != 'cuda':
        print("Forcing model to GPU...")
        base_model = base_model.to(device)

    # Apply QLoRA directly to the base model
    print("Configuring QLoRA...")

    # Use target modules from config
    target_modules = LORA_TARGET_MODULES

    print(f"Using target modules from config: {target_modules}")

    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply QLoRA to the base model
    lora_model = get_peft_model(base_model, lora_config)
    print("QLoRA applied to PaliGemma model.")
    lora_model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    if hasattr(lora_model, "enable_input_require_grads"):
        lora_model.enable_input_require_grads()

    # Make sure we're using the right dtype
    if hasattr(lora_model, "config") and hasattr(lora_model.config, "torch_dtype"):
        print(f"Model is using dtype: {lora_model.config.torch_dtype}")
    else:
        print("Could not determine model dtype from config")

    # Add TSN to config
    CONFIG["tsn"]["use_tsn"] = True

    # Create the final model by wrapping the LoRA-tuned PaliGemma
    final_model = create_tsn_paligemma_model(lora_model, CONFIG)
    print("Created PaliGemma-LoRA model with TSN")

    # Ensure model is on the correct device
    final_model = final_model.to(device)

    # Prepare datasets
    print("Preparing datasets...")
    # Use max_train_samples and max_val_samples from config
    train_dataset = RISCTSNDataset(
        csv_path=TRAIN_CSV,
        img_dir=RISC_DATASET_PATH,
        processor=processor,
        split='train',
        max_samples=MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES > 0 else None  # Use config parameter
    )

    val_dataset = RISCTSNDataset(
        csv_path=VAL_CSV,
        img_dir=RISC_DATASET_PATH,
        processor=processor,
        split='val',
        max_samples=MAX_VAL_SAMPLES if MAX_VAL_SAMPLES > 0 else None  # Use config parameter
    )

    # Check if datasets are empty
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty. Check paths and CSV content.")
        return

    if len(val_dataset) == 0:
        print("WARNING: Validation dataset is empty. Evaluation will be skipped.")
        globals()["EVAL_DURING_TRAINING"] = False

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

    # Create DataLoaders with weighted sampling for balanced training
    if BALANCE_DATASET and USE_WEIGHTED_SAMPLING and train_dataset.sample_weights is not None:
        print("Using weighted random sampling for balanced training (enabled in config)")
        # Weighted random sampler for balanced training
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,  # Use sampler instead of shuffle
            collate_fn=custom_data_collator
        )
    else:
        if not BALANCE_DATASET:
            print("Dataset balancing is disabled in config")
        elif not USE_WEIGHTED_SAMPLING:
            print("Weighted sampling is disabled in config")
        else:
            print("No sample weights available, using regular shuffling")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=custom_data_collator
        )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_data_collator
    )

    # Set up training
    print("Setting up training...")
    # Set up optimizer parameters
    optimizer_params = []

    # Add TSN parameters if available
    if hasattr(final_model, 'tsn') and final_model.tsn is not None:
        optimizer_params.append({'params': final_model.tsn.parameters(), 'lr': LEARNING_RATE * 0.1})
        print("Added TSN parameters to optimizer")

    # Add PaliGemma parameters
    optimizer_params.append({'params': final_model.paligemma.parameters(), 'lr': LEARNING_RATE})
    print("Added PaliGemma parameters to optimizer")

    optimizer = AdamW(optimizer_params, weight_decay=WEIGHT_DECAY)

    # Set up mixed precision training (always enabled for GPU)
    USE_MIXED_PRECISION = True  # Force mixed precision for GPU training
    scaler = GradScaler()

    # Create learning rate scheduler with warmup
    from transformers import get_cosine_schedule_with_warmup
    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)  # %10 warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"Using cosine learning rate scheduler with {num_warmup_steps} warmup steps")

    print("Using mixed precision training for faster training and reduced memory usage")

    # Training loop
    print("Starting training loop...")
    best_val_loss = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, "best_model")
    os.makedirs(best_model_path, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*20} Epoch {epoch+1}/{NUM_EPOCHS} {'='*20}")
        final_model.train()
        total_loss = 0
        epoch_start_time = time.time()
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            if batch is None:  # Skip empty batches
                continue

            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass with mixed precision (always enabled for GPU)
            with autocast(device_type='cuda'):
                outputs = final_model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Handle loss
                loss = outputs.loss
                if loss is None:
                    print("WARNING: Loss is None, skipping batch")
                    continue

                # Apply label smoothing manually if needed
                # Note: This is only needed if the model doesn't support label_smoothing in its config
                # Most HuggingFace models handle this internally when label_smoothing is in config

                # Check for zero loss
                if loss.item() == 0:
                    print("WARNING: Zero loss detected, skipping batch")

                # Get batch metadata for category-based loss weighting
                batch_categories = batch.get('category', None)
                batch_sources = batch.get('source', None)

                # Apply category-based loss weighting if enabled in config
                loss_weight = 1.0
                if BALANCE_DATASET and USE_LOSS_WEIGHTING and batch_categories is not None:
                    # Count categories in this batch
                    category_counts = {}
                    for cat in batch_categories:
                        if cat in category_counts:
                            category_counts[cat] += 1
                        else:
                            category_counts[cat] = 1

                    # Calculate inverse frequency weight for this batch
                    if len(category_counts) > 0:
                        # Use rare categories from config
                        # Count rare and common categories in batch
                        rare_count = sum(category_counts.get(cat, 0) for cat in RARE_CATEGORIES)
                        common_count = sum(category_counts.get(cat, 0) for cat in COMMON_CATEGORIES)

                        # Apply higher weight if batch has rare categories
                        if rare_count > 0:
                            rare_ratio = rare_count / (rare_count + common_count + 1e-10)
                            # Scale weight based on config parameter
                            loss_weight = 1.0 + (RARE_CATEGORY_WEIGHT - 1.0) * rare_ratio

                            if step % 50 == 0:  # Reduced frequency
                                print(f"Rare categories: {rare_count}, weight: {loss_weight:.2f}")

                # Apply source-based loss weighting if enabled in config
                if BALANCE_DATASET and USE_LOSS_WEIGHTING and batch_sources is not None:
                    # Count sources in this batch
                    source_counts = {}
                    for src in batch_sources:
                        if src in source_counts:
                            source_counts[src] += 1
                        else:
                            source_counts[src] = 1

                    # Apply higher weight to underrepresented sources from config
                    rare_source_count = sum(source_counts.get(src, 0) for src in RARE_SOURCES)
                    if rare_source_count > 0 and len(source_counts) > 1:
                        rare_source_ratio = rare_source_count / len(batch_sources)
                        # Scale weight based on config parameter
                        source_weight = 1.0 + (SOURCE_WEIGHT - 1.0) * rare_source_ratio
                        loss_weight *= source_weight

                # Apply the weight to the loss
                if loss_weight != 1.0:
                    loss = loss * loss_weight

                # Get original loss value for reporting
                original_loss = loss.item()

                # Scale loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS

                # Get scaled loss for reporting (this matches validation loss scale)
                current_loss = original_loss / GRADIENT_ACCUMULATION_STEPS

                # Skip zero loss for better training
                if current_loss == 0:
                    print(f"WARNING: Zero loss detected at step {step}. Skipping this batch.")
                    continue

                # Add to total loss for epoch average calculation (using scaled loss)
                # This ensures train_loss and val_loss are on the same scale
                total_loss += current_loss

                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'AvgLoss': f'{total_loss / (step + 1):.4f}'
                })

                # Log loss occasionally
                if step % 20 == 0:
                    print(f"Step {step}, Loss: {current_loss:.4f}")

                    # Generate and show model output during training (every 100 steps)
                    if step % 100 == 0:
                        print("\n=== Training Sample Generation ===")
                        # Get a single image from the batch
                        train_image = pixel_values[0].unsqueeze(0)

                        # Use prompt from config
                        train_prompt = INPUT_PROMPT

                        # Process the image
                        with torch.no_grad():
                            # Prepare the image - normalize to [0, 1] range
                            train_image_cpu = train_image.cpu().clone()

                            # Convert from [-1, 1] to [0, 1] range if needed
                            if train_image_cpu.min() < 0:
                                train_image_cpu = (train_image_cpu + 1) / 2

                            # Ensure values are clamped to [0, 1]
                            train_image_cpu = torch.clamp(train_image_cpu, 0, 1)

                            # Prepare inputs
                            train_inputs = processor(
                                images=train_image_cpu,
                                text=train_prompt,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=512  # Use longer context length for better results
                            ).to(device)

                            # Generate with the model
                            train_generation_inputs = train_inputs.copy()
                            train_generation_inputs.update({
                                'max_new_tokens': GENERATE_MAX_LENGTH,
                                'num_beams': NUM_BEAMS
                            })

                            # Generate text
                            try:
                                # Make sure we're in eval mode for generation
                                final_model.eval()
                                train_generated_ids = final_model.generate(**train_generation_inputs)
                                # Return to train mode after generation
                                final_model.train()

                                # Decode the generated caption
                                train_captions = processor.batch_decode(train_generated_ids, skip_special_tokens=True)

                                # Show raw model output
                                print(f"Raw model output: {train_captions[0]}")

                                # Get reference caption if available
                                if 'caption' in batch and len(batch['caption']) > 0:
                                    print(f"Reference: {batch['caption'][0]}")

                                print("=== End of Training Sample ===\n")
                            except Exception as e:
                                print(f"Error generating during training: {e}")

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Update parameters after accumulating gradients
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_dataloader) - 1:
                # Clip gradients to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)  # Increased from 0.5 for better convergence

                # Update parameters with scaler
                scaler.step(optimizer)
                scaler.update()

                # Step the learning rate scheduler
                scheduler.step()

                # Log current learning rate occasionally
                if step % 50 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Current learning rate: {current_lr:.2e}")

                optimizer.zero_grad()

        # Calculate average loss for the epoch
        avg_epoch_loss = total_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s. Average Training Loss: {avg_epoch_loss:.4f}")

        # Log metrics to Weights & Biases if enabled
        if USE_WANDB:
            # Get current learning rate from scheduler
            current_lr = scheduler.get_last_lr()[0]

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "epoch_time": epoch_time,
                "learning_rate": current_lr,  # Log actual current learning rate
                "mixed_precision": USE_MIXED_PRECISION,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS
            })

        # Save checkpoint for this epoch
        checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save PaliGemma model
        final_model.paligemma.save_pretrained(os.path.join(checkpoint_path, "paligemma"))

        # Save TSN model if available
        if hasattr(final_model, 'tsn') and final_model.tsn is not None:
            torch.save(final_model.tsn.state_dict(), os.path.join(checkpoint_path, "tsn_model.pt"))

        print(f"Saved checkpoint to {checkpoint_path}")

        # Run validation after each epoch if enabled
        if EVAL_DURING_TRAINING and len(val_dataset) > 0:
            print(f"\nRunning validation for epoch {epoch+1}...")
            val_loss, predictions, references = evaluate_model(final_model, val_dataloader, processor)
            print(f"Validation Loss: {val_loss:.4f}")

            # Print validation examples with more detailed information
            print("\n=== Validation Examples ===")

            # Check if we have any predictions
            if len(predictions) == 0:
                print("WARNING: No predictions generated during validation!")
                print("This is a critical issue that needs to be fixed.")
                print("Check the logs above for error messages during generation.")
            else:
                print(f"Generated {len(predictions)} predictions during validation.")

                # Print a few examples
                for i in range(min(3, len(predictions))):
                    print(f"\nExample {i+1}:")

                    # Print the prediction with clear formatting
                    print(f"  Prediction: \"{predictions[i]}\"")

                    # Print the reference with clear formatting
                    print(f"  Reference: \"{references[i]}\"")

                    # Print the image ID for reference if available
                    if 'image_id' in val_dataset.annotations.columns and i < len(val_dataset):
                        try:
                            image_id = val_dataset.annotations.iloc[i]['image_id']
                            print(f"  Image ID: {image_id}")
                        except Exception as e:
                            print(f"  Error retrieving image ID: {e}")

                    # Print prediction length for analysis
                    pred_words = len(predictions[i].split())
                    ref_words = len(references[i].split())
                    print(f"  Prediction length: {pred_words} words")
                    print(f"  Reference length: {ref_words} words")

                    print()

            # Log validation metrics to Weights & Biases if enabled
            if USE_WANDB:
                # Log validation loss for hyperparameter optimization
                wandb.log({
                    "val_loss": val_loss,
                    "epoch": epoch + 1
                })

                # Log some example predictions
                if len(predictions) > 0:
                    example_table = wandb.Table(columns=["Example", "Prediction", "Reference"])
                    for i in range(min(5, len(predictions))):
                        example_table.add_data(i+1, predictions[i], references[i])
                    wandb.log({"validation_examples": example_table})

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")

                # Save PaliGemma model
                final_model.paligemma.save_pretrained(os.path.join(best_model_path, "paligemma"))

                # Save TSN model if available
                if hasattr(final_model, 'tsn') and final_model.tsn is not None:
                    torch.save(final_model.tsn.state_dict(), os.path.join(best_model_path, "tsn_model.pt"))

                print(f"Saved best model to {best_model_path}")

    print("Training finished.")

    # Save final model
    final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    # Save PaliGemma model
    final_model.paligemma.save_pretrained(os.path.join(final_model_dir, "paligemma"))

    # Save TSN model if available
    if hasattr(final_model, 'tsn') and final_model.tsn is not None:
        torch.save(final_model.tsn.state_dict(), os.path.join(final_model_dir, "tsn_model.pt"))

    print(f"Final model saved to {final_model_dir}")

    # Close Weights & Biases if it was used and not part of hyperparameter optimization
    if USE_WANDB and os.environ.get('WANDB_SWEEP_ID') is None:
        wandb.finish()
        print("Weights & Biases logging completed")
    elif USE_WANDB:
        print("Keeping Weights & Biases run open for hyperparameter optimization")

if __name__ == "__main__":
    main()
