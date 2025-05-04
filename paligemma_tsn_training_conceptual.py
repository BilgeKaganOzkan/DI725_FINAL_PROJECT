import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import os
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import warnings
import pretrainedmodels
import wandb
warnings.filterwarnings("ignore")

# --- Load Configuration from YAML ---
def load_config(config_path="config/config.yaml"):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        # Default configuration
        return {
            "model": {
                "paligemma_model_id": "google/paligemma-3b-mix-224",
                "feature_dim": 1024
            },
            "dataset": {
                "captions_path": "dataset/captions.csv",
                "risc_dataset_path": "dataset/resized",
                "max_train_samples": 100,
                "max_val_samples": 20
            },
            "training": {
                "num_epochs": 1,
                "batch_size": 2,
                "learning_rate": 1.0e-5,
                "gradient_accumulation_steps": 4,
                "weight_decay": 0.01,
                "warmup_steps": 100
            },
            "lora": {
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            },
            "output": {
                "output_dir": "paligemma_finetuned_tsn_corrected",
                "save_every_n_epochs": 1
            },
            "evaluation": {
                "eval_during_training": True,
                "generate_max_length": 128,
                "num_beams": 4
            }
        }

# Load configuration
CONFIG = load_config()

# --- Configuration ---
# Model and data paths
PALIGEMMA_MODEL_ID = CONFIG["model"]["paligemma_model_id"]
CAPTIONS_PATH = CONFIG["dataset"]["captions_path"]
RISC_DATASET_PATH = CONFIG["dataset"]["risc_dataset_path"]
OUTPUT_DIR = CONFIG["output"]["output_dir"]

# Training parameters
NUM_EPOCHS = CONFIG["training"]["num_epochs"]
BATCH_SIZE = CONFIG["training"]["batch_size"]
LEARNING_RATE = CONFIG["training"]["learning_rate"]
GRADIENT_ACCUMULATION_STEPS = CONFIG["training"]["gradient_accumulation_steps"]
WEIGHT_DECAY = CONFIG["training"].get("weight_decay", 0.01)
WARMUP_STEPS = CONFIG["training"].get("warmup_steps", 100)

# LoRA parameters
LORA_R = CONFIG["lora"]["r"]
LORA_ALPHA = CONFIG["lora"]["lora_alpha"]
LORA_DROPOUT = CONFIG["lora"]["lora_dropout"]
LORA_TARGET_MODULES = CONFIG["lora"]["target_modules"]

# Evaluation parameters
EVAL_DURING_TRAINING = CONFIG["evaluation"]["eval_during_training"]
GENERATE_MAX_LENGTH = CONFIG["evaluation"]["generate_max_length"]
NUM_BEAMS = CONFIG["evaluation"]["num_beams"]

# Dataset parameters
MAX_TRAIN_SAMPLES = CONFIG["dataset"]["max_train_samples"]
MAX_VAL_SAMPLES = CONFIG["dataset"]["max_val_samples"]

# Weights & Biases parameters
USE_WANDB = CONFIG.get("wandb", {}).get("use_wandb", False)
WANDB_PROJECT = CONFIG.get("wandb", {}).get("project", "paligemma-tsn-risc")
WANDB_NAME = CONFIG.get("wandb", {}).get("name", "paligemma-tsn-training")
WANDB_TAGS = CONFIG.get("wandb", {}).get("tags", ["paligemma", "tsn", "risc"])
WANDB_LOG_MODEL = CONFIG.get("wandb", {}).get("log_model", False)
WANDB_LOG_FREQ = CONFIG.get("wandb", {}).get("log_freq", 10)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Initialize Weights & Biases ---
if USE_WANDB:
    print(f"Initializing Weights & Biases with project: {WANDB_PROJECT}, run: {WANDB_NAME}")
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_NAME,
        tags=WANDB_TAGS,
        config={
            "model": {
                "paligemma_model_id": PALIGEMMA_MODEL_ID,
                "feature_dim": CONFIG["model"]["feature_dim"],
            },
            "training": {
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": WARMUP_STEPS,
            },
            "lora": {
                "r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
            },
            "dataset": {
                "max_train_samples": MAX_TRAIN_SAMPLES,
                "max_val_samples": MAX_VAL_SAMPLES,
            },
            "evaluation": {
                "eval_during_training": EVAL_DURING_TRAINING,
                "generate_max_length": GENERATE_MAX_LENGTH,
                "num_beams": NUM_BEAMS,
            },
        }
    )

# --- 1. Feature Extractor with Spatial Segmentation (Using BNInception) ---
class BNInceptionFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=1024):
        super().__init__()
        print("INFO: Using pretrained BNInception backbone from pretrainedmodels library.")
        # Load pretrained BNInception model from pretrainedmodels
        self.bninception_backbone = pretrainedmodels.bninception(pretrained='imagenet')
        self.bninception_backbone.eval() # Set to evaluation mode

        # Get the actual output dimension from the model
        # BNInception from pretrainedmodels outputs 1024-dim features
        self.feature_dim = feature_dim
        self.bninception_output_dim = 1024  # Actual output dimension from BNInception

        # If feature_dim is different from the model's output, add a projection layer
        if self.feature_dim != self.bninception_output_dim:
            self.feature_projection = nn.Linear(self.bninception_output_dim, feature_dim)
        else:
            self.feature_projection = nn.Identity()

        # Preprocessing matching BNInception training from pretrainedmodels
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # ToTensor is applied later in dataset
            transforms.Normalize(
                mean=self.bninception_backbone.mean,
                std=self.bninception_backbone.std
            ),
        ])
        print(f"BNInception Backbone loaded. Output dimension: {self.bninception_output_dim}, projected to {feature_dim}-dim.")

    def spatial_segment(self, image_tensor):
        # Input: image_tensor (Batch x 3 x H x W)
        # Output: dictionary of segments { '1x1': [tensor], '2x2': [4 tensors], '4x4': [16 tensors] }
        segments = {}
        _, _, h, w = image_tensor.shape

        # 1x1 grid (whole image)
        segments['1x1'] = [image_tensor]

        # 2x2 grid
        segments['2x2'] = []
        for i in range(2):
            for j in range(2):
                segment = image_tensor[..., i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                segments['2x2'].append(segment)

        # 4x4 grid
        segments['4x4'] = []
        for i in range(4):
            for j in range(4):
                segment = image_tensor[..., i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
                segments['4x4'].append(segment)

        return segments

    def extract_segment_features(self, segments_dict):
        # Input: dictionary of segments
        # Output: dictionary of features { '1x1': [feat_tensor], '2x2': [4 feat_tensors], '4x4': [16 feat_tensors] }
        segment_features = {}
        with torch.no_grad():
            for scale, segment_list in segments_dict.items():
                scale_features = []
                for segment in segment_list:
                    # Preprocess segment (Apply normalization)
                    # Assuming segments are already cropped, apply normalization
                    segment = self.preprocess(segment) # Apply normalization
                    segment = segment.to(device)

                    # Extract features using BNInception backbone
                    # For pretrainedmodels BNInception, we can use the features method
                    # or directly call the model which returns features
                    feature = self.bninception_backbone(segment)

                    # Project features to target dimension if needed
                    projected_feature = self.feature_projection(feature)
                    scale_features.append(projected_feature.cpu()) # Move back to CPU

                segment_features[scale] = scale_features

        return segment_features

# --- 2. Spatial Attention Mechanism (MLP-based) ---
class SpatialAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        # Simple MLP for attention scoring
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        print("Spatial Attention module (MLP-based) initialized.")

    def forward(self, segment_features_list):
        # Input: list of segment features [tensor1, tensor2, ...], each [Batch, FeatureDim]
        # Output: attention weights [Batch, NumSegments], weighted context vector [Batch, FeatureDim]

        # Stack features: [Batch, NumSegments, FeatureDim]
        segment_features_tensor = torch.stack(segment_features_list, dim=1).to(device)
        batch_size, num_segments, feature_dim = segment_features_tensor.shape

        # Print debug info
        # print(f"Segment features tensor shape: {segment_features_tensor.shape}")
        # print(f"Expected feature dim: {self.feature_dim}, Actual feature dim: {feature_dim}")

        # Calculate attention scores
        # Reshape for MLP: [Batch * NumSegments, FeatureDim]
        # Use the actual feature dimension from the tensor, not the expected one
        attn_input = segment_features_tensor.reshape(batch_size * num_segments, feature_dim)

        # If feature dimensions don't match, we need to adapt the attention network
        if feature_dim != self.feature_dim and not hasattr(self, 'adapted_attention_net'):
            print(f"Adapting attention network for feature dim {feature_dim} (was {self.feature_dim})")
            self.adapted_attention_net = nn.Sequential(
                nn.Linear(feature_dim, self.attention_dim),
                nn.Tanh(),
                nn.Linear(self.attention_dim, 1)
            ).to(device)
            self.feature_dim = feature_dim  # Update the feature_dim attribute

        # Use the appropriate attention network
        if hasattr(self, 'adapted_attention_net'):
            scores = self.adapted_attention_net(attn_input)
        else:
            scores = self.attention_net(attn_input)

        # Reshape scores: [Batch, NumSegments]
        scores = scores.view(batch_size, num_segments)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)

        # Calculate weighted context vector
        # weights: [Batch, NumSegments, 1], features: [Batch, NumSegments, FeatureDim]
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * segment_features_tensor, dim=1)
        # Output: [Batch, FeatureDim]

        # print("Attention weights calculated and context vector generated.")
        return attention_weights, context_vector

# --- 3. Feature Adapter ---
class FeatureAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Simple linear projection + LayerNorm
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim  # Store output dimension for reference
        print(f"Feature Adapter initialized: {input_dim} -> {output_dim}")

    def forward(self, aggregated_features):
        # Input: Aggregated feature vector [Batch, input_dim]
        # Output: Adapted features ready for PaliGemma [Batch, output_dim]
        aggregated_features = aggregated_features.to(device)

        # Check if input dimension matches expected dimension
        if aggregated_features.shape[1] != self.input_dim and not hasattr(self, 'adapted_projection'):
            actual_dim = aggregated_features.shape[1]
            print(f"Adapting feature projection for input dim {actual_dim} (was {self.input_dim})")
            self.adapted_projection = nn.Linear(actual_dim, self.output_dim).to(device)
            self.input_dim = actual_dim  # Update the input_dim attribute

        # Use the appropriate projection
        if hasattr(self, 'adapted_projection'):
            projected = self.adapted_projection(aggregated_features)
        else:
            projected = self.projection(aggregated_features)

        adapted_features = self.norm(projected)
        return adapted_features # Keep on device for model input

# --- 4. Dataset and DataLoader ---
class RISCTSNFeatureDataset(Dataset):
    def __init__(self, captions_csv, img_dir, feature_extractor, attention_module, adapter, processor, split='train', max_samples=None):
        try:
            self.annotations = pd.read_csv(captions_csv)
        except FileNotFoundError:
            print(f"ERROR: Captions file not found at {captions_csv}")
            self.annotations = pd.DataFrame(columns=['source', 'split', 'image_id', 'caption_1']) # Empty dataframe

        # Filter by split
        self.annotations = self.annotations[self.annotations['split'] == split]
        if max_samples:
             self.annotations = self.annotations.head(max_samples)

        self.img_dir = img_dir
        self.feature_extractor = feature_extractor
        self.attention = attention_module
        self.adapter = adapter
        self.processor = processor
        self.split = split

        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        print(f"RISC Dataset loaded for {split} split with {len(self.annotations)} samples.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_id = row['image']
        image_path = os.path.join(self.img_dir, image_id)

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image) # [3, H, W]
            image_tensor = image_tensor.unsqueeze(0) # Add batch dim [1, 3, H, W]

            # 1. Spatial Segmentation
            segments = self.feature_extractor.spatial_segment(image_tensor)

            # 2. Feature Extraction
            segment_features_dict = self.feature_extractor.extract_segment_features(segments)

            # Flatten features from all scales/segments for attention
            all_features = [] # List of tensors, each [1, FeatureDim]
            for scale in ['1x1', '2x2', '4x4']:
                all_features.extend(segment_features_dict[scale])

            # Ensure all features are on CPU before stacking for attention input (if needed)
            all_features_cpu = [f.cpu() for f in all_features]

            # Use attention mechanism
            # Input needs to be list of [Batch, FeatureDim], here Batch=1
            _, aggregated_feature = self.attention(all_features_cpu)
            # aggregated_feature is [1, FeatureDim]

            # 3. Feature Adaptation
            adapted_feature = self.adapter(aggregated_feature) # Output [1, PaliGemmaVisionDim]
            adapted_feature = adapted_feature.squeeze(0) # Remove batch dim [PaliGemmaVisionDim]

            # 4. Prepare Text Input
            captions = [row[f'caption_{i}'] for i in range(1, 6) if f'caption_{i}' in row and pd.notna(row[f'caption_{i}'])]
            if not captions: captions = ["An image"] # Default caption
            caption = np.random.choice(captions) # Select one caption

            # Process text using PaliGemma processor
            # PaliGemma processor requires both image and text with <image> token
            text_prompt = f"<image> caption en {caption}" # Add <image> token at the beginning

            # Create a dummy image for the processor
            dummy_image = Image.new('RGB', (224, 224), color='black')

            # Process both image and text
            inputs = self.processor(
                images=dummy_image,
                text=text_prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            input_ids = inputs.input_ids.squeeze(0) # Remove batch dim
            attention_mask = inputs.attention_mask.squeeze(0) # Remove batch dim

            # Create labels by shifting input_ids
            labels = input_ids.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 # Ignore padding in loss

            return {
                "adapted_vision_features": adapted_feature.detach(),
                "input_ids": input_ids.detach(),
                "attention_mask": attention_mask.detach(),
                "labels": labels.detach()
            }

        except FileNotFoundError:
            print(f"Warning: Image file not found {image_path}. Skipping item {idx}.")
            # Try the next item with a limit to avoid infinite recursion
            if not hasattr(self, '_skip_count'):
                self._skip_count = 0
            self._skip_count += 1

            if self._skip_count > 10:  # Limit recursion depth
                self._skip_count = 0
                raise RuntimeError("Too many skipped items in a row. Check your dataset.")

            return self.__getitem__((idx + 1) % len(self))

        except Exception as e:
            print(f"Error processing item {idx} (image: {image_id}): {e}. Skipping.")
            # Try the next item with a limit to avoid infinite recursion
            if not hasattr(self, '_skip_count'):
                self._skip_count = 0
            self._skip_count += 1

            if self._skip_count > 10:  # Limit recursion depth
                self._skip_count = 0
                raise RuntimeError("Too many skipped items in a row. Check your dataset.")

            return self.__getitem__((idx + 1) % len(self))

# Custom Data Collator
def custom_data_collator(batch):
    # Filter out None items resulting from errors
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if batch is empty

    # Pad sequences to max length in batch
    input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    adapted_features = torch.stack([item['adapted_vision_features'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'external_features': adapted_features
    }

# --- 5. Custom PaliGemma Model Wrapper for External Features ---
class CustomPaliGemmaModelWrapper(nn.Module):
    def __init__(self, model_id, adapter_output_dim):
        super().__init__()
        # Load base PaliGemma model
        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            # Use torch_dtype=torch.float16 if bfloat16 not supported but float16 is
        )
        self.config = self.paligemma.config
        self.adapter_output_dim = adapter_output_dim
        self.paligemma_vision_dim = self.config.vision_config.hidden_size

        # Check if adapter output dim matches PaliGemma vision dim
        if self.adapter_output_dim != self.paligemma_vision_dim:
             print(f"Warning: Adapter output dim ({self.adapter_output_dim}) does not match PaliGemma vision dim ({self.paligemma_vision_dim}). Adding projection.")
             self.vision_projection = nn.Linear(self.adapter_output_dim, self.paligemma_vision_dim)
        else:
             self.vision_projection = nn.Identity()
             print("Adapter output dim matches PaliGemma vision dim. Using Identity projection.")

        print(f"Custom PaliGemma Wrapper initialized for {model_id}.")

    def forward(self, input_ids, attention_mask, external_features=None, labels=None, **kwargs):
        # external_features are the output from the FeatureAdapter [Batch, AdapterOutputDim]
        if external_features is not None:
            # Project features if dimensions don't match
            projected_features = self.vision_projection(external_features.to(self.paligemma.device))
            # Reshape to [Batch, 1, PaliGemmaVisionDim] for PaliGemma's expected format
            vision_hidden_states = projected_features.unsqueeze(1)

            # Pass features as vision_hidden_states, bypassing the internal vision encoder
            outputs = self.paligemma(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_hidden_states=vision_hidden_states,
                labels=labels,
                **kwargs
            )
        else:
            # Standard forward pass if no external features provided
            outputs = self.paligemma(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        return outputs

# --- 6. Initialization ---
print("Initializing components...")
# Initialize Processor & Tokenizer
processor = PaliGemmaProcessor.from_pretrained(PALIGEMMA_MODEL_ID)
# Initialize Feature Extractor, Attention, Adapter
feature_extractor = BNInceptionFeatureExtractor(feature_dim=1024).to(device)
attention_module = SpatialAttention(feature_dim=1024).to(device)
paligemma_hidden_size = PaliGemmaForConditionalGeneration.from_pretrained(PALIGEMMA_MODEL_ID).config.vision_config.hidden_size
adapter = FeatureAdapter(input_dim=1024, output_dim=paligemma_hidden_size).to(device)

# Initialize Custom Model Wrapper
model = CustomPaliGemmaModelWrapper(PALIGEMMA_MODEL_ID, adapter_output_dim=paligemma_hidden_size).to(device)

# --- 7. LoRA Configuration ---
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM # PaliGemma is causal LM
)

# Apply LoRA to the PaliGemma part of the wrapper
model.paligemma = get_peft_model(model.paligemma, lora_config)
print("LoRA applied to PaliGemma model.")
model.paligemma.print_trainable_parameters()

# Set up wandb to watch the model
if USE_WANDB:
    wandb.watch(model, log_freq=WANDB_LOG_FREQ, log="all")

# --- 8. Prepare Dataset and DataLoader ---
print("Preparing datasets...")
# Training dataset
train_dataset = RISCTSNFeatureDataset(
    captions_csv=CAPTIONS_PATH,
    img_dir=RISC_DATASET_PATH,
    feature_extractor=feature_extractor,
    attention_module=attention_module,
    adapter=adapter,
    processor=processor,
    split='train',
    max_samples=MAX_TRAIN_SAMPLES
)

# Validation dataset
val_dataset = RISCTSNFeatureDataset(
    captions_csv=CAPTIONS_PATH,
    img_dir=RISC_DATASET_PATH,
    feature_extractor=feature_extractor,
    attention_module=attention_module,
    adapter=adapter,
    processor=processor,
    split='val',
    max_samples=MAX_VAL_SAMPLES
)

# Check if datasets are empty
if len(train_dataset) == 0:
    print("ERROR: Training dataset is empty. Check paths and CSV content.")
    exit()

if len(val_dataset) == 0:
    print("WARNING: Validation dataset is empty. Evaluation will be skipped.")
    EVAL_DURING_TRAINING = False

print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

# Create DataLoaders
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

# --- Evaluation Function ---
def evaluate_model(model, dataloader, processor, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for step, batch in enumerate(progress_bar):
            if batch is None:  # Skip empty batches
                continue

            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            external_features = batch['external_features'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                external_features=external_features
            )

            loss = outputs.loss
            if loss is not None:
                total_loss += loss.item()

            # Generate predictions for a few samples (for qualitative evaluation)
            if step < 2:  # Only generate for first 2 batches to save time
                # Generate captions
                generated_ids = model.paligemma.generate(
                    input_ids=input_ids[:, :1],  # Only use the first token (usually <image>)
                    attention_mask=attention_mask[:, :1],
                    vision_hidden_states=external_features.unsqueeze(1),
                    max_length=GENERATE_MAX_LENGTH,
                    num_beams=NUM_BEAMS,
                    early_stopping=True
                )

                # Decode generated captions
                generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

                # Decode reference captions (labels)
                label_captions = processor.batch_decode(
                    labels.masked_fill(labels == -100, processor.tokenizer.pad_token_id),
                    skip_special_tokens=True
                )

                # Store predictions and references
                all_preds.extend(generated_captions)
                all_labels.extend(label_captions)

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)

    # Print some examples
    print("\n=== Validation Examples ===")
    for i in range(min(3, len(all_preds))):
        print(f"Example {i+1}:")
        print(f"  Prediction: {all_preds[i]}")
        print(f"  Reference: {all_labels[i]}")
        print()

    # Return to training mode
    model.train()

    return avg_loss, all_preds, all_labels

# --- 9. Training Loop ---
print("Setting up training...")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Move model components requiring gradients to train mode
model.train()
# Feature extractor is frozen
feature_extractor.eval()

print("Starting training loop...")
global_step = 0
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    model.train()  # Ensure model is in training mode
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(progress_bar):
        if batch is None:  # Skip empty batches from collator
            continue

        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        external_features = batch['external_features'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            external_features=external_features
        )

        loss = outputs.loss
        if loss is None:
            print("Warning: Loss is None. Check model output and labels.")
            continue

        loss = loss / GRADIENT_ACCUMULATION_STEPS  # Scale loss
        loss.backward()  # Accumulate gradients

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()  # Update parameters
            optimizer.zero_grad()  # Reset gradients
            global_step += 1
            current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            total_loss += current_loss

            # Log metrics to wandb
            if USE_WANDB and global_step % WANDB_LOG_FREQ == 0:
                wandb.log({
                    "train/loss": current_loss,
                    "train/avg_loss": total_loss / (global_step * GRADIENT_ACCUMULATION_STEPS),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                })

            progress_bar.set_postfix({'Loss': f'{current_loss:.4f}', 'AvgLoss': f'{total_loss / (global_step * GRADIENT_ACCUMULATION_STEPS):.4f}'})

    avg_epoch_loss = total_loss / (len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_epoch_loss:.4f}")

    # Validation
    if EVAL_DURING_TRAINING and len(val_dataset) > 0:
        print(f"Running validation for epoch {epoch+1}...")
        val_loss, val_preds, val_labels = evaluate_model(model, val_dataloader, processor, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Log validation metrics to wandb
        if USE_WANDB:
            wandb.log({
                "val/loss": val_loss,
                "val/epoch": epoch + 1,
                "val/examples": len(val_preds),
            })

            # Log validation examples as a table
            if len(val_preds) > 0:
                examples_table = wandb.Table(columns=["Prediction", "Reference"])
                for pred, ref in zip(val_preds[:min(5, len(val_preds))], val_labels[:min(5, len(val_labels))]):
                    examples_table.add_data(pred, ref)
                wandb.log({"val/examples_table": examples_table})

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = os.path.join(OUTPUT_DIR, "best_model")
            model.paligemma.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"New best model saved to {checkpoint_dir}")

            # Log best model to wandb
            if USE_WANDB and WANDB_LOG_MODEL:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
                # Log model as artifact
                model_artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
                model_artifact.add_dir(checkpoint_dir)
                wandb.log_artifact(model_artifact)

    # Save checkpoint at specified intervals
    if (epoch + 1) % CONFIG["output"]["save_every_n_epochs"] == 0:
        checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        model.paligemma.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")

print("Training finished.")

# --- 10. Save Final Model --- (Save LoRA adapters)
final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
model.paligemma.save_pretrained(final_model_dir)
processor.save_pretrained(final_model_dir)
print(f"Final LoRA adapters and processor saved to {final_model_dir}")

# Log final model to wandb
if USE_WANDB and WANDB_LOG_MODEL:
    final_model_artifact = wandb.Artifact(f"final-model-{wandb.run.id}", type="model")
    final_model_artifact.add_dir(final_model_dir)
    wandb.log_artifact(final_model_artifact)

    # Add final metrics to summary
    wandb.run.summary["final_train_loss"] = avg_epoch_loss
    wandb.run.summary["total_epochs"] = NUM_EPOCHS
    wandb.run.summary["total_steps"] = global_step

    # Finish the wandb run
    wandb.finish()

print("Code execution finished conceptually.") 