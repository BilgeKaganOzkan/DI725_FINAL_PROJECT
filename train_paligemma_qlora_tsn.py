#!/usr/bin/env python3
import os
import torch
import yaml
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb
import warnings
import logging
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.amp import autocast, GradScaler

# Import TSN models
from models.tsn_paligemma_model import create_tsn_paligemma_model, TSNModule
from models.tsn_paligemma_adapter import create_tsn_paligemma_adapter
from models.tsn_paligemma_direct import create_tsn_paligemma_direct_model
from models.tsn_paligemma_enhanced import create_tsn_paligemma_enhanced_model

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="You are passing both `text` and `images` to `PaliGemmaProcessor`")
logging.getLogger("transformers").setLevel(logging.ERROR)

class RISCDataset(Dataset):
    """
    Dataset class for RISC (Remote Sensing Image Captioning) dataset.
    """
    def __init__(self, csv_path, img_dir, processor, split='train', max_samples=None, max_length=20):
        self.annotations = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.processor = processor
        self.split = split
        self.max_length = max_length

        # Limit samples if specified
        if max_samples is not None and max_samples > 0:
            self.annotations = self.annotations.sample(min(max_samples, len(self.annotations)), random_state=42)

        print(f"Loaded {len(self.annotations)} samples for {split}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get image path and caption
        row = self.annotations.iloc[idx]
        img_path = row['image_path']
        caption = row['caption']

        # Normalize path separators for cross-platform compatibility
        img_path = os.path.normpath(img_path).replace('\\', '/')

        # Load and preprocess image
        try:
            # Open image and convert to RGB
            image = Image.open(img_path).convert('RGB')

            # Convert to numpy array and normalize to [0, 1] range
            image_np = np.array(image).astype(np.float32) / 255.0

            # Convert back to PIL image
            image = Image.fromarray((image_np * 255).astype(np.uint8))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder black image
            image = Image.new('RGB', (224, 224), color='black')

        # Get additional metadata if available
        metadata = {}
        for col in ['source', 'category', 'caption_length']:
            if col in row:
                metadata[col] = row[col]

        # Process image and text
        try:
            # Use the processor directly with the image and caption
            encoding = self.processor(
                images=image,
                text=caption,  # Use the actual caption directly
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

            # Set labels to be the same as input_ids for training
            encoding['labels'] = encoding['input_ids'].clone()

            # Remove batch dimension
            for k, v in encoding.items():
                encoding[k] = v.squeeze()

            # Add metadata
            encoding.update(metadata)
        except Exception as e:
            print(f"Error processing sample: {e}")
            # Fallback processing
            try:
                # Process image only
                pixel_values = self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"]

                # Process text only - use a simple caption
                text_encoding = self.processor.tokenizer(
                    "This is an image.",  # Simple placeholder
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                # Create encoding dictionary
                encoding = {
                    'pixel_values': pixel_values.squeeze(),
                    'input_ids': text_encoding['input_ids'].squeeze(),
                    'attention_mask': text_encoding['attention_mask'].squeeze(),
                    'labels': text_encoding['input_ids'].squeeze().clone()
                }

                # Add metadata
                encoding.update(metadata)
            except Exception as e:
                print(f"Critical error in fallback processing: {e}")
                # Last resort - create empty tensors
                encoding = {
                    'pixel_values': torch.zeros((3, 224, 224)),
                    'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                    'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                    'labels': torch.zeros(self.max_length, dtype=torch.long)
                }
                encoding.update(metadata)

        return encoding

def custom_collate_fn(batch):
    """
    Custom collate function that handles tensors of different dimensions.

    Args:
        batch: Examples in the batch

    Returns:
        Combined batch
    """
    # Check for empty batch
    if len(batch) == 0:
        return {}

    # Get all keys
    keys = batch[0].keys()

    # Collect values for each key
    result = {}
    for key in keys:
        # Collect metadata fields as lists
        if key in ['source', 'category', 'caption_length']:
            result[key] = [item[key] for item in batch if key in item]
        else:
            # Stack tensor fields
            try:
                # First check if all tensors have the same shape
                tensors = [item[key] for item in batch if key in item]
                if all(t.shape == tensors[0].shape for t in tensors):
                    result[key] = torch.stack(tensors)
                else:
                    # Apply padding for tensors of different dimensions
                    max_dim = max(t.shape[0] for t in tensors)
                    padded_tensors = []

                    for t in tensors:
                        if t.shape[0] < max_dim:
                            # Apply padding
                            if t.dim() == 1:
                                padding = torch.zeros(max_dim - t.shape[0], dtype=t.dtype, device=t.device)
                                padded_t = torch.cat([t, padding])
                            else:
                                padding_shape = list(t.shape)
                                padding_shape[0] = max_dim - t.shape[0]
                                padding = torch.zeros(padding_shape, dtype=t.dtype, device=t.device)
                                padded_t = torch.cat([t, padding], dim=0)
                            padded_tensors.append(padded_t)
                        else:
                            padded_tensors.append(t)

                    result[key] = torch.stack(padded_tensors)
            except Exception as e:
                print(f"Error stacking tensors for key {key}: {e}")
                # Skip this field in case of error
                continue

    return result

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train():
    # Load configuration
    config_path = "config/config.yaml"
    config = load_config(config_path)

    # Extract configuration values
    PALIGEMMA_MODEL_ID = config['model']['model_id']
    TRAIN_CSV = config['data']['train_csv']
    VAL_CSV = config['data']['val_csv']
    RISC_DATASET_PATH = config['data']['dataset_path']
    OUTPUT_DIR = config['output']['output_dir']
    BATCH_SIZE = config['training']['batch_size']
    GRADIENT_ACCUMULATION_STEPS = config['training']['gradient_accumulation_steps']
    LEARNING_RATE = config['training']['learning_rate']
    NUM_EPOCHS = config['training']['num_epochs']
    WEIGHT_DECAY = config['training']['weight_decay']
    MAX_TRAIN_SAMPLES = config['training'].get('max_train_samples', -1)
    MAX_VAL_SAMPLES = config['training'].get('max_val_samples', -1)
    MIXED_PRECISION = config['training'].get('mixed_precision', True)
    LABEL_SMOOTHING = config['training'].get('label_smoothing', 0.1)

    # LoRA configuration
    LORA_R = config['lora']['r']
    LORA_ALPHA = config['lora']['lora_alpha']
    LORA_DROPOUT = config['lora']['lora_dropout']
    LORA_TARGET_MODULES = config['lora']['target_modules']

    # Evaluation configuration
    EVAL_DURING_TRAINING = config['evaluation']['eval_during_training']
    GENERATE_MAX_LENGTH = config['evaluation']['generate_max_length']
    MIN_LENGTH = config['evaluation']['min_length']
    NUM_BEAMS = config['evaluation']['num_beams']

    # Prompt configuration is no longer needed as we use direct captions

    # Wandb configuration
    USE_WANDB = config['training'].get('use_wandb', True)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Weights & Biases
    if USE_WANDB:
        # Get WandB configuration from config file
        wandb_config = config.copy()
        wandb_project = config.get('wandb', {}).get('project', "2697134-tsn-paligemma-project")
        wandb_run_name = config.get('wandb', {}).get('run_name', "tsn-inception_v3-lora-r8")

        print(f"Initializing WandB with run name: {wandb_run_name}")

        # Initialize WandB with configuration from config file
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config
        )

    # Load processor
    print("Loading PaliGemma processor...")
    processor = PaliGemmaProcessor.from_pretrained(PALIGEMMA_MODEL_ID)

    # Set up quantization config for QLoRA
    print("Setting up quantization config for QLoRA...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model with quantization
    print(f"Loading PaliGemma model: {PALIGEMMA_MODEL_ID}...")
    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": {"": 0},
        "torch_dtype": torch.bfloat16
    }

    # Load the model
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        PALIGEMMA_MODEL_ID,
        **model_kwargs
    )

    # Print model configuration
    print(f"Model loaded with label_smoothing={LABEL_SMOOTHING}")

    # Set a fixed maximum length for PaliGemma
    # Print model configuration
    print(f"Model config: {base_model.config}")

    # Determine the actual maximum length of the model
    # PaliGemma model's max_length appears to be 20, we'll use this value
    MAX_SEQ_LENGTH = 20  # Expected value for the model

    print(f"Using MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")

    # If max_length exists in the model configuration, use it
    if hasattr(base_model.config, 'max_length'):
        MAX_SEQ_LENGTH = base_model.config.max_length
        print(f"Using model's max_length: {MAX_SEQ_LENGTH}")

    # Configure LoRA for efficient fine-tuning
    print("Configuring QLoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
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

    # Select TSN integration method
    TSN_INTEGRATION_METHOD = config.get('tsn', {}).get('integration_method', 'enhanced')
    print(f"Using TSN integration method: {TSN_INTEGRATION_METHOD}")

    # Create model based on selected integration method
    if TSN_INTEGRATION_METHOD == 'adapter':
        # Use adapter approach
        final_model = create_tsn_paligemma_adapter(lora_model, config)
        print("Created PaliGemma-LoRA model with TSN Adapter")
    elif TSN_INTEGRATION_METHOD == 'direct':
        # Use direct output manipulation approach
        final_model = create_tsn_paligemma_direct_model(lora_model, config)
        print("Created PaliGemma-LoRA model with TSN Direct Output Manipulation")
    elif TSN_INTEGRATION_METHOD == 'enhanced':
        # Use enhanced encoder output approach
        final_model = create_tsn_paligemma_enhanced_model(lora_model, config)
        print("Created PaliGemma-LoRA model with TSN Enhanced Encoder Integration")
    else:
        # Use vision tower replacement approach
        final_model = create_tsn_paligemma_model(lora_model, config)
        print("Created PaliGemma-LoRA model with TSN (vision tower replacement)")

    # Ensure model is on the correct device
    final_model = final_model.to(device)

    # Prepare datasets
    print("Preparing datasets...")
    # Pass maximum length to RISCDataset class
    train_dataset = RISCDataset(
        csv_path=TRAIN_CSV,
        img_dir=RISC_DATASET_PATH,
        processor=processor,
        split='train',
        max_samples=MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES > 0 else None,
        max_length=MAX_SEQ_LENGTH  # Pass maximum length
    )

    val_dataset = RISCDataset(
        csv_path=VAL_CSV,
        img_dir=RISC_DATASET_PATH,
        processor=processor,
        split='val',
        max_samples=MAX_VAL_SAMPLES if MAX_VAL_SAMPLES > 0 else None,
        max_length=MAX_SEQ_LENGTH  # Pass maximum length
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate_fn,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn,
        drop_last=False
    )

    # Prepare optimizer and scheduler
    optimizer = AdamW(
        final_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Create learning rate scheduler
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if MIXED_PRECISION else None

    # Training loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        # Training phase
        final_model.train()
        train_loss = 0.0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass with mixed precision
            if MIXED_PRECISION:
                with autocast(device_type=device.type):
                    outputs = final_model(
                        pixel_values=batch['pixel_values'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(final_model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            else:
                # Standard training without mixed precision
                outputs = final_model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(final_model.parameters(), 0.1)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Update metrics
            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            train_steps += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})

            # Log to wandb
            if USE_WANDB and (step + 1) % 10 == 0:
                wandb.log({
                    "train/loss": loss.item() * GRADIENT_ACCUMULATION_STEPS,
                    "train/learning_rate": lr_scheduler.get_last_lr()[0]
                })

            # Generate sample output every 100 steps
            if (step + 1) % 100 == 0:
                # Get a sample image from the batch
                sample_image = batch['pixel_values'][0].unsqueeze(0)

                # Generate caption
                final_model.eval()
                with torch.no_grad():
                    # Convert tensor to PIL image for proper normalization
                    sample_image_cpu = sample_image.cpu()

                    # Ensure values are in [0, 1] range
                    sample_image_cpu = torch.clamp(sample_image_cpu, 0, 1)

                    # Convert to numpy and then to PIL
                    sample_np = sample_image_cpu.squeeze(0).permute(1, 2, 0).numpy()
                    sample_pil = Image.fromarray((sample_np * 255).astype(np.uint8))

                    # Get prompt from config
                    input_prompt = config.get('prompt', {}).get('input_prompt', 'Caption this image:')
                    print(f"\nUsing prompt from config: '{input_prompt}'")

                    # Process with processor - use starting text (prompt)
                    sample_inputs = processor(
                        images=sample_pil,  # Use PIL image
                        text=input_prompt,  # Use prompt from config
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=MAX_SEQ_LENGTH
                    ).to(device)

                    # Debug - decode input_ids
                    input_text = processor.tokenizer.decode(sample_inputs['input_ids'][0], skip_special_tokens=True)
                    print(f"Input text: '{input_text}'")

                    # Debug info - before generate
                    print(f"\nBefore generate - pixel_values shape: {sample_inputs['pixel_values'].shape}")
                    print(f"Before generate - input_ids shape: {sample_inputs['input_ids'].shape}")
                    print(f"Before generate - input_ids: {sample_inputs['input_ids']}")
                    print(f"Before generate - attention_mask shape: {sample_inputs['attention_mask'].shape}")

                    # First test using the PaliGemma model directly
                    print("\n--- Testing with direct PaliGemma model ---")
                    with torch.no_grad():
                        # Use PaliGemma model directly
                        paligemma_outputs = final_model.paligemma.generate(
                            pixel_values=sample_inputs['pixel_values'],
                            input_ids=sample_inputs['input_ids'],
                            attention_mask=sample_inputs['attention_mask'],
                            max_new_tokens=GENERATE_MAX_LENGTH,
                            num_beams=NUM_BEAMS,
                            do_sample=False,
                            temperature=1.0,
                            top_p=1.0,
                            no_repeat_ngram_size=2,
                            min_length=MIN_LENGTH
                        )

                        # Decode PaliGemma output
                        paligemma_caption = processor.batch_decode(paligemma_outputs, skip_special_tokens=True)[0]
                        print(f"PaliGemma direct output: '{paligemma_caption}'")

                    # Now use the TSN-integrated model
                    print("\n--- Testing with TSN-integrated PaliGemma model ---")
                    # Call generate function based on TSN integration method
                    if TSN_INTEGRATION_METHOD == 'direct':
                        # Pass processor for direct model
                        sample_outputs = final_model.generate(
                            pixel_values=sample_inputs['pixel_values'],
                            input_ids=sample_inputs['input_ids'],
                            attention_mask=sample_inputs['attention_mask'],
                            max_new_tokens=GENERATE_MAX_LENGTH,
                            num_beams=NUM_BEAMS,
                            do_sample=False,
                            temperature=1.0,
                            top_p=1.0,
                            no_repeat_ngram_size=2,
                            min_length=MIN_LENGTH,
                            processor=processor  # Pass processor (for direct model)
                        )
                    else:
                        # Don't pass processor for other models
                        sample_outputs = final_model.generate(
                            pixel_values=sample_inputs['pixel_values'],
                            input_ids=sample_inputs['input_ids'],
                            attention_mask=sample_inputs['attention_mask'],
                            max_new_tokens=GENERATE_MAX_LENGTH,
                            num_beams=NUM_BEAMS,
                            do_sample=False,
                            temperature=1.0,
                            top_p=1.0,
                            no_repeat_ngram_size=2,
                            min_length=MIN_LENGTH
                        )

                    # Debug info - after generate
                    print(f"After generate - sample_outputs shape: {sample_outputs.shape}")
                    print(f"After generate - sample_outputs: {sample_outputs}")

                    # Add debug info
                    print(f"\nRaw sample outputs shape: {sample_outputs.shape}")
                    print(f"Raw sample outputs: {sample_outputs}")

                    # Decode each token separately
                    print("\nToken-by-token decoding:")
                    for i in range(sample_outputs.shape[1]):
                        token_id = sample_outputs[0, i].item()
                        token = processor.tokenizer.convert_ids_to_tokens(token_id)
                        print(f"Token {i}: ID={token_id}, Token='{token}'")

                    # Decode generated caption
                    generated_caption = processor.batch_decode(sample_outputs, skip_special_tokens=True)[0]
                    print(f"Decoded with skip_special_tokens=True: '{generated_caption}'")

                    # Decode without skipping special tokens
                    full_caption = processor.batch_decode(sample_outputs, skip_special_tokens=False)[0]
                    print(f"Decoded with skip_special_tokens=False: '{full_caption}'")

                    # Get ground truth caption
                    # Replace -100 values with tokenizer's pad_token_id
                    labels = batch['labels'][0].unsqueeze(0).clone()
                    labels[labels == -100] = processor.tokenizer.pad_token_id
                    ground_truth = processor.batch_decode(labels, skip_special_tokens=True)[0]

                    print(f"\nSample generation:")
                    print(f"Generated: {generated_caption}")
                    print(f"Ground truth: {ground_truth}")

                    if USE_WANDB:
                        wandb.log({
                            "train/sample_generation": wandb.Html(
                                f"<p><b>Generated:</b> {generated_caption}</p>"
                                f"<p><b>Ground truth:</b> {ground_truth}</p>"
                            )
                        })

                # Switch back to training mode
                final_model.train()

        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average training loss: {avg_train_loss:.4f}")

        # Validation phase
        if EVAL_DURING_TRAINING:
            final_model.eval()
            val_loss = 0.0
            val_steps = 0

            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Forward pass
                with torch.no_grad():
                    outputs = final_model(
                        pixel_values=batch['pixel_values'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss

                # Update metrics
                val_loss += loss.item()
                val_steps += 1

                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})

                # Log validation loss to wandb for each batch
                if USE_WANDB:
                    wandb.log({
                        "val/loss": loss.item()
                    })

                # Generate sample output for the first batch
                if step == 0:
                    # Get a sample image from the batch
                    sample_image = batch['pixel_values'][0].unsqueeze(0)

                    # Generate caption
                    with torch.no_grad():
                        # Convert tensor to PIL image for proper normalization
                        sample_image_cpu = sample_image.cpu()

                        # Ensure values are in [0, 1] range
                        sample_image_cpu = torch.clamp(sample_image_cpu, 0, 1)

                        # Convert to numpy and then to PIL
                        sample_np = sample_image_cpu.squeeze(0).permute(1, 2, 0).numpy()
                        sample_pil = Image.fromarray((sample_np * 255).astype(np.uint8))

                        # Get prompt from config (validation)
                        input_prompt = config.get('prompt', {}).get('input_prompt', 'Caption this image:')
                        print(f"\nValidation - Using prompt from config: '{input_prompt}'")

                        # Process with processor - use starting text (prompt) (validation)
                        sample_inputs = processor(
                            images=sample_pil,  # Use PIL image
                            text=input_prompt,  # Use prompt from config
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=MAX_SEQ_LENGTH
                        ).to(device)

                        # Debug - decode input_ids (validation)
                        input_text = processor.tokenizer.decode(sample_inputs['input_ids'][0], skip_special_tokens=True)
                        print(f"Validation - Input text: '{input_text}'")

                        # Debug info - before generate (validation)
                        print(f"\nValidation - Before generate - pixel_values shape: {sample_inputs['pixel_values'].shape}")
                        print(f"Validation - Before generate - input_ids shape: {sample_inputs['input_ids'].shape}")
                        print(f"Validation - Before generate - input_ids: {sample_inputs['input_ids']}")
                        print(f"Validation - Before generate - attention_mask shape: {sample_inputs['attention_mask'].shape}")

                        # First test using the PaliGemma model directly (validation)
                        print("\n--- Validation - Testing with direct PaliGemma model ---")
                        with torch.no_grad():
                            # Use PaliGemma model directly
                            paligemma_outputs = final_model.paligemma.generate(
                                pixel_values=sample_inputs['pixel_values'],
                                input_ids=sample_inputs['input_ids'],
                                attention_mask=sample_inputs['attention_mask'],
                                max_new_tokens=GENERATE_MAX_LENGTH,
                                num_beams=NUM_BEAMS,
                                do_sample=False,
                                temperature=1.0,
                                top_p=1.0,
                                no_repeat_ngram_size=2,
                                min_length=MIN_LENGTH
                            )

                            # Decode PaliGemma output
                            paligemma_caption = processor.batch_decode(paligemma_outputs, skip_special_tokens=True)[0]
                            print(f"Validation - PaliGemma direct output: '{paligemma_caption}'")

                        # Now use the TSN-integrated model (validation)
                        print("\n--- Validation - Testing with TSN-integrated PaliGemma model ---")
                        # Call generate function based on TSN integration method (validation)
                        if TSN_INTEGRATION_METHOD == 'direct':
                            # Pass processor for direct model
                            sample_outputs = final_model.generate(
                                pixel_values=sample_inputs['pixel_values'],
                                input_ids=sample_inputs['input_ids'],
                                attention_mask=sample_inputs['attention_mask'],
                                max_new_tokens=GENERATE_MAX_LENGTH,
                                num_beams=NUM_BEAMS,
                                do_sample=False,
                                temperature=1.0,
                                top_p=1.0,
                                no_repeat_ngram_size=2,
                                min_length=MIN_LENGTH,
                                processor=processor  # Pass processor (for direct model)
                            )
                        else:
                            # Don't pass processor for other models
                            sample_outputs = final_model.generate(
                                pixel_values=sample_inputs['pixel_values'],
                                input_ids=sample_inputs['input_ids'],
                                attention_mask=sample_inputs['attention_mask'],
                                max_new_tokens=GENERATE_MAX_LENGTH,
                                num_beams=NUM_BEAMS,
                                do_sample=False,
                                temperature=1.0,
                                top_p=1.0,
                                no_repeat_ngram_size=2,
                                min_length=MIN_LENGTH
                            )

                        # Debug info - after generate (validation)
                        print(f"Validation - After generate - sample_outputs shape: {sample_outputs.shape}")
                        print(f"Validation - After generate - sample_outputs: {sample_outputs}")

                        # Add debug info
                        print(f"\nValidation - Raw sample outputs shape: {sample_outputs.shape}")
                        print(f"Validation - Raw sample outputs: {sample_outputs}")

                        # Decode each token separately (validation)
                        print("\nValidation - Token-by-token decoding:")
                        for i in range(sample_outputs.shape[1]):
                            token_id = sample_outputs[0, i].item()
                            token = processor.tokenizer.convert_ids_to_tokens(token_id)
                            print(f"Token {i}: ID={token_id}, Token='{token}'")

                        # Decode generated caption
                        generated_caption = processor.batch_decode(sample_outputs, skip_special_tokens=True)[0]
                        print(f"Validation - Decoded with skip_special_tokens=True: '{generated_caption}'")

                        # Decode without skipping special tokens
                        full_caption = processor.batch_decode(sample_outputs, skip_special_tokens=False)[0]
                        print(f"Validation - Decoded with skip_special_tokens=False: '{full_caption}'")

                        # Get ground truth caption
                        # Replace -100 values with tokenizer's pad_token_id
                        labels = batch['labels'][0].unsqueeze(0).clone()
                        labels[labels == -100] = processor.tokenizer.pad_token_id
                        ground_truth = processor.batch_decode(labels, skip_special_tokens=True)[0]

                        print(f"\nValidation sample generation:")
                        print(f"Generated: {generated_caption}")
                        print(f"Ground truth: {ground_truth}")

                        if USE_WANDB:
                            wandb.log({
                                "val/sample_generation": wandb.Html(
                                    f"<p><b>Generated:</b> {generated_caption}</p>"
                                    f"<p><b>Ground truth:</b> {ground_truth}</p>"
                                )
                            })

            # Calculate average validation loss
            avg_val_loss = val_loss / val_steps
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average validation loss: {avg_val_loss:.4f}")

            # Log to wandb
            if USE_WANDB:
                wandb.log({
                    "train/epoch_loss": avg_train_loss,
                    "val/epoch_loss": avg_val_loss,  # Only log epoch average as val/epoch_loss
                    "epoch": epoch + 1
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

    print("Training finished.")

    # Save final model
    final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    # Save PaliGemma model
    final_model.paligemma.save_pretrained(os.path.join(final_model_dir, "paligemma"))

    # Save TSN model if available
    if hasattr(final_model, 'tsn') and final_model.tsn is not None:
        torch.save(final_model.tsn.state_dict(), os.path.join(final_model_dir, "tsn_model.pt"))

    print(f"Saved final model to {final_model_dir}")

    # Close wandb
    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    train()
