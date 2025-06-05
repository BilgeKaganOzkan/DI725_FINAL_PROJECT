#!/usr/bin/env python3

# Core PyTorch and system imports
import os
import torch
import yaml
import pandas as pd
import numpy as np
import random
from PIL import Image
import wandb
import warnings

# PyTorch utilities for data handling
from torch.utils.data import Dataset

# HuggingFace transformers for model training
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

# PEFT (Parameter-Efficient Fine-Tuning) for LoRA
from peft import LoraConfig, get_peft_model

# Custom TSN integration module
from models.tsn_paligemma_model import create_tsn_paligemma_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class ValidationLossCallback(TrainerCallback):
    """
    Custom callback to periodically evaluate model performance on validation set.
    
    This callback:
    - Runs validation at specific training steps
    - Calculates average validation loss
    - Logs results to WandB if enabled
    - Uses a subset of validation data for efficiency
    """

    def __init__(self, val_dataset, processor, device, use_wandb=False):
        """
        Initialize validation callback.
        
        Args:
            val_dataset: Validation dataset for evaluation
            processor: PaliGemma processor for text/image handling
            device: GPU/CPU device for computation
            use_wandb (bool): Whether to log to Weights & Biases
        """
        self.val_dataset = val_dataset
        self.processor = processor
        self.device = device
        self.use_wandb = use_wandb
        # Define specific steps for validation (strategic intervals)
        self.validation_steps = [200, 400, 600, 800, 1000, 1200, 1400]

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Called at the end of each training step.
        Performs validation if current step matches validation schedule.
        
        Args:
            args: Training arguments
            state: Current training state
            control: Training control flags
            model: Current model being trained
        """
        # Check if current step requires validation
        if state.global_step in self.validation_steps and self.val_dataset is not None:
            model.eval()  # Set model to evaluation mode
            total_loss = 0.0
            num_samples = min(100, len(self.val_dataset))  # Limit samples for efficiency

            # Disable gradient computation for validation
            with torch.no_grad():
                # Process subset of validation samples
                for i in range(num_samples):
                    try:
                        # Get validation sample
                        sample = self.val_dataset[i]
                        image = sample['image']
                        caption = sample['caption']

                        # Prepare input text with standard PaliGemma format
                        text = "<image> <bos> describe this image.}"
                        
                        # Try suffix-based processing first
                        try:
                            inputs = self.processor(
                                text=[text],
                                images=[image],
                                suffix=[caption],
                                return_tensors="pt",
                                padding="longest"
                            ).to(self.device)
                        except (AttributeError, TypeError):
                            # Fallback to manual text construction
                            full_text = text + caption
                            inputs = self.processor(
                                text=[full_text],
                                images=[image],
                                return_tensors="pt",
                                padding="longest"
                            ).to(self.device)
                            # Set labels for loss calculation
                            inputs["labels"] = inputs["input_ids"].clone()

                        # Forward pass with mixed precision
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            outputs = model(**inputs)
                            loss = outputs.loss
                            total_loss += loss.item()

                    except Exception:
                        # Skip problematic samples
                        continue

            # Calculate and log average validation loss
            if num_samples > 0:
                avg_val_loss = total_loss / num_samples

                # Log to WandB if enabled
                if self.use_wandb:
                    import wandb
                    if wandb.run is not None:
                        try:
                            wandb.log({
                                "validation/loss": float(avg_val_loss),
                                "validation/samples_used": int(num_samples),
                                "validation/step": int(state.global_step)
                            }, step=state.global_step)
                        except Exception:
                            pass

            model.train()  # Return model to training mode

class BestModelCallback(TrainerCallback):
    """
    Callback to automatically save the best performing model during training.
    
    This callback:
    - Monitors training loss continuously
    - Saves model when loss improves
    - Handles TSN wrapper extraction for proper saving
    - Verifies essential files are saved correctly
    """

    def __init__(self, output_dir, processor, use_wandb=False):
        """
        Initialize best model callback.
        
        Args:
            output_dir (str): Directory to save best model
            processor: PaliGemma processor to save
            use_wandb (bool): Whether to log to WandB
        """
        self.output_dir = output_dir
        self.processor = processor
        self.use_wandb = use_wandb
        self.best_loss = float('inf')  # Track best loss seen so far
        self.best_model_dir = os.path.join(output_dir, "best_model")
        os.makedirs(self.best_model_dir, exist_ok=True)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        Called when training logs are updated.
        Saves model if training loss has improved.
        
        Args:
            args: Training arguments
            state: Current training state
            control: Training control flags
            model: Current model being trained
            logs: Dictionary of logged metrics
        """
        # Check if training loss is available and improved
        if logs and "train_loss" in logs:
            current_loss = logs["train_loss"]

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                print(f"\n[BEST] New best model! Loss: {current_loss:.4f} (Step: {state.global_step})")

                try:
                    # Determine which model to save (handle TSN wrapper)
                    model_to_save = model

                    # Extract PaliGemma model from TSN wrapper if present
                    if hasattr(model, 'paligemma'):
                        model_to_save = model.paligemma
                        print("[CONFIG] Detected TSN wrapper, extracting PaliGemma model")
                    elif hasattr(model, 'module') and hasattr(model.module, 'paligemma'):
                        # Handle DataParallel case
                        model_to_save = model.module.paligemma
                        print("[CONFIG] Detected TSN wrapper in module, extracting PaliGemma model")
                    else:
                        print("[WARNING] No TSN wrapper detected, using model directly")

                    # Save LoRA adapter model
                    print(f"[SAVING] Saving model to: {self.best_model_dir}")
                    model_to_save.save_pretrained(self.best_model_dir, safe_serialization=False)
                    print(f"[SUCCESS] Best model saved to: {self.best_model_dir}")

                    # Save processor (handles tokenization and image preprocessing)
                    self.processor.save_pretrained(self.best_model_dir)
                    print(f"[SUCCESS] Processor saved to: {self.best_model_dir}")

                    # Verify essential LoRA files are present
                    essential_files = ["adapter_config.json", "adapter_model.bin"]
                    for file in essential_files:
                        if os.path.exists(os.path.join(self.best_model_dir, file)):
                            print(f"[SUCCESS] {file} saved successfully")
                        else:
                            print(f"[ERROR] Missing: {file}")

                    # Log to WandB if enabled
                    if self.use_wandb:
                        import wandb
                        if wandb.run is not None:
                            try:
                                wandb.log({
                                    "best_model/loss": float(current_loss),
                                    "best_model/step": int(state.global_step),
                                    "best_model/saved": True
                                }, step=state.global_step)
                            except Exception as e:
                                print(f"[ERROR] WandB logging error: {e}")

                except Exception as e:
                    print(f"[ERROR] Failed to save best model: {e}")
                    import traceback
                    traceback.print_exc()

class SamplingCallback(TrainerCallback):
    """
    Callback for periodic caption generation during training.
    
    This callback:
    - Generates sample captions every 200 steps
    - Tracks mixing ratios for TSN models
    - Monitors caption quality over time
    - Creates sampling tables for analysis
    """

    def __init__(self, processor, val_dataset, device, use_wandb=False):
        """
        Initialize sampling callback.
        
        Args:
            processor: PaliGemma processor
            val_dataset: Validation dataset for sampling
            device: GPU/CPU device
            use_wandb (bool): Whether to log to WandB
        """
        self.processor = processor
        self.val_dataset = val_dataset
        self.device = device
        self.sample_count = 0
        self.use_wandb = use_wandb
        self.sampling_data = []  # Store sampling results for analysis

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Called at end of each training step.
        Updates TSN mixing ratios and generates sample captions periodically.
        
        Args:
            args: Training arguments
            state: Current training state
            control: Training control flags
            model: Current model being trained
        """
        # Update TSN mixing ratios based on training progress
        if hasattr(model, 'update_training_step'):
            model.update_training_step(state.global_step)
        elif hasattr(model, 'module') and hasattr(model.module, 'update_training_step'):
            model.module.update_training_step(state.global_step)

        # Generate sample captions every 200 steps
        if state.global_step % 200 == 0 and state.global_step > 0:
            self.sample_count += 1

            # Get current TSN mixing ratio if available
            if hasattr(model, 'get_current_mixing_ratios'):
                _, tsn_ratio = model.get_current_mixing_ratios()
            elif hasattr(model, 'module') and hasattr(model.module, 'get_current_mixing_ratios'):
                _, tsn_ratio = model.module.get_current_mixing_ratios()
            else:
                tsn_ratio = 0.0

            # Generate sample caption
            model.eval()
            with torch.no_grad():
                # Select random validation sample
                sample_idx = random.randint(0, len(self.val_dataset) - 1)
                sample = self.val_dataset[sample_idx]
                image = sample['image']
                ground_truth = sample['caption']

                try:
                    # Prepare input for caption generation
                    inputs = self.processor(
                        images=image,
                        text="<image>",
                        return_tensors="pt",
                        padding=False
                    ).to(self.device)

                    # Generate caption with mixed precision
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            num_beams=1,
                            do_sample=False,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id
                        )

                    generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    generated_caption = generated_text.replace("<image>", "").strip()

                    if self.use_wandb:
                        import wandb

                        if wandb.run is None:
                            return

                        quality_score = 1.0 if len(generated_caption) >= 5 and "sorry" not in generated_caption.lower() and "cannot" not in generated_caption.lower() else 0.0

                        tsn_ratio = 0.0
                        if hasattr(model, 'get_current_mixing_ratios'):
                            _, tsn_ratio = model.get_current_mixing_ratios()
                        elif hasattr(model, 'module') and hasattr(model.module, 'get_current_mixing_ratios'):
                            _, tsn_ratio = model.module.get_current_mixing_ratios()

                        clean_ground_truth = ground_truth.replace('\n', ' ').replace('\r', ' ').strip()
                        clean_generated = generated_caption.replace('\n', ' ').replace('\r', ' ').strip()

                        try:
                            numerical_metrics = {
                                "sampling/caption_length": int(len(generated_caption)),
                                "sampling/quality_score": float(quality_score),
                                "sampling/tsn_ratio": float(tsn_ratio),
                                "sampling/sample_count": int(self.sample_count),
                                "sampling/ground_truth_length": int(len(ground_truth)),
                                "sampling/sample_id": int(sample_idx)
                            }

                            wandb.log(numerical_metrics, step=state.global_step)

                            self.sampling_data.append([
                                int(state.global_step),
                                int(sample_idx),
                                clean_ground_truth[:100] + "..." if len(clean_ground_truth) > 100 else clean_ground_truth,
                                clean_generated[:100] + "..." if len(clean_generated) > 100 else clean_generated,
                                int(len(ground_truth)),
                                int(len(generated_caption)),
                                float(quality_score),
                                f"{tsn_ratio:.3f}"
                            ])

                            consolidated_table = wandb.Table(
                                columns=["Step", "Sample_ID", "Ground_Truth", "Generated", "GT_Length", "Gen_Length", "Quality", "TSN_Ratio"],
                                data=self.sampling_data
                            )

                            wandb.log({"sampling/all_examples": consolidated_table}, step=state.global_step)

                        except Exception:
                            try:
                                wandb.log({
                                    "sampling/quality": float(quality_score),
                                    "sampling/length": int(len(generated_caption))
                                }, step=state.global_step)
                            except Exception:
                                pass

                except Exception:
                    pass

            model.train()

class RISCDataset(Dataset):

    def __init__(self, csv_path, processor, split='train', max_samples=None):
        self.annotations = pd.read_csv(csv_path)
        self.processor = processor
        self.split = split

        if max_samples is not None and max_samples > 0:
            self.annotations = self.annotations.sample(min(max_samples, len(self.annotations)), random_state=42)

        print(f"Loaded {len(self.annotations)} samples for {split}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = row['image_path']
        caption = row['caption']

        img_path = os.path.normpath(img_path).replace('\\', '/')

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        return {
            'image': image,
            'caption': caption,
            'image_path': img_path
        }

def collate_fn(processor, device, use_quantization=False):
    def _collate_fn(examples):
        texts = ["<image> <bos> describe this image.}" for _ in examples]
        labels = [example['caption'] for example in examples]
        images = [example["image"].convert("RGB") for example in examples]

        try:
            tokens = processor(text=texts, images=images, suffix=labels,
                              return_tensors="pt", padding="longest")
        except (AttributeError, TypeError):
            full_texts = [texts[i] + labels[i] for i in range(len(examples))]
            tokens = processor(text=full_texts, images=images,
                              return_tensors="pt", padding="longest")
            tokens["labels"] = tokens["input_ids"].clone()

        if not use_quantization:
            tokens = tokens.to(torch.bfloat16).to(device)

        return tokens
    return _collate_fn

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config_path="config/config.yaml"):
    config = load_config(config_path)

    seed = config.get('seed', None)
    if seed is None and 'training' in config:
        seed = config['training'].get('seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    PALIGEMMA_MODEL_ID = config['model']['model_id']
    TRAIN_CSV = config['data']['train_csv']
    OUTPUT_DIR = config['output']['output_dir']
    BATCH_SIZE = config['training']['batch_size']
    GRADIENT_ACCUMULATION_STEPS = config['training']['gradient_accumulation_steps']
    LEARNING_RATE = config['training']['learning_rate']
    NUM_EPOCHS = config['training']['num_epochs']
    WEIGHT_DECAY = config['training']['weight_decay']
    MAX_TRAIN_SAMPLES = config['data'].get('max_train_samples', -1)
    MAX_VAL_SAMPLES = config['data'].get('max_val_samples', -1)
    WARMUP_STEPS = config['training'].get('warmup_steps', 50)

    LORA_R = config['lora']['r']
    LORA_TARGET_MODULES = config['lora']['target_modules']

    USE_WANDB = config['training'].get('use_wandb', True)
    WANDB_PROJECT = config['wandb'].get('project', "paligemma-qlora-project")
    WANDB_RUN_NAME = config['wandb'].get('run_name', "paligemma-qlora-training")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "wandb"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if USE_WANDB:
        os.environ["WANDB_DIR"] = "."
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config=config,
            dir=".",
            save_code=True,
            tags=["paligemma", "tsn", "qlora", "remote-sensing"]
        )

    processor = PaliGemmaProcessor.from_pretrained(PALIGEMMA_MODEL_ID)

    # Configure quantization if enabled
    quantization_config = config.get('quantization', {})
    use_quantization = quantization_config.get('enable_quantization', False)

    # Load model with or without quantization
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            PALIGEMMA_MODEL_ID,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16
        )
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            PALIGEMMA_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Integrate TSN module
    print("\nIntegrating TSN for enhanced vision-language performance...")
    model = create_tsn_paligemma_model(model, config)
    print("TSN-PaliGemma model created successfully!")

    if not os.path.exists(TRAIN_CSV):
        print(f"Train CSV file not found: {TRAIN_CSV}")
        return

    train_dataset = RISCDataset(
        csv_path=TRAIN_CSV,
        processor=processor,
        split='train',
        max_samples=MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES > 0 else None
    )

    # Load validation dataset if available
    val_csv = config['data']['val_csv']
    val_dataset = None
    if os.path.exists(val_csv):
        val_dataset = RISCDataset(
            csv_path=val_csv,
            processor=processor,
            split='validation',
            max_samples=MAX_VAL_SAMPLES if MAX_VAL_SAMPLES > 0 else None
        )
        print(f"Loaded validation dataset with {len(val_dataset)} samples")

    args = TrainingArguments(
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        adam_beta2=0.999,
        logging_steps=50,
        optim="adamw_8bit",
        save_strategy="steps",
        save_steps=500,
        eval_strategy="no",
        save_total_limit=3,
        load_best_model_at_end=False,
        metric_for_best_model="train_loss",
        greater_is_better=False,
        save_safetensors=False,
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        bf16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        report_to=["wandb"] if USE_WANDB else ["tensorboard"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=True,
        disable_tqdm=False,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        run_name=WANDB_RUN_NAME,
    )

    callbacks = []

    # Add best model callback
    best_model_callback = BestModelCallback(OUTPUT_DIR, processor, USE_WANDB)
    callbacks.append(best_model_callback)

    # Add validation and sampling callbacks if validation dataset exists
    if val_dataset is not None:
        # Add validation loss callback
        validation_callback = ValidationLossCallback(val_dataset, processor, device, USE_WANDB)
        callbacks.append(validation_callback)

        # Add sampling callback
        sampling_callback = SamplingCallback(processor, val_dataset, device, USE_WANDB)
        callbacks.append(sampling_callback)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=collate_fn(processor, device, use_quantization),
        args=args,
        callbacks=callbacks
    )

    train_result = trainer.train()

    print("\n" + "="*60)
    print("SAVING FINAL MODEL")
    print("="*60)

    final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")

    # Check if best model was saved during training
    best_model_exists = os.path.exists(best_model_dir) and len(os.listdir(best_model_dir)) > 0

    if best_model_exists:
        print("[SUCCESS] Best model found, copying to final model directory...")
        try:
            import shutil
            if os.path.exists(final_model_dir):
                shutil.rmtree(final_model_dir)
            shutil.copytree(best_model_dir, final_model_dir)
            print(f"[SUCCESS] Best model copied to: {final_model_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to copy best model: {e}")
            print("[LOADING] Saving current model as fallback...")
            try:
                os.makedirs(final_model_dir, exist_ok=True)

                # Get the correct model to save (handle TSN wrapper)
                model_to_save = model
                if hasattr(model, 'paligemma'):
                    model_to_save = model.paligemma
                    print("[CONFIG] Detected TSN wrapper, extracting PaliGemma model")
                elif hasattr(model, 'module') and hasattr(model.module, 'paligemma'):
                    model_to_save = model.module.paligemma
                    print("[CONFIG] Detected TSN wrapper in module, extracting PaliGemma model")

                model_to_save.save_pretrained(final_model_dir, safe_serialization=False)
                processor.save_pretrained(final_model_dir)
                print(f"[SUCCESS] Fallback model saved to: {final_model_dir}")
            except Exception as e2:
                print(f"[ERROR] Fallback save failed: {e2}")
    else:
        print("[WARNING]  No best model found, saving current model...")
        try:
            os.makedirs(final_model_dir, exist_ok=True)
            os.makedirs(best_model_dir, exist_ok=True)

            # Get the correct model to save (handle TSN wrapper)
            model_to_save = model
            if hasattr(model, 'paligemma'):
                model_to_save = model.paligemma
                print("[CONFIG] Detected TSN wrapper, extracting PaliGemma model")
            elif hasattr(model, 'module') and hasattr(model.module, 'paligemma'):
                model_to_save = model.module.paligemma
                print("[CONFIG] Detected TSN wrapper in module, extracting PaliGemma model")

            # Save to both directories
            model_to_save.save_pretrained(final_model_dir, safe_serialization=False)
            processor.save_pretrained(final_model_dir)
            print(f"[SUCCESS] Final model saved to: {final_model_dir}")

            model_to_save.save_pretrained(best_model_dir, safe_serialization=False)
            processor.save_pretrained(best_model_dir)
            print(f"[SUCCESS] Best model saved to: {best_model_dir}")

        except Exception as e:
            print(f"[ERROR] Model save failed: {e}")
            import traceback
            traceback.print_exc()
            print("Model weights are still available in trainer checkpoints")

    # Verify saved files
    print("\n" + "="*60)
    print("VERIFYING SAVED MODELS")
    print("="*60)

    for model_type, model_dir in [("Final", final_model_dir), ("Best", best_model_dir)]:
        if os.path.exists(model_dir):
            print(f"\n{model_type} model contents:")
            try:
                files = os.listdir(model_dir)
                if not files:
                    print("   [ERROR] Directory is empty!")
                    continue

                # Check for essential files
                essential_files = ["adapter_config.json", "adapter_model.bin"]
                processor_files = ["preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"]

                for file in sorted(files):
                    file_path = os.path.join(model_dir, file)
                    if os.path.isfile(file_path):
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        status = "[SUCCESS]" if file in essential_files + processor_files else "FILE"
                        print(f"   {status} {file} ({size_mb:.1f} MB)")

                # Check if all essential files are present
                missing_essential = [f for f in essential_files if f not in files]
                missing_processor = [f for f in processor_files if f not in files]

                if missing_essential:
                    print(f"   [ERROR] Missing essential files: {missing_essential}")
                else:
                    print(f"   [SUCCESS] All essential model files present")

                if missing_processor:
                    print(f"   [WARNING]  Missing processor files: {missing_processor}")
                else:
                    print(f"   [SUCCESS] All processor files present")

            except Exception as e:
                print(f"   [ERROR] Could not list files: {e}")
        else:
            print(f"\n[ERROR] {model_type} model directory does not exist: {model_dir}")

    # Log training completion to WandB
    if USE_WANDB and wandb.run is not None:
        try:
            print("\nLogging training completion to WandB...")
            training_summary = {
                "training/final_loss": float(train_result.training_loss) if train_result.training_loss is not None else 0.0,
                "training/total_steps": int(train_result.global_step) if train_result.global_step is not None else 0,
                "training/epochs_completed": int(NUM_EPOCHS),
                "training/completed": True
            }
            wandb.log(training_summary)
            print("[SUCCESS] Training summary logged to WandB")
        except Exception as e:
            print(f"[ERROR] WandB logging error: {e}")

    validate_and_sample(model, processor, config, device)

    if USE_WANDB:
        wandb.finish()

def validate_and_sample(model, processor, config, device):
    val_csv = config['data']['val_csv']
    if not os.path.exists(val_csv):
        return

    # Use config for max validation samples
    max_val_samples = config['data'].get('max_val_samples', -1)
    val_dataset = RISCDataset(
        csv_path=val_csv,
        processor=processor,
        split='validation',
        max_samples=max_val_samples if max_val_samples > 0 else None
    )

    print(f"Final validation using {len(val_dataset)} total validation samples")

    model.eval()
    # Use more samples for final validation (up to 50 or 10% of dataset)
    total_samples = min(50, max(10, len(val_dataset) // 10))
    print(f"Running final validation on {total_samples} samples")

    validation_metrics = {
        "validation/total_samples": total_samples,
        "validation/good_generations": 0,
        "validation/short_generations": 0,
        "validation/refusing_generations": 0,
        "validation/avg_generation_length": 0
    }

    generation_lengths = []

    print(f"\nValidation Results ({total_samples} samples):")
    print("=" * 60)

    with torch.no_grad():
        for i in range(total_samples):
            sample = val_dataset[i]
            image = sample['image']
            ground_truth = sample['caption']

            try:
                inputs = processor(
                    images=image,
                    text="<image>",
                    return_tensors="pt",
                    padding=False
                ).to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=1,
                        do_sample=False,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )

                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

                generated_caption = generated_text.replace("<image>", "").strip()

                print(f"\nValidation Sample {i+1}:")
                print(f"Ground Truth: {ground_truth}")
                print(f"Generated:    {generated_caption}")

                generation_lengths.append(len(generated_caption))

                if len(generated_caption) < 5:
                    print("   WARNING: Very short generation!")
                    validation_metrics["validation/short_generations"] += 1
                elif "sorry" in generated_caption.lower() or "cannot" in generated_caption.lower():
                    print("   WARNING: Model refusing to generate!")
                    validation_metrics["validation/refusing_generations"] += 1
                else:
                    print("   Good generation length and content")
                    validation_metrics["validation/good_generations"] += 1
                print("-" * 80)

            except Exception as e:
                print(f"Error {i+1}: {e}")
                continue

    if generation_lengths:
        validation_metrics["validation/avg_generation_length"] = sum(generation_lengths) / len(generation_lengths)

    USE_WANDB = config['training'].get('use_wandb', True)
    if USE_WANDB:
        import wandb

        # Calculate rates
        success_rate = validation_metrics["validation/good_generations"] / total_samples if total_samples > 0 else 0
        failure_rate = (validation_metrics["validation/short_generations"] +
                       validation_metrics["validation/refusing_generations"]) / total_samples if total_samples > 0 else 0

        # Ensure wandb is properly initialized
        if wandb.run is not None:
            try:
                # Get current step from wandb run
                current_step = wandb.run.step if hasattr(wandb.run, 'step') else 0

                # Log final validation metrics with proper types
                final_validation_metrics = {
                    "final_validation/total_samples": int(total_samples),
                    "final_validation/good_generations": int(validation_metrics["validation/good_generations"]),
                    "final_validation/short_generations": int(validation_metrics["validation/short_generations"]),
                    "final_validation/refusing_generations": int(validation_metrics["validation/refusing_generations"]),
                    "final_validation/avg_generation_length": float(validation_metrics["validation/avg_generation_length"]),
                    "final_validation/success_rate": float(success_rate),
                    "final_validation/failure_rate": float(failure_rate),
                    "final_validation/completed": True
                }

                # Log with explicit step
                wandb.log(final_validation_metrics, step=current_step)
                print(f"Successfully logged final validation metrics to WandB")

                # Create and log summary table
                try:
                    validation_summary = wandb.Table(
                        columns=["Metric", "Value", "Percentage"],
                        data=[
                            ["Total Samples", str(total_samples), "100%"],
                            ["Good Generations", str(validation_metrics["validation/good_generations"]), f"{success_rate*100:.1f}%"],
                            ["Short Generations", str(validation_metrics["validation/short_generations"]), f"{validation_metrics['validation/short_generations']/total_samples*100:.1f}%" if total_samples > 0 else "0%"],
                            ["Refusing Generations", str(validation_metrics["validation/refusing_generations"]), f"{validation_metrics['validation/refusing_generations']/total_samples*100:.1f}%" if total_samples > 0 else "0%"],
                            ["Avg Generation Length", f"{validation_metrics['validation/avg_generation_length']:.1f}", "-"]
                        ]
                    )

                    wandb.log({"final_validation/summary_table": validation_summary}, step=current_step)
                    print("Successfully logged validation summary table to WandB")

                except Exception as e:
                    print(f"Error logging validation table to WandB: {e}")

            except Exception as e:
                print(f"Error logging final validation metrics to WandB: {e}")
        else:
            print("Warning: WandB not initialized, skipping final validation logging")

    print(f"\nValidation completed!")
    print(f"Good: {validation_metrics['validation/good_generations']}/{total_samples}")
    print(f"Short: {validation_metrics['validation/short_generations']}/{total_samples}")
    print(f"Refusing: {validation_metrics['validation/refusing_generations']}/{total_samples}")
    print(f"Avg Length: {validation_metrics['validation/avg_generation_length']:.1f}")
    print("=" * 60)

if __name__ == "__main__":
    train()
