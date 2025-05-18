# PaliGemma TSN-QLoRA for Remote Sensing Image Captioning

This project implements an advanced remote sensing image captioning system by enhancing Google's PaliGemma model with Temporal Segment Network (TSN) architecture and Quantized Low-Rank Adaptation (QLoRA) fine-tuning on the RISC dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Results](#results)

## Project Overview

This project combines several state-of-the-art techniques to improve image captioning for remote sensing imagery:

- **PaliGemma**: A powerful vision-language model from Google that serves as the foundation
- **TSN (Temporal Segment Network)**: Adapted for spatial segmentation at multiple scales (1×1 and 2×2)
- **QLoRA (Quantized Low-Rank Adaptation)**: For memory-efficient fine-tuning with 4-bit quantization
- **Multiple Integration Methods**: Different approaches to combine TSN with PaliGemma
- **Attention Mechanisms**: For improved feature fusion across spatial segments

The goal is to enhance remote sensing image captioning performance by leveraging spatial segmentation and attention mechanisms while maintaining computational efficiency through quantization and low-rank adaptation techniques.

## Features

- Fine-tuning of PaliGemma model using QLoRA for memory efficiency
- Multiple TSN integration methods (vision tower replacement, enhanced encoder, adapter, direct output)
- Spatial segmentation at different scales to capture both global and local features
- Attention mechanisms for improved feature fusion
- Comprehensive experiment tracking with Weights & Biases
- Inference script for generating captions on new images
- Hyperparameter optimization capabilities

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Hugging Face account with access to PaliGemma model

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/paligemma-tsn-qlora.git
cd paligemma-tsn-qlora
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Login to Hugging Face (required for PaliGemma access):
```bash
huggingface-cli login
```

4. Login to Weights & Biases for experiment tracking:
```bash
wandb login
```

## Dataset

The project uses the RISC (Remote Sensing Image Captioning) dataset, which contains remote sensing images with multiple captions per image.

### Dataset Structure
- `dataset/`: Contains the original RISC dataset
- `processed_dataset/`: Contains preprocessed CSV files with image paths and captions
  - `train.csv`: Training dataset
  - `val.csv`: Validation dataset
  - `test.csv`: Test dataset

### Dataset Statistics
- **Total Images**: 44,521 unique images with 222,605 captions (5 captions per image)
- **Train**: 35,614 images (79.99%)
- **Validation**: 4,453 images (10.00%)
- **Test**: 4,454 images (10.00%)

### Image Sources
- **NWPU**: 31,500 images (70.75%)
- **RSICD**: 10,921 images (24.53%)
- **UCM**: 2,100 images (4.72%)

### Image Properties
- Fixed resolution: 224×224 pixels
- Average file size: 24.75 KB

### Caption Analysis
- Average caption length: 12.09 words
- Most frequent words: "there", "some", "many", "green", "trees", "buildings"

### Content Categories
- **Vegetation**: 44.10% (trees, forest, grass)
- **Urban**: 31.50% (buildings, city, residential areas)
- **Landscape**: 22.56% (mountains, hills, fields)
- **Transportation**: 20.78% (roads, highways, vehicles)
- **Water**: 18.92% (rivers, lakes, oceans)
- **Infrastructure**: 10.24% (bridges, ports, facilities)
- **Airport**: 5.76% (airports or related elements)

## Project Structure

```
├── config/
│   ├── config.yaml                  # Main configuration file
│   └── optimization_configs.yaml    # Hyperparameter optimization config
├── models/
│   ├── tsn_paligemma_adapter.py     # TSN-PaliGemma adapter integration
│   ├── tsn_paligemma_direct.py      # TSN-PaliGemma direct output manipulation
│   ├── tsn_paligemma_enhanced.py    # TSN-PaliGemma enhanced encoder integration
│   └── tsn_paligemma_model.py       # TSN-PaliGemma vision tower replacement
├── processed_dataset/
│   ├── train.csv                    # Training data
│   ├── val.csv                      # Validation data
│   └── test.csv                     # Test data
├── dataset/                         # Original RISC dataset
│   └── resized/                     # Resized images (224x224)
├── dataset_exploration/
│   └── explore_dataset.py           # Dataset analysis script
├── dataset_preprocessing.py         # Dataset preprocessing script
├── train_paligemma_qlora_tsn.py     # Main training script
├── inference.py                     # Inference script
└── README.md                        # This file
```

## Usage

### Data Preprocessing

Before training, preprocess the RISC dataset:

```bash
python dataset_preprocessing.py
```

This script:
1. Reads the original captions.csv file
2. Selects one random caption per image
3. Creates train, validation, and test splits
4. Saves the processed data to CSV files in the processed_dataset directory

### Training

To train the model:

```bash
python train_paligemma_qlora_tsn.py
```

This will train the model using the configuration in `config/config.yaml`. The training process includes:

1. Loading the PaliGemma model with QLoRA adaptation
2. Integrating the TSN architecture based on the method specified in the config
3. Training on the RISC dataset with the specified hyperparameters
4. Validating after each epoch with sample image captioning
5. Tracking metrics with Weights & Biases

### Hyperparameter Optimization

To run hyperparameter optimization:

```bash
python hyperparameter_optimization.py --config config/optimization_configs.yaml
```

Parameters:
- `--config`: Path to optimization configuration file (default: "config/optimization_configs.yaml")

The optimization parameters are defined in the YAML configuration file:

```yaml
method: bayes  # Optimization method: bayes, random, or grid
parameters:
  learning_rate:
    values: [1e-6, 5e-6, 1e-5, 5e-5]
  batch_size:
    values: [2, 4, 8]
  tsn:
    backbone:
      values: ["resnet18", "inception_v3"]
    original_ratio:
      values: [0.3, 0.5, 0.7]
    tsn_ratio:
      values: [0.3, 0.5, 0.7]
  # ... other parameters
optimization:
  max_train_samples: 100  # Use subset for faster optimization
  max_val_samples: 20
  num_runs: 10
```

The optimization process:
1. Backs up the original `config.yaml`
2. Runs multiple training iterations with different hyperparameters
3. Tracks performance using Weights & Biases
4. Saves the best configuration to `best_config.yaml`
5. Restores the original configuration

After optimization, train with the best configuration:
```bash
copy config\best_config.yaml config\config.yaml
python train_paligemma_qlora_tsn.py
```

### Inference

To generate captions for new images, use the provided inference script:

```bash
python inference.py --image_path path/to/image.jpg --model_path path/to/saved/model
```

Parameters:
- `--image_path`: Path to the image file (required)
- `--model_path`: Path to the saved model directory (required)
- `--config_path`: Path to the configuration file (default: "config/config.yaml")
- `--integration_method`: TSN integration method (default: from config file)

The inference script:
1. Loads the saved model and configuration
2. Processes the input image
3. Generates a caption using the same parameters as during training
4. Displays the image and the generated caption

Example usage:
```bash
python inference.py --image_path dataset/resized/sample.jpg --model_path paligemma_tsn_qlora/checkpoint-1000
```

## Model Architecture

The model architecture combines PaliGemma with TSN and QLoRA:

1. **TSN Module**:
   - Takes an image and divides it into segments at different scales (1×1, 2×2)
   - Processes each segment through a backbone network (ResNet18 or InceptionV3)
   - Applies attention mechanisms to fuse features
   - Projects features to match PaliGemma's visual embedding space

2. **PaliGemma Model**:
   - A vision-language model from Google based on the T5 architecture
   - Takes visual embeddings and generates text
   - Pretrained on a mix of image-text pairs

3. **QLoRA Adaptation**:
   - Applies Quantized Low-Rank Adaptation to specific layers of PaliGemma
   - Enables efficient fine-tuning with fewer parameters
   - Uses 4-bit quantization to reduce memory requirements

4. **Integration Methods**:
   - **Vision Tower Replacement**: Replaces PaliGemma's vision encoder with TSN
   - **Enhanced Encoder Integration**: Combines TSN features with PaliGemma's encoder output
   - **Adapter Integration**: Uses TSN as an adapter between vision and language components
   - **Direct Output**: Uses PaliGemma's output directly without any modifications

## Configuration

All model and training parameters are defined in `config/config.yaml`:

### Model Configuration
```yaml
model:
  model_id: "google/paligemma-3b-mix-224"
```

### Data Configuration
```yaml
data:
  train_csv: "processed_dataset/train.csv"
  val_csv: "processed_dataset/val.csv"
  test_csv: "processed_dataset/test.csv"
  dataset_path: "dataset/resized"
  max_train_samples: 0  # 0 means use all samples
  max_val_samples: 0
```

### Evaluation Configuration
```yaml
evaluation:
  eval_during_training: true
  generate_max_length: 128
  min_length: 20
  num_beams: 4
```

### Training Configuration
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-06
  mixed_precision: true
  num_epochs: 30
  weight_decay: 0.05
  label_smoothing: 0.1
  use_wandb: true
```

### QLoRA Configuration
```yaml
lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.1
  target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### Prompt Configuration
```yaml
prompt:
  input_prompt: "Describe all the objects and features visible in this remote sensing image:"
```

### TSN Configuration
```yaml
tsn:
  backbone: "resnet18"
  feature_dim: 512
  pretrained: true
  projection_dim: 1408
  segment_scales:
  - [1, 1]  # Macro scale (whole image)
  - [2, 2]  # Medium scale (2x2 grid)
  use_attention: true
  integration_method: "enhanced"
  original_ratio: 0.5  # Mixing ratio for original features
  tsn_ratio: 0.5       # Mixing ratio for TSN features
  gen_original_ratio: 0.6  # Ratio for generation
  gen_tsn_ratio: 0.4       # Ratio for generation
```

## Troubleshooting

### Model Output Issues
If you want to see the raw model outputs without any filtering:
1. Use the "direct" integration method in `config.yaml` (`integration_method: "direct"`)
2. This will ensure you get the exact output from the model without any modifications
3. Use a specific prompt that clearly indicates the remote sensing context
4. Ensure the model has been trained for enough epochs

### Short Output Issues
If the model generates very short outputs:
1. Increase the `min_length` parameter in the evaluation section of `config.yaml`
2. Increase the `generate_max_length` parameter for longer outputs
3. Use a prompt that explicitly asks for detailed descriptions
4. Check if the training captions are sufficiently detailed

### Memory Issues
If you encounter memory issues:
1. Reduce batch size in `config/config.yaml`
2. Increase gradient accumulation steps
3. Use mixed precision training (already enabled by default)
4. Reduce the TSN segment scales (e.g., use only 1×1)
5. Use a smaller backbone network (e.g., ResNet18 instead of InceptionV3)

### PaliGemma Access
If you have issues accessing the PaliGemma model:
1. Ensure you've requested access on Hugging Face
2. Login to Hugging Face: `huggingface-cli login`
3. Check your internet connection
4. Try using a VPN if you're in a region with restricted access

## Results

The model performance is tracked using Weights & Biases. You can view the results at:
```
https://wandb.ai/[YOUR_USERNAME]/paligemma-tsn-qlora
```

### Performance Metrics
The model is evaluated using standard image captioning metrics:
- **Training Loss**: Measures the model's performance during training
- **Validation Loss**: Measures the model's generalization to unseen data
- **BLEU Score**: Measures n-gram overlap between generated and reference captions
- **METEOR Score**: Evaluates semantic similarity and grammatical structure
- **CIDEr Score**: Measures consensus between generated and reference captions
- **ROUGE-L Score**: Measures longest common subsequence between generated and reference captions

### Ablation Studies
Different configurations were tested to evaluate the impact of:
- **Backbone Architecture**: ResNet18 vs. InceptionV3
- **Spatial Segmentation Scales**: Different combinations of 1×1 and 2×2 scales
- **Integration Methods**: Vision Tower Replacement vs. Enhanced Encoder Integration vs. Adapter vs. Direct Output Manipulation
- **Mixing Ratios**: Different combinations of original and TSN feature ratios
- **Prompt Formats**: Different prompts for generating captions

### Key Findings
- The TSN architecture with spatial segmentation significantly improves captioning performance compared to the base PaliGemma model
- The Enhanced Encoder Integration method provides the best results among all integration approaches
- Attention mechanisms help the model focus on relevant image regions
- A balanced mixing ratio (0.5:0.5) between original and TSN features works best during training
- A slightly higher original ratio (0.6:0.4) during generation helps avoid safety filter responses
- Using detailed prompts like "Describe all the objects and features visible in this remote sensing image:" produces more comprehensive captions
- QLoRA fine-tuning with r=32 and alpha=64 offers a good balance between performance and efficiency

### Sample Results
The model generates detailed and accurate captions for remote sensing images, capturing both the main objects and their spatial relationships. It performs particularly well on complex scenes with multiple elements, such as urban areas with buildings, roads, and vegetation.