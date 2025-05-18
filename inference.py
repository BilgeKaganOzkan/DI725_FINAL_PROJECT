#!/usr/bin/env python3
import os
import torch
import yaml
import argparse
import pandas as pd
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import matplotlib.pyplot as plt
from models.tsn_paligemma_model import create_tsn_paligemma_model

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def load_model(model_path, config):
    """
    Load the fine-tuned PaliGemma model with TSN.

    Args:
        model_path: Path to the saved model
        config: Configuration dictionary

    Returns:
        processor: PaliGemma processor
        model: Fine-tuned model
    """
    # Load processor
    processor = PaliGemmaProcessor.from_pretrained(config['model']['model_id'])

    # Load PaliGemma model
    paligemma_path = os.path.join(model_path, "paligemma")

    # Check if the model exists
    if not os.path.exists(paligemma_path):
        raise FileNotFoundError(f"Model not found at {paligemma_path}")

    # Load the model
    paligemma_model = PeftModel.from_pretrained(
        PaliGemmaForConditionalGeneration.from_pretrained(
            config['model']['model_id'],
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        ),
        paligemma_path,
        torch_dtype=torch.bfloat16
    )

    # Load TSN model
    tsn_path = os.path.join(model_path, "tsn_model.pt")

    # Create the combined model
    model = create_tsn_paligemma_model(paligemma_model, config)

    # Load TSN weights if available
    if os.path.exists(tsn_path):
        model.tsn.load_state_dict(torch.load(tsn_path))
        print(f"Loaded TSN weights from {tsn_path}")
    else:
        print(f"TSN weights not found at {tsn_path}, using initialized weights")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return processor, model

def generate_caption(image_path, processor, model, config):
    """
    Generate a caption for an image.

    Args:
        image_path: Path to the image
        processor: PaliGemma processor
        model: Fine-tuned model
        config: Configuration dictionary

    Returns:
        caption: Generated caption
    """
    # Normalize path separators for cross-platform compatibility
    image_path = os.path.normpath(image_path).replace('\\', '/')

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Get prompt from config
    prompt = config['prompt']['input_prompt']

    # Process image and text
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate caption
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=inputs['pixel_values'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=config['evaluation']['generate_max_length'],
            num_beams=config['evaluation']['num_beams']
        )

    # Decode caption
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    return caption

def visualize_result(image_path, caption):
    """
    Visualize the image and its generated caption.

    Args:
        image_path: Path to the image
        caption: Generated caption
    """
    # Normalize path separators for cross-platform compatibility
    image_path = os.path.normpath(image_path).replace('\\', '/')

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Caption: {caption}", fontsize=12, wrap=True)
    plt.tight_layout()

    # Save figure
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.', '_caption.'))
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

    # Show figure
    plt.show()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate captions for remote sensing images")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image")
    parser.add_argument("--visualize", action="store_true", help="Visualize the result")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model
    processor, model = load_model(args.model_path, config)

    # Generate caption
    caption = generate_caption(args.image_path, processor, model, config)

    # Print caption
    print(f"Generated caption: {caption}")

    # Visualize result if requested
    if args.visualize:
        visualize_result(args.image_path, caption)

def batch_inference(image_dir, output_csv, model_path, config_path, test_csv=None):
    """
    Generate captions for all images in a directory or in a test CSV file.

    Args:
        image_dir: Directory containing images
        output_csv: Path to output CSV file
        model_path: Path to the saved model
        config_path: Path to config file
        test_csv: Path to test CSV file (optional)
    """
    # Load configuration
    config = load_config(config_path)

    # Load model
    processor, model = load_model(model_path, config)

    # Create output directory
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Generate captions
    results = []

    # If a test CSV file is specified, use it
    if test_csv:
        # Read the CSV file
        df = pd.read_csv(test_csv)

        # Generate a caption for each image
        for _, row in df.iterrows():
            image_path = row['image_path']
            # Normalize path separators for cross-platform compatibility
            image_path = os.path.normpath(image_path).replace('\\', '/')
            image_file = os.path.basename(image_path)

            try:
                # Generate caption
                caption = generate_caption(image_path, processor, model, config)

                # Add to results
                results.append({
                    'image': image_file,
                    'caption': caption,
                    'ground_truth': row['caption'] if 'caption' in row else ''
                })

                print(f"Generated caption for {image_file}: {caption}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results.append({
                    'image': image_file,
                    'caption': "Error: " + str(e),
                    'ground_truth': row['caption'] if 'caption' in row else ''
                })
    else:
        # Get all image files in the directory
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Generate a caption for each image
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            # Normalize path separators for cross-platform compatibility
            image_path = os.path.normpath(image_path).replace('\\', '/')

            try:
                # Generate caption
                caption = generate_caption(image_path, processor, model, config)

                # Add to results
                results.append({
                    'image': image_file,
                    'caption': caption
                })

                print(f"Generated caption for {image_file}: {caption}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results.append({
                    'image': image_file,
                    'caption': "Error: " + str(e)
                })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(results)} captions to {output_csv}")

if __name__ == "__main__":
    # Check if batch inference is requested
    parser = argparse.ArgumentParser(description="Generate captions for remote sensing images")
    parser.add_argument("--batch", action="store_true", help="Perform batch inference")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, help="Path to the saved model")
    parser.add_argument("--image_path", type=str, help="Path to the image or directory of images")
    parser.add_argument("--output_csv", type=str, help="Path to output CSV file (for batch inference)")
    parser.add_argument("--test_csv", type=str, help="Path to test CSV file (for batch inference)")
    parser.add_argument("--visualize", action="store_true", help="Visualize the result (for single image)")

    args, _ = parser.parse_known_args()

    if args.batch:
        if not args.image_path or not args.output_csv or not args.model_path:
            parser.error("--batch requires --image_path, --output_csv, and --model_path")

        batch_inference(args.image_path, args.output_csv, args.model_path, args.config, args.test_csv)
    else:
        main()
