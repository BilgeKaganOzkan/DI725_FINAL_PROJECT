#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tsn_paligemma_model import TSNModule

class TSNPaliGemmaDirectModel(nn.Module):
    """
    A different approach to integrating TSN with PaliGemma.
    Instead of modifying the vision tower, this model:
    1. Processes the image through both PaliGemma and TSN separately
    2. Uses PaliGemma's output directly for both training and generation
    3. No fallback or safety message detection - returns direct model outputs
    """
    def __init__(self, paligemma_model, config):
        super(TSNPaliGemmaDirectModel, self).__init__()

        # Store PaliGemma model
        self.paligemma = paligemma_model

        # Initialize TSN module
        tsn_config = config.get('tsn', {})
        self.tsn = TSNModule(tsn_config)

        # Store configuration
        self.config = config

        # Get hidden dimension from PaliGemma model
        if hasattr(self.paligemma, 'config') and hasattr(self.paligemma.config, 'hidden_size'):
            self.hidden_dim = self.paligemma.config.hidden_size
        else:
            # Default value if not available
            self.hidden_dim = 1408  # Common hidden size for PaliGemma

        # Create projection layer for TSN features
        self.feature_projection = nn.Linear(self.tsn.projection_dim, self.hidden_dim)

        # Create a layer to combine TSN features with text features
        self.combination_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # No safety message detection - direct model output only

        print("Initialized TSNPaliGemmaDirectModel - Using direct model output without fallbacks")

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass through the combined model.
        For training, we just use PaliGemma's output directly.
        """
        # Process image through TSN module (for monitoring only during training)
        if pixel_values is not None:
            tsn_features, _ = self.tsn(pixel_values)
            projected_features = self.feature_projection(tsn_features)

            # Debug information
            print(f"\nTSN features (direct) - shape: {tsn_features.shape}")
            print(f"TSN features (direct) - mean: {tsn_features.mean().item()}, std: {tsn_features.std().item()}")
            print(f"Projected features (direct) - shape: {projected_features.shape}")

        # For training, just use PaliGemma's output directly
        return self.paligemma(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, pixel_values=None, input_ids=None, attention_mask=None, processor=None, **kwargs):
        """
        Generate text based on image and text inputs.
        This method directly returns the PaliGemma model outputs without any modifications.
        """
        # Generate with PaliGemma directly
        paligemma_outputs = self.paligemma.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Debug information
        print(f"PaliGemma direct generate - outputs shape: {paligemma_outputs.shape}")

        # If processor is provided, decode and print the output for debugging
        if processor is not None and pixel_values is not None:
            # Decode the output
            decoded_output = processor.batch_decode(paligemma_outputs, skip_special_tokens=True)[0]
            print(f"PaliGemma direct output: '{decoded_output}'")

        # Return the original output directly
        return paligemma_outputs

def create_tsn_paligemma_direct_model(paligemma_model, config):
    """
    Create a TSNPaliGemmaDirectModel instance.

    Args:
        paligemma_model: Pre-trained PaliGemma model
        config: Configuration dictionary

    Returns:
        TSNPaliGemmaDirectModel instance
    """
    return TSNPaliGemmaDirectModel(paligemma_model, config)
