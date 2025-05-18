#!/usr/bin/env python3
import torch
import torch.nn as nn
from models.tsn_paligemma_model import TSNModule

class TSNPaliGemmaEnhancedModel(nn.Module):
    """
    Enhanced integration of TSN with PaliGemma.
    This model directly integrates TSN features with PaliGemma's encoder outputs
    to enhance the model's understanding of remote sensing images.
    """
    def __init__(self, paligemma_model, config):
        super(TSNPaliGemmaEnhancedModel, self).__init__()

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

        # Create a gating mechanism to control TSN feature influence
        self.feature_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )

        # Get mixing ratios from config or use defaults
        tsn_config = config.get('tsn', {})
        self.original_ratio = tsn_config.get('original_ratio', 0.7)
        self.tsn_ratio = tsn_config.get('tsn_ratio', 0.3)

        # Separate mixing ratios for generation
        self.gen_original_ratio = tsn_config.get('gen_original_ratio', 0.8)
        self.gen_tsn_ratio = tsn_config.get('gen_tsn_ratio', 0.2)

        print(f"Using mixing ratios - Forward: {self.original_ratio}:{self.tsn_ratio}, Generate: {self.gen_original_ratio}:{self.gen_tsn_ratio}")
        print("Initialized TSNPaliGemmaEnhancedModel - Using direct encoder enhancement")

        # Store original encoder and decoder forward methods
        if hasattr(self.paligemma, 'encoder'):
            self.original_encoder_forward = self.paligemma.encoder.forward
            # Replace encoder forward method
            self._patch_encoder()

        # Patch decoder if available
        if hasattr(self.paligemma, 'decoder'):
            self.original_decoder_forward = self.paligemma.decoder.forward
            # Replace decoder forward method
            self._patch_decoder()

    def _patch_encoder(self):
        """
        Patch the encoder's forward method to integrate TSN features.
        """
        original_forward = self.paligemma.encoder.forward
        model = self  # Store reference to self

        # Define new forward method
        def enhanced_forward(*args, **kwargs):
            # Call original forward method
            outputs = original_forward(*args, **kwargs)

            # Check if hidden_states is in outputs
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state

                # Get the current batch's pixel_values from kwargs
                pixel_values = kwargs.get('pixel_values', None)

                # Only enhance if pixel_values is provided
                if pixel_values is not None and model.tsn is not None:
                    # Process through TSN module
                    tsn_features, _ = model.tsn(pixel_values)

                    # Project TSN features to match hidden dimension
                    projected_features = model.feature_projection(tsn_features)

                    # Cache TSN features for decoder to use
                    model.current_tsn_features = projected_features

                    # Debug information
                    print(f"\nTSN features (enhanced) - shape: {tsn_features.shape}")
                    print(f"TSN features (enhanced) - mean: {tsn_features.mean().item()}, std: {tsn_features.std().item()}")
                    print(f"Projected features (enhanced) - shape: {projected_features.shape}")
                    print(f"Hidden states (enhanced) - shape: {hidden_states.shape}")

                    # Reshape TSN features to match hidden states
                    batch_size = hidden_states.size(0)
                    seq_len = hidden_states.size(1)

                    # Expand TSN features to match sequence length
                    expanded_features = projected_features.unsqueeze(1).expand(-1, seq_len, -1)

                    # Compute gating values - determine how much TSN features to use
                    # for each position in the sequence
                    concat_features = torch.cat([hidden_states, expanded_features], dim=-1)
                    gate_values = model.feature_gate(concat_features)

                    # Apply gating - use more TSN features for visual tokens, less for text tokens
                    # First 14 tokens are typically image tokens in PaliGemma
                    visual_tokens_mask = torch.zeros_like(gate_values)
                    if seq_len > 14:
                        # Mark more tokens as visual tokens (first 20 tokens)
                        visual_tokens_mask[:, :20, :] = 1.0

                    # Adjust gate values - increase for visual tokens (higher weight: 1.0 -> 2.0)
                    adjusted_gate_values = gate_values * (1.0 + 2.0 * visual_tokens_mask)

                    # Combine original hidden states with TSN features using gating
                    enhanced_hidden_states = (
                        model.original_ratio * hidden_states +
                        model.tsn_ratio * adjusted_gate_values * expanded_features
                    )

                    # Replace hidden states in outputs
                    outputs.last_hidden_state = enhanced_hidden_states

            return outputs

        # Replace the encoder's forward method
        self.paligemma.encoder.forward = enhanced_forward

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass through the combined model.
        The encoder has been patched to integrate TSN features.
        """
        # Call PaliGemma with our enhanced encoder
        return self.paligemma(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        """
        Generate text based on image and text inputs.
        The encoder has been patched to integrate TSN features.
        """
        # Temporarily adjust mixing ratios for generation
        original_ratio = self.original_ratio
        tsn_ratio = self.tsn_ratio

        self.original_ratio = self.gen_original_ratio
        self.tsn_ratio = self.gen_tsn_ratio

        try:
            # Call PaliGemma's generate method with our enhanced encoder
            outputs = self.paligemma.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        finally:
            # Restore original mixing ratios
            self.original_ratio = original_ratio
            self.tsn_ratio = tsn_ratio

        return outputs

    def _patch_decoder(self):
        """
        Patch the decoder's forward method to integrate TSN features.
        """
        original_forward = self.paligemma.decoder.forward
        model = self  # Store reference to self

        # Define new forward method
        def enhanced_forward(*args, **kwargs):
            # Call original forward method
            outputs = original_forward(*args, **kwargs)

            # Check if hidden_states is in outputs
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state

                # Get the current batch's pixel_values from kwargs
                pixel_values = kwargs.get('encoder_hidden_states', None)

                # Only enhance if we have encoder hidden states
                if pixel_values is not None and model.tsn is not None:
                    # Debug information
                    print(f"\nDecoder hidden states (enhanced) - shape: {hidden_states.shape}")

                    # Get TSN features from the model's cache
                    if hasattr(model, 'current_tsn_features'):
                        projected_features = model.current_tsn_features

                        # Reshape TSN features to match hidden states
                        seq_len = hidden_states.size(1)

                        # Expand TSN features to match sequence length
                        expanded_features = projected_features.unsqueeze(1).expand(-1, seq_len, -1)

                        # Compute gating values for decoder
                        concat_features = torch.cat([hidden_states, expanded_features], dim=-1)
                        gate_values = model.feature_gate(concat_features)

                        # Apply gating - use less TSN features for decoder (focus on language generation)
                        adjusted_gate_values = gate_values * 0.5

                        # Combine original hidden states with TSN features using gating
                        enhanced_hidden_states = (
                            model.original_ratio * hidden_states +
                            model.tsn_ratio * adjusted_gate_values * expanded_features
                        )

                        # Replace hidden states in outputs
                        outputs.last_hidden_state = enhanced_hidden_states

            return outputs

        # Replace the decoder's forward method
        self.paligemma.decoder.forward = enhanced_forward

    def __del__(self):
        """
        Restore original encoder and decoder forward methods when the model is deleted.
        """
        if hasattr(self, 'original_encoder_forward') and hasattr(self.paligemma, 'encoder'):
            self.paligemma.encoder.forward = self.original_encoder_forward

        if hasattr(self, 'original_decoder_forward') and hasattr(self.paligemma, 'decoder'):
            self.paligemma.decoder.forward = self.original_decoder_forward

def create_tsn_paligemma_enhanced_model(paligemma_model, config):
    """
    Create a TSNPaliGemmaEnhancedModel instance.

    Args:
        paligemma_model: Pre-trained PaliGemma model
        config: Configuration dictionary

    Returns:
        TSNPaliGemmaEnhancedModel instance
    """
    return TSNPaliGemmaEnhancedModel(paligemma_model, config)
