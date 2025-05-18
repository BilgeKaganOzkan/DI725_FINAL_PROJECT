#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SpatialAttention(nn.Module):
    """
    Spatial attention module for focusing on important regions in feature maps.
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # Generate attention map
        attention = self.conv(x)
        attention = torch.sigmoid(attention)

        # Apply attention
        return x * attention.expand_as(x), attention

class TSNModule(nn.Module):
    """
    Temporal Segment Network (TSN) module adapted for spatial segmentation.
    Processes image at different spatial scales and applies attention.
    """
    def __init__(self, config):
        super(TSNModule, self).__init__()

        # Extract configuration
        self.backbone_name = config.get('backbone', 'inception_v3')
        self.pretrained = config.get('pretrained', True)
        self.segment_scales = config.get('segment_scales', [[1, 1], [2, 2], [4, 4]])
        self.feature_dim = config.get('feature_dim', 2048)
        self.use_attention = config.get('use_attention', True)
        self.projection_dim = config.get('projection_dim', 1408)  # Match PaliGemma's visual embedding dimension

        # Initialize backbone
        print(f"Initializing TSN with backbone: {self.backbone_name}")

        # Helper function to initialize backbone with proper weights
        def init_backbone(model_class, weights_class=None, feature_dim=None):
            if weights_class and hasattr(weights_class, 'IMAGENET1K_V1') and self.pretrained:
                # New torchvision version
                backbone = model_class(weights='IMAGENET1K_V1')
            else:
                # Older torchvision version
                backbone = model_class(pretrained=self.pretrained)

            # Update feature dimension if provided
            if feature_dim:
                self.feature_dim = feature_dim

            return backbone

        # Select backbone based on name
        if self.backbone_name == 'resnet50':
            backbone = init_backbone(models.resnet50, models.ResNet50_Weights)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avg pool and fc

        elif self.backbone_name == 'resnet101':
            backbone = init_backbone(models.resnet101, models.ResNet101_Weights)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avg pool and fc

        elif self.backbone_name == 'resnet152':
            backbone = init_backbone(models.resnet152, models.ResNet152_Weights)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avg pool and fc

        elif self.backbone_name == 'inception_v3':
            backbone = init_backbone(models.inception_v3, models.Inception_V3_Weights)
            # Disable aux_logits
            backbone.aux_logits = False
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final classifier

        elif self.backbone_name == 'resnet18':
            backbone = init_backbone(models.resnet18, models.ResNet18_Weights, feature_dim=512)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avg pool and fc

        elif self.backbone_name == 'efficientnet_b0':
            backbone = init_backbone(models.efficientnet_b0, models.EfficientNet_B0_Weights, feature_dim=1280)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final classifier

        elif self.backbone_name == 'efficientnet_b3':
            backbone = init_backbone(models.efficientnet_b3, models.EfficientNet_B3_Weights, feature_dim=1536)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final classifier

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        print(f"TSN backbone initialized with feature dimension: {self.feature_dim}")

        # Initialize attention modules if needed
        if self.use_attention:
            self.attention_modules = nn.ModuleList([
                SpatialAttention(self.feature_dim) for _ in range(len(self.segment_scales))
            ])

        # Projection layer to match PaliGemma's visual embedding dimension
        self.projection = nn.Linear(self.feature_dim * len(self.segment_scales), self.projection_dim)

    def forward(self, x):
        """
        Forward pass through the TSN module.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Tensor of shape [batch_size, projection_dim]
        """
        batch_size = x.size(0)
        all_features = []
        attention_maps = []

        # Check input image size
        if x.size(2) < 32 or x.size(3) < 32:
            print(f"Warning: Input image size is too small: {x.shape}")
            # Resize image
            x = F.interpolate(x, size=(max(224, x.size(2)), max(224, x.size(3))),
                             mode='bilinear', align_corners=False)
            print(f"Resized input image to: {x.shape}")

        # Process each spatial scale
        for i, scale in enumerate(self.segment_scales):
            h_segments, w_segments = scale

            # Calculate segment size
            h_size = max(1, x.size(2) // h_segments)
            w_size = max(1, x.size(3) // w_segments)

            scale_features = []
            scale_attention_maps = []

            # Process each segment
            for h_idx in range(h_segments):
                for w_idx in range(w_segments):
                    # Extract segment
                    h_start = h_idx * h_size
                    h_end = (h_idx + 1) * h_size if h_idx < h_segments - 1 else x.size(2)
                    w_start = w_idx * w_size
                    w_end = (w_idx + 1) * w_size if w_idx < w_segments - 1 else x.size(3)

                    segment = x[:, :, h_start:h_end, w_start:w_end]

                    # Check segment size and skip if too small
                    if segment.size(2) < 32 or segment.size(3) < 32:
                        # Segment is too small, skip this segment
                        continue

                    # Resize segment to match backbone input size (minimum 224x224)
                    segment = F.interpolate(segment, size=(max(224, segment.size(2)), max(224, segment.size(3))),
                                           mode='bilinear', align_corners=False)

                    # Process segment through backbone
                    try:
                        features = self.backbone(segment)
                    except Exception as e:
                        print(f"Error processing segment of size {segment.shape}: {e}")
                        # In case of error, skip this segment
                        continue

                    # Apply attention if enabled
                    if self.use_attention:
                        features, attention = self.attention_modules[i](features)
                        scale_attention_maps.append(attention)

                    # Global average pooling
                    features = F.adaptive_avg_pool2d(features, (1, 1))
                    features = features.view(batch_size, -1)

                    scale_features.append(features)

            # Combine features from all segments at this scale
            if len(scale_features) > 1:
                scale_features = torch.stack(scale_features, dim=1)
                scale_features = torch.mean(scale_features, dim=1)
            elif len(scale_features) == 1:
                scale_features = scale_features[0]
            else:
                # If no segments were processed, create a zero tensor
                scale_features = torch.zeros(batch_size, self.feature_dim, device=x.device)

            all_features.append(scale_features)

            if self.use_attention and len(scale_attention_maps) > 0:
                attention_maps.append(scale_attention_maps)

        # Concatenate features from all scales
        combined_features = torch.cat(all_features, dim=1)

        # Project to match PaliGemma's visual embedding dimension
        projected_features = self.projection(combined_features)

        # Debug information
        print(f"\nTSN Module - combined_features shape: {combined_features.shape}")
        print(f"TSN Module - combined_features mean: {combined_features.mean().item()}, std: {combined_features.std().item()}")
        print(f"TSN Module - projected_features shape: {projected_features.shape}")
        print(f"TSN Module - projected_features mean: {projected_features.mean().item()}, std: {projected_features.std().item()}")
        print(f"TSN Module - projected_features min: {projected_features.min().item()}, max: {projected_features.max().item()}")

        return projected_features, attention_maps if self.use_attention else None

class TSNPaliGemmaModel(nn.Module):
    """
    Combined model with TSN module and PaliGemma.
    TSN processes the image at different spatial scales and feeds enhanced features to PaliGemma.
    """
    def __init__(self, paligemma_model, config):
        super(TSNPaliGemmaModel, self).__init__()

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

        # Create static projection layer for TSN features if needed
        self.feature_projection = nn.Linear(self.tsn.projection_dim, self.hidden_dim)

        # Mixing ratio for combining original and TSN features
        # Get mixing ratios from config or use defaults
        tsn_config = config.get('tsn', {})
        self.original_ratio = tsn_config.get('original_ratio', 0.95)
        self.tsn_ratio = tsn_config.get('tsn_ratio', 0.05)

        # Separate mixing ratios for generation
        self.gen_original_ratio = tsn_config.get('gen_original_ratio', 0.98)
        self.gen_tsn_ratio = tsn_config.get('gen_tsn_ratio', 0.02)

        print(f"Using mixing ratios - Forward: {self.original_ratio}:{self.tsn_ratio}, Generate: {self.gen_original_ratio}:{self.gen_tsn_ratio}")

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass through the combined model.

        Args:
            pixel_values: Image tensor of shape [batch_size, channels, height, width]
            input_ids: Input token IDs
            attention_mask: Attention mask for input tokens
            labels: Target token IDs for training

        Returns:
            Output from PaliGemma model
        """
        # Process image through TSN module
        if pixel_values is not None:
            # Store the current batch's pixel_values for use in the closure
            current_pixel_values = pixel_values.clone()

            # Process through TSN module
            tsn_features, _ = self.tsn(current_pixel_values)

            # Project TSN features to match hidden dimension if needed
            projected_tsn_features = self.feature_projection(tsn_features)

            # Apply simple normalization (just z-score)
            tsn_mean = projected_tsn_features.mean()
            tsn_std = projected_tsn_features.std() + 1e-6
            normalized_tsn_features = (projected_tsn_features - tsn_mean) / tsn_std

            # Debug information
            print(f"\nTSN features processed - shape: {normalized_tsn_features.shape}")
            print(f"TSN features processed - mean: {normalized_tsn_features.mean().item()}, std: {normalized_tsn_features.std().item()}")
            print(f"TSN features processed - min: {normalized_tsn_features.min().item()}, max: {normalized_tsn_features.max().item()}")

            # Check if PaliGemma has a vision tower
            if hasattr(self.paligemma, 'vision_tower') and self.paligemma.vision_tower is not None:
                # Store original forward method
                original_forward = self.paligemma.vision_tower.forward

                # Get a reference to the normalized features for this batch
                batch_tsn_features = normalized_tsn_features

                # Define a new forward method that combines original vision tower output with TSN features
                def new_forward(*args, **kwargs):
                    # Get original vision tower output
                    original_output = original_forward(*args, **kwargs)

                    # Debug information
                    print(f"Original vision tower output shape: {original_output.shape}")
                    print(f"Original vision tower output mean: {original_output.mean().item()}, std: {original_output.std().item()}")

                    # Check dimensions and prepare for combination
                    if original_output.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                        batch_size = original_output.size(0)
                        seq_len = original_output.size(1)
                        hidden_dim = original_output.size(2)

                        # Reshape TSN features to match vision tower output
                        reshaped_features = batch_tsn_features.view(batch_size, 1, hidden_dim)

                        # Combine with original output
                        if seq_len > 1:
                            # If multiple tokens, keep original sequence and don't modify
                            combined_output = original_output
                        else:
                            # If single token, use weighted sum with higher weight for original
                            combined_output = self.original_ratio * original_output + self.tsn_ratio * reshaped_features

                        print(f"Combined output shape: {combined_output.shape}")
                        print(f"Combined output mean: {combined_output.mean().item()}, std: {combined_output.std().item()}")

                        return combined_output
                    else:
                        # If dimensions don't match, return original output
                        print("Dimension mismatch: Using original vision tower output")
                        return original_output

                # Replace the forward method temporarily
                self.paligemma.vision_tower.forward = new_forward

                try:
                    # Call PaliGemma with our modified vision tower
                    outputs = self.paligemma(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        **kwargs
                    )
                finally:
                    # Always restore the original forward method, even if an error occurs
                    self.paligemma.vision_tower.forward = original_forward

                return outputs

        # If no pixel values or no vision tower, just pass through PaliGemma
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

        Args:
            pixel_values: Image tensor
            input_ids: Input token IDs
            attention_mask: Attention mask for input tokens

        Returns:
            Generated token IDs
        """
        # Process image through TSN module
        if pixel_values is not None:
            # Store the current batch's pixel_values for use in the closure
            current_pixel_values = pixel_values.clone()

            # Process through TSN module
            tsn_features, _ = self.tsn(current_pixel_values)

            # Project TSN features to match hidden dimension
            projected_tsn_features = self.feature_projection(tsn_features)

            # Apply simple normalization (just z-score)
            tsn_mean = projected_tsn_features.mean()
            tsn_std = projected_tsn_features.std() + 1e-6
            normalized_tsn_features = (projected_tsn_features - tsn_mean) / tsn_std

            # Debug information
            print(f"\nTSN features processed (generate) - shape: {normalized_tsn_features.shape}")
            print(f"TSN features processed (generate) - mean: {normalized_tsn_features.mean().item()}, std: {normalized_tsn_features.std().item()}")
            print(f"TSN features processed (generate) - min: {normalized_tsn_features.min().item()}, max: {normalized_tsn_features.max().item()}")

            # Check if PaliGemma has a vision tower
            if hasattr(self.paligemma, 'vision_tower') and self.paligemma.vision_tower is not None:
                # Store original forward method
                original_forward = self.paligemma.vision_tower.forward

                # Get a reference to the normalized features for this batch
                batch_tsn_features = normalized_tsn_features

                # Define a new forward method that combines original vision tower output with TSN features
                def new_forward(*args, **kwargs):
                    # Get original vision tower output
                    original_output = original_forward(*args, **kwargs)

                    # Debug information
                    print(f"Original vision tower output shape (generate): {original_output.shape}")
                    print(f"Original vision tower output mean (generate): {original_output.mean().item()}, std: {original_output.std().item()}")

                    # Check dimensions and prepare for combination
                    if original_output.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                        batch_size = original_output.size(0)
                        seq_len = original_output.size(1)
                        hidden_dim = original_output.size(2)

                        # Reshape TSN features to match vision tower output
                        reshaped_features = batch_tsn_features.view(batch_size, 1, hidden_dim)

                        # Use the generation-specific mixing ratios
                        original_ratio = self.gen_original_ratio
                        tsn_ratio = self.gen_tsn_ratio

                        # Combine with original output - for generation, always use weighted sum
                        # even with multiple tokens, to minimize disruption
                        combined_output = original_output.clone()  # Create a copy to avoid in-place modification

                        # Only modify the first token with a very small contribution from TSN
                        if seq_len > 0:
                            combined_output[:, 0, :] = original_ratio * original_output[:, 0, :] + tsn_ratio * reshaped_features.squeeze(1)

                        print(f"Combined output shape (generate): {combined_output.shape}")
                        print(f"Combined output mean (generate): {combined_output.mean().item()}, std: {combined_output.std().item()}")

                        return combined_output
                    else:
                        # If dimensions don't match, return original output
                        print("Dimension mismatch (generate): Using original vision tower output")
                        return original_output

                # Replace the forward method temporarily
                self.paligemma.vision_tower.forward = new_forward

                # Debug information
                print(f"\nTSN Generate - input_ids shape: {input_ids.shape if input_ids is not None else None}")
                print(f"TSN Generate - attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
                print(f"TSN Generate - kwargs: {kwargs}")

                try:
                    # Call PaliGemma's generate method
                    outputs = self.paligemma.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **kwargs
                    )

                    # Debug information
                    print(f"TSN Generate - outputs shape: {outputs.shape}")
                    print(f"TSN Generate - outputs: {outputs}")
                finally:
                    # Always restore the original forward method, even if an error occurs
                    self.paligemma.vision_tower.forward = original_forward

                return outputs

        # If no pixel values or no vision tower, just pass through PaliGemma
        return self.paligemma.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

def create_tsn_paligemma_model(paligemma_model, config):
    """
    Create a TSNPaliGemmaModel instance.

    Args:
        paligemma_model: Pre-trained PaliGemma model
        config: Configuration dictionary

    Returns:
        TSNPaliGemmaModel instance
    """
    return TSNPaliGemmaModel(paligemma_model, config)
