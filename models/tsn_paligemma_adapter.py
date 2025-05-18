#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class TSNPaliGemmaAdapter(nn.Module):
    """
    Adapter module to integrate TSN features with PaliGemma model.
    Instead of replacing the vision tower, this adapter adds TSN features
    as additional input to the model.
    """
    def __init__(self, paligemma_model, tsn_module, config):
        super(TSNPaliGemmaAdapter, self).__init__()
        
        # Store PaliGemma model and TSN module
        self.paligemma = paligemma_model
        self.tsn = tsn_module
        
        # Get hidden dimension from PaliGemma model
        if hasattr(self.paligemma, 'config') and hasattr(self.paligemma.config, 'hidden_size'):
            self.hidden_dim = self.paligemma.config.hidden_size
        else:
            # Default value if not available
            self.hidden_dim = 1408  # Common hidden size for PaliGemma
        
        # Create adapter layers
        self.adapter_projection = nn.Linear(self.tsn.projection_dim, self.hidden_dim)
        self.adapter_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # Store configuration
        self.config = config
        
        # Adapter scaling factor (how much to weight the TSN features)
        self.adapter_scale = 0.3
        
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
        # First, get the PaliGemma outputs normally
        paligemma_outputs = self.paligemma(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # If no pixel values or not in training mode, return original outputs
        if pixel_values is None or not self.training:
            return paligemma_outputs
        
        # Process image through TSN module
        tsn_features, _ = self.tsn(pixel_values)
        
        # Debug information
        print(f"\nTSN Adapter - TSN features shape: {tsn_features.shape}")
        print(f"TSN Adapter - TSN features mean: {tsn_features.mean().item()}, std: {tsn_features.std().item()}")
        
        # Project TSN features to match hidden dimension
        projected_features = self.adapter_projection(tsn_features)
        normalized_features = self.adapter_layer_norm(projected_features)
        
        print(f"TSN Adapter - Projected features shape: {projected_features.shape}")
        print(f"TSN Adapter - Projected features mean: {projected_features.mean().item()}, std: {projected_features.std().item()}")
        
        # Get hidden states from PaliGemma outputs
        if hasattr(paligemma_outputs, 'hidden_states') and paligemma_outputs.hidden_states is not None:
            # Use the last hidden state
            hidden_states = paligemma_outputs.hidden_states[-1]
            
            print(f"TSN Adapter - Hidden states shape: {hidden_states.shape}")
            
            # Add TSN features to the hidden states (only to the first token)
            # Scale the contribution of TSN features
            if hidden_states.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                # Add TSN features to the first token (usually the [CLS] token)
                hidden_states[:, 0, :] = hidden_states[:, 0, :] + self.adapter_scale * normalized_features
                
                # Recompute the logits with the modified hidden states
                # This depends on the specific PaliGemma architecture
                if hasattr(self.paligemma, 'lm_head'):
                    modified_logits = self.paligemma.lm_head(hidden_states)
                    
                    # Create a new output object with modified logits
                    # This is a simplified approach and might need adjustment based on the actual model
                    paligemma_outputs.logits = modified_logits
        
        return paligemma_outputs
    
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
        # For generation, we use the original PaliGemma model
        # TSN features are not used during generation in this adapter approach
        return self.paligemma.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

def create_tsn_paligemma_adapter(paligemma_model, tsn_module, config):
    """
    Create a TSNPaliGemmaAdapter instance.
    
    Args:
        paligemma_model: Pre-trained PaliGemma model
        tsn_module: Pre-trained TSN module
        config: Configuration dictionary
        
    Returns:
        TSNPaliGemmaAdapter instance
    """
    return TSNPaliGemmaAdapter(paligemma_model, tsn_module, config)
