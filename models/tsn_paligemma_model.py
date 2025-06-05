#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VisionLanguageAttention(nn.Module):
    """Enhanced attention for vision-language tasks"""
    def __init__(self, in_channels, reduction=16):
        super(VisionLanguageAttention, self).__init__()
        # Channel attention for semantic features
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention for fine-grained details
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Cross-scale feature fusion
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x_ca = x * ca

        # Spatial attention
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x_sa = x_ca * sa

        # Feature fusion
        enhanced = self.fusion_conv(x_sa)
        output = x + enhanced  # Residual connection

        return output, sa

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    def __init__(self, in_channels, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for _ in range(3)
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(3)
        ])

    def forward(self, features):
        # features: list of [P3, P4, P5] from different scales
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode='bilinear', align_corners=False
            )

        # Final convolutions
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        return outputs

class TSNModule(nn.Module):
    def __init__(self, config):
        super(TSNModule, self).__init__()

        self.backbone_name = config.get('backbone', 'resnet50')
        self.pretrained = config.get('pretrained', True)
        self.segment_scales = config.get('segment_scales', [[1, 1], [2, 2], [3, 3]])
        self.feature_dim = config.get('feature_dim', 2048)
        self.use_attention = config.get('use_attention', True)
        self.use_fpn = config.get('use_fpn', True)
        self.projection_dim = config.get('projection_dim', 1152)
        self.dropout_rate = float(config.get('dropout_rate', 0.1))
        self.layer_norm_eps = float(config.get('layer_norm_eps', 1e-6))

        def init_backbone(model_class, weights_class=None, feature_dim=None):
            if weights_class and hasattr(weights_class, 'IMAGENET1K_V1') and self.pretrained:
                backbone = model_class(weights='IMAGENET1K_V1')
            else:
                backbone = model_class(pretrained=self.pretrained)

            if feature_dim:
                self.feature_dim = feature_dim

            return backbone

        if self.backbone_name == 'resnet50':
            backbone = init_backbone(models.resnet50, models.ResNet50_Weights)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        elif self.backbone_name == 'resnet101':
            backbone = init_backbone(models.resnet101, models.ResNet101_Weights)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        elif self.backbone_name == 'resnet152':
            backbone = init_backbone(models.resnet152, models.ResNet152_Weights)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        elif self.backbone_name == 'inception_v3':
            backbone = init_backbone(models.inception_v3, models.Inception_V3_Weights)
            backbone.aux_logits = False
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        elif self.backbone_name == 'resnet18':
            backbone = init_backbone(models.resnet18, models.ResNet18_Weights, feature_dim=512)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        elif self.backbone_name == 'efficientnet_b0':
            backbone = init_backbone(models.efficientnet_b0, models.EfficientNet_B0_Weights, feature_dim=1280)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        elif self.backbone_name == 'efficientnet_b3':
            backbone = init_backbone(models.efficientnet_b3, models.EfficientNet_B3_Weights, feature_dim=1536)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        if self.use_attention:
            self.attention_modules = nn.ModuleList([
                VisionLanguageAttention(self.feature_dim) for _ in range(len(self.segment_scales))
            ])

        # Feature Pyramid Network for multi-scale fusion
        if self.use_fpn:
            self.fpn = FeaturePyramidNetwork(self.feature_dim, self.feature_dim // 2)
            fpn_output_dim = (self.feature_dim // 2) * len(self.segment_scales)
        else:
            fpn_output_dim = self.feature_dim * len(self.segment_scales)

        # Enhanced projection with residual connections
        self.projection = nn.Sequential(
            nn.Linear(fpn_output_dim, fpn_output_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(fpn_output_dim // 2, self.projection_dim),
            nn.LayerNorm(self.projection_dim, eps=self.layer_norm_eps)
        )

        # Adaptive feature weighting
        self.scale_weights = nn.Parameter(torch.ones(len(self.segment_scales)))

        # Cross-scale feature interaction
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim, num_heads=8, dropout=self.dropout_rate, batch_first=True
        )

    def forward(self, x):
        batch_size = x.size(0)
        all_features = []
        all_raw_features = []
        attention_maps = []

        # Ensure minimum input size
        if x.size(2) < 224 or x.size(3) < 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        for i, scale in enumerate(self.segment_scales):
            h_segments, w_segments = scale
            h_size = max(1, x.size(2) // h_segments)
            w_size = max(1, x.size(3) // w_segments)

            scale_features = []
            scale_raw_features = []
            scale_attention_maps = []

            for h_idx in range(h_segments):
                for w_idx in range(w_segments):
                    h_start = h_idx * h_size
                    h_end = (h_idx + 1) * h_size if h_idx < h_segments - 1 else x.size(2)
                    w_start = w_idx * w_size
                    w_end = (w_idx + 1) * w_size if w_idx < w_segments - 1 else x.size(3)

                    segment = x[:, :, h_start:h_end, w_start:w_end]

                    # Skip small segments
                    if segment.size(2) < 32 or segment.size(3) < 32:
                        continue

                    # Resize to standard input size
                    segment = F.interpolate(segment, size=(224, 224), mode='bilinear', align_corners=False)

                    try:
                        # Extract features
                        raw_features = self.backbone(segment)
                        scale_raw_features.append(raw_features)

                        # Apply attention if enabled
                        if self.use_attention:
                            enhanced_features, attention = self.attention_modules[i](raw_features)
                            scale_attention_maps.append(attention)
                        else:
                            enhanced_features = raw_features

                        # Global average pooling
                        pooled_features = F.adaptive_avg_pool2d(enhanced_features, (1, 1))
                        pooled_features = pooled_features.view(batch_size, -1)

                        # Ensure dtype consistency
                        if pooled_features.dtype != x.dtype:
                            pooled_features = pooled_features.to(dtype=x.dtype)

                        scale_features.append(pooled_features)

                    except Exception:
                        continue

            # Aggregate features for this scale
            if len(scale_features) > 1:
                # Use weighted average with cross-scale attention
                scale_features = torch.stack(scale_features, dim=1)
                attended_features, _ = self.cross_scale_attention(
                    scale_features, scale_features, scale_features
                )
                scale_features = torch.mean(attended_features, dim=1)
            elif len(scale_features) == 1:
                scale_features = scale_features[0]
            else:
                # Fallback: zero features
                scale_features = torch.zeros(batch_size, self.feature_dim, device=x.device, dtype=x.dtype)

            # Apply scale-specific weighting
            scale_weight = torch.softmax(self.scale_weights, dim=0)[i]
            scale_features = scale_features * scale_weight

            all_features.append(scale_features)
            if scale_raw_features:
                all_raw_features.append(scale_raw_features[0])

            if self.use_attention and len(scale_attention_maps) > 0:
                attention_maps.append(scale_attention_maps)

        # Feature Pyramid Network processing
        if self.use_fpn and len(all_raw_features) >= 2:
            try:
                fpn_features = self.fpn(all_raw_features[:3])
                # Convert FPN features to same format as all_features
                fpn_pooled = []
                for fpn_feat in fpn_features:
                    pooled = F.adaptive_avg_pool2d(fpn_feat, (1, 1))
                    pooled = pooled.view(batch_size, -1)
                    if pooled.dtype != x.dtype:
                        pooled = pooled.to(dtype=x.dtype)
                    fpn_pooled.append(pooled)
                combined_features = torch.cat(fpn_pooled, dim=1)
            except Exception:
                # Fallback to original features
                combined_features = torch.cat(all_features, dim=1)
        else:
            combined_features = torch.cat(all_features, dim=1)

        # Project to target dimension
        try:
            projected_features = self.projection(combined_features)
        except Exception:
            # Emergency fallback
            projected_features = torch.zeros(
                batch_size, self.projection_dim, device=x.device, dtype=x.dtype
            )

        return projected_features, attention_maps

class TSNPaliGemmaModel(nn.Module):
    def __init__(self, paligemma_model, config):
        super(TSNPaliGemmaModel, self).__init__()

        self.paligemma = paligemma_model
        tsn_config = config.get('tsn', {})
        self.tsn = TSNModule(tsn_config)
        self.config = config

        if hasattr(self.paligemma, 'vision_tower') and hasattr(self.paligemma.vision_tower, 'config'):
            if hasattr(self.paligemma.vision_tower.config, 'hidden_size'):
                self.vision_hidden_dim = self.paligemma.vision_tower.config.hidden_size
            else:
                self.vision_hidden_dim = 1152
        else:
            self.vision_hidden_dim = 1152

        if hasattr(self.paligemma, 'config') and hasattr(self.paligemma.config, 'hidden_size'):
            self.text_hidden_dim = self.paligemma.config.hidden_size
        else:
            self.text_hidden_dim = 2048

        self.feature_projection = nn.Linear(self.tsn.projection_dim, self.vision_hidden_dim)

        self.adaptive_mixing = tsn_config.get('adaptive_mixing', True)
        self.spatial_awareness = tsn_config.get('spatial_awareness', True)
        self.feature_enhancement = tsn_config.get('feature_enhancement', True)

        if self.adaptive_mixing:
            self.mixing_attention = nn.Sequential(
                nn.Linear(self.vision_hidden_dim * 2, self.vision_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.vision_hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        self.initial_original_ratio = tsn_config.get('initial_original_ratio', 0.98)
        self.initial_tsn_ratio = tsn_config.get('initial_tsn_ratio', 0.02)
        self.final_original_ratio = tsn_config.get('final_original_ratio', 0.85)
        self.final_tsn_ratio = tsn_config.get('final_tsn_ratio', 0.15)
        self.gen_original_ratio = tsn_config.get('gen_original_ratio', 0.90)
        self.gen_tsn_ratio = tsn_config.get('gen_tsn_ratio', 0.10)
        self.progressive_steps = tsn_config.get('progressive_steps', 2000)
        self.warmup_tsn_steps = tsn_config.get('warmup_tsn_steps', 500)

        self.training_step = 0

    def get_current_mixing_ratios(self):
        if self.training_step < self.warmup_tsn_steps:
            return 1.0, 0.0

        progress_step = self.training_step - self.warmup_tsn_steps
        progress_ratio = min(1.0, progress_step / self.progressive_steps)

        current_original = self.initial_original_ratio + progress_ratio * (self.final_original_ratio - self.initial_original_ratio)
        current_tsn = self.initial_tsn_ratio + progress_ratio * (self.final_tsn_ratio - self.initial_tsn_ratio)

        return current_original, current_tsn

    def update_training_step(self, step):
        self.training_step = step

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if pixel_values is not None:
            current_pixel_values = pixel_values.clone()

            tsn_dtype = next(self.tsn.parameters()).dtype
            if current_pixel_values.dtype != tsn_dtype:
                current_pixel_values = current_pixel_values.to(dtype=tsn_dtype)

            tsn_features, _ = self.tsn(current_pixel_values)
            projected_tsn_features = self.feature_projection(tsn_features)
            projected_tsn_features = F.layer_norm(projected_tsn_features, projected_tsn_features.shape[-1:])

            if hasattr(self.paligemma, 'get_image_features'):
                original_get_image_features = self.paligemma.get_image_features
                batch_tsn_features = projected_tsn_features

                def new_get_image_features(pixel_values_vt):
                    nonlocal batch_tsn_features
                    original_features = original_get_image_features(pixel_values_vt)

                    batch_size = original_features.size(0)
                    seq_len = original_features.size(1)

                    if batch_tsn_features.size(0) == batch_size:
                        if batch_tsn_features.dtype != original_features.dtype:
                            batch_tsn_features = batch_tsn_features.to(dtype=original_features.dtype)

                        reshaped_tsn = batch_tsn_features.unsqueeze(1).expand(-1, seq_len, -1)
                        current_original_ratio, current_tsn_ratio = self.get_current_mixing_ratios()
                        normalized_tsn = F.layer_norm(reshaped_tsn, reshaped_tsn.shape[-1:])

                        if self.adaptive_mixing and hasattr(self, 'mixing_attention'):
                            combined_for_attention = torch.cat([original_features, normalized_tsn], dim=-1)
                            attention_weights = self.mixing_attention(combined_for_attention)
                            adaptive_tsn_ratio = current_tsn_ratio * attention_weights
                            adaptive_original_ratio = 1.0 - adaptive_tsn_ratio
                            mixed_features = adaptive_original_ratio * original_features + adaptive_tsn_ratio * normalized_tsn
                        else:
                            mixed_features = current_original_ratio * original_features + current_tsn_ratio * normalized_tsn

                        if self.feature_enhancement:
                            enhancement = 0.1 * current_tsn_ratio * normalized_tsn
                            mixed_features = mixed_features + enhancement

                        residual_strength = 0.02 * (1.0 - current_tsn_ratio)
                        mixed_features = mixed_features + residual_strength * original_features

                        return mixed_features
                    else:
                        return original_features

                self.paligemma.get_image_features = new_get_image_features

                try:
                    outputs = self.paligemma(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        **kwargs
                    )
                finally:
                    self.paligemma.get_image_features = original_get_image_features

                return outputs

        return self.paligemma(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        generation_kwargs = kwargs.copy()
        eval_cfg = self.config.get('evaluation', {})
        generation_kwargs.setdefault('max_new_tokens', eval_cfg.get('max_new_tokens', 20))
        generation_kwargs.setdefault('num_beams', eval_cfg.get('num_beams', 1))
        generation_kwargs.setdefault('repetition_penalty', eval_cfg.get('repetition_penalty', 1.2))
        generation_kwargs.setdefault('no_repeat_ngram_size', eval_cfg.get('no_repeat_ngram_size', 3))
        generation_kwargs.setdefault('early_stopping', eval_cfg.get('early_stopping', True))
        generation_kwargs.setdefault('min_length', eval_cfg.get('min_length', None))
        generation_kwargs.setdefault('do_sample', eval_cfg.get('do_sample', False))
        if generation_kwargs['do_sample']:
            generation_kwargs.setdefault('temperature', eval_cfg.get('temperature', 1.0))
            generation_kwargs.setdefault('top_p', eval_cfg.get('top_p', 1.0))
        if eval_cfg.get('generate_max_length') is not None:
            generation_kwargs['max_length'] = eval_cfg.get('generate_max_length')
        if pixel_values is not None:
            current_pixel_values = pixel_values.clone()
            tsn_dtype = next(self.tsn.parameters()).dtype
            if current_pixel_values.dtype != tsn_dtype:
                current_pixel_values = current_pixel_values.to(dtype=tsn_dtype)
            tsn_features, _ = self.tsn(current_pixel_values)
            projected_tsn = self.feature_projection(tsn_features)
            projected_tsn = F.layer_norm(projected_tsn, projected_tsn.shape[-1:])
            if hasattr(self.paligemma, 'get_image_features'):
                original_get_image_features = self.paligemma.get_image_features
                batch_tsn = projected_tsn
                def new_get_image_features(pixel_values_vt):
                    orig_feats = original_get_image_features(pixel_values_vt)
                    if batch_tsn.size(0) == orig_feats.size(0):
                        if batch_tsn.dtype != orig_feats.dtype:
                            batch_tsn.to(dtype=orig_feats.dtype)
                        tsn_expanded = batch_tsn.unsqueeze(1).expand_as(orig_feats)
                        return self.gen_original_ratio * orig_feats + self.gen_tsn_ratio * tsn_expanded
                    return orig_feats
                self.paligemma.get_image_features = new_get_image_features
                try:
                    outputs = self.paligemma.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_kwargs
                    )
                finally:
                    self.paligemma.get_image_features = original_get_image_features
                return outputs
        return self.paligemma.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )

def create_tsn_paligemma_model(paligemma_model, config):
    tsn_model = TSNPaliGemmaModel(paligemma_model, config)

    device = next(paligemma_model.parameters()).device
    dtype = next(paligemma_model.parameters()).dtype

    tsn_model.tsn = tsn_model.tsn.to(device=device, dtype=dtype)
    tsn_model.feature_projection = tsn_model.feature_projection.to(device=device, dtype=dtype)

    if hasattr(tsn_model.tsn, 'backbone'):
        tsn_model.tsn.backbone = tsn_model.tsn.backbone.to(device=device, dtype=dtype)

    return tsn_model
