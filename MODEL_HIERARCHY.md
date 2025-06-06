# TSN-PaliGemma Model Hierarchy Documentation

## Overview
This document provides a comprehensive visualization of the TSN-Enhanced PaliGemma model architecture, showing class relationships, data flow, and system hierarchy.

## 1. Main System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TSN-PALIGEMMA SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Image (224x224x3)                                       │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   TSNModule     │  ◄──── Core Enhancement Engine            │
│  │                 │                                           │
│  │  ┌─────────────┐│                                           │
│  │  │  Backbone   ││  ◄──── ResNet/EfficientNet               │
│  │  │   CNN       ││                                           │
│  │  └─────────────┘│                                           │
│  │         │       │                                           │
│  │         ▼       │                                           │
│  │  ┌─────────────┐│                                           │
│  │  │ VisionLang  ││  ◄──── Channel + Spatial Attention       │
│  │  │ Attention   ││                                           │
│  │  └─────────────┘│                                           │
│  │         │       │                                           │
│  │         ▼       │                                           │
│  │  ┌─────────────┐│                                           │
│  │  │ Feature     ││  ◄──── Multi-Scale Fusion                │
│  │  │ Pyramid     ││                                           │
│  │  │ Network     ││                                           │
│  │  └─────────────┘│                                           │
│  └─────────────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Feature         │  ◄──── Project to PaliGemma Dimensions   │
│  │ Projection      │                                           │
│  └─────────────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ TSNPaliGemma    │  ◄──── Integration Wrapper               │
│  │ Model           │                                           │
│  │                 │                                           │
│  │ ┌─────────────┐ │                                           │
│  │ │ PaliGemma   │ │  ◄──── Base Vision-Language Model        │
│  │ │ Base Model  │ │                                           │
│  │ └─────────────┘ │                                           │
│  │         │       │                                           │
│  │         ▼       │                                           │
│  │ ┌─────────────┐ │                                           │
│  │ │ Adaptive    │ │  ◄──── Dynamic Feature Mixing            │
│  │ │ Mixing      │ │                                           │
│  │ └─────────────┘ │                                           │
│  └─────────────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  Generated Caption Text                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Class Hierarchy Tree

```
TSNPaliGemmaModel (Main Integration Wrapper)
├── PaliGemmaForConditionalGeneration (Base Model)
│   ├── Vision Tower (SigLIP Vision Encoder)
│   ├── Language Model (Gemma-2B)
│   └── Multi-Modal Projector
│
├── TSNModule (Spatial Enhancement Engine)
│   ├── Backbone CNN
│   │   ├── ResNet-50/101/152 (Default Options)
│   │   └── EfficientNet-B0 to B7 (Alternative Options)
│   │
│   ├── VisionLanguageAttention (3 instances - one per scale)
│   │   ├── Channel Attention
│   │   │   ├── AdaptiveAvgPool2d
│   │   │   ├── Conv2d (Reduction Layer)
│   │   │   ├── ReLU
│   │   │   ├── Conv2d (Expansion Layer)
│   │   │   └── Sigmoid
│   │   │
│   │   ├── Spatial Attention
│   │   │   ├── Mean Pooling (Channel-wise)
│   │   │   ├── Max Pooling (Channel-wise)
│   │   │   ├── Concatenation
│   │   │   ├── Conv2d (7x7 kernel)
│   │   │   └── Sigmoid
│   │   │
│   │   └── Fusion Convolution
│   │
│   ├── FeaturePyramidNetwork
│   │   ├── Lateral Convolutions (3 scales: P3, P4, P5)
│   │   ├── Top-Down Pathway
│   │   └── Final Refinement Convolutions
│   │
│   ├── Cross-Scale Attention
│   │   └── Multi-Head Attention Module
│   │
│   ├── Scale Weighting
│   │   └── Learnable Parameters (3 scales)
│   │
│   └── Projection Layer
│       ├── Linear Layer (features → 1152)
│       ├── LayerNorm
│       └── Dropout
│
├── Feature Projection (TSN → PaliGemma)
│   └── Linear (projection_dim → vision_hidden_dim)
│
└── Adaptive Mixing Module
    ├── Mixing Attention
    │   ├── Linear (vision_hidden_dim * 2 → vision_hidden_dim // 2)
    │   ├── ReLU
    │   ├── Dropout
    │   ├── Linear (vision_hidden_dim // 2 → 1)
    │   └── Sigmoid
    │
    └── Progressive Ratio Controller
        ├── Warmup Phase (500 steps)
        ├── Progressive Phase (2000 steps)
        └── Final Phase (85% original + 15% TSN)
```

## 3. Multi-Scale Processing Flow

```
Input Image (B, 3, H, W)
         │
         ▼
┌────────────────────┐
│   Size Check &     │ ──────► Resize to (224, 224) if needed
│   Preprocessing    │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  Multi-Scale       │ ──────► Segment Scales: [1,1], [2,2], [3,3]
│  Segmentation      │
└────────────────────┘
         │
         ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │   Scale 1   │    │   Scale 2   │    │   Scale 3   │
  │   (1x1)     │    │   (2x2)     │    │   (3x3)     │
  │             │    │             │    │             │
  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │
  │ │Segment 1│ │    │ │Segment 1│ │    │ │Segment 1│ │
  │ └─────────┘ │    │ ├─────────┤ │    │ ├─────────┤ │
  │             │    │ │Segment 2│ │    │ │Segment 2│ │
  │             │    │ ├─────────┤ │    │ ├─────────┤ │
  │             │    │ │Segment 3│ │    │ │   ...   │ │
  │             │    │ ├─────────┤ │    │ ├─────────┤ │
  │             │    │ │Segment 4│ │    │ │Segment 9│ │
  │             │    │ └─────────┘ │    │ └─────────┘ │
  └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  Backbone   │    │  Backbone   │    │  Backbone   │
  │  CNN        │    │  CNN        │    │  CNN        │
  │  Processing │    │  Processing │    │  Processing │
  └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │ VisionLang  │    │ VisionLang  │    │ VisionLang  │
  │ Attention 1 │    │ Attention 2 │    │ Attention 3 │
  └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  Global     │    │  Global     │    │  Global     │
  │  Average    │    │  Average    │    │  Average    │
  │  Pooling    │    │  Pooling    │    │  Pooling    │
  └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                   ┌─────────────────┐
                   │  Cross-Scale    │
                   │  Attention &    │
                   │  Aggregation    │
                   └─────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │ Feature Pyramid │
                   │ Network (FPN)   │
                   │ P3, P4, P5      │
                   └─────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │   Projection    │
                   │ (feat_dim →     │
                   │  1152)          │
                   └─────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │  TSN Features   │
                   │ (B, 1152)       │
                   └─────────────────┘
```

## 4. VisionLanguageAttention Mechanism

```
Input Features (B, C, H, W)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                  VISION LANGUAGE ATTENTION                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │  CHANNEL ATTENTION  │    │      SPATIAL ATTENTION      │ │
│  │                     │    │                             │ │
│  │  Input (B,C,H,W)    │    │   Channel Weighted          │ │
│  │         │           │    │   Features (B,C,H,W)        │ │
│  │         ▼           │    │            │                │ │
│  │  AdaptiveAvgPool2d  │    │            ▼                │ │
│  │  (B,C,1,1)          │    │   ┌─────────────────────┐   │ │
│  │         │           │    │   │  Channel Statistics │   │ │
│  │         ▼           │    │   │                     │   │ │
│  │  Conv2d(C→C//16)    │    │   │  Mean: (B,1,H,W)    │   │ │
│  │         │           │    │   │   +                 │   │ │
│  │         ▼           │    │   │  Max:  (B,1,H,W)    │   │ │
│  │       ReLU          │    │   │         │           │   │ │
│  │         │           │    │   │         ▼           │   │ │
│  │         ▼           │    │   │  Concat: (B,2,H,W)  │   │ │
│  │  Conv2d(C//16→C)    │    │   └─────────────────────┘   │ │
│  │         │           │    │            │                │ │
│  │         ▼           │    │            ▼                │ │
│  │      Sigmoid        │    │   Conv2d(2→1, 7x7)         │ │
│  │         │           │    │            │                │ │
│  │         ▼           │    │            ▼                │ │
│  │  Channel Weights    │    │      Sigmoid               │ │
│  │  (B,C,1,1)          │    │            │                │ │
│  └─────────────────────┘    │            ▼                │ │
│           │                 │   Spatial Weights           │ │
│           │                 │   (B,1,H,W)                 │ │
│           │                 └─────────────────────────────┘ │
│           │                            │                    │
│           ▼                            ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FEATURE FUSION                         │   │
│  │                                                     │   │
│  │  Original * Channel_Weights * Spatial_Weights      │   │
│  │                         │                           │   │
│  │                         ▼                           │   │
│  │                 Fusion Conv1x1                     │   │
│  │                         │                           │   │
│  │                         ▼                           │   │
│  │              Residual Connection                    │   │
│  │         (Enhanced + Original)                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│           Enhanced Features (B,C,H,W)                       │
│                    +                                        │
│           Attention Map (B,1,H,W)                           │
└─────────────────────────────────────────────────────────────┘
```

## 5. Progressive Training Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Warmup (Steps 1-500)                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Mixing Ratio: 100% PaliGemma + 0% TSN                 │   │
│  │  Purpose: Establish stable baseline                     │   │
│  │  Benefits: Prevents training instability               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  Phase 2: Progressive (Steps 501-2500)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Mixing Ratio: 98%→85% PaliGemma + 2%→15% TSN          │   │
│  │  Purpose: Gradual TSN integration                       │   │
│  │  Benefits: Smooth learning transition                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  Phase 3: Final (Steps 2501+)                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Mixing Ratio: 85% PaliGemma + 15% TSN                 │   │
│  │  Purpose: Optimal performance balance                   │   │
│  │  Benefits: Maximum enhancement without degradation      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Generation Phase (Inference)                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Mixing Ratio: 90% PaliGemma + 10% TSN                 │   │
│  │  Purpose: Conservative enhancement for generation       │   │
│  │  Benefits: Reliable caption quality                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 6. Performance Impact Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PERFORMANCE METRICS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Base PaliGemma Model Results:                                  │
│  ├── BLEU Score: 0.0176 (Very Poor)                           │
│  ├── ROUGE-L: 0.0471 (Poor)                                   │
│  ├── Word Overlap: 0.0602 (Poor)                              │
│  ├── METEOR: 0.0000 (No Semantic Understanding)               │
│  ├── CIDEr: 0.0000 (No Consensus)                             │
│  └── Common Response: "Sorry, as a base VLM..."               │
│                                                                 │
│           ▼ AFTER TSN ENHANCEMENT ▼                            │
│                                                                 │
│  TSN-Enhanced PaliGemma Results:                                │
│  ├── BLEU Score: 0.1809 (+929.26% improvement)                │
│  ├── ROUGE-L: 0.3144 (+567.02% improvement)                   │
│  ├── Word Overlap: 0.5499 (+813.12% improvement)              │
│  ├── METEOR: 0.0459 (∞% improvement from 0)                   │
│  ├── CIDEr: 0.1047 (∞% improvement from 0)                    │
│  └── 100% Successful Caption Generation (4,454/4,454)         │
│                                                                 │
│  Key Success Factors:                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Multi-Scale Spatial Analysis                          │   │
│  │ • Attention-Driven Feature Enhancement                  │   │
│  │ • Progressive Training Strategy                         │   │
│  │ • Non-Destructive Integration                           │   │
│  │ • Remote Sensing Domain Adaptation                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 7. File Structure and Code Locations

```
project-di/
├── models/
│   └── tsn_paligemma_model.py ◄─── CORE MODEL ARCHITECTURE
│       ├── VisionLanguageAttention (Lines 7-78)
│       │   ├── __init__ (Lines 20-49)
│       │   └── forward (Lines 50-78)
│       ├── FeaturePyramidNetwork (Lines 79-137)
│       │   ├── __init__ (Lines 91-109)  
│       │   └── forward (Lines 111-137)
│       ├── TSNModule (Lines 138-396)
│       │   ├── __init__ (Lines 151-268)
│       │   └── forward (Lines 269-396)
│       ├── TSNPaliGemmaModel (Lines 397-585)
│       │   ├── __init__ (Lines 398-444)
│       │   ├── get_current_mixing_ratios (Lines 445-457)
│       │   ├── forward (Lines 460-533)
│       │   └── generate (Lines 534-585)
│       └── create_tsn_paligemma_model (Lines 586-599)
│
├── train_paligemma_qlora_tsn.py ◄─── TRAINING SYSTEM
│   ├── ValidationLossCallback (Lines 36-120)
│   ├── BestModelCallback (Lines 122-240)
│   ├── SamplingCallback (Lines 241-392)
│   ├── RISCDataset (Lines 393-426)
│   └── Main Training Loop (Lines 453-755)
│
└── config/config.yaml ◄─── CONFIGURATION PARAMETERS
    ├── Model Settings
    ├── Training Parameters
    └── TSN Configuration
```