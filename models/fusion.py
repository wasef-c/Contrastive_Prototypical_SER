#!/usr/bin/env python3
"""
Multimodal fusion mechanisms for combining audio and text features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion mechanism for audio and text features
    Allows each modality to attend to the other modality
    """

    def __init__(self, audio_dim=768, text_dim=768, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project audio and text to common hidden dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Cross-attention: Audio attends to Text
        self.audio_to_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: Text attends to Audio
        self.text_to_audio_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.audio_norm1 = nn.LayerNorm(hidden_dim)
        self.text_norm1 = nn.LayerNorm(hidden_dim)

        # Feed-forward networks
        self.audio_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Final layer norms
        self.audio_norm2 = nn.LayerNorm(hidden_dim)
        self.text_norm2 = nn.LayerNorm(hidden_dim)

        # Output projection to combine both modalities
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        print(f"✅ CrossAttentionFusion initialized (hidden_dim={hidden_dim}, heads={num_heads})")

    def forward(self, audio_features, text_features):
        """
        Fuse audio and text features using cross-attention

        Args:
            audio_features: [batch_size, audio_dim]
            text_features: [batch_size, text_dim]

        Returns:
            fused_features: [batch_size, hidden_dim]
        """
        batch_size = audio_features.shape[0]

        # Project to common dimension
        audio_proj = self.audio_proj(audio_features)  # [batch_size, hidden_dim]
        text_proj = self.text_proj(text_features)  # [batch_size, hidden_dim]

        # Add sequence dimension for attention (treating each as single-token sequence)
        audio_seq = audio_proj.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        text_seq = text_proj.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Cross-attention: Audio attends to Text
        audio_attended, _ = self.audio_to_text_attention(
            query=audio_seq,
            key=text_seq,
            value=text_seq
        )  # [batch_size, 1, hidden_dim]
        audio_attended = audio_attended.squeeze(1)  # [batch_size, hidden_dim]

        # Residual connection + norm
        audio_enhanced = self.audio_norm1(audio_proj + audio_attended)

        # Feed-forward + residual
        audio_out = self.audio_norm2(audio_enhanced + self.audio_ffn(audio_enhanced))

        # Cross-attention: Text attends to Audio
        text_attended, _ = self.text_to_audio_attention(
            query=text_seq,
            key=audio_seq,
            value=audio_seq
        )  # [batch_size, 1, hidden_dim]
        text_attended = text_attended.squeeze(1)  # [batch_size, hidden_dim]

        # Residual connection + norm
        text_enhanced = self.text_norm1(text_proj + text_attended)

        # Feed-forward + residual
        text_out = self.text_norm2(text_enhanced + self.text_ffn(text_enhanced))

        # Concatenate and fuse
        combined = torch.cat([audio_out, text_out], dim=-1)  # [batch_size, hidden_dim * 2]
        fused = self.fusion_layer(combined)  # [batch_size, hidden_dim]

        return fused


class SimpleConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion
    Concatenates audio and text features and projects to output dimension
    """

    def __init__(self, audio_dim=768, text_dim=768, output_dim=512, dropout=0.1):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.output_dim = output_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(audio_dim + text_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        print(f"✅ SimpleConcatFusion initialized (output_dim={output_dim})")

    def forward(self, audio_features, text_features):
        """
        Fuse audio and text features by concatenation

        Args:
            audio_features: [batch_size, audio_dim]
            text_features: [batch_size, text_dim]

        Returns:
            fused_features: [batch_size, output_dim]
        """
        combined = torch.cat([audio_features, text_features], dim=-1)
        fused = self.fusion_layer(combined)
        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism that learns to weight audio and text contributions
    """

    def __init__(self, audio_dim=768, text_dim=768, hidden_dim=512, dropout=0.1):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # Project to common dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        print(f"✅ GatedFusion initialized (hidden_dim={hidden_dim})")

    def forward(self, audio_features, text_features):
        """
        Fuse audio and text features using learned gates

        Args:
            audio_features: [batch_size, audio_dim]
            text_features: [batch_size, text_dim]

        Returns:
            fused_features: [batch_size, hidden_dim]
        """
        # Project to common dimension
        audio_proj = self.audio_proj(audio_features)  # [batch_size, hidden_dim]
        text_proj = self.text_proj(text_features)  # [batch_size, hidden_dim]

        # Compute gates
        combined = torch.cat([audio_proj, text_proj], dim=-1)  # [batch_size, hidden_dim * 2]
        gates = self.gate(combined)  # [batch_size, 2]

        audio_gate = gates[:, 0:1]  # [batch_size, 1]
        text_gate = gates[:, 1:2]  # [batch_size, 1]

        # Weighted combination
        fused = audio_gate * audio_proj + text_gate * text_proj  # [batch_size, hidden_dim]

        # Normalize and dropout
        fused = self.norm(fused)
        fused = self.dropout(fused)

        return fused


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that can dynamically select fusion strategy
    Useful for handling missing modalities or varying modality quality
    """

    def __init__(self, audio_dim=768, text_dim=768, hidden_dim=512, dropout=0.1):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # Individual modality encoders
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        print(f"✅ AdaptiveFusion initialized (hidden_dim={hidden_dim})")

    def forward(self, audio_features=None, text_features=None):
        """
        Adaptively fuse features, handling missing modalities

        Args:
            audio_features: [batch_size, audio_dim] or None
            text_features: [batch_size, text_dim] or None

        Returns:
            fused_features: [batch_size, hidden_dim]
        """
        has_audio = audio_features is not None
        has_text = text_features is not None

        if has_audio and has_text:
            # Both modalities available
            audio_encoded = self.audio_encoder(audio_features)
            text_encoded = self.text_encoder(text_features)
            combined = torch.cat([audio_encoded, text_encoded], dim=-1)
            return self.fusion_net(combined)

        elif has_audio:
            # Only audio available
            return self.audio_encoder(audio_features)

        elif has_text:
            # Only text available
            return self.text_encoder(text_features)

        else:
            raise ValueError("At least one modality must be provided")


def get_fusion_module(fusion_type, audio_dim=768, text_dim=768, hidden_dim=512,
                      num_heads=8, dropout=0.1):
    """
    Factory function to get the appropriate fusion module

    Args:
        fusion_type: Type of fusion ("cross_attention", "concat", "gated", "adaptive")
        audio_dim: Audio feature dimension
        text_dim: Text feature dimension
        hidden_dim: Hidden dimension for fusion
        num_heads: Number of attention heads (for cross-attention)
        dropout: Dropout rate

    Returns:
        Fusion module instance
    """
    if fusion_type == "cross_attention":
        return CrossAttentionFusion(audio_dim, text_dim, hidden_dim, num_heads, dropout)
    elif fusion_type == "concat":
        return SimpleConcatFusion(audio_dim, text_dim, hidden_dim, dropout)
    elif fusion_type == "gated":
        return GatedFusion(audio_dim, text_dim, hidden_dim, dropout)
    elif fusion_type == "adaptive":
        return AdaptiveFusion(audio_dim, text_dim, hidden_dim, dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


if __name__ == "__main__":
    # Test fusion modules
    print("Testing fusion modules...")

    batch_size = 8
    audio_dim = 768
    text_dim = 768

    # Create dummy features
    audio_features = torch.randn(batch_size, audio_dim)
    text_features = torch.randn(batch_size, text_dim)

    print(f"\nInput shapes:")
    print(f"  Audio: {audio_features.shape}")
    print(f"  Text: {text_features.shape}")

    # Test each fusion type
    fusion_types = ["cross_attention", "concat", "gated", "adaptive"]

    for fusion_type in fusion_types:
        print(f"\n{'='*50}")
        print(f"Testing {fusion_type} fusion...")

        fusion_module = get_fusion_module(
            fusion_type=fusion_type,
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=512
        )

        if fusion_type == "adaptive":
            # Test with both modalities
            fused = fusion_module(audio_features, text_features)
            print(f"Fused features (both): {fused.shape}")

            # Test with only audio
            fused_audio = fusion_module(audio_features=audio_features, text_features=None)
            print(f"Fused features (audio only): {fused_audio.shape}")

            # Test with only text
            fused_text = fusion_module(audio_features=None, text_features=text_features)
            print(f"Fused features (text only): {fused_text.shape}")
        else:
            fused = fusion_module(audio_features, text_features)
            print(f"Fused features: {fused.shape}")
            print(f"Feature statistics:")
            print(f"  Mean: {fused.mean().item():.4f}")
            print(f"  Std: {fused.std().item():.4f}")

    print("\n✅ All fusion module tests passed!")
