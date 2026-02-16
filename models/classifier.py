#!/usr/bin/env python3
"""
Clean emotion recognition classifier with embedding extraction for contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import FrozenBERTEncoder
from models.fusion import get_fusion_module


class EmotionClassifier(nn.Module):
    """
    Multimodal emotion classifier with built-in embedding extraction

    Supports:
    - Audio-only, text-only, or multimodal (audio + text)
    - Classification (4 classes) or VAD regression (3 outputs)
    - Embedding extraction at [batch, 1024] for contrastive learning
    """

    def __init__(
        self,
        audio_dim=768,
        text_model_name="bert-base-uncased",
        hidden_dim=1024,
        num_classes=4,
        dropout=0.1,
        modality="both",  # "audio", "text", or "both"
        fusion_type="cross_attention",
        fusion_hidden_dim=512,
        num_attention_heads=8,
        task_type="classification",  # "classification" or "regression"
        vad_output_dim=3
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.modality = modality
        self.task_type = task_type

        # Output dimension
        self.output_dim = vad_output_dim if task_type == "regression" else num_classes

        # Text encoder
        if modality in ["text", "both"]:
            self.text_encoder = FrozenBERTEncoder(model_name=text_model_name)
            self.text_dim = self.text_encoder.get_output_dim()
        else:
            self.text_encoder = None
            self.text_dim = None

        # Build model based on modality
        if modality == "audio":
            self._build_audio_only()
        elif modality == "text":
            self._build_text_only()
        elif modality == "both":
            self._build_multimodal(fusion_type, fusion_hidden_dim, num_attention_heads, dropout)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _build_audio_only(self):
        """Audio-only: [batch, 768] → [batch, 1024] → [batch, output_dim]"""
        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def _build_text_only(self):
        """Text-only: [batch, 768] → [batch, 1024] → [batch, output_dim]"""
        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def _build_multimodal(self, fusion_type, fusion_hidden_dim, num_heads, dropout):
        """Multimodal: fusion → [batch, fusion_hidden_dim] → [batch, 1024] → [batch, output_dim]"""
        # Fusion module
        self.fusion_module = get_fusion_module(
            fusion_type=fusion_type,
            audio_dim=self.audio_dim,
            text_dim=self.text_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(fusion_hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, audio_features=None, text_input_ids=None, text_attention_mask=None, return_embeddings=False):
        """
        Forward pass with optional embedding extraction

        Args:
            audio_features: [batch_size, audio_dim] - preextracted audio features
            text_input_ids: [batch_size, seq_len] - text token IDs
            text_attention_mask: [batch_size, seq_len] - text attention mask
            return_embeddings: bool - if True, return (logits, embeddings)

        Returns:
            logits: [batch_size, output_dim]
            embeddings: [batch_size, hidden_dim] - only if return_embeddings=True
        """
        if self.modality == "audio":
            return self._forward_audio(audio_features, return_embeddings)
        elif self.modality == "text":
            return self._forward_text(text_input_ids, text_attention_mask, return_embeddings)
        elif self.modality == "both":
            return self._forward_multimodal(audio_features, text_input_ids, text_attention_mask, return_embeddings)

    def _forward_audio(self, audio_features, return_embeddings=False):
        """Audio-only forward pass"""
        if audio_features is None:
            raise ValueError("audio_features required for audio mode")

        # Get embeddings
        embeddings = self.embedding_layer(audio_features)  # [batch, 1024]

        # Get logits
        logits = self.output_layer(embeddings)  # [batch, output_dim]

        if return_embeddings:
            return logits, embeddings
        return logits

    def _forward_text(self, text_input_ids, text_attention_mask, return_embeddings=False):
        """Text-only forward pass"""
        if text_input_ids is None or text_attention_mask is None:
            raise ValueError("text inputs required for text mode")

        # Extract text features
        text_features = self.text_encoder(text_input_ids, text_attention_mask)  # [batch, 768]

        # Get embeddings
        embeddings = self.embedding_layer(text_features)  # [batch, 1024]

        # Get logits
        logits = self.output_layer(embeddings)  # [batch, output_dim]

        if return_embeddings:
            return logits, embeddings
        return logits

    def _forward_multimodal(self, audio_features, text_input_ids, text_attention_mask, return_embeddings=False):
        """Multimodal forward pass"""
        if audio_features is None:
            raise ValueError("audio_features required for multimodal mode")
        if text_input_ids is None or text_attention_mask is None:
            raise ValueError("text inputs required for multimodal mode")

        # Extract text features
        text_features = self.text_encoder(text_input_ids, text_attention_mask)  # [batch, 768]

        # Fuse modalities
        fused_features = self.fusion_module(audio_features, text_features)  # [batch, fusion_hidden_dim]

        # Get embeddings
        embeddings = self.embedding_layer(fused_features)  # [batch, 1024]

        # Get logits
        logits = self.output_layer(embeddings)  # [batch, output_dim]

        if return_embeddings:
            return logits, embeddings
        return logits


def create_model(config):
    """
    Factory function to create model from config

    Args:
        config: Configuration object or dict

    Returns:
        EmotionClassifier instance
    """
    # Handle both dict and object configs
    if hasattr(config, '__dict__'):
        cfg = config.__dict__
    else:
        cfg = config

    return EmotionClassifier(
        audio_dim=cfg.get('audio_dim', 768),
        text_model_name=cfg.get('text_model_name', 'bert-base-uncased'),
        hidden_dim=cfg.get('hidden_dim', 1024),
        num_classes=cfg.get('num_classes', 4),
        dropout=cfg.get('dropout', 0.1),
        modality=cfg.get('modality', 'both'),
        fusion_type=cfg.get('fusion_type', 'cross_attention'),
        fusion_hidden_dim=cfg.get('fusion_hidden_dim', 512),
        num_attention_heads=cfg.get('num_attention_heads', 8),
        task_type=cfg.get('task_type', 'classification'),
        vad_output_dim=cfg.get('vad_output_dim', 3)
    )
