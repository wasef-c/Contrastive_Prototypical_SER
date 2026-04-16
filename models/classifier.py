#!/usr/bin/env python3
"""
Clean emotion recognition classifier with embedding extraction for contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import FrozenBERTEncoder
from models.audio_encoder import Wav2Vec2Encoder, Emotion2VecEncoder
from models.fusion import get_fusion_module


class EmotionClassifier(nn.Module):
    """
    Multimodal emotion classifier with built-in embedding extraction

    Supports:
    - Audio-only, text-only, or multimodal (audio + text)
    - Classification (4 classes) or VAD regression (3 outputs)
    - Embedding extraction at [batch, 1024] for contrastive learning
    - Preextracted audio features or raw waveforms via Wav2Vec2
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
        vad_output_dim=3,
        projection_dim=128,
        projection_hidden_dim=512,
        use_projection_head=True,
        unfreeze_bert_layers=0,
        audio_encoder_type="preextracted",  # "preextracted", "wav2vec2", or "emotion2vec"
        audio_model_name="facebook/wav2vec2-base-960h",
        unfreeze_audio_layers=0,
        emotion2vec_upstream_dir="/home/rml/Documents/pythontest/emotion2vec/upstream",
        use_cross_modal_projection=False,
        cross_modal_dim=256,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.modality = modality
        self.task_type = task_type
        self.audio_encoder_type = audio_encoder_type

        # Output dimension
        self.output_dim = vad_output_dim if task_type == "regression" else num_classes

        # Audio encoder (replaces preextracted features)
        if audio_encoder_type == "wav2vec2" and modality in ["audio", "both"]:
            self.audio_encoder = Wav2Vec2Encoder(
                model_name=audio_model_name,
                unfreeze_layers=unfreeze_audio_layers,
            )
            self.audio_dim = self.audio_encoder.get_output_dim()
        elif audio_encoder_type == "emotion2vec" and modality in ["audio", "both"]:
            self.audio_encoder = Emotion2VecEncoder(
                model_name=audio_model_name,
                unfreeze_layers=unfreeze_audio_layers,
            )
            self.audio_dim = self.audio_encoder.get_output_dim()
        else:
            self.audio_encoder = None

        # Text encoder
        if modality in ["text", "both"]:
            self.text_encoder = FrozenBERTEncoder(
                model_name=text_model_name,
                unfreeze_layers=unfreeze_bert_layers,
            )
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

        # Projection head for contrastive learning (separate from classifier)
        # Maps embeddings to a lower-dim space where contrastive loss operates
        # Classification head still uses the raw embeddings
        self.use_projection_head = use_projection_head
        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim, projection_hidden_dim),
                nn.BatchNorm1d(projection_hidden_dim),
                nn.ReLU(),
                nn.Linear(projection_hidden_dim, projection_dim),
            )
            print(f"   Projection head: {hidden_dim} -> {projection_hidden_dim} -> {projection_dim}")

        # Cross-modal projection: learned mapping to shared space for agreement signal
        if use_cross_modal_projection and modality == "both" and use_projection_head:
            self.audio_cross_proj = nn.Linear(audio_dim, cross_modal_dim)
            self.text_cross_proj = nn.Linear(768, cross_modal_dim)  # BERT output is always 768
            print(f"   Cross-modal projection: {audio_dim}/{768} -> {cross_modal_dim}")

    def _build_audio_only(self):
        """Audio-only: [batch, audio_dim] -> [batch, 1024] -> [batch, output_dim]"""
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
        """Text-only: [batch, 768] -> [batch, 1024] -> [batch, output_dim]"""
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
        """Multimodal: fusion -> [batch, fusion_hidden_dim] -> [batch, 1024] -> [batch, output_dim]"""
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

    def forward(self, audio_features=None, text_input_ids=None, text_attention_mask=None,
                audio_waveforms=None, audio_attention_mask=None, return_embeddings=False,
                detach_modal_features=True):
        """
        Forward pass with optional embedding extraction

        Args:
            audio_features: [batch_size, audio_dim] - preextracted audio features
            text_input_ids: [batch_size, seq_len] - text token IDs
            text_attention_mask: [batch_size, seq_len] - text attention mask
            audio_waveforms: [batch_size, num_samples] - raw waveforms (wav2vec2 mode)
            audio_attention_mask: [batch_size, num_samples] - waveform attention mask
            return_embeddings: bool - if True, return (logits, embeddings)

        Returns:
            logits: [batch_size, output_dim]
            embeddings: [batch_size, hidden_dim] - only if return_embeddings=True
        """
        # If wav2vec2 mode, encode waveforms to get audio_features
        if self.audio_encoder is not None and audio_waveforms is not None:
            audio_features = self.audio_encoder(audio_waveforms, audio_attention_mask)

        if self.modality == "audio":
            return self._forward_audio(audio_features, return_embeddings)
        elif self.modality == "text":
            return self._forward_text(text_input_ids, text_attention_mask, return_embeddings)
        elif self.modality == "both":
            return self._forward_multimodal(audio_features, text_input_ids, text_attention_mask,
                                            return_embeddings, detach_modal_features=detach_modal_features)

    def _project(self, embeddings):
        """Project embeddings through projection head for contrastive learning"""
        if self.use_projection_head:
            return self.projection_head(embeddings)
        return embeddings

    def _forward_audio(self, audio_features, return_embeddings=False):
        """Audio-only forward pass"""
        if audio_features is None:
            raise ValueError("audio_features required for audio mode")

        embeddings = self.embedding_layer(audio_features)  # [batch, 1024]
        logits = self.output_layer(embeddings)  # [batch, output_dim]

        if return_embeddings:
            projected = self._project(embeddings)
            return logits, projected, embeddings, None
        return logits

    def _forward_text(self, text_input_ids, text_attention_mask, return_embeddings=False):
        """Text-only forward pass"""
        if text_input_ids is None or text_attention_mask is None:
            raise ValueError("text inputs required for text mode")

        text_features = self.text_encoder(text_input_ids, text_attention_mask)  # [batch, 768]
        embeddings = self.embedding_layer(text_features)  # [batch, 1024]
        logits = self.output_layer(embeddings)  # [batch, output_dim]

        if return_embeddings:
            projected = self._project(embeddings)
            return logits, projected, embeddings, None
        return logits

    def _forward_multimodal(self, audio_features, text_input_ids, text_attention_mask,
                             return_embeddings=False, detach_modal_features=True):
        """Multimodal forward pass"""
        if audio_features is None:
            raise ValueError("audio_features required for multimodal mode")
        if text_input_ids is None or text_attention_mask is None:
            raise ValueError("text inputs required for multimodal mode")

        text_features = self.text_encoder(text_input_ids, text_attention_mask)  # [batch, 768]
        fused_features = self.fusion_module(audio_features, text_features)  # [batch, fusion_hidden_dim]
        embeddings = self.embedding_layer(fused_features)  # [batch, 1024]
        logits = self.output_layer(embeddings)  # [batch, output_dim]

        if return_embeddings:
            projected = self._project(embeddings)
            if hasattr(self, 'audio_cross_proj'):
                # Learned cross-modal projection: detach source features so only
                # the projection layers get gradients from cross-modal alignment loss
                modal_features = {
                    'audio': self.audio_cross_proj(audio_features.detach()),  # [B, cross_modal_dim]
                    'text': self.text_cross_proj(text_features.detach()),     # [B, cross_modal_dim]
                }
            elif detach_modal_features:
                modal_features = {
                    'audio': audio_features.detach(),  # [B, 768]
                    'text': text_features.detach(),     # [B, 768]
                }
            else:
                # Keep gradients flowing back into encoders (for domain adversarial training)
                modal_features = {
                    'audio': audio_features,  # [B, 768]
                    'text': text_features,     # [B, 768]
                }
            return logits, projected, embeddings, modal_features
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
        vad_output_dim=cfg.get('vad_output_dim', 3),
        projection_dim=cfg.get('projection_dim', 128),
        projection_hidden_dim=cfg.get('projection_hidden_dim', 512),
        use_projection_head=cfg.get('use_contrastive', False),
        unfreeze_bert_layers=cfg.get('unfreeze_bert_layers', 0),
        audio_encoder_type=cfg.get('audio_encoder_type', 'preextracted'),
        audio_model_name=cfg.get('audio_model_name', 'facebook/wav2vec2-base-960h'),
        unfreeze_audio_layers=cfg.get('unfreeze_audio_layers', 0),
        emotion2vec_upstream_dir=cfg.get(
            'emotion2vec_upstream_dir',
            '/home/rml/Documents/pythontest/emotion2vec/upstream'
        ),
        use_cross_modal_projection=cfg.get('use_cross_modal_projection', False),
        cross_modal_dim=cfg.get('cross_modal_dim', 256),
    )
