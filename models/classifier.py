#!/usr/bin/env python3
"""
Clean emotion recognition classifier with embedding extraction for contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import FrozenBERTEncoder
from models.audio_encoder import Wav2Vec2Encoder, Emotion2VecEncoder
from models.attention_pool import AttentionPool, MeanPoolControl
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
        audio_pooling="mean",
        num_attn_heads_pool=4,
        attn_pool_hidden=256,
        use_mean_pool_control=False,
        emotion2vec_upstream_dir="/home/rml/Documents/pythontest/emotion2vec/upstream",
        use_cross_modal_projection=False,
        cross_modal_dim=256,
        use_proto_atyp_split=False,
        use_evidence_heads=False,
        use_hierarchical_head=False,
        use_latent_prototypes=False,
        prototypes_per_class=2,
        use_aux_vad_cluster=False,
        use_salience_gate=False,
        aux_vad_cluster_k=8,
        aux_vad_head_depth=1,
        aux_vad_task="cluster",
        aux_vad_cluster_scope="global",
        aux_vad_clusters_per_class=2,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.modality = modality
        self.task_type = task_type
        self.audio_encoder_type = audio_encoder_type
        self.use_proto_atyp_split = use_proto_atyp_split

        # Output dimension. When the proto/atyp split is on we double the
        # classification head so each class becomes (proto, atypical). Eval
        # collapses back to num_classes via softmax pair sum. Evidence heads
        # instead drop neutral entirely: one sigmoid head per emotional
        # class, and neutral is the state where no head fires.
        _head_flags = [use_proto_atyp_split, use_evidence_heads,
                       use_hierarchical_head, use_latent_prototypes]
        if sum(bool(f) for f in _head_flags) > 1:
            raise ValueError(
                "use_proto_atyp_split, use_evidence_heads and "
                "use_hierarchical_head all reinterpret the output head; "
                "enable at most one."
            )
        if task_type == "regression":
            self.output_dim = vad_output_dim
        elif use_proto_atyp_split:
            self.output_dim = 2 * num_classes
        elif use_latent_prototypes:
            # K prototypes per class. Only the parent class is supervised;
            # which prototype a sample uses is latent.
            self.output_dim = num_classes * int(prototypes_per_class)
            print(f"   Latent prototypes: {prototypes_per_class} per class "
                  f"-> {self.output_dim} logits, no sub-label supervision")
        elif use_evidence_heads:
            self.output_dim = num_classes - 1
            print(f"   Evidence heads: {num_classes - 1} sigmoid outputs, "
                  f"neutral = nothing fires")
        else:
            self.output_dim = num_classes
        self.use_evidence_heads = use_evidence_heads
        self.use_latent_prototypes = use_latent_prototypes
        self.prototypes_per_class = int(prototypes_per_class)
        # Hierarchical keeps num_classes outputs: column 0 is the
        # emotion-presence logit, the rest are emotion logits. Only the
        # interpretation changes, so no shape override is needed.
        self.use_hierarchical_head = use_hierarchical_head
        if use_hierarchical_head:
            print(f"   Hierarchical head: 1 presence logit + "
                  f"{num_classes - 1} emotion logits, no threshold")

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
                pooling=audio_pooling,
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

        # Frame-level input needs a module that collapses time before
        # fusion. AttentionPool learns which frames matter; MeanPoolControl
        # is its capacity-matched placebo, producing the same width from
        # the plain mean. Which one is active decides whether a gain can be
        # attributed to attending over time or merely to a wider layer.
        self.frame_pool = None
        if audio_pooling == "frames" and modality in ("audio", "both"):
            pool_cls = MeanPoolControl if use_mean_pool_control else AttentionPool
            self.frame_pool = pool_cls(
                input_dim=768,
                num_heads=int(num_attn_heads_pool),
                hidden_dim=int(attn_pool_hidden),
                dropout=dropout,
            )
            self.audio_dim = self.frame_pool.output_dim
            kind = "mean-pool CONTROL" if use_mean_pool_control else "attention pool"
            print(f"   Frame pooling: {kind}, {num_attn_heads_pool} heads "
                  f"-> audio_dim {self.audio_dim}")

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
                nn.LayerNorm(projection_hidden_dim),
                nn.ReLU(),
                nn.Linear(projection_hidden_dim, projection_dim),
            )
            print(f"   Projection head: {hidden_dim} -> {projection_hidden_dim} -> {projection_dim}")

        # Cross-modal projection: learned mapping to shared space for agreement signal
        if use_cross_modal_projection and modality == "both" and use_projection_head:
            self.audio_cross_proj = nn.Linear(audio_dim, cross_modal_dim)
            self.text_cross_proj = nn.Linear(768, cross_modal_dim)  # BERT output is always 768
            print(f"   Cross-modal projection: {audio_dim}/{768} -> {cross_modal_dim}")

        # Auxiliary VAD-cluster head: predicts which VAD k-means cluster the
        # sample belongs to. Small linear or 2-layer MLP on top of the shared
        # embedding. Consumed only during training; eval ignores this head.
        self.use_aux_vad_cluster = use_aux_vad_cluster
        self.aux_vad_task = aux_vad_task

        # Salience gate: predicted VAD becomes an explicit intensity signal
        # on the neutral logit. Requires the regression aux head, since the
        # cluster variant does not produce a VAD vector.
        self.use_salience_gate = bool(use_salience_gate) and use_aux_vad_cluster
        if self.use_salience_gate:
            if aux_vad_task != "regression":
                raise ValueError("use_salience_gate requires aux_vad_task='regression'")
            self.register_buffer("salience_vad_mean", torch.zeros(3))
            self.register_buffer("salience_vad_scale", torch.ones(3))
            self.register_buffer("salience_neutral_centre", torch.zeros(3))
            self.salience_gate_w = nn.Parameter(torch.zeros(1))
            self.salience_gate_b = nn.Parameter(torch.zeros(1))
            print("   Salience gate: predicted-VAD intensity -> neutral logit "
                  "(2 learned params, init 0)")
        if use_aux_vad_cluster:
            # Regression variant predicts (V, A, D) directly. Cluster variant
            # predicts one of k cluster IDs, where per-class scope gives
            # num_classes * clusters_per_class subtype labels.
            if aux_vad_task == "regression":
                k_out = 3
            elif aux_vad_cluster_scope == "per_class":
                k_out = int(num_classes) * int(aux_vad_clusters_per_class)
            else:
                k_out = int(aux_vad_cluster_k)
            depth = int(aux_vad_head_depth)
            if depth <= 1:
                self.aux_vad_head = nn.Linear(hidden_dim, k_out)
                print(f"   Aux VAD {aux_vad_task} head: {hidden_dim} -> {k_out}")
            else:
                mid = max(k_out, hidden_dim // 2)
                self.aux_vad_head = nn.Sequential(
                    nn.Linear(hidden_dim, mid),
                    nn.ReLU(),
                    nn.Linear(mid, k_out),
                )
                print(f"   Aux VAD {aux_vad_task} head: {hidden_dim} -> {mid} -> {k_out}")

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
            if not self.use_salience_gate:
                return self._forward_multimodal(audio_features, text_input_ids, text_attention_mask,
                                                return_embeddings, detach_modal_features=detach_modal_features)
            # The gate needs the shared embedding, so always request it and
            # drop it again when the caller did not ask for it.
            logits, projected, embeddings, modal_features = self._forward_multimodal(
                audio_features, text_input_ids, text_attention_mask,
                True, detach_modal_features=detach_modal_features)
            logits = self._apply_salience_gate(logits, embeddings)
            if return_embeddings:
                return logits, projected, embeddings, modal_features
            return logits

    def _project(self, embeddings):
        """Project embeddings through projection head for contrastive learning"""
        if self.use_projection_head:
            return self.projection_head(embeddings)
        return embeddings

    def set_salience_reference(self, mean, scale, neutral_centre):
        """Install the VAD standardisation and neutral centroid for the gate.

        These come from the training split only and are constants, not
        learned, so the gate's learned parameters stay interpretable: the
        only freedom is how strongly predicted intensity shifts the
        neutral logit.

        Args:
            mean: [3] per-dimension VAD mean of the training split.
            scale: [3] per-dimension VAD standard deviation.
            neutral_centre: [3] neutral class centroid in standardised space.
        """
        device = self.salience_vad_mean.device
        self.salience_vad_mean.copy_(torch.as_tensor(mean, device=device))
        self.salience_vad_scale.copy_(torch.as_tensor(scale, device=device))
        self.salience_neutral_centre.copy_(
            torch.as_tensor(neutral_centre, device=device))

    def _apply_salience_gate(self, logits, embeddings):
        """Shift the neutral logit by a learned affine function of intensity.

        Predicted VAD gives a label-free emotional intensity, the distance
        from the neutral centroid. Low intensity is evidence for neutral
        and high intensity is evidence against it, and this lets the model
        use that evidence explicitly rather than having to rediscover it.

        Only the neutral logit is touched, because the error analysis found
        predicted intensity informative for neutral and angry but redundant
        with confidence for happy and sad. Two parameters, so capacity is
        essentially unchanged against the plain auxiliary arm, and a learned
        weight near zero is the model reporting the signal is not useful.

        Args:
            logits: [B, num_classes] classification logits.
            embeddings: [B, hidden_dim] shared embedding.

        Returns:
            [B, num_classes] logits with the neutral column adjusted.
        """
        vad_pred = self.aux_vad_forward(embeddings)
        if vad_pred is None or vad_pred.size(-1) != 3:
            return logits
        z = (vad_pred - self.salience_vad_mean) / self.salience_vad_scale
        intensity = torch.linalg.norm(z - self.salience_neutral_centre, dim=1)
        adjust = torch.zeros_like(logits)
        adjust[:, 0] = self.salience_gate_w * intensity + self.salience_gate_b
        return logits + adjust

    def aux_vad_forward(self, embeddings):
        """Auxiliary VAD head output from the shared embedding.

        Args:
            embeddings: [B, hidden_dim] fused embedding tensor.

        Returns:
            [B, aux_vad_cluster_k] cluster logits, or [B, 3] VAD predictions
            when aux_vad_task is "regression". None if aux head disabled.
        """
        if not self.use_aux_vad_cluster:
            return None
        return self.aux_vad_head(embeddings)

    def _forward_audio(self, audio_features, return_embeddings=False):
        """Audio-only forward pass"""
        if audio_features is None:
            raise ValueError("audio_features required for audio mode")

        audio_features = self._pool_frames(audio_features)
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

    def _pool_frames(self, audio_features):
        """Collapse [B, T, D] frame input to [B, D'] before fusion.

        A no-op when the input is already pooled, so the same forward path
        serves every pooling mode. Padding is inferred from all-zero rows,
        which the encoder guarantees for frames mode.

        Args:
            audio_features: [B, T, D] frames or [B, D] pooled features.

        Returns:
            [B, D'] pooled features.
        """
        if self.frame_pool is None or audio_features.dim() != 3:
            return audio_features
        mask = (audio_features.abs().sum(dim=-1) > 0)
        return self.frame_pool(audio_features, mask)

    def _forward_multimodal(self, audio_features, text_input_ids, text_attention_mask,
                             return_embeddings=False, detach_modal_features=True):
        """Multimodal forward pass"""
        if audio_features is None:
            raise ValueError("audio_features required for multimodal mode")
        if text_input_ids is None or text_attention_mask is None:
            raise ValueError("text inputs required for multimodal mode")

        audio_features = self._pool_frames(audio_features)
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
        audio_pooling=cfg.get('audio_pooling', 'mean'),
        num_attn_heads_pool=cfg.get('num_attn_heads_pool', 4),
        attn_pool_hidden=cfg.get('attn_pool_hidden', 256),
        use_mean_pool_control=cfg.get('use_mean_pool_control', False),
        emotion2vec_upstream_dir=cfg.get(
            'emotion2vec_upstream_dir',
            '/home/rml/Documents/pythontest/emotion2vec/upstream'
        ),
        use_cross_modal_projection=cfg.get('use_cross_modal_projection', False),
        cross_modal_dim=cfg.get('cross_modal_dim', 256),
        use_proto_atyp_split=cfg.get('use_proto_atyp_split', False),
        use_evidence_heads=cfg.get('use_evidence_heads', False),
        use_hierarchical_head=cfg.get('use_hierarchical_head', False),
        use_latent_prototypes=cfg.get('use_latent_prototypes', False),
        prototypes_per_class=cfg.get('prototypes_per_class', 2),
        use_aux_vad_cluster=cfg.get('use_aux_vad_cluster', False),
        use_salience_gate=cfg.get('use_salience_gate', False),
        aux_vad_cluster_k=cfg.get('aux_vad_cluster_k', 8),
        aux_vad_head_depth=cfg.get('aux_vad_head_depth', 1),
        aux_vad_task=cfg.get('aux_vad_task', 'cluster'),
        aux_vad_cluster_scope=cfg.get('aux_vad_cluster_scope', 'global'),
        aux_vad_clusters_per_class=cfg.get('aux_vad_clusters_per_class', 2),
    )
