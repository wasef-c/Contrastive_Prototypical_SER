#!/usr/bin/env python3
"""
Audio encoders: Wav2Vec2 and Emotion2Vec with optional partial unfreezing.
Both mirror the FrozenBERTEncoder pattern from models/encoder.py.
"""

from dataclasses import dataclass
from typing import List, Optional

import fairseq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor


@dataclass
class _UserDirModule:
    """Minimal dataclass required by fairseq.utils.import_user_module."""

    user_dir: str


class Wav2Vec2Encoder(nn.Module):
    """Wav2Vec2 encoder with optional partial unfreezing of top transformer layers"""

    def __init__(self, model_name="facebook/wav2vec2-base-960h", unfreeze_layers=0):
        super().__init__()

        self.model_name = model_name
        self.unfreeze_layers = unfreeze_layers

        print(f"Loading Wav2Vec2 model: {model_name}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        self.output_dim = self.wav2vec2.config.hidden_size  # 768 for base

        # Freeze all parameters first
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        # Unfreeze top N encoder layers if requested
        if unfreeze_layers > 0:
            encoder_layers = self.wav2vec2.encoder.layers
            total_layers = len(encoder_layers)
            unfreeze_from = total_layers - unfreeze_layers
            for i in range(unfreeze_from, total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True

            # Enable gradient checkpointing to save activation memory
            self.wav2vec2.gradient_checkpointing_enable()
            trainable = sum(p.numel() for p in self.wav2vec2.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.wav2vec2.parameters())
            print(f"  {model_name}: unfroze top {unfreeze_layers}/{total_layers} layers ({trainable:,}/{total:,} params trainable)")
            print(f"  Gradient checkpointing enabled")
        else:
            self.wav2vec2.eval()
            print(f"  {model_name} loaded and frozen (output_dim={self.output_dim})")

    def forward(self, waveforms, attention_mask=None):
        """
        Forward pass: raw waveforms -> mean-pooled features

        Args:
            waveforms: [batch_size, num_samples] raw audio at 16kHz
            attention_mask: [batch_size, num_samples] 1=real, 0=padding

        Returns:
            features: [batch_size, output_dim] mean-pooled representation
        """
        if self.unfreeze_layers == 0:
            self.wav2vec2.eval()
            with torch.no_grad():
                outputs = self.wav2vec2(
                    input_values=waveforms,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                hidden_states = outputs.last_hidden_state  # [B, T, 768]
        else:
            outputs = self.wav2vec2(
                input_values=waveforms,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state  # [B, T, 768]

        # Mean pool over time dimension
        # If attention_mask provided, only pool over non-padded frames
        if attention_mask is not None:
            # Wav2Vec2 downsamples the input, so we need the output lengths
            # Use the model's internal length computation
            output_lengths = self.wav2vec2._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1).long()
            )
            # Create mask for the hidden state time dimension
            max_T = hidden_states.size(1)
            frame_mask = torch.arange(max_T, device=hidden_states.device).unsqueeze(0) < output_lengths.unsqueeze(1)
            frame_mask = frame_mask.unsqueeze(-1).float()  # [B, T, 1]
            pooled = (hidden_states * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)  # [B, 768]

        return pooled

    def process_waveforms(self, waveforms, sampling_rate=16000):
        """
        Use processor for input normalization (feature extraction).

        Args:
            waveforms: list of numpy arrays or single numpy array
            sampling_rate: int, audio sampling rate

        Returns:
            input_values: normalized tensor ready for model
        """
        inputs = self.processor(
            waveforms,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        return inputs.input_values

    def get_audio_params(self):
        """Return only the unfrozen Wav2Vec2 parameters (for differential LR)"""
        return [p for p in self.wav2vec2.parameters() if p.requires_grad]

    def get_output_dim(self):
        """Get output feature dimension"""
        return self.output_dim


class Emotion2VecEncoder(nn.Module):
    """
    Finetunable Emotion2Vec encoder (data2vec_multi architecture).

    Loads emotion2vec_base.pt via fairseq and provides the same interface
    as Wav2Vec2Encoder: raw waveforms in, mean-pooled [B, 768] out.

    Supports optional unfreezing of the top N main transformer blocks
    (model.blocks) with the same freeze-all-then-unfreeze-top-N pattern.
    """

    def __init__(
        self,
        checkpoint_path: str,
        upstream_dir: str,
        unfreeze_layers: int = 0,
    ):
        """
        Args:
            checkpoint_path: Path to emotion2vec_base.pt checkpoint file.
            upstream_dir: Path to the emotion2vec upstream models directory
                          (contains models/emotion2vec.py, etc.).
            unfreeze_layers: Number of top transformer blocks to unfreeze.
                             0 = fully frozen. Targets model.blocks[-N:].
        """
        super().__init__()

        self.unfreeze_layers = unfreeze_layers

        # Register the upstream model code with fairseq so it can
        # find and instantiate the data2vec_multi architecture.
        _mod = _UserDirModule(upstream_dir)
        fairseq.utils.import_user_module(_mod)

        print(f"Loading Emotion2Vec model: {checkpoint_path}")
        models, _cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = models[0]

        # task.cfg.normalize controls per-sample layer-norm preprocessing
        self.normalize: bool = task.cfg.normalize
        self.output_dim: int = 768

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        if unfreeze_layers > 0:
            total_blocks = len(self.model.blocks)
            for blk in self.model.blocks[-unfreeze_layers:]:
                for param in blk.parameters():
                    param.requires_grad = True
            # Unfreeze final norm layer if it exists (layer_norm_first=True configs)
            if self.model.norm is not None:
                for param in self.model.norm.parameters():
                    param.requires_grad = True

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(
                f"  emotion2vec: unfroze top {unfreeze_layers}/{total_blocks} blocks "
                f"({trainable:,}/{total:,} params trainable)"
            )
        else:
            self.model.eval()
            print(f"  emotion2vec loaded and frozen (output_dim={self.output_dim})")

    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: raw waveforms -> mean-pooled features.

        Args:
            waveforms: [batch_size, num_samples] raw audio at 16kHz.
            attention_mask: [batch_size, num_samples] 1=real audio, 0=padding.

        Returns:
            features: [batch_size, 768] mean-pooled representation.
        """
        # Per-sample layer-norm preprocessing (matches original feature extraction)
        if self.normalize:
            waveforms = F.layer_norm(waveforms, waveforms.shape[1:])

        # Convert attention_mask (1=real) to fairseq padding_mask (True=pad)
        padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0

        if self.unfreeze_layers == 0:
            self.model.eval()
            with torch.no_grad():
                return self._extract(waveforms, padding_mask)
        return self._extract(waveforms, padding_mask)

    def _extract(
        self,
        waveforms: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run extract_features and mean-pool the frame-level output."""
        res = self.model.extract_features(
            waveforms,
            padding_mask=padding_mask,
            mask=False,
            remove_extra_tokens=True,
        )
        x = res["x"]                      # [B, T', 768]
        frame_pm = res["padding_mask"]    # [B, T'] True=pad, or None

        if frame_pm is not None and frame_pm.any():
            real_mask = (~frame_pm).unsqueeze(-1).float()   # [B, T', 1]
            return (x * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)
        return x.mean(dim=1)

    def get_audio_params(self) -> List[torch.nn.Parameter]:
        """Return only the unfrozen parameters (for differential LR optimizer)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim
