#!/usr/bin/env python3
"""
Wav2Vec2 audio encoder with optional partial unfreezing of top layers.
Mirrors the FrozenBERTEncoder pattern from models/encoder.py.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


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

            trainable = sum(p.numel() for p in self.wav2vec2.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.wav2vec2.parameters())
            print(f"  {model_name}: unfroze top {unfreeze_layers}/{total_layers} layers ({trainable:,}/{total:,} params trainable)")
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
