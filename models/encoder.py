#!/usr/bin/env python3
"""
Text encoder module using frozen BERT for text feature extraction
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import warnings


class FrozenBERTEncoder(nn.Module):
    """BERT encoder with optional partial unfreezing of top layers"""

    def __init__(self, model_name="bert-base-uncased", output_dim=768, unfreeze_layers=0):
        super().__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.unfreeze_layers = unfreeze_layers

        # Load pretrained BERT model and tokenizer
        print(f"📚 Loading BERT model: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Auto-detect output dim from model config
        self.output_dim = self.bert.config.hidden_size

        # Find the encoder layers (works for BERT, RoBERTa, DeBERTa, etc.)
        if hasattr(self.bert, 'encoder') and hasattr(self.bert.encoder, 'layer'):
            self.encoder_layers = self.bert.encoder.layer
        elif hasattr(self.bert, 'encoder') and hasattr(self.bert.encoder, 'layers'):
            self.encoder_layers = self.bert.encoder.layers
        else:
            raise ValueError(f"Cannot find encoder layers for model: {model_name}")

        # Freeze all parameters first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze top N layers if requested
        if unfreeze_layers > 0:
            total_layers = len(self.encoder_layers)
            unfreeze_from = total_layers - unfreeze_layers
            for i in range(unfreeze_from, total_layers):
                for param in self.encoder_layers[i].parameters():
                    param.requires_grad = True
            # Also unfreeze pooler if it exists
            if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
            # Enable gradient checkpointing to save activation memory
            self.bert.gradient_checkpointing_enable()
            trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.bert.parameters())
            print(f"🔓 {model_name}: unfroze top {unfreeze_layers}/{total_layers} layers ({trainable:,}/{total:,} params trainable)")
            print(f"  Gradient checkpointing enabled")
        else:
            self.bert.eval()  # Set to eval mode permanently when fully frozen
            print(f"✅ {model_name} loaded and frozen (output_dim={self.output_dim})")

    def get_bert_params(self):
        """Return only the unfrozen BERT parameters (for differential LR)"""
        return [p for p in self.bert.parameters() if p.requires_grad]

    def forward(self, input_ids, attention_mask):
        """
        Extract features from text using BERT

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            features: [batch_size, output_dim] CLS token representation
        """
        if self.unfreeze_layers == 0:
            # Fully frozen — no grad, eval mode
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                cls_features = outputs.last_hidden_state[:, 0, :]
        else:
            # Partially unfrozen — need gradients
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            cls_features = outputs.last_hidden_state[:, 0, :]

        return cls_features

    def tokenize_batch(self, texts, max_length=128, device='cpu'):
        """
        Tokenize a batch of text strings

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            device: Device to place tensors on

        Returns:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        return input_ids, attention_mask

    def encode_texts(self, texts, max_length=128, device='cpu'):
        """
        Complete pipeline: tokenize + encode

        Args:
            texts: List of text strings or single string
            max_length: Maximum sequence length
            device: Device to place tensors on

        Returns:
            features: [batch_size, output_dim]
        """
        input_ids, attention_mask = self.tokenize_batch(texts, max_length, device)
        features = self.forward(input_ids, attention_mask)
        return features

    def get_output_dim(self):
        """Get output feature dimension"""
        return self.output_dim


class TextFeatureExtractor:
    """
    Utility class for extracting and caching text features
    Useful for pre-extracting features to avoid repeated computation
    """

    def __init__(self, model_name="bert-base-uncased", device='cuda'):
        self.encoder = FrozenBERTEncoder(model_name).to(device)
        self.device = device
        self.feature_cache = {}

    def extract_features(self, texts, max_length=128, use_cache=True):
        """
        Extract features with optional caching

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            use_cache: Whether to use caching

        Returns:
            features: [batch_size, output_dim]
        """
        if use_cache:
            # Check cache
            cache_keys = [text[:100] for text in texts]  # Use first 100 chars as key
            cached_features = []
            uncached_indices = []
            uncached_texts = []

            for i, (text, key) in enumerate(zip(texts, cache_keys)):
                if key in self.feature_cache:
                    cached_features.append((i, self.feature_cache[key]))
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

            # Extract uncached features
            if uncached_texts:
                new_features = self.encoder.encode_texts(
                    uncached_texts,
                    max_length=max_length,
                    device=self.device
                )

                # Update cache
                for text, feat in zip(uncached_texts, new_features):
                    key = text[:100]
                    self.feature_cache[key] = feat.cpu()

            # Combine cached and new features in original order
            all_features = [None] * len(texts)

            # Place cached features
            for i, feat in cached_features:
                all_features[i] = feat.to(self.device)

            # Place new features
            if uncached_texts:
                for i, feat in zip(uncached_indices, new_features):
                    all_features[i] = feat

            return torch.stack(all_features)
        else:
            # Direct extraction without cache
            return self.encoder.encode_texts(texts, max_length=max_length, device=self.device)

    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
        print("🧹 Text feature cache cleared")

    def cache_size(self):
        """Get number of cached features"""
        return len(self.feature_cache)


if __name__ == "__main__":
    # Test the text encoder
    print("Testing FrozenBERTEncoder...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test basic encoding
    encoder = FrozenBERTEncoder().to(device)

    test_texts = [
        "I am very happy today!",
        "This makes me so angry.",
        "I feel sad and disappointed.",
        "Everything is neutral and calm."
    ]

    print("\nTest texts:")
    for i, text in enumerate(test_texts):
        print(f"  {i}: {text}")

    # Extract features
    features = encoder.encode_texts(test_texts, device=device)
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Feature statistics:")
    print(f"  Mean: {features.mean().item():.4f}")
    print(f"  Std: {features.std().item():.4f}")
    print(f"  Min: {features.min().item():.4f}")
    print(f"  Max: {features.max().item():.4f}")

    # Test feature extractor with cache
    print("\n" + "="*50)
    print("Testing TextFeatureExtractor with caching...")
    extractor = TextFeatureExtractor(device=device)

    # First extraction
    features1 = extractor.extract_features(test_texts, use_cache=True)
    print(f"First extraction - Cache size: {extractor.cache_size()}")

    # Second extraction (should use cache)
    features2 = extractor.extract_features(test_texts, use_cache=True)
    print(f"Second extraction - Cache size: {extractor.cache_size()}")

    # Verify features are identical
    print(f"Features identical: {torch.allclose(features1, features2)}")

    print("\n✅ Text encoder tests passed!")
