#!/usr/bin/env python3
"""
Compare online Emotion2Vec (FunASR) features against precomputed features
stored in the HuggingFace dataset for a few samples.
"""
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from datasets import load_dataset
from funasr import AutoModel

DATASET = "cairocode/IEMO_Audio_Text_Merged"
N_SAMPLES = 5

print(f"Loading dataset: {DATASET}")
ds = load_dataset(DATASET, split="train", trust_remote_code=True)

print("Loading FunASR emotion2vec model...")
auto_model = AutoModel(model="iic/emotion2vec_base")
model = auto_model.model
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

normalize = model.cfg.normalize
print(f"  normalize={normalize}, device={device}")

cosine_sims = []
l2_dists = []

for i in range(N_SAMPLES):
    item = ds[i]

    # --- Precomputed feature ---
    precomp = np.array(item["emotion2vec_features"][0]["feats"], dtype=np.float32)
    precomp_t = torch.tensor(precomp)

    # --- Online feature ---
    audio = item["audio"]
    waveform = torch.tensor(audio["array"], dtype=torch.float32)
    if audio["sampling_rate"] != 16000:
        waveform = torchaudio.functional.resample(waveform, audio["sampling_rate"], 16000)

    waveform = waveform.to(device)
    if normalize:
        waveform = F.layer_norm(waveform, waveform.shape)
    waveform = waveform.unsqueeze(0)  # [1, T]

    with torch.no_grad():
        res = model.extract_features(waveform, padding_mask=None, mask=False, remove_extra_tokens=True)
        online = res["x"].squeeze(0).mean(dim=0).cpu()  # [768]

    cos = F.cosine_similarity(precomp_t.unsqueeze(0), online.unsqueeze(0)).item()
    l2 = (precomp_t - online).norm().item()
    cosine_sims.append(cos)
    l2_dists.append(l2)
    print(f"  Sample {i}: cosine_sim={cos:.6f}, L2={l2:.6f}")

print(f"\nMean cosine similarity: {np.mean(cosine_sims):.6f}")
print(f"Mean L2 distance:       {np.mean(l2_dists):.6f}")
print("\nIf cosine_sim ≈ 1.0 and L2 ≈ 0, features match precomputed exactly.")
