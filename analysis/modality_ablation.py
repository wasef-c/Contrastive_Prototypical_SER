"""Separate the audio and text contributions of a trained fusion model.

The fusion is symmetric cross-attention: audio attends to text, text attends to
audio, and the two results are concatenated and projected. Neither branch has
its own classifier head, so a per-modality prediction has to be obtained by
intervention rather than by reading something off.

The intervention substitutes one branch's feature vector with its dataset MEAN
at the fusion input, leaving the other branch real. The mean is used rather
than zeros so the trained fusion sees an in-distribution but uninformative
input; zeros are far outside the range it was trained on and would measure
extrapolation rather than reliance.

Writes one npz per arm/seed/corpus with three logit sets: full, audio-only
(text replaced by its mean) and text-only (audio replaced by its mean).

Usage:
    python -m analysis.modality_ablation --arm uf4_whitevad --seeds 42
    python -m analysis.modality_ablation --arm uf4_whitevad --corpus IEMO
"""

import argparse
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.collate import vad_collate_fn
from data.dataset import EmotionDataset
from models.classifier import create_model
from train import _prepare_model_inputs
from utils.config import Config

OUT_DIR = pathlib.Path("analysis/modality_ablation")
WEIGHTS = pathlib.Path("saved_models")


def load_arm(arm: str, seed: int, device: torch.device) -> Tuple[torch.nn.Module, Config]:
    """Rebuild a trained model from its saved weights and embedded config.

    Args:
        arm: arm name without the seed suffix.
        seed: run seed.
        device: target device.

    Returns:
        Tuple of the loaded model in eval mode and its Config.

    Raises:
        FileNotFoundError: when no weight file matches.
    """
    cands = sorted(WEIGHTS.glob(f"{arm}_seed{seed}*.pt"))
    if not cands:
        raise FileNotFoundError(f"no weights for {arm} seed {seed} in {WEIGHTS}")
    blob = torch.load(cands[0], map_location="cpu", weights_only=False)
    config = Config(**blob["config"])
    model = create_model(config).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model, config


def collect_features(model: torch.nn.Module, loader: DataLoader, config: Config,
                     device: torch.device) -> Dict[str, np.ndarray]:
    """Run the encoders once and keep both branches' fusion inputs.

    Args:
        model: loaded multimodal model.
        loader: unshuffled dataloader over one corpus.
        config: the model's config.
        device: target device.

    Returns:
        Dict with audio [N, D], text [N, D] and labels [N].
    """
    audio, text, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = _prepare_model_inputs(batch, config, model, device)
            feats = inputs.get("audio_features")
            if feats is None:
                feats = model.audio_encoder(inputs["audio_waveforms"],
                                            inputs.get("audio_attention_mask"))
            feats = model._pool_frames(feats)
            t = model.text_encoder(inputs["text_input_ids"],
                                   inputs["text_attention_mask"])
            audio.append(feats.detach().cpu().float().numpy())
            text.append(t.detach().cpu().float().numpy())
            labels.append(batch["label"].numpy())
    return {
        "audio": np.concatenate(audio, axis=0),
        "text": np.concatenate(text, axis=0),
        "labels": np.concatenate(labels, axis=0),
    }


def logits_from(model: torch.nn.Module, audio: np.ndarray, text: np.ndarray,
                device: torch.device, chunk: int = 256) -> np.ndarray:
    """Push a pair of branch features through fusion, embedding and classifier.

    Args:
        model: loaded model.
        audio: [N, D] audio branch features.
        text: [N, D] text branch features.
        device: target device.
        chunk: rows per forward chunk.

    Returns:
        [N, C] logits.
    """
    out = []
    with torch.no_grad():
        for i in range(0, len(audio), chunk):
            a = torch.from_numpy(audio[i:i + chunk]).to(device)
            t = torch.from_numpy(text[i:i + chunk]).to(device)
            fused = model.fusion_module(a, t)
            emb = model.embedding_layer(fused)
            out.append(model.classify_from_embeddings(emb).detach().cpu().float().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    """Run the ablation for one arm across seeds and corpora."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--corpus", nargs="+", default=["IEMO"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        model, config = load_arm(args.arm, seed, device)
        for corpus in args.corpus:
            ds = EmotionDataset(corpus, split="train", config=config)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=vad_collate_fn, num_workers=2,
                                pin_memory=False)
            feats = collect_features(model, loader, config, device)
            a, t = feats["audio"], feats["text"]
            a_mean = np.repeat(a.mean(axis=0, keepdims=True), len(a), axis=0)
            t_mean = np.repeat(t.mean(axis=0, keepdims=True), len(t), axis=0)

            full = logits_from(model, a, t, device)
            audio_only = logits_from(model, a, t_mean, device)
            text_only = logits_from(model, a_mean, t, device)

            dest = OUT_DIR / f"{args.arm}_seed{seed}_{corpus}.npz"
            np.savez_compressed(
                dest,
                # Branch features kept so the INFORMATION in each
                # representation can be probed separately from how much the
                # fusion happens to rely on it. A branch can be informative
                # and under-used, which is a different problem from a branch
                # that is uninformative.
                audio_feat=a.astype(np.float32),
                text_feat=t.astype(np.float32),
                full=full.astype(np.float32),
                audio_only=audio_only.astype(np.float32),
                text_only=text_only.astype(np.float32),
                labels=feats["labels"].astype(np.int64),
                arm=np.array(args.arm), seed=np.array(seed),
                corpus=np.array(corpus))
            agree = (full.argmax(1) == np.load(
                f"checkpoints/{args.arm}_seed{seed}_seed{seed}/predictions/{corpus}.npz",
                allow_pickle=True)["logits"].argmax(1)).mean() if pathlib.Path(
                f"checkpoints/{args.arm}_seed{seed}_seed{seed}/predictions/{corpus}.npz"
            ).exists() else float("nan")
            print(f"  {args.arm} seed {seed} {corpus}: wrote {dest.name} "
                  f"(N={len(full)}, reproduces dumped logits on "
                  f"{100 * agree:.1f}% of rows)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
