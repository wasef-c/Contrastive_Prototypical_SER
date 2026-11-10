#!/usr/bin/env python3
"""Select GenLabel density hyperparameters on the development split.

GenLabel's generative model has two choices this setting forces and the
paper does not cover: how many dimensions to fit in, and how hard to
regularise the covariance. Both are selected here on the MSP-Podcast
development split and never on any test corpus.

The selection criterion is how well the class-conditional densities
classify held-out embeddings, which is exactly the job the densities do
when they relabel a mixed point. Densities are fitted on the TRAINING
split only; the development split is scored.

Embeddings come from an already-trained checkpoint, by default the vanilla
mixup control, so the tuning does not look at the representation of the
method under test. The train/development split is reproduced exactly as
train.py builds it: class-stratified on label4 under RandomState(seed).
"""

import argparse
import io
import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from analysis.modality_ablation import load_arm
from analysis.vad_explained_split import uar
from data.collate import vad_collate_fn
from data.dataset import EmotionDataset
from train import _prepare_model_inputs
from utils.config import Config
from utils.genlabel import EmbeddingClassDensity

OUT_DIR = pathlib.Path("analysis/genlabel_tuning")


def reproduce_split(dataset: EmotionDataset, config: Config) -> Tuple[List[int], List[int]]:
    """Rebuild train.py's class-stratified train/validation split.

    Args:
        dataset: the training corpus dataset.
        config: the run config, supplying seed and val_split.

    Returns:
        Tuple of (train indices, validation indices).
    """
    total = len(dataset)
    class_indices: Dict[int, List[int]] = defaultdict(list)
    for idx in range(total):
        row = dataset.data[idx]
        class_indices[row.get("label4", row["label"])].append(idx)

    split_rng = np.random.RandomState(int(config.seed))
    train_idx: List[int] = []
    val_idx: List[int] = []
    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        split_rng.shuffle(indices)
        val_size = max(1, int(len(indices) * config.val_split))
        val_idx.extend(indices[:val_size])
        train_idx.extend(indices[val_size:])
    return train_idx, val_idx


def collect_embeddings(model: torch.nn.Module, loader: DataLoader,
                       config: Config,
                       device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Collect post-fusion embeddings and labels for one split.

    Args:
        model: loaded model in eval mode.
        loader: unshuffled dataloader.
        config: the model's config.
        device: target device.

    Returns:
        Tuple of embeddings [N, D] and labels [N].
    """
    embs: List[np.ndarray] = []
    labs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = _prepare_model_inputs(batch, config, model, device)
            _, _, raw, _ = model(**inputs, return_embeddings=True)
            embs.append(raw.detach().cpu().float().numpy())
            labs.append(batch["label"].numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labs, axis=0)


def apply_selection(config_path: str, pca_dim: int, shrinkage: float) -> None:
    """Write the selected values into the uf4_genlabel block of a config.

    Exactly one occurrence of each key must exist, so this cannot silently
    edit a different arm. The file is rewritten as text, leaving anchors,
    comments and formatting untouched.

    Args:
        config_path: path to the YAML config.
        pca_dim: selected projection dimensionality.
        shrinkage: selected shrinkage coefficient.

    Raises:
        AssertionError: if a key is missing or appears more than once.
    """
    text = io.open(config_path, encoding="utf-8").read()
    for key, value in (("genlabel_pca_dim", str(pca_dim)),
                       ("genlabel_shrinkage", repr(shrinkage))):
        hits = [ln for ln in text.splitlines()
                if ln.strip().startswith(key + ":")]
        assert len(hits) == 1, f"{key} matched {len(hits)} lines, refusing"
        text = text.replace(hits[0], "    %s: %s" % (key, value), 1)
    io.open(config_path, "w", encoding="utf-8").write(text)
    print("  Applied to %s: pca_dim=%d shrinkage=%g"
          % (config_path, pca_dim, shrinkage))


def main() -> None:
    """Grid-search pca_dim and shrinkage against development UAR."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="uf4_softmix_vanilla",
                    help="checkpoint whose embeddings are used for tuning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", default="MSPP")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--pca-dims", nargs="+", type=int,
                    default=[0, 16, 32, 64, 128, 256])
    ap.add_argument("--shrinkages", nargs="+", type=float,
                    default=[0.0, 0.05, 0.1, 0.3])
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--apply-config", default=None,
                    help="config file whose uf4_genlabel block is updated "
                         "in place with the selected values")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / f"{args.arm}_seed{args.seed}_{args.corpus}_emb.npz"

    if cache.exists():
        print(f"  Reusing cached embeddings at {cache}")
        blob = np.load(cache)
        ztr, ytr = blob["ztr"], blob["ytr"]
        zva, yva = blob["zva"], blob["yva"]
    else:
        model, config = load_arm(args.arm, args.seed, device)
        ds = EmotionDataset(args.corpus, split="train", config=config)
        tr_idx, va_idx = reproduce_split(ds, config)
        print(f"  {args.corpus}: {len(ds)} rows -> train {len(tr_idx)}, "
              f"dev {len(va_idx)} (val_split={config.val_split}, "
              f"seed={config.seed})")
        out = []
        for name, idx in (("train", tr_idx), ("dev", va_idx)):
            loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size,
                                shuffle=False, collate_fn=vad_collate_fn,
                                num_workers=args.num_workers, pin_memory=False)
            z, y = collect_embeddings(model, loader, config, device)
            print(f"    {name}: embeddings {z.shape}")
            out.append((z, y))
        (ztr, ytr), (zva, yva) = out
        np.savez_compressed(cache, ztr=ztr.astype(np.float32), ytr=ytr,
                            zva=zva.astype(np.float32), yva=yva)
        print(f"  Cached to {cache}")

    n_cls = int(max(ytr.max(), yva.max())) + 1
    dim = ztr.shape[1]
    t_tr = torch.as_tensor(ztr)
    t_va = torch.as_tensor(zva)
    y_tr = torch.as_tensor(ytr).long()

    print()
    print("  Development UAR of the class-conditional density classifier")
    print("  (fitted on train embeddings, scored on dev embeddings)")
    print("  %-10s %-11s %9s %14s" % ("pca_dim", "shrinkage", "dev UAR", "mean max post"))
    best = (-1.0, None)
    for pca in args.pca_dims:
        for rho in args.shrinkages:
            den = EmbeddingClassDensity(n_cls, dim, torch.device("cpu"),
                                        pca_dim=pca, shrinkage=rho,
                                        tol=args.tol)
            for i in range(0, len(t_tr), 4096):
                den.accumulate(t_tr[i:i + 4096], y_tr[i:i + 4096])
            if not den.refit():
                print("  %-10s %-11s %9s" % (pca or "none", rho, "refit failed"))
                continue
            post = den.posterior(t_va)
            u = 100 * uar(yva, post.argmax(1).numpy())
            conf = post.max(1).values.mean().item()
            flag = ""
            if conf > 0.99:
                flag = "  <- degenerate, target is effectively one-hot"
            print("  %-10s %-11s %8.2f%% %13.3f%s"
                  % (pca or "none", rho, u, conf, flag))
            if u > best[0]:
                best = (u, (pca, rho))
    print()
    if best[1] is None:
        return
    pca, rho = best[1]
    print("  SELECTED on dev: genlabel_pca_dim=%d genlabel_shrinkage=%g "
          "(dev UAR %.2f%%)" % (pca, rho, best[0]))
    json.dump({"pca_dim": int(pca), "shrinkage": float(rho),
               "dev_uar": float(best[0]), "arm": args.arm, "seed": args.seed},
              io.open(OUT_DIR / "selected.json", "w"), indent=2)

    if args.apply_config:
        apply_selection(args.apply_config, int(pca), float(rho))


if __name__ == "__main__":
    main()
