#!/usr/bin/env python3
"""
Experiment runner with multi-seed averaging and multi-experiment support.
Wraps train.train() to handle seeds, averaging, and batch experiment configs.
"""

import numpy as np
import wandb
import argparse

from typing import Any, Dict, List, Tuple

from utils.config import Config
from data.dataset import EmotionDataset, MultiCorpusDataset
from train import train, is_run_finished, load_finished_results


def _needs_raw_audio(config) -> bool:
    """True when experiment requires raw waveforms (unfrozen audio encoder)."""
    audio_type = getattr(config, 'audio_encoder_type', 'preextracted')
    unfreeze = getattr(config, 'unfreeze_audio_layers', 0)
    return (audio_type in ('wav2vec2', 'emotion2vec')) and unfreeze > 0


def _corpus_key(corpus_name: str, config) -> Tuple[Any, ...]:
    """
    Cache key for a single corpus instance. Two experiments that resolve to the
    same key can share one EmotionDataset object.

    Includes audio_model_name and needs_raw_audio because frozen experiments
    mutate the dataset via cache_encoder_features (which bakes in model-specific
    features and frees raw audio), so those variants cannot be shared with
    experiments that need raw waveforms or a different encoder.
    """
    return (
        corpus_name,
        getattr(config, 'audio_encoder_type', 'preextracted'),
        getattr(config, 'audio_model_name', None),
        getattr(config, 'modality', 'both'),
        getattr(config, 'task_type', 'classification'),
        _needs_raw_audio(config),
    )


def _resolve_train_names(config) -> List[str]:
    names = config.train_dataset
    if isinstance(names, str):
        names = [names]
    return list(names)


def _resolve_test_names(config, train_names: List[str]) -> List[str]:
    """Mirror the test-name resolution logic from data.dataset.create_datasets."""
    task_type = getattr(config, 'task_type', 'classification')
    test_names = list(getattr(config, 'test_datasets', []) or [])

    if not test_names:
        test_names = [d for d in EmotionDataset.DATASET_MAP.keys() if d not in train_names]

    test_names = [d for d in test_names if d not in train_names]

    if task_type == "regression":
        test_names = [d for d in test_names if d in EmotionDataset.DATASETS_WITH_VAD]

    return test_names


def _get_or_load_corpus(
    corpus_name: str,
    config,
    corpus_cache: Dict[Tuple[Any, ...], EmotionDataset],
) -> EmotionDataset:
    """Return a cached EmotionDataset for this corpus/config, loading on miss."""
    key = _corpus_key(corpus_name, config)
    ds = corpus_cache.get(key)
    if ds is None:
        ds = EmotionDataset(
            corpus_name,
            split="train",
            config=config,
            task_type=getattr(config, 'task_type', 'classification'),
        )
        corpus_cache[key] = ds
    return ds


def build_datasets_from_cache(
    config,
    corpus_cache: Dict[Tuple[Any, ...], EmotionDataset],
) -> Tuple[Any, List[EmotionDataset]]:
    """
    Compose (train_dataset, test_datasets) for an experiment, reusing any
    already-loaded corpora from corpus_cache. Populates the cache on miss.
    """
    train_names = _resolve_train_names(config)
    test_names = _resolve_test_names(config, train_names)

    train_list = [_get_or_load_corpus(n, config, corpus_cache) for n in train_names]
    if len(train_list) == 1:
        train_dataset = train_list[0]
    else:
        train_dataset = MultiCorpusDataset(train_list)

    test_datasets = [_get_or_load_corpus(n, config, corpus_cache) for n in test_names]

    print(f"  Training: {train_dataset.dataset_name} -> {test_names}")
    return train_dataset, test_datasets


def preload_datasets(yaml_path) -> Dict[Tuple[Any, ...], EmotionDataset]:
    """
    Walk every experiment and pre-load each unique corpus exactly once.
    Returns a per-corpus cache keyed by _corpus_key(name, config).
    """
    experiments = Config.list_experiments(yaml_path)
    if experiments is None:
        return {}

    # Collect the union of required corpora across all experiments.
    required: Dict[Tuple[Any, ...], Tuple[str, Any]] = {}
    for exp in experiments:
        config = Config.from_yaml(yaml_path, experiment_id=exp['index'])
        train_names = _resolve_train_names(config)
        test_names = _resolve_test_names(config, train_names)
        for name in list(train_names) + list(test_names):
            key = _corpus_key(name, config)
            if key not in required:
                required[key] = (name, config)

    print(f"  Pre-loading {len(required)} unique corpus instances...")
    corpus_cache: Dict[Tuple[Any, ...], EmotionDataset] = {}
    for i, (key, (name, config)) in enumerate(required.items()):
        audio_type = getattr(config, 'audio_encoder_type', 'preextracted')
        print(f"\n  [{i+1}/{len(required)}] Loading corpus: {name} (audio={audio_type}, raw={_needs_raw_audio(config)})")
        corpus_cache[key] = EmotionDataset(
            name,
            split="train",
            config=config,
            task_type=getattr(config, 'task_type', 'classification'),
        )

    print(f"\n  All corpora pre-loaded.\n")
    return corpus_cache


def _run_or_load(config, datasets):
    """Run train() or load cached results if this run is already finished."""
    if is_run_finished(config):
        print(f"  Run already finished: {config.experiment_name}_seed{config.seed} — loading cached results")
        return load_finished_results(config)
    return train(config, datasets=datasets)


def run_with_seeds(config, datasets=None):
    """
    Run experiment with multiple seeds and compute averaged results.
    Creates WandB runs for each seed and a final _AVERAGED run.
    """
    seeds = getattr(config, 'seeds', [config.seed])

    if isinstance(seeds, (int, float)):
        seeds = [int(seeds)]

    if len(seeds) == 1:
        config.seed = seeds[0]
        return _run_or_load(config, datasets)

    # Multi-seed experiment
    print(f"\n{'='*60}")
    print(f"MULTI-SEED EXPERIMENT: {config.experiment_name}")
    print(f"   Running with {len(seeds)} seeds: {seeds}")
    print(f"{'='*60}\n")

    all_results = []
    original_exp_name = config.experiment_name

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed_idx+1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")

        config.seed = int(seed)
        config.experiment_name = f"{original_exp_name}_seed{seed}"

        result = _run_or_load(config, datasets)
        all_results.append(result)

    # Compute and log averaged results
    print(f"\n{'='*60}")
    print(f"COMPUTING AVERAGED RESULTS ACROSS {len(seeds)} SEEDS")
    print(f"{'='*60}")

    averaged = compute_averaged_results(all_results, config)
    log_averaged_to_wandb(averaged, config, original_exp_name, seeds)

    return averaged


def compute_averaged_results(all_results, config):
    """Compute mean and std across multiple seed runs"""
    averaged = {}
    task_type = getattr(config, 'task_type', 'classification')

    if task_type == "regression":
        # Validation
        val_maes = [r["validation"]["overall_mae"] for r in all_results]
        val_cccs = [r["validation"]["overall_ccc"] for r in all_results]
        averaged["validation"] = {
            "overall_mae_mean": np.mean(val_maes),
            "overall_mae_std": np.std(val_maes),
            "overall_ccc_mean": np.mean(val_cccs),
            "overall_ccc_std": np.std(val_cccs),
        }

        # Test results per dataset
        test_datasets = [tr["dataset"] for tr in all_results[0]["test_results"]]
        averaged["test_results"] = []

        for dataset_name in test_datasets:
            maes, cccs = [], []
            v_cccs, a_cccs, d_cccs = [], [], []

            for result in all_results:
                for tr in result["test_results"]:
                    if tr["dataset"] == dataset_name:
                        maes.append(tr["results"]["overall_mae"])
                        cccs.append(tr["results"]["overall_ccc"])
                        v_cccs.append(tr["results"].get("valence_ccc", 0))
                        a_cccs.append(tr["results"].get("arousal_ccc", 0))
                        d_cccs.append(tr["results"].get("dominance_ccc", 0))
                        break

            averaged["test_results"].append({
                "dataset": dataset_name,
                "overall_mae_mean": np.mean(maes),
                "overall_mae_std": np.std(maes),
                "overall_ccc_mean": np.mean(cccs),
                "overall_ccc_std": np.std(cccs),
                "valence_ccc_mean": np.mean(v_cccs),
                "arousal_ccc_mean": np.mean(a_cccs),
                "dominance_ccc_mean": np.mean(d_cccs),
            })
    else:
        # Classification - Validation
        val_accs = [r["validation"]["accuracy"] for r in all_results]
        val_uars = [r["validation"]["uar"] for r in all_results]
        averaged["validation"] = {
            "accuracy_mean": np.mean(val_accs),
            "accuracy_std": np.std(val_accs),
            "uar_mean": np.mean(val_uars),
            "uar_std": np.std(val_uars),
        }

        # Test results per dataset
        test_datasets = [tr["dataset"] for tr in all_results[0]["test_results"]]
        averaged["test_results"] = []

        for dataset_name in test_datasets:
            accs, uars, f1s = [], [], []

            for result in all_results:
                for tr in result["test_results"]:
                    if tr["dataset"] == dataset_name:
                        accs.append(tr["results"]["accuracy"])
                        uars.append(tr["results"]["uar"])
                        f1s.append(tr["results"].get("f1_weighted", 0))
                        break

            averaged["test_results"].append({
                "dataset": dataset_name,
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "uar_mean": np.mean(uars),
                "uar_std": np.std(uars),
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
            })

    return averaged


def log_averaged_to_wandb(averaged, config, experiment_name, seeds):
    """Log averaged results to WandB as a separate _AVERAGED run"""
    train_ds = config.train_dataset if isinstance(config.train_dataset, str) else "+".join(config.train_dataset)

    wandb.init(
        project=config.wandb_project,
        name=f"{experiment_name}_AVERAGED",
        config={
            **config.to_dict(),
            "seeds": seeds,
            "num_seeds": len(seeds),
            "averaged_run": True,
        },
        tags=["averaged", "multi-seed"],
    )

    print(f"\nAVERAGED RESULTS ACROSS {len(seeds)} SEEDS")

    task_type = getattr(config, 'task_type', 'classification')

    if task_type == "regression":
        val = averaged["validation"]
        print(f"\nValidation:")
        print(f"  MAE: {val['overall_mae_mean']:.4f} +/- {val['overall_mae_std']:.4f}")
        print(f"  CCC: {val['overall_ccc_mean']:.4f} +/- {val['overall_ccc_std']:.4f}")

        wandb.log({
            f"AVERAGED_{train_ds}/val_mae_mean": val["overall_mae_mean"],
            f"AVERAGED_{train_ds}/val_mae_std": val["overall_mae_std"],
            f"AVERAGED_{train_ds}/val_ccc_mean": val["overall_ccc_mean"],
            f"AVERAGED_{train_ds}/val_ccc_std": val["overall_ccc_std"],
        })

        for tr in averaged["test_results"]:
            ds = tr["dataset"]
            print(f"\n{train_ds} -> {ds}:")
            print(f"  MAE: {tr['overall_mae_mean']:.4f} +/- {tr['overall_mae_std']:.4f}")
            print(f"  CCC: {tr['overall_ccc_mean']:.4f} +/- {tr['overall_ccc_std']:.4f}")
            print(f"  V/A/D CCC: {tr['valence_ccc_mean']:.4f} / {tr['arousal_ccc_mean']:.4f} / {tr['dominance_ccc_mean']:.4f}")

            prefix = f"AVERAGED_{train_ds}/{train_ds}to{ds}"
            wandb.log({
                f"{prefix}_mae_mean": tr["overall_mae_mean"],
                f"{prefix}_mae_std": tr["overall_mae_std"],
                f"{prefix}_ccc_mean": tr["overall_ccc_mean"],
                f"{prefix}_ccc_std": tr["overall_ccc_std"],
                f"{prefix}_valence_ccc_mean": tr["valence_ccc_mean"],
                f"{prefix}_arousal_ccc_mean": tr["arousal_ccc_mean"],
                f"{prefix}_dominance_ccc_mean": tr["dominance_ccc_mean"],
            })
    else:
        val = averaged["validation"]
        print(f"\nValidation:")
        print(f"  Accuracy: {val['accuracy_mean']:.4f} +/- {val['accuracy_std']:.4f}")
        print(f"  UAR: {val['uar_mean']:.4f} +/- {val['uar_std']:.4f}")

        wandb.log({
            f"AVERAGED_{train_ds}/val_accuracy_mean": val["accuracy_mean"],
            f"AVERAGED_{train_ds}/val_accuracy_std": val["accuracy_std"],
            f"AVERAGED_{train_ds}/val_uar_mean": val["uar_mean"],
            f"AVERAGED_{train_ds}/val_uar_std": val["uar_std"],
        })

        for tr in averaged["test_results"]:
            ds = tr["dataset"]
            print(f"\n{train_ds} -> {ds}:")
            print(f"  Accuracy: {tr['accuracy_mean']:.4f} +/- {tr['accuracy_std']:.4f}")
            print(f"  UAR: {tr['uar_mean']:.4f} +/- {tr['uar_std']:.4f}")

            prefix = f"AVERAGED_{train_ds}/{train_ds}to{ds}"
            wandb.log({
                f"{prefix}_accuracy_mean": tr["accuracy_mean"],
                f"{prefix}_accuracy_std": tr["accuracy_std"],
                f"{prefix}_uar_mean": tr["uar_mean"],
                f"{prefix}_uar_std": tr["uar_std"],
                f"{prefix}_f1_mean": tr["f1_mean"],
                f"{prefix}_f1_std": tr["f1_std"],
            })

    wandb.finish()


def run_all_experiments(yaml_path):
    """Run all experiments from a multi-experiment YAML file"""
    experiments = Config.list_experiments(yaml_path)
    if experiments is None:
        raise ValueError("YAML file doesn't contain multiple experiments")

    # Pre-load every unique corpus once; experiments compose train/test from this cache.
    corpus_cache = preload_datasets(yaml_path)

    print(f"Running {len(experiments)} experiments from {yaml_path}")
    results = []

    for exp in experiments:
        exp_name = exp['name']
        exp_idx = exp['index']
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_idx+1}/{len(experiments)}: {exp_name}")
        print(f"{'='*60}")

        try:
            config = Config.from_yaml(yaml_path, experiment_id=exp_idx)

            # Compose this experiment's train/test bundle from the per-corpus cache.
            datasets = build_datasets_from_cache(config, corpus_cache)

            result = run_with_seeds(config, datasets=datasets)
            results.append({
                'index': exp_idx,
                'name': exp_name,
                'status': 'completed',
                'result': result,
            })
        except Exception as e:
            import traceback
            print(f"Experiment {exp_name} failed: {e}")
            traceback.print_exc()
            # Close the wandb run so it doesn't leak into the next experiment
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
            results.append({
                'index': exp_idx,
                'name': exp_name,
                'status': 'failed',
                'error': str(e),
            })

    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r['status'] == 'completed' else "FAILED"
        print(f"   [{status}] {r['name']}")

    completed = sum(1 for r in results if r['status'] == 'completed')
    print(f"\nCompleted: {completed}/{len(results)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run emotion recognition experiments")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--experiment', '-e', type=str, default=None,
                        help='Experiment ID/name/index for multi-experiment configs')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Run all experiments in a multi-experiment config')
    args = parser.parse_args()

    if args.all:
        run_all_experiments(args.config)
    else:
        experiment_id = args.experiment
        if experiment_id is not None and experiment_id.isdigit():
            experiment_id = int(experiment_id)
        config = Config.from_yaml(args.config, experiment_id=experiment_id)
        run_with_seeds(config)
