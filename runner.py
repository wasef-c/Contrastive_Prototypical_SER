#!/usr/bin/env python3
"""
Experiment runner with multi-seed averaging and multi-experiment support.
Wraps train.train() to handle seeds, averaging, and batch experiment configs.
"""

import numpy as np
import wandb
import argparse

from utils.config import Config
from data.dataset import create_datasets
from train import train


def _dataset_cache_key(config):
    """
    Compute a cache key for dataset loading.
    Experiments with the same key can share loaded datasets.
    """
    train_ds = config.train_dataset
    if isinstance(train_ds, list):
        train_ds = tuple(sorted(train_ds))
    else:
        train_ds = (train_ds,)

    test_ds = tuple(sorted(getattr(config, 'test_datasets', [])))
    audio_type = getattr(config, 'audio_encoder_type', 'preextracted')
    modality = getattr(config, 'modality', 'both')
    task_type = getattr(config, 'task_type', 'classification')

    return (train_ds, test_ds, audio_type, modality, task_type)


def preload_datasets(yaml_path):
    """
    Pre-load all unique dataset combinations needed across experiments.
    Returns a dict mapping cache_key -> (train_dataset, test_datasets).
    """
    experiments = Config.list_experiments(yaml_path)
    if experiments is None:
        return {}

    # Find unique dataset configs
    seen_keys = {}
    for exp in experiments:
        config = Config.from_yaml(yaml_path, experiment_id=exp['index'])
        key = _dataset_cache_key(config)
        if key not in seen_keys:
            seen_keys[key] = config

    print(f"  Pre-loading {len(seen_keys)} unique dataset configurations...")
    cache = {}
    for i, (key, config) in enumerate(seen_keys.items()):
        train_ds_name = config.train_dataset if isinstance(config.train_dataset, str) else "+".join(config.train_dataset)
        audio_type = getattr(config, 'audio_encoder_type', 'preextracted')
        print(f"\n  [{i+1}/{len(seen_keys)}] Loading: train={train_ds_name}, audio={audio_type}")
        datasets = create_datasets(config)
        cache[key] = datasets

    print(f"\n  All datasets pre-loaded.\n")
    return cache


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
        return train(config, datasets=datasets)

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

        result = train(config, datasets=datasets)
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

    # Pre-load all datasets once
    dataset_cache = preload_datasets(yaml_path)

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

            # Look up cached datasets
            key = _dataset_cache_key(config)
            datasets = dataset_cache.get(key)

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
