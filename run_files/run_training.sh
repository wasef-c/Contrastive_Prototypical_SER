#!/bin/bash
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:12:00
#SBATCH --gres=gpu:h100_3g.40gb:1

# For a full H100 instead of a MIG slice:
# #SBATCH --gres=gpu:h100:1

set -euo pipefail

# --- wandb ---
# API key is expected to be in ~/.bashrc or sourced from a gitignored file.
# Never commit WANDB_API_KEY to this script.
export WANDB_MODE=offline
export WANDB_DIR=$SLURM_TMPDIR/wandb

# --- HuggingFace / ModelScope caches (persistent, on scratch) ---
export HF_HOME_SRC=/home/carolw/links/scratch/HF
export MODELSCOPE_CACHE=$SCRATCH/modelscope_cache

# --- Offline mode: compute nodes have no internet ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- Python behavior ---
export PYTHONUNBUFFERED=1

# --- Clear stale ModelScope lock from previous killed jobs ---
rm -rf "$MODELSCOPE_CACHE/.lock" || true

# --- Stage HF cache to node-local NVMe so DataLoader reads hit local disk ---
# One-time sequential copy at job start, then every epoch's audio reads are fast.
echo "[stage] copying HF cache to \$SLURM_TMPDIR ..."
mkdir -p "$SLURM_TMPDIR/HF"
cp -r "$HF_HOME_SRC/." "$SLURM_TMPDIR/HF/"
export HF_HOME=$SLURM_TMPDIR/HF
export HF_DATASETS_CACHE=$SLURM_TMPDIR/HF/Datasets
export TRANSFORMERS_CACHE=$SLURM_TMPDIR/HF/Transformers
echo "[stage] done. HF_HOME=$HF_HOME"

# --- Args: first positional is the config path; rest passed through ---
CONFIG_FILE=${1:-"configs/sample.yaml"}
shift 1
ADDITIONAL_ARGS="$@"

# --- Modules ---
module purge
module load gcc
module load cuda/12.6
module load opencv/4.9.0
module load ffmpeg/7.1.1

export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$EBROOTCUDA/extras/CUPTI/lib64:$EBROOTFFMPEG/lib:${LD_LIBRARY_PATH:-}

source .venv/bin/activate

# --- Train ---
python -u runner.py --config "$CONFIG_FILE" --experiment $ADDITIONAL_ARGS
EXIT_CODE=$?

# --- Sync wandb offline runs back to scratch for later `wandb sync` on login ---
if [[ -d "$SLURM_TMPDIR/wandb" ]]; then
    mkdir -p "$SCRATCH/wandb_runs"
    cp -r "$SLURM_TMPDIR/wandb" "$SCRATCH/wandb_runs/job_${SLURM_JOB_ID}"
    echo "[wandb] offline runs copied to $SCRATCH/wandb_runs/job_${SLURM_JOB_ID}"
fi

exit $EXIT_CODE
