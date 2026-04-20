#!/bin/bash
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=8:12:00    
#SBATCH --gres=gpu:h100_3g.40gb:1

#gpu:h100:1         

export WANDB_MODE=offline
export WANDB_DIR=./wandb
export WANDB_API_KEY=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

export HF_DATASETS_CACHE=/home/carolw/links/scratch/HF/Datasets
export TRANSFORMERS_CACHE=/home/carolw/links/scratch/HF/Transformers
export HF_HOME=/home/carolw/links/scratch/HF
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG_FILE=${1:-"configs/sample.yaml"}


# Capture all additional arguments after the first two
shift 1
ADDITIONAL_ARGS="$@"

module purge

module load opencv/4.9.0

source venv/bin/activate

python runer.py --cfg $CONFIG_FILE --all  $ADDITIONAL_ARGS


