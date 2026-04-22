#!/bin/bash
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=16:12:00    
#SBATCH --gres=gpu:h100_3g.40gb:1

#gpu:h100:1         

export WANDB_MODE=offline
export WANDB_DIR=./wandb
export WANDB_API_KEY=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
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1                                                                
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1                                                           
export MODELSCOPE_CACHE=$SCRATCH/modelscope_cache
export HF_HOME=$SCRATCH/hf_cache                                                                                                                                                     
export PYTHONUNBUFFERED=1 
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
module load gcc
module load cuda/12.6
module load opencv/4.9.0
module load ffmpeg/7.1.1

export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$EBROOTCUDA/extras/CUPTI/lib64:$EBROOTFFMPEG/lib:$LD_LIBRARY_PATH

source .venv/bin/activate

python runner.py --config $CONFIG_FILE --experiment $ADDITIONAL_ARGS