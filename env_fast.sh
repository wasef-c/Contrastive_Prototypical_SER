# Point the heavy caches at the NVMe. Source this before any training run:
#   source env_fast.sh
#
# The spinning sda held every cache until 2026-08-19, which capped feature
# extraction at 3 MB/s with the GPU idle at 2 percent. Nothing here changes
# behaviour, only where bytes live.
export HF_HOME=/mnt/fast/huggingface
export HF_DATASETS_CACHE=/mnt/fast/huggingface/datasets
export MODELSCOPE_CACHE=/mnt/fast/modelscope
export FRAME_FEATURE_CACHE_DIR=/mnt/fast/flash/frame_feature_cache
