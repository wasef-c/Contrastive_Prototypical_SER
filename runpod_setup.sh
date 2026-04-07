#!/bin/bash
# RunPod A40 setup script for enhanced_proto_sweep
# Usage: ssh into pod, then:
#   git clone <your-repo> && cd Emotion2Vec_Contrastive && bash runpod_setup.sh

set -e

echo "=== Installing dependencies ==="
pip install -q funasr==1.2.6 modelscope==1.26.0 datasets==3.3.2 \
    transformers==4.49.0 wandb==0.20.1 librosa==0.11.0 soundfile==0.13.1 \
    scikit-learn==1.6.1 pyyaml seaborn

echo "=== Logging into wandb ==="
wandb login

echo "=== Pre-downloading models ==="
python -c "
from funasr import AutoModel
AutoModel(model='iic/emotion2vec_base', disable_update=True)
print('emotion2vec OK')
"
python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('bert-base-uncased')
AutoTokenizer.from_pretrained('bert-base-uncased')
print('BERT OK')
"

echo "=== Setup complete ==="
echo "Run the sweep with:"
echo "  python runner.py --config configs/enhanced_proto_sweep.yaml --all"
echo ""
echo "Or a single experiment:"
echo "  python runner.py --config configs/enhanced_proto_sweep.yaml -e 0"
