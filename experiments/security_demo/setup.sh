#!/bin/bash
# Setup script for parameter-golf security demo on runpod.
# Run from the parameter-golf repo root.
set -e

echo "=== Installing parameter-golf dependencies ==="
pip install -r requirements.txt

echo "=== Downloading data (10 training shards for demo) ==="
python data/cached_challenge_fineweb.py --train-shards 10

echo "=== Installing ZKML dependencies ==="
pip install ezkl onnx onnxruntime

echo "=== Setup complete ==="
echo "Installed: torch, ezkl, onnx, onnxruntime"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import ezkl; print(f'EZKL: installed')"
