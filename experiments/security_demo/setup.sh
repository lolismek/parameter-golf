#!/bin/bash
# Setup script for parameter-golf security demo on runpod.
# Run from the parameter-golf repo root.
set -e

echo "=== Installing parameter-golf dependencies ==="
pip install -r requirements.txt

echo "=== Downloading data (10 training shards for demo) ==="
python data/cached_challenge_fineweb.py --train-shards 10

echo "=== Installing ZKML dependencies ==="
pip install ezkl onnx onnxruntime onnxscript

echo "=== Installing ZKTorch (build from source) ==="
if ! command -v cargo &>/dev/null; then
    echo "  Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
if [ ! -d "$HOME/zk-torch" ]; then
    git clone https://github.com/uiuc-kang-lab/zk-torch.git "$HOME/zk-torch"
    cd "$HOME/zk-torch"
    rustup override set nightly
    cargo build --release --bin zk_torch --features fold
    cd -
else
    echo "  ZKTorch already cloned at ~/zk-torch"
fi

echo "=== Setup complete ==="
echo "Installed: torch, ezkl, onnx, onnxruntime, zktorch"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import ezkl; print(f'EZKL: installed')"
echo "ZKTorch: $(ls $HOME/zk-torch/target/release/zk_torch 2>/dev/null && echo 'installed' || echo 'not found')"
