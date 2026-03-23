#!/bin/bash
# Full security demo: train with checkpoints + ZKML benchmark.
# Run from the parameter-golf repo root.
set -e

NGPU=${1:-1}  # Pass GPU count as first arg, default 1

echo "============================================================"
echo "  Security Demo: parameter-golf verification benchmark"
echo "  GPUs: $NGPU"
echo "============================================================"

# --- Step 1: Setup (skip if already done) ---
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo ""
    echo ">>> Step 1: Setup"
    bash experiments/security_demo/setup.sh
else
    echo ""
    echo ">>> Step 1: Setup (skipped, data already exists)"
    pip install -q ezkl onnx onnxruntime 2>/dev/null || true
fi

# --- Step 2: Patch training script ---
echo ""
echo ">>> Step 2: Patch training script"
python experiments/security_demo/patch_training.py

# --- Step 3: Run training (10 min) ---
echo ""
echo ">>> Step 3: Training (10 min wallclock, $NGPU GPUs)"
echo "    Checkpoints saved to ./checkpoints/ every 60s"
echo "    Loss log saved to ./loss_log.json"
torchrun --nproc_per_node=$NGPU train_gpt_patched.py

# --- Step 4: ZKML benchmark ---
echo ""
echo ">>> Step 4: ZKML benchmark"

# Try multiple sequence lengths to see scaling
for SEQ_LEN in 1 4 8; do
    echo ""
    echo "--- ZKML benchmark: seq_len=$SEQ_LEN ---"
    python experiments/security_demo/zkml_benchmark.py \
        --model checkpoints/ckpt_final.pt \
        --seq-len $SEQ_LEN \
        --work-dir "zkml_bench_seq${SEQ_LEN}" \
        || echo "  (failed for seq_len=$SEQ_LEN)"
done

# --- Summary ---
echo ""
echo "============================================================"
echo "  Demo complete!"
echo ""
echo "  Checkpoints:  ls -la checkpoints/"
ls -la checkpoints/ 2>/dev/null || true
echo ""
echo "  Loss log:     head loss_log.json"
head -20 loss_log.json 2>/dev/null || true
echo ""
echo "  ZKML results:"
for d in zkml_bench_seq*; do
    if [ -f "$d/benchmark_results.json" ]; then
        echo "    $d:"
        cat "$d/benchmark_results.json"
    fi
done
echo "============================================================"
