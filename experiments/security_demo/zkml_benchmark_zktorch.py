#!/usr/bin/env python3
"""
ZKML Benchmark using ZKTorch (https://github.com/uiuc-kang-lab/zk-torch).

Exports the trained model to ONNX, then runs ZKTorch's prove/verify pipeline,
timing each step. ZKTorch is a Rust binary invoked via subprocess.

Run from the parameter-golf repo root:
    python experiments/security_demo/zkml_benchmark_zktorch.py [options]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timer(name, timings):
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self
        def __exit__(self, *exc):
            timings[name] = time.time() - self.t0
            print(f"      {timings[name]:.2f}s")
    return _Timer()


def find_zktorch_binary():
    """Locate the zk_torch binary."""
    candidates = [
        os.path.expanduser("~/zk-torch/target/release/zk_torch"),
        shutil.which("zk_torch"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def load_and_export_onnx(model_path, seq_len, work_dir):
    """Load model, create inference wrapper, export to ONNX. Returns (onnx_path, vocab_size)."""
    sys.path.insert(0, "records/track_10min_16mb/2026-03-17_NaiveBaseline")
    import train_gpt as tg
    import torch

    hp = tg.Hyperparameters()
    model = tg.GPT(
        vocab_size=hp.vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings,
        tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
    )

    if os.path.exists(model_path):
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"  Loaded checkpoint: {model_path}")
    else:
        print(f"  WARNING: {model_path} not found, using random init")

    model.eval().float()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Import the inference wrapper from the EZKL benchmark
    sys.path.insert(0, "experiments/security_demo")
    from zkml_benchmark import make_inference_model

    inf_model = make_inference_model(model)
    inf_model.eval()

    # Export to ONNX
    onnx_path = os.path.join(work_dir, "model.onnx")
    dummy = torch.randint(0, hp.vocab_size, (1, seq_len))
    torch.onnx.export(
        inf_model,
        (dummy,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    # Post-process: replace Div nodes with Reciprocal+Mul.
    # ZKTorch's Div implementation crashes on certain tensor shapes.
    # Pattern: Div(Constant(1.0), Sqrt(x)) → Reciprocal(Sqrt(x)) i.e. Rsqrt
    # General: Div(a, b) → Mul(a, Reciprocal(b))
    import onnx
    from onnx import helper
    model_proto = onnx.load(onnx_path)

    # Also check if ZKTorch supports Reciprocal — if not, we may need Pow(x, -1)
    new_nodes = []
    replaced = 0
    for node in model_proto.graph.node:
        if node.op_type == "Div":
            # Replace Div(a, b) → Mul(a, Pow(b, -1))
            # Pow(x, -1) is more universally supported than Reciprocal
            neg_one_name = node.output[0] + "_neg1"
            inv_name = node.output[0] + "_inv"

            # Create constant -1
            neg_one = helper.make_node(
                "Constant", [], [neg_one_name],
                value=helper.make_tensor("neg1", onnx.TensorProto.FLOAT, [], [-1.0]),
            )
            # b^(-1)
            pow_node = helper.make_node("Pow", [node.input[1], neg_one_name], [inv_name])
            # a * b^(-1)
            mul_node = helper.make_node(
                "Mul", [node.input[0], inv_name], list(node.output), name=node.name,
            )
            new_nodes.extend([neg_one, pow_node, mul_node])
            replaced += 1
        else:
            new_nodes.append(node)
    del model_proto.graph.node[:]
    model_proto.graph.node.extend(new_nodes)
    onnx.save(model_proto, onnx_path)
    print(f"  Post-processed: replaced {replaced} Div nodes with Pow+Mul")

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ONNX exported: {size_mb:.1f} MB")

    return onnx_path, hp.vocab_size


# ---------------------------------------------------------------------------
# ZKTorch pipeline
# ---------------------------------------------------------------------------

def write_config(work_dir, onnx_path, input_path):
    """Write a ZKTorch config.yaml matching its expected nested YAML structure."""
    abs_work = os.path.abspath(work_dir)
    zktorch_dir = find_zktorch_binary()
    zktorch_root = os.path.dirname(os.path.dirname(os.path.dirname(zktorch_dir))) if zktorch_dir else ""
    # ZKTorch ships a 'challenge' ptau file in its repo root
    ptau_path = os.path.join(zktorch_root, "challenge") if zktorch_root else ""

    # Create output directories
    for d in ("models", "setups", "modelsEnc", "inputsEnc", "outputsEnc",
              "proofs", "acc_proofs", "final_proofs"):
        os.makedirs(os.path.join(abs_work, d), exist_ok=True)

    config_yaml = f"""task: sample
onnx:
  model_path: "{os.path.abspath(onnx_path)}"
  input_path: "{os.path.abspath(input_path) if input_path else ''}"
ptau:
  ptau_path: "{ptau_path}"
  pow_len_log: 7
  loaded_pow_len_log: 7
sf:
  scale_factor_log: 3
  cq_range_log: 6
  cq_range_lower_log: 5
prover:
  model_path: "{os.path.join(abs_work, 'models')}"
  setup_path: "{os.path.join(abs_work, 'setups')}"
  enc_model_path: "{os.path.join(abs_work, 'modelsEnc')}"
  enc_input_path: "{os.path.join(abs_work, 'inputsEnc')}"
  enc_output_path: "{os.path.join(abs_work, 'outputsEnc')}"
  proof_path: "{os.path.join(abs_work, 'proofs')}"
  acc_proof_path: "{os.path.join(abs_work, 'acc_proofs')}"
  final_proof_path: "{os.path.join(abs_work, 'final_proofs')}"
  enable_layer_setup: true
verifier:
  enc_model_path: "{os.path.join(abs_work, 'modelsEnc')}"
  enc_input_path: "{os.path.join(abs_work, 'inputsEnc')}"
  enc_output_path: "{os.path.join(abs_work, 'outputsEnc')}"
  proof_path: "{os.path.join(abs_work, 'proofs')}"
"""

    config_path = os.path.join(abs_work, "config.yaml")
    with open(config_path, "w") as f:
        f.write(config_yaml)
    print(f"  Config written to {config_path}")
    print(f"  PTAU: {ptau_path}")
    return config_path


def run_zktorch(zktorch_bin, config_path, work_dir, timings):
    """Run the ZKTorch binary and capture output."""
    cmd = [zktorch_bin, os.path.abspath(config_path)]
    env = os.environ.copy()

    print(f"  Running: {' '.join(cmd)}")
    with timer("zktorch_total", timings):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,  # 1 hour max
        )

    print(f"  Return code: {result.returncode}")
    if result.stdout:
        print("  --- stdout ---")
        for line in result.stdout.strip().split("\n")[-30:]:
            print(f"    {line}")
    if result.stderr:
        print("  --- stderr (last 20 lines) ---")
        for line in result.stderr.strip().split("\n")[-20:]:
            print(f"    {line}")

    # Save full output
    with open(os.path.join(work_dir, "zktorch_stdout.txt"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(work_dir, "zktorch_stderr.txt"), "w") as f:
        f.write(result.stderr)

    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ZKML benchmark using ZKTorch")
    parser.add_argument(
        "--model",
        default="checkpoints/ckpt_final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Sequence length for ZKML proof (default: 8)",
    )
    parser.add_argument(
        "--work-dir",
        default="zktorch_bench",
        help="Working directory for ZKTorch artifacts",
    )
    args = parser.parse_args()

    timings = {}

    print("=" * 60)
    print(f"ZKTorch Benchmark  (seq_len={args.seq_len})")
    print("=" * 60)

    # ---- Check ZKTorch binary ----
    zktorch_bin = find_zktorch_binary()
    if not zktorch_bin:
        print("\nERROR: zk_torch binary not found.")
        print("Install with:")
        print("  git clone https://github.com/uiuc-kang-lab/zk-torch.git ~/zk-torch")
        print("  cd ~/zk-torch && rustup override set nightly")
        print("  cargo build --release --bin zk_torch --features fold")
        return
    print(f"\nZKTorch binary: {zktorch_bin}")

    # ---- ONNX export ----
    os.makedirs(args.work_dir, exist_ok=True)

    print("\nStep 1: Load model + ONNX export")
    try:
        with timer("onnx_export", timings):
            onnx_path, vocab_size = load_and_export_onnx(
                args.model, args.seq_len, args.work_dir
            )
    except Exception as e:
        print(f"  ONNX export FAILED: {e}")
        import traceback; traceback.print_exc()
        _save_results(args, timings, success=False, error=f"onnx_export: {e}")
        return

    # ---- Create input data ----
    print("\nStep 2: Create input data")
    import numpy as np
    input_data = np.random.randint(0, vocab_size, (1, args.seq_len)).tolist()
    input_path = os.path.join(args.work_dir, "input.json")
    with open(input_path, "w") as f:
        json.dump({"input_data": input_data}, f)

    # ---- Write config ----
    print("\nStep 3: Write ZKTorch config")
    config_path = write_config(args.work_dir, onnx_path, input_path)
    print(f"  Config: {config_path}")

    # ---- Run ZKTorch ----
    print("\nStep 4: Run ZKTorch prove + verify")
    try:
        success = run_zktorch(zktorch_bin, config_path, args.work_dir, timings)
    except subprocess.TimeoutExpired:
        print("  ZKTorch TIMED OUT (1 hour limit)")
        _save_results(args, timings, success=False, error="timeout")
        return
    except Exception as e:
        print(f"  ZKTorch FAILED: {e}")
        import traceback; traceback.print_exc()
        _save_results(args, timings, success=False, error=f"zktorch: {e}")
        return

    _save_results(args, timings, success=success)


def _save_results(args, timings, success, error=None):
    print("\n" + "=" * 60)
    print("TIMING SUMMARY (ZKTorch)")
    print("=" * 60)
    for k, v in timings.items():
        print(f"  {k:20s}  {v:8.2f}s")
    total = sum(timings.values())
    print(f"  {'TOTAL':20s}  {total:8.2f}s")

    if "zktorch_total" in timings:
        t = timings["zktorch_total"]
        print(f"\n  Extrapolations (based on total={t:.1f}s per sample):")
        print(f"    50 samples (final eval):        {t * 50:8.1f}s = {t * 50 / 60:.1f} min")
        print(f"    5 samples x 10 checkpoints:     {t * 50:8.1f}s = {t * 50 / 60:.1f} min")

    results = {
        "framework": "zktorch",
        "success": success,
        "seq_len": args.seq_len,
        "model": args.model,
        "timings": timings,
    }
    if error:
        results["error"] = error

    results_path = os.path.join(args.work_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
