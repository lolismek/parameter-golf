#!/usr/bin/env python3
"""
ZKML Benchmark for parameter-golf model.

Exports the trained model to ONNX, then runs the full EZKL pipeline
(settings, calibration, compile, setup, witness, prove, verify),
timing each step.

Run from the parameter-golf repo root:
    python experiments/security_demo/zkml_benchmark.py [options]
"""

import argparse
import json
import os
import sys
import time
import shutil

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timer(name, timings):
    """Context manager that records elapsed seconds into timings[name]."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self
        def __exit__(self, *exc):
            timings[name] = time.time() - self.t0
            print(f"      {timings[name]:.2f}s")
    return _Timer()


def load_model(model_path):
    """Load the baseline GPT model and optionally restore a checkpoint."""
    sys.path.insert(0, "records/track_10min_16mb/2026-03-17_NaiveBaseline")
    import train_gpt as tg

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

    import torch
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"  Loaded checkpoint: {model_path}")
    else:
        print(f"  WARNING: {model_path} not found, using random init weights")

    model.eval().float()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    return model, hp


# ---------------------------------------------------------------------------
# Inference wrapper (returns logits, ONNX-friendly)
# ---------------------------------------------------------------------------

def make_inference_model(model):
    """
    Wrap the GPT so forward(input_ids) -> logits.

    Internally replaces F.rms_norm and F.scaled_dot_product_attention with
    decomposed ops so ONNX export (and subsequently EZKL) can handle them.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class InferenceGPT(nn.Module):
        def __init__(self, gpt):
            super().__init__()
            self.gpt = gpt

        @staticmethod
        def _rms_norm(x, eps=1e-6):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        def forward(self, input_ids):
            g = self.gpt
            x = g.tok_emb(input_ids)
            x = self._rms_norm(x)
            x0 = x
            skips = []

            for i in range(g.num_encoder_layers):
                x = self._block_forward(g.blocks[i], x, x0)
                skips.append(x)

            for i in range(g.num_decoder_layers):
                if skips:
                    sw = g.skip_weights[i].unsqueeze(0).unsqueeze(0)
                    x = x + sw * skips.pop()
                x = self._block_forward(
                    g.blocks[g.num_encoder_layers + i], x, x0
                )

            x = self._rms_norm(x)

            if g.tie_embeddings:
                logits = F.linear(x, g.tok_emb.weight)
            else:
                logits = g.lm_head(x)

            logits = g.logit_softcap * torch.tanh(logits / g.logit_softcap)
            return logits

        def _block_forward(self, block, x, x0):
            mix = block.resid_mix
            x = mix[0].unsqueeze(0).unsqueeze(0) * x + mix[1].unsqueeze(0).unsqueeze(0) * x0
            attn_out = self._attn_forward(block.attn, self._rms_norm(x))
            x = x + block.attn_scale.unsqueeze(0).unsqueeze(0) * attn_out
            mlp_out = self._mlp_forward(block.mlp, self._rms_norm(x))
            x = x + block.mlp_scale.unsqueeze(0).unsqueeze(0) * mlp_out
            return x

        def _mlp_forward(self, mlp, x):
            x = torch.relu(F.linear(x, mlp.fc.weight))
            return F.linear(x.square(), mlp.proj.weight)

        def _attn_forward(self, attn, x):
            bsz, seqlen, dim = x.shape
            head_dim = attn.head_dim
            num_heads = attn.num_heads
            num_kv_heads = attn.num_kv_heads

            q = F.linear(x, attn.c_q.weight).reshape(bsz, seqlen, num_heads, head_dim).permute(0, 2, 1, 3)
            k = F.linear(x, attn.c_k.weight).reshape(bsz, seqlen, num_kv_heads, head_dim).permute(0, 2, 1, 3)
            v = F.linear(x, attn.c_v.weight).reshape(bsz, seqlen, num_kv_heads, head_dim).permute(0, 2, 1, 3)

            # Manual RMS norm on q, k
            q = self._rms_norm(q)
            k = self._rms_norm(k)

            # RoPE
            cos, sin = attn.rotary(seqlen, x.device, q.dtype)
            half = head_dim // 2
            q1, q2 = q[..., :half], q[..., half:]
            q = torch.cat((q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos), dim=-1)
            k1, k2 = k[..., :half], k[..., half:]
            k = torch.cat((k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos), dim=-1)

            # Q gain
            q = q * attn.q_gain.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            # GQA: expand kv heads
            if num_kv_heads != num_heads:
                repeats = num_heads // num_kv_heads
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)

            # Manual scaled dot-product attention (causal)
            scale = head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask
            mask = torch.triu(
                torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1)

            out = torch.matmul(attn_weights, v)
            out = out.permute(0, 2, 1, 3).reshape(bsz, seqlen, dim)
            return F.linear(out, attn.proj.weight)

    return InferenceGPT(model)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(inf_model, seq_len, vocab_size, onnx_path, timings):
    import torch

    dummy = torch.randint(0, vocab_size, (1, seq_len))
    print(f"  Exporting ONNX (batch=1, seq_len={seq_len}) ...")

    with timer("onnx_export", timings):
        # Use legacy TorchScript-based exporter (dynamo=False) for EZKL compatibility.
        # PyTorch 2.10's new dynamo exporter produces ONNX ops that EZKL/tract can't parse.
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

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ONNX file: {size_mb:.1f} MB")
    return True


# ---------------------------------------------------------------------------
# EZKL pipeline
# ---------------------------------------------------------------------------

def run_ezkl_pipeline(onnx_path, seq_len, vocab_size, work_dir, timings):
    import ezkl
    import torch
    import numpy as np

    settings_path = os.path.join(work_dir, "settings.json")
    cal_path = os.path.join(work_dir, "calibration.json")
    compiled_path = os.path.join(work_dir, "model.compiled")
    vk_path = os.path.join(work_dir, "vk.key")
    pk_path = os.path.join(work_dir, "pk.key")
    witness_path = os.path.join(work_dir, "witness.json")
    proof_path = os.path.join(work_dir, "proof.json")

    # Create calibration / input data
    input_data = np.random.randint(0, vocab_size, (1, seq_len)).tolist()
    with open(cal_path, "w") as f:
        json.dump({"input_data": input_data}, f)

    # --- gen_settings ---
    print("  4a. gen_settings ...")
    with timer("gen_settings", timings):
        ezkl.gen_settings(onnx_path, settings_path)

    # --- calibrate ---
    print("  4b. calibrate_settings ...")
    with timer("calibrate", timings):
        ezkl.calibrate_settings(
            cal_path, onnx_path, settings_path, target="resources"
        )

    # --- compile ---
    print("  4c. compile_circuit ...")
    with timer("compile", timings):
        ezkl.compile_circuit(onnx_path, compiled_path, settings_path)

    # --- setup (keygen) ---
    print("  4d. setup (key generation) ...")
    with timer("setup_keygen", timings):
        ezkl.setup(compiled_path, vk_path, pk_path)

    # --- witness ---
    print("  4e. gen_witness ...")
    with timer("gen_witness", timings):
        ezkl.gen_witness(cal_path, compiled_path, witness_path)

    # --- prove ---
    print("  4f. prove ...")
    with timer("prove", timings):
        ezkl.prove(witness_path, compiled_path, pk_path, proof_path)

    # --- verify ---
    print("  4g. verify ...")
    with timer("verify", timings):
        verified = ezkl.verify(proof_path, settings_path, vk_path)
    print(f"      Verified: {verified}")

    proof_size = os.path.getsize(proof_path) if os.path.exists(proof_path) else 0
    print(f"      Proof size: {proof_size / 1024:.1f} KB")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ZKML benchmark for parameter-golf")
    parser.add_argument(
        "--model",
        default="checkpoints/ckpt_final.pt",
        help="Path to model checkpoint (default: checkpoints/ckpt_final.pt)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Sequence length for ZKML proof (default: 8). Start small!",
    )
    parser.add_argument(
        "--work-dir",
        default="zkml_bench",
        help="Working directory for EZKL artifacts",
    )
    args = parser.parse_args()

    timings = {}

    # ---- Step 1: Load model ----
    print("=" * 60)
    print(f"ZKML Benchmark  (seq_len={args.seq_len})")
    print("=" * 60)
    print("\nStep 1: Load model")
    model, hp = load_model(args.model)

    # ---- Step 2: Inference wrapper ----
    print("\nStep 2: Create inference wrapper")
    inf_model = make_inference_model(model)
    inf_model.eval()

    # Quick sanity check
    import torch
    dummy = torch.randint(0, hp.vocab_size, (1, args.seq_len))
    with torch.no_grad():
        logits = inf_model(dummy)
    print(f"  Sanity check: input {dummy.shape} -> logits {logits.shape}")

    # ---- Step 3: ONNX export ----
    os.makedirs(args.work_dir, exist_ok=True)
    onnx_path = os.path.join(args.work_dir, "model.onnx")

    print("\nStep 3: ONNX export")
    try:
        export_onnx(inf_model, args.seq_len, hp.vocab_size, onnx_path, timings)
    except Exception as e:
        print(f"  ONNX export FAILED: {e}")
        import traceback; traceback.print_exc()
        _save_results(args, timings, success=False, error=f"onnx_export: {e}")
        return

    # ---- Step 4: EZKL pipeline ----
    print("\nStep 4: EZKL pipeline")
    try:
        run_ezkl_pipeline(
            onnx_path, args.seq_len, hp.vocab_size, args.work_dir, timings
        )
    except Exception as e:
        print(f"\n  EZKL pipeline FAILED at some step: {e}")
        import traceback; traceback.print_exc()
        _save_results(args, timings, success=False, error=f"ezkl: {e}")
        return

    # ---- Summary ----
    _save_results(args, timings, success=True)


def _save_results(args, timings, success, error=None):
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for k, v in timings.items():
        print(f"  {k:20s}  {v:8.2f}s")
    total = sum(timings.values())
    print(f"  {'TOTAL':20s}  {total:8.2f}s")

    if "prove" in timings:
        t = timings["prove"]
        print(f"\n  Extrapolations (based on prove={t:.1f}s per sample):")
        print(f"    50 samples (final eval):        {t * 50:8.1f}s = {t * 50 / 60:.1f} min")
        print(f"    5 samples x 10 checkpoints:     {t * 50:8.1f}s = {t * 50 / 60:.1f} min")
        print(f"    Total verification overhead:     {t * 100:8.1f}s = {t * 100 / 60:.1f} min")

    results = {
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
