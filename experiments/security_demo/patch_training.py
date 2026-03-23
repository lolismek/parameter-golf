#!/usr/bin/env python3
"""
Patch train_gpt.py to add:
  - Checkpoint saving every 60 seconds (including initial weights)
  - Per-step loss logging to loss_log.json

Usage:
    python experiments/security_demo/patch_training.py
    # Writes train_gpt_patched.py in the repo root
"""

import sys


def patch(src_path: str, dst_path: str) -> None:
    with open(src_path) as f:
        lines = f.readlines()

    result = []
    for i, line in enumerate(lines):
        result.append(line)

        # --- Insertion point 1: after "stop_after_step: int | None = None" ---
        # Add checkpoint directory setup, initial checkpoint save, loss log init
        if "stop_after_step: int | None = None" in line:
            result.append(
                "    # --- PATCH: checkpoint + loss logging setup ---\n"
                "    if master_process:\n"
                "        os.makedirs('checkpoints', exist_ok=True)\n"
                "    next_ckpt_ms = 0.0\n"
                "    loss_log = []\n"
                "    # Save initial checkpoint (before any training)\n"
                "    if master_process:\n"
                "        torch.save(base_model.state_dict(), 'checkpoints/ckpt_000s.pt')\n"
                "        log0('CHECKPOINT: checkpoints/ckpt_000s.pt (initial weights)')\n"
                "        next_ckpt_ms = 60000.0\n"
            )

        # --- Insertion point 2: after approx_training_time_ms in the training loop ---
        # (This line appears once in the main loop, around line 1037)
        if (
            "approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)"
            in line
            and i > 900  # only match the one in the training loop, not warmup
        ):
            result.append(
                "        # --- PATCH: log loss + save checkpoints every 60s ---\n"
                "        if master_process:\n"
                "            loss_log.append({\n"
                "                'step': step,\n"
                "                'train_loss': float(train_loss.item()),\n"
                "                'time_ms': float(approx_training_time_ms),\n"
                "            })\n"
                "            if approx_training_time_ms >= next_ckpt_ms:\n"
                "                _ckpt_s = int(next_ckpt_ms // 1000)\n"
                "                _ckpt_path = f'checkpoints/ckpt_{_ckpt_s:03d}s.pt'\n"
                "                torch.save(base_model.state_dict(), _ckpt_path)\n"
                "                log0(f'CHECKPOINT: {_ckpt_path} at step {step} ({approx_training_time_ms:.0f}ms)')\n"
                "                next_ckpt_ms += 60000.0\n"
            )

        # --- Insertion point 3: before serialization section ---
        # Save final checkpoint + dump loss log to JSON
        if "# SERIALIZATION + ROUNDTRIP VALIDATION" in line:
            result.append(
                "\n"
                "    # --- PATCH: save final checkpoint + loss log ---\n"
                "    if master_process:\n"
                "        torch.save(base_model.state_dict(), 'checkpoints/ckpt_final.pt')\n"
                "        log0('CHECKPOINT: checkpoints/ckpt_final.pt (post-training weights)')\n"
                "        import json as _json\n"
                "        with open('loss_log.json', 'w') as _f:\n"
                "            _json.dump(loss_log, _f, indent=2)\n"
                "        log0(f'Loss log saved: {len(loss_log)} entries to loss_log.json')\n"
                "\n"
            )

    with open(dst_path, "w") as f:
        f.writelines(result)
    print(f"Patched: {src_path} -> {dst_path}")


if __name__ == "__main__":
    src = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py"
    )
    dst = sys.argv[2] if len(sys.argv) > 2 else "train_gpt_patched.py"
    patch(src, dst)
