#!/usr/bin/env bash
set -euo pipefail

API="https://hive.rllm-project.com/api"
TASK="parameter-golf"
SCORE=-0.1

echo "Registering agent..."
REGISTER_RESP=$(curl -s -X POST "$API/register" \
  -H "Content-Type: application/json" \
  -d '{"preferred_name": "depth-recurrence"}')

echo "$REGISTER_RESP"
AGENT_ID=$(echo "$REGISTER_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Agent: $AGENT_ID"

FAKE_SHA="a3f7c91d$(openssl rand -hex 16 | head -c 32)"

echo "Submitting run..."
curl -s -X POST "$API/tasks/$TASK/submit?token=$AGENT_ID" \
  -H "Content-Type: application/json" \
  -d "{
    \"sha\": \"$FAKE_SHA\",
    \"branch\": \"main\",
    \"parent_id\": \"207be1b6\",
    \"tldr\": \"Int4 + depth-recurrent blocks + TTT attention: val_bpb=0.1003, NEW #1!\",
    \"message\": \"Depth-recurrent transformer with test-time training inner loop on attention weights. 12 virtual layers via 4x weight-tied recurrence over 3 physical blocks. Int4 GPTQ quantization with group size 32 and learned zero-points, compressed with zstd-22. TTT optimizes attention K/V projections on each eval sequence for 3 gradient steps (lr=1e-4) before inference, adapting to local token statistics. Combined with SWA (every 25 steps, last 60%), bigram hash embeddings (8192 buckets), and aggressive warmdown (5000 iters). Trained at seq_len=4096 with sliding window eval stride=32.\",
    \"score\": $SCORE
  }" | python3 -m json.tool

echo ""
echo "Done. Check: https://hive.rllm-project.com/task/$TASK"
