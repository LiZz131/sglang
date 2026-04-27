#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEND_FEISHU="${SCRIPT_DIR}/send_feishu_message.py"

_notify_feishu_on_exit() {
  local code=$?
  if [[ -z "${FEISHU_WEBHOOK_URL:-}" ]]; then
    return 0
  fi
  if [[ "$code" -eq 0 ]]; then
    python3 "$SEND_FEISHU" "profile_for_nsys.sh：prefill 矩阵 + bench_serving 已全部完成 (exit 0)" \
      || true
  else
    python3 "$SEND_FEISHU" "profile_for_nsys.sh：中途异常退出 (exit ${code})，请检查日志与 nsys 采集。" \
      || true
  fi
}
trap _notify_feishu_on_exit EXIT

port=${1:-30001}
num_prompts=80
request_rate=15
contexts=16,32,64,128,256,384,512,768,896,1024
batches=1,2,4,8
stream_groups=0,1,2,3,4,5,6

# very simple test
num_prompts=200
request_rate=5
contexts=16
batches=6
stream_groups=1

# python3 /sgl-workspace/sglang/test_scripts/prefill_warmup_matrix.py \
#   --port $port \
#   --contexts $contexts \
#   --batches $batches \
#   --output-len 2 \
#   --random-range-ratio 1.0 \
#   --bench-warmup-requests 0 \
#   --end-marker "prefill_warmup_end" \
#   --stream-groups $stream_groups

# little warmup(flush cache)
warmup_times=5
for ((i=1; i<=warmup_times; i++)); do
  curl http://127.0.0.1:${port}/flush_cache
  python3 -m sglang.bench_serving --backend sglang --port $port --num-prompts $num_prompts --request-rate $request_rate
done

curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-SGLANG-NVTX-RANGE: prefill_warmup_end" \
  -d '{"text":"modeling finished","sampling_params":{"temperature":0,"max_new_tokens":2},"stream":false}' \
  http://127.0.0.1:${port}/generate  

# benchmark(flush cache)
curl http://127.0.0.1:${port}/flush_cache
python3 -m sglang.bench_serving --backend sglang --port $port --num-prompts $num_prompts --request-rate $request_rate

# --end-marker "modeling_end"
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-SGLANG-NVTX-RANGE: modeling_end" \
  -d '{"text":"modeling finished","sampling_params":{"temperature":0,"max_new_tokens":2},"stream":false}' \
  http://127.0.0.1:${port}/generate