# arg: port, default: 30001
port=${1:-30001}
# python3 /sgl-workspace/sglang/test_scripts/prefill_warmup_matrix.py \
#   --port $port \
#   --contexts 16,32,64,128,256,384,512,768,896,1024 \
#   --batches 1,2,4,8 \
#   --output-len 2 \
#   --random-range-ratio 1.0 \
#   --bench-warmup-requests 0 \
#   --end-marker "prefill_warmup_end" \
#   --stream-groups 0,1,2,3,4,5,6

# python3 /sgl-workspace/sglang/test_scripts/prefill_warmup_matrix.py \
#   --port $port \
#   --contexts 16 \
#   --batches 1,2,4 \
#   --output-len 2 \
#   --random-range-ratio 1.0 \
#   --bench-warmup-requests 0 \
#   --end-marker "prefill_warmup_end" \
#   --stream-groups 0,1

# 1-25
# for ((i=1;i<=25;i++)); do
#     curl http://localhost:$port/flush_cache
#     python3 -m sglang.bench_serving --backend sglang --port $port --num-prompts 500 --request-rate $i
# done

# 1-40
out_it_times=10
stride=1    # 可修改为任意正整数


for ((j=1; j<=out_it_times; j++)); do
    for ((offset=1; offset<=stride; offset++)); do
        for ((i=offset; i<=40; i+=stride)); do
            curl http://localhost:$port/flush_cache
            python3 -m sglang.bench_serving --backend sglang --port $port --num-prompts 500 --request-rate $i
        done
    done
done

# python3 -m sglang.bench_serving --backend sglang --port 31001 --num-prompts 500 --request-rate 10

for ((i=1;i<=15;i++)); do
    rate=$(awk "BEGIN {printf \"%.2f\", $i * 0.04 + 0.04}")
    curl http://localhost:$port/flush_cache
    python /sgl-workspace/sglang/benchmark/pdmux/bench_serving.py --dataset-name loogle --num-prompts 20 --model /sgl-workspace/data/DeepSeek-V2-Lite --backend sglang --request-rate $rate --port $port --output-file loogle_pdmux_bench_serving.json
done