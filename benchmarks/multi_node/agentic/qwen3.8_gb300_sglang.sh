#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for Qwen3.8-2.4T FP8/NVFP4 on the verified GB300
# low-latency SGLang topologies, with capacity scaled for each AgentX point.

source "$(dirname "$0")/../../benchmark_lib.sh"

# Loading 2.4T weights from shared storage can skew TP ranks beyond SGLang's
# fixed 480-second guard. Change only the ephemeral container overlay.
SGLANG_LOAD_MODEL_UTILS=/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py
sed -i 's/^UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480 /UNBALANCED_MODEL_LOADING_TIMEOUT_S = 3600 /' "$SGLANG_LOAD_MODEL_UTILS"
grep -q '^UNBALANCED_MODEL_LOADING_TIMEOUT_S = 3600 ' "$SGLANG_LOAD_MODEL_UTILS"

check_env_vars \
    MODEL MODEL_PATH PRECISION TP NNODES DIST_INIT_ADDR CONC EP_SIZE \
    KV_OFFLOADING SPEC_DECODING ACCEPTANCE_MODE RESULT_DIR DURATION
[[ "$ACCEPTANCE_MODE" == natural ]]
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    RANKS_PER_NODE=$((TP / NNODES))
    HICACHE_ALIGNMENT_RESERVE_GB=$RANKS_PER_NODE
    HICACHE_USABLE_TOTAL_GB=$((TOTAL_CPU_DRAM_GB - HICACHE_ALIGNMENT_RESERVE_GB))
    MAX_HICACHE_SIZE_GB=$((HICACHE_USABLE_TOTAL_GB * 15 / RANKS_PER_NODE / 31))
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -lt 1 ] || [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB outside 1..$MAX_HICACHE_SIZE_GB" >&2
        exit 1
    fi
    CACHE_ARGS=(
        --page-size 64
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-io-backend kernel
        --hicache-mem-layout page_first
        --hicache-write-policy write_through_selective
    )
fi

NODE_RANK="${SLURM_PROCID:?Launch one Slurm task per node}"
[[ "$SLURM_NTASKS" == "$NNODES" ]]

# AgentX concurrency counts live session trees rather than individual HTTP
# requests. Leave room for subagent fan-out.
MAX_RUNNING_REQUESTS=$((2 * CONC))

case "$PRECISION:$TP:$NNODES" in
    fp8:16:4)
        MAX_MAMBA_CACHE_SIZE=$((5 * MAX_RUNNING_REQUESTS))
        CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
        [ "$CUDA_GRAPH_MAX_BS" -gt 96 ] && CUDA_GRAPH_MAX_BS=96
        export SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1
        export NCCL_NVLS_ENABLE=1
        PRECISION_ARGS=(
            --kv-cache-dtype fp8_e4m3
            --attention-backend trtllm_mha
            --moe-runner-backend flashinfer_trtllm
            --mem-fraction-static 0.90
            --max-running-requests "$MAX_RUNNING_REQUESTS"
            --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
            --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
        )
        ;;
    fp4:8:2)
        export SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1
        export NCCL_MNNVL_ENABLE=1
        export NCCL_CUMEM_ENABLE=1
        export NCCL_NVLS_ENABLE=1
        PRECISION_ARGS=(
            --quantization modelopt_fp4
            --fp4-gemm-backend flashinfer_cutlass
            --kv-cache-dtype fp8_e4m3
            --attention-backend trtllm_mha
            --linear-attn-prefill-backend flashinfer
            --moe-runner-backend flashinfer_trtllm
            --mem-fraction-static 0.90
            --max-running-requests "$MAX_RUNNING_REQUESTS"
            --cuda-graph-backend-prefill breakable
            --cuda-graph-backend-decode full
        )
        ;;
    *)
        echo "Error: unsupported PRECISION:TP:NNODES=$PRECISION:$TP:$NNODES" >&2
        exit 1
        ;;
esac

SPEC_ARGS=()
case "$SPEC_DECODING" in
    mtp)
        SPEC_ARGS=(
            --speculative-algorithm NEXTN
            --speculative-num-steps 3
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 4
            --enable-linear-replayssm-spec
        )
        ;;
    none) ;;
    *) echo "Error: unsupported SPEC_DECODING=$SPEC_DECODING" >&2; exit 1 ;;
esac

export PYTHONNOUSERSITE=1
export SGLANG_TIMEOUT_KEEP_ALIVE=1800
export RUNAI_STREAMER_DIST_GLOBAL=1

SERVER_LOG="$RESULT_DIR/server-node${NODE_RANK}.log"
STOP_FILE="$RESULT_DIR/server.stop"
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --load-format runai_streamer
    --model-loader-extra-config '{"distributed":true}'
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp-size "$TP"
    --nnodes "$NNODES"
    --node-rank "$NODE_RANK"
    --dist-init-addr "$DIST_INIT_ADDR"
    --mamba-ssm-dtype bfloat16
    --mamba-radix-cache-strategy extra_buffer
    --chunked-prefill-size 8192
    --max-prefill-tokens 8192
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --enable-metrics
    --enable-cache-report
    "${CACHE_ARGS[@]}"
    "${PRECISION_ARGS[@]}"
    "${SPEC_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" > "$RESULT_DIR/sglang_command.node${NODE_RANK}.txt"
printf '\n' >> "$RESULT_DIR/sglang_command.node${NODE_RANK}.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if [[ "$NODE_RANK" == 0 ]]; then
        touch "$STOP_FILE"
    fi
    stop_background_process_tree "$SERVER_PID" "SGLang node $NODE_RANK" 30
}
trap cleanup EXIT TERM INT

if [[ "$NODE_RANK" != 0 ]]; then
    while [[ ! -e "$STOP_FILE" ]]; do
        if ! _background_process_is_running "$SERVER_PID"; then
            set +e
            wait "$SERVER_PID"
            server_rc=$?
            set -e
            echo "Error: SGLang node $NODE_RANK exited before rank 0 finished (rc=$server_rc)" >&2
            exit 1
        fi
        sleep 2
    done
    exit 0
fi

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

build_replay_cmd "$RESULT_DIR"
REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
