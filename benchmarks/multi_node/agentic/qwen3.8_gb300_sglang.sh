#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for Qwen3.8-2.4T FP8/NVFP4 on GB300, with capacity
# scaled for each AgentX point.

source "$(dirname "$0")/../../benchmark_lib.sh"

# Loading 2.4T weights from shared storage can skew TP ranks beyond SGLang's
# fixed 480-second guard. Change only the ephemeral container overlay.
SGLANG_LOAD_MODEL_UTILS=/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py
sed -i 's/^UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480 /UNBALANCED_MODEL_LOADING_TIMEOUT_S = 3600 /' "$SGLANG_LOAD_MODEL_UTILS"
grep -q '^UNBALANCED_MODEL_LOADING_TIMEOUT_S = 3600 ' "$SGLANG_LOAD_MODEL_UTILS"

check_env_vars \
    MODEL MODEL_PATH PRECISION TP NNODES DIST_INIT_ADDR CONC DP_SIZE EP_SIZE \
    DP_ATTENTION \
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
CHUNKED_PREFILL_SIZE=8192
MAX_PREFILL_TOKENS=8192
MAMBA_RADIX_CACHE_STRATEGY=extra_buffer
LOAD_ARGS=(
    --load-format runai_streamer
    --model-loader-extra-config '{"distributed":true}'
)

case "$PRECISION:$TP:$NNODES:$DP_SIZE:$EP_SIZE:$DP_ATTENTION" in
    fp8:16:4:1:1:false)
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
    fp4:8:2:1:1:false)
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
    fp8:16:4:4:16:true)
        SGLANG_DEEPEP_V2=/sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py
        SGLANG_MOE_HOOK=/sgl-workspace/sglang/python/sglang/srt/arg_groups/moe_hook.py
        patch --silent --batch -d /sgl-workspace/sglang/python -p0 <<'PATCH'
--- sglang/srt/layers/moe/token_dispatcher/deepep_v2.py
+++ sglang/srt/layers/moe/token_dispatcher/deepep_v2.py
@@ -237,0 +238,13 @@
+        masked_cap = int(
+            os.environ.get(
+                "SGLANG_DEEPEP_V2_MASKED_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
+                str(num_max_dispatch_tokens_per_rank),
+            )
+        )
+        if masked_cap < 1 or masked_cap > num_max_dispatch_tokens_per_rank:
+            raise ValueError(
+                "SGLANG_DEEPEP_V2_MASKED_NUM_MAX_DISPATCH_TOKENS_PER_RANK "
+                f"must be in [1, {num_max_dispatch_tokens_per_rank}], got "
+                f"{masked_cap}"
+            )
+        self.masked_num_max_dispatch_tokens_per_rank = masked_cap
@@ -293,3 +306,16 @@
-        # Decode uses expanded/masked layout; extend uses contiguous in both modes.
-        use_expand_layout = not get_is_extend_in_batch()
-        use_masked = use_expand_layout
+        force_expand_prefill = (
+            os.environ.get("SGLANG_DEEPEP_V2_EXPAND_PREFILL") == "1"
+        )
+        use_expand_layout = force_expand_prefill or not get_is_extend_in_batch()
+        use_masked = use_expand_layout and not get_is_extend_in_batch()
+        if (
+            use_masked
+            and hidden_states.shape[0]
+            > self.masked_num_max_dispatch_tokens_per_rank
+        ):
+            raise ValueError(
+                "DeepEP v2 masked decode input exceeds the per-rank slab "
+                f"capacity {self.masked_num_max_dispatch_tokens_per_rank}, got "
+                f"{hidden_states.shape[0]}. Increase "
+                "SGLANG_DEEPEP_V2_MASKED_NUM_MAX_DISPATCH_TOKENS_PER_RANK."
+            )
@@ -298 +324,3 @@
-        self._pad_empty_combine = (not use_masked) and hidden_states.shape[0] == 0
+        self._pad_empty_combine = (
+            not use_expand_layout and hidden_states.shape[0] == 0
+        )
@@ -385 +413,3 @@
-            masked_max_m = self.num_max_dispatch_tokens_per_rank * ep_group_size
+            masked_max_m = (
+                self.masked_num_max_dispatch_tokens_per_rank * ep_group_size
+            )
--- sglang/srt/arg_groups/moe_hook.py
+++ sglang/srt/arg_groups/moe_hook.py
@@ -28 +28 @@
-from sglang.srt.runtime_context import get_platform
+from sglang.srt.runtime_context import derive_attention_widths, get_platform
@@ -552,0 +553,8 @@
+        attn_dp_size, _ = derive_attention_widths(
+            tp_size=view.tp_size,
+            attn_cp_size=view.attn_cp_size,
+            dp_size=view.dp_size,
+            enable_dp_attention=view.enable_dp_attention,
+        )
+        scatter_ranks = max(1, view.tp_size // attn_dp_size)
+        prefill_tokens = -(-prefill_tokens // scatter_ranks)
@@ -612,0 +621 @@
+        "Qwen3_5MoeForCausalLM",
PATCH
        python3 -m py_compile "$SGLANG_DEEPEP_V2" "$SGLANG_MOE_HOOK"
        export SGLANG_DEEPEP_V2_EXPAND_PREFILL=1
        export SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048
        export SGLANG_DEEPEP_V2_MASKED_NUM_MAX_DISPATCH_TOKENS_PER_RANK=384
        export EP_DISABLE_GIN=1
        CHUNKED_PREFILL_SIZE=32768
        MAX_MAMBA_CACHE_SIZE=$((5 * MAX_RUNNING_REQUESTS))
        CUDA_GRAPH_MAX_BS=$((MAX_RUNNING_REQUESTS / DP_SIZE))
        PRECISION_ARGS=(
            --kv-cache-dtype fp8_e4m3
            --linear-attn-prefill-backend flashinfer
            --moe-a2a-backend deepep_v2
            --deepep-v2-mode hybrid
            --enable-eplb
            --mem-fraction-static 0.93
            --max-total-tokens 2800000
            --max-running-requests "$MAX_RUNNING_REQUESTS"
            --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
            --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
        )
        ;;
    fp4:8:2:8:8:true)
        # Preserve one request slot per AgentX lane and a full 256K context in
        # the memory-constrained 8-GPU configuration.
        MAX_RUNNING_REQUESTS=$CONC
        MAX_MAMBA_CACHE_SIZE=$((5 * MAX_RUNNING_REQUESTS))
        CHUNKED_PREFILL_SIZE=16384
        MAX_PREFILL_TOKENS=2048
        LOAD_ARGS=(
            --load-format auto
            --model-loader-extra-config '{"enable_multithread_load":false}'
            --weight-loader-drop-cache-after-load
        )
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048
        export NCCL_MNNVL_ENABLE=1
        export NCCL_CUMEM_ENABLE=1
        export NCCL_NET_GDR_C2C=1
        PRECISION_ARGS=(
            --quantization modelopt_fp4
            --fp4-gemm-backend flashinfer_cutlass
            --kv-cache-dtype fp8_e4m3
            --linear-attn-prefill-backend flashinfer
            --moe-runner-backend flashinfer_trtllm_routed
            --moe-a2a-backend flashinfer
            --ep-dispatch-algorithm static
            --mem-fraction-static 0.98
            --max-total-tokens 262144
            --max-running-requests "$MAX_RUNNING_REQUESTS"
            --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
            --disable-cuda-graph
            --skip-server-warmup
        )
        ;;
    fp4:16:4:16:16:true)
        MAX_MAMBA_CACHE_SIZE=$((5 * MAX_RUNNING_REQUESTS))
        CUDA_GRAPH_MAX_BS=$((MAX_RUNNING_REQUESTS / DP_SIZE))
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8192
        export NCCL_MNNVL_ENABLE=1
        export NCCL_CUMEM_ENABLE=1
        export NCCL_NET_GDR_C2C=1
        CHUNKED_PREFILL_SIZE=131072
        PRECISION_ARGS=(
            --quantization modelopt_fp4
            --fp4-gemm-backend flashinfer_cutlass
            --kv-cache-dtype fp8_e4m3
            --linear-attn-prefill-backend flashinfer
            --moe-runner-backend flashinfer_trtllm_routed
            --moe-a2a-backend flashinfer
            --ep-dispatch-algorithm static
            --mem-fraction-static 0.95
            --max-total-tokens 1048576
            --max-running-requests "$MAX_RUNNING_REQUESTS"
            --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
            --disable-prefill-cuda-graph
            --cuda-graph-backend-decode full
            --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
            --skip-server-warmup
        )
        ;;
    *)
        echo "Error: unsupported parallelism" >&2
        exit 1
        ;;
esac

PARALLEL_ARGS=(--tp-size "$TP" --ep-size "$EP_SIZE")
if [[ "$DP_ATTENTION" == true ]]; then
    PARALLEL_ARGS+=(
        --dp-size "$DP_SIZE"
        --enable-dp-attention
        --enable-dp-lm-head
        --enable-dp-attention-local-control-broadcast
        --moe-dense-tp-size 1
    )
fi

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
USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
ROUTER_PID=""
if [[ "$DP_ATTENTION" == true ]]; then
    USE_SGLANG_ROUTER=true
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    ROUTER_LOG="$RESULT_DIR/router.log"
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    "${LOAD_ARGS[@]}"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --nnodes "$NNODES"
    --node-rank "$NODE_RANK"
    --dist-init-addr "$DIST_INIT_ADDR"
    --mamba-ssm-dtype bfloat16
    --mamba-radix-cache-strategy "$MAMBA_RADIX_CACHE_STRATEGY"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
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
        if [[ -n "$ROUTER_PID" ]]; then
            stop_background_process_tree "$ROUTER_PID" "SGLang router" 30
        fi
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
wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$USE_SGLANG_ROUTER" == true ]]; then
    python3 -m sglang_router.launch_router \
        --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
        --policy consistent_hashing \
        --request-id-headers x-correlation-id \
        --dp-aware \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$SGLANG_ROUTER_METRICS_PORT" \
        --connect-timeout-secs 900 \
        --request-timeout-secs 14400 \
        --disable-health-check \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"

    # /health becomes ready before the DP workers do. Match srtctl's
    # aggregated SGLang readiness rule before sending warmup traffic.
    set +x
    router_workers_deadline=$((SECONDS + 900))
    until curl --silent --fail "http://localhost:$PORT/workers" \
        | python3 -c 'import json, sys; stats = json.load(sys.stdin)["stats"]; sys.exit(stats.get("decode_count", 0) + stats.get("regular_count", 0) < int(sys.argv[1]))' "$DP_SIZE" 2>/dev/null; do
        if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
            echo "SGLang router died before its DP workers became ready." >&2
            exit 1
        fi
        if [[ "$SECONDS" -ge "$router_workers_deadline" ]]; then
            echo "Timed out waiting for $DP_SIZE SGLang DP workers." >&2
            exit 1
        fi
        sleep 2
    done
    set -x
fi

build_replay_cmd "$RESULT_DIR"
REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
