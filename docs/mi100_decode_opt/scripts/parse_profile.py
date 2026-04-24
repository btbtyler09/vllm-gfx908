#!/usr/bin/env python3
"""Parse vLLM/torch profiler traces into per-bucket decode time breakdowns.

Usage: parse_profile.py TRACE_JSON [TRACE_JSON...]
"""
import json, sys, gzip
from collections import defaultdict

# Buckets evaluated in order — first match wins.
BUCKETS = [
    # --- Exclude top-level markers ---
    ("__exclude_marker__", ["execute_context_", "vllm::unified_kv_cache_update"]),

    # --- linear attention (DeltaNet / gated delta rule) — Qwen3.6 hybrid arch ---
    ("linear-attn",  ["chunk_gated_delta_rule", "chunk_fwd_kernel_o", "recompute_w_u", "fused_recurrent_gated_delta_rule",
                      "chunk_scaled_dot_kkt", "merge_16x16_to_64x64", "_causal_conv1d", "ChunkGatedDeltaRule", "gdn_attention"]),

    # --- transformer attention (unified attention kernel) ---
    ("attention",    ["unified_attention", "paged_attention", "flash_attn", "triton_attn",
                      "_fwd_kernel_stage", "attention_2d"]),

    # --- all-reduce / collectives ---
    ("all-reduce",   ["ncclDevKernel", "ncclAllReduce", "rccl", "one_shot_", "two_shot_", "custom_ar"]),

    # --- MoE kernels (routing + fused GEMM) ---
    ("moe-gemm",     ["fused_moe_kernel", "moe_stage1", "fmoe_stage1", "moe_gemm_stage1",
                      "moe_stage2", "fmoe_stage2", "moe_gemm_stage2"]),
    ("moe-routing",  ["topkGating", "moe_align_block_size", "moe_sorting", "count_and_sort_expert",
                      "moe_permute", "moe_unpermute", "act_and_mul_kernel", "moe_reduce"]),

    # --- linear projections (rocBLAS Tensile kernels — QKV, O, router, shared experts) ---
    ("linear-rocblas", ["Cijk_", "PostGSU"]),

    # --- norms / activations / rope ---
    ("norm",         ["rms_norm", "rmsnorm", "layer_norm", "layernorm", "fused_norm"]),
    ("rope",         ["rope", "rotary", "reshape_and_cache", "_and_cache"]),

    # --- sampler ---
    ("sampler",      ["_topk", "top_k", "top_p", "sample", "categorical", "sampler",
                      "SoftMax", "_softmax"]),

    # --- memory ops ---
    ("memcpy",       ["Memcpy", "memcpy", "memset"]),

    # --- small triton fused (catch-all) ---
    ("triton-misc",  ["triton_poi_", "triton_red_", "triton_per_"]),

    # --- pytorch elementwise ops ---
    ("elementwise",  ["elementwise_kernel", "vectorized_elementwise", "bitwise", "masked_fill",
                      "aten::", "index_elementwise"]),
]

def bucket_for(name):
    for bname, subs in BUCKETS:
        for s in subs:
            if s in name:
                return bname
    return "other"

def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt"
    with opener(path, mode) as f:
        return json.load(f)

def collect_kernels(trace):
    """Yield (name, dur_us) for GPU kernel events only."""
    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("ph") != "X":
            continue
        cat = (ev.get("cat") or "").lower()
        # Only GPU kernel events. Exclude user_annotation explicitly — that's execute_context markers.
        if cat not in ("kernel", "gpu_op", "gpu_memcpy", "gpu_memset"):
            continue
        dur = ev.get("dur")
        if dur is None:
            continue
        name = ev.get("name") or ""
        yield name, float(dur)

def main():
    if len(sys.argv) < 2:
        print("usage: parse_profile.py TRACE_JSON [TRACE_JSON...]", file=sys.stderr)
        sys.exit(2)

    per_bucket = defaultdict(float)   # us
    per_kernel = defaultdict(float)   # us
    kernel_calls = defaultdict(int)
    total = 0.0
    excluded = 0.0
    events_seen = 0

    for path in sys.argv[1:]:
        try:
            trace = load(path)
        except Exception as e:
            print(f"# skip {path}: {e}", file=sys.stderr)
            continue
        for name, dur in collect_kernels(trace):
            events_seen += 1
            b = bucket_for(name)
            if b == "__exclude_marker__":
                excluded += dur
                continue
            per_bucket[b] += dur
            per_kernel[name] += dur
            kernel_calls[name] += 1
            total += dur

    if events_seen == 0:
        print("# No GPU kernel events found. Are these torch profiler traces?", file=sys.stderr)
        sys.exit(3)

    print(f"# Files loaded: {len(sys.argv)-1}, GPU kernel events: {events_seen}, excluded (execute_context markers): {excluded/1000:.2f} ms")
    print(f"# Total GPU kernel time (bucketed): {total/1000:.2f} ms")
    print()
    print("## Per-bucket breakdown")
    print()
    print("| Bucket | GPU ms | % of total |")
    print("|---|---:|---:|")
    bucket_order = [b for b, _ in BUCKETS if b != "__exclude_marker__"] + ["other"]
    for bname in bucket_order:
        us = per_bucket.get(bname, 0.0)
        pct = (us / total * 100.0) if total > 0 else 0.0
        if us > 0:
            print(f"| {bname} | {us/1000:.2f} | {pct:.1f}% |")
    print(f"| **TOTAL** | **{total/1000:.2f}** | **100.0%** |")
    print()
    print("## Top 30 kernels by total GPU time")
    print()
    print("| Kernel | Bucket | Calls | Total ms | Avg us |")
    print("|---|---|---:|---:|---:|")
    top = sorted(per_kernel.items(), key=lambda kv: kv[1], reverse=True)[:30]
    for name, us in top:
        calls = kernel_calls[name]
        avg = us / calls if calls else 0
        b = bucket_for(name)
        disp = name if len(name) <= 95 else name[:92] + "..."
        disp = disp.replace("|", "\\|")
        print(f"| `{disp}` | {b} | {calls} | {us/1000:.2f} | {avg:.1f} |")

if __name__ == "__main__":
    main()
