"""One-shot boot banner for the gfx908 (MI100) serving stack.

Printed once by the model runner after the model is loaded, so the log says
which gfx908 kernel paths are actually active instead of 48 per-layer lines.
Every probe is wrapped: a module that is not importable simply reports "n/a".
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)


def _try(fn, default="n/a"):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - banner must never break a boot
        return default


def _onoff(v) -> str:
    if v == "n/a":
        return v
    return "on" if v else "off"


def gfx908_boot_summary(model=None) -> None:
    try:
        from vllm.platforms.rocm import on_gfx908

        if not on_gfx908():
            return
    except Exception:  # noqa: BLE001
        return

    from vllm.model_executor.layers.fused_moe import gfx908_w4a8 as w4a8

    rows: list[tuple[str, str]] = []

    def w4a8_line():
        mode = w4a8.w4a8_mode()
        fold = w4a8.prep_fold_enabled()
        shared = w4a8.shared_as_expert_enabled()
        return f"on ({mode} dot, prep fold {'on' if fold else 'off'}, shared expert {'folded' if shared else 'separate'})"

    rows.append(("MoE decode GEMV  W4A8", _try(lambda: w4a8_line() if w4a8.w4a8_enabled() else "off")))

    def mr_line():
        from vllm.model_executor.layers.fused_moe import gfx908_moe_hip as hip

        top = getattr(hip, "MOE_HIP_MAX_TOKENS", "?")
        if w4a8.moe_mr_enabled():
            return f"on (rows {w4a8.MOE_MR_MIN_M + 1}..{top}; HIP path to {top} rows, Triton above)"
        return f"off (HIP path to {top} rows)"

    rows.append(("MoE multi-row kernel", _try(mr_line)))
    rows.append(("MoE prefill GEMM   fp16", _onoff(os.environ.get("VLLM_GFX908_MOE_FP16_COMPUTE", "1") == "1")))

    def w8_line():
        from vllm.model_executor.layers import gfx908_w8a16 as w8

        if not w8.w8a16_enabled():
            return "off"
        mf = w8.w8a16_mfma_enabled()
        return f"on (gs{os.environ.get('VLLM_GFX908_W8A16_GS', '128')}; MFMA 5..64 rows {'on, M=1 swizzled GEMV' if mf else 'off'})"

    rows.append(("GDN/lm_head int8   W8A16", _try(w8_line)))

    def gdn_line():
        from vllm.model_executor.layers.mamba.gdn import gfx908_gdn_fused as g

        n = getattr(g, "ENABLED_LAYERS", 0)
        spec = os.environ.get("VLLM_GFX908_GDN_FUSED_SPEC", "0") == "1"
        return f"on ({n} layers{', spec branch' if spec else ''})" if g.gdn_fused_enabled() else "off"

    rows.append(("GDN fused decode glue", _try(gdn_line)))

    def qsa_line():
        from vllm.models.qwen4_exp.amd import gfx908_qsa_glue as q

        if not q.qsa_glue_enabled():
            return "off"
        n = getattr(q, "ENABLED_LAYERS", 0)
        return f"on ({n} layers, up to {q.qsa_glue_max_q()} query tokens per request)"

    rows.append(("QSA fused decode glue", _try(qsa_line)))

    def hc_line():
        from vllm.models.qwen4_exp.amd import gfx908_hc_fused as h

        if not h.hc_fused_enabled():
            return "off"
        w8 = h.hc_w8_enabled()
        return f"on (rows <= {h.hc_fused_max_m()}; {'W8 mixes' if w8 else 'bf16 mixes'})"

    rows.append(("HC fused mix chain", _try(hc_line)))

    def ar_line():
        from vllm.distributed.device_communicators import gfx908_push_ar as p

        if p.push_ar_requested():
            fused = os.environ.get("VLLM_GFX908_HC_AR_FUSED", "0") == "1"
            return "custom push all-reduce (xGMI sentinel)" + (", consumer fused into HC combine" if fused else "")
        return "custom one-shot all-reduce"

    rows.append(("TP all-reduce", _try(ar_line)))
    rows.append(("Logits gather", "custom xGMI all-gather" if os.environ.get("VLLM_GFX908_CUSTOM_AG", "0") == "1" else "RCCL"))
    rows.append(("Logits in decode graph", _onoff(os.environ.get("VLLM_GFX908_LOGITS_IN_GRAPH", "0") == "1")))
    rows.append(("Router", "fused GEMV+softmax+top-k" if os.environ.get("VLLM_GFX908_ROUTER_FUSED", "1") == "1" else "stock"))
    rows.append(("Sampler top-k/top-p", "radix fast path (<=64)" if os.environ.get("VLLM_GFX908_SAMPLER_FASTK", "1") == "1" else "stock"))
    rows.append(("PLE embeddings", "zero-copy pinned host gather" if os.environ.get("VLLM_PLE_ZEROCOPY", "1") == "1" else "device"))
    rows.append(("PLE fused decode glue", _onoff(os.environ.get("VLLM_GFX908_PLE_GLUE", "0") == "1")))
    rows.append(("Stable QSA top-k", _onoff(os.environ.get("VLLM_GFX908_QSA_STABLE_TOPK", "1") == "1")))
    rows.append(("Extension loader", "strict" if os.environ.get("VLLM_GFX908_STRICT_EXT", "1") == "1" else "lenient"))

    width = max(len(k) for k, _ in rows)
    lines = ["gfx908 (MI100) serving paths:"] + [f"  {k.ljust(width)}  {v}" for k, v in rows]
    logger.info("\n".join(lines))
