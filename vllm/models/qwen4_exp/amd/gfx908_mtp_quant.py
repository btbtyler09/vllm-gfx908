# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908: load-time quantization of the Qwen4Exp MTP draft layer.

The GPTQ checkpoints of this family leave ``mtp.*`` in bf16 (4.97 GB: a 512-expert
MoE of 4.8 GB plus ~110 MB of dense projections).  ``VLLM_GFX908_MTP_QUANT``
selects what the draft is built from at load time (default ``off`` = stock bf16):

  ``w8``     dense draft projections (fc_embedding, fc_hidden, qkv_proj, o_proj,
             indexer.index_qk_proj, shared_expert.*) take the existing W8A16 int8
             group-128 side copy (``gfx908_w8a16.py``, per-layer opt-in).  Experts
             stay bf16.  Bytes only; no new kernels.
  ``w8moe``  ``w8`` + routed experts as int8 group-128 in the GPTQ-format layout the
             stock ``MoeWNA16`` method loads (Triton WNA16 kernel; no HIP GEMV,
             which is int4-only).  Mostly a VRAM lever (-0.6 GB/rank).
  ``w4``     dense as ``w8``; routed experts AND the shared expert as W4 group-32
             symmetric RTN in the exact GPTQ layout of the target's experts, so the
             draft MoE runs the same tuned path as the 48 target MoE layers (HIP
             W4A8 GEMV, shared-as-expert, fused router).  -0.95 GB/rank.  Plain RTN
             at 4 bits is ~11% relative error on these tensors (rtn_error_mtp.log)
             against ~1% for int8, so this mode must pass the acceptance gate
             (REPORT.md section 6) before it is defaulted.

The routed experts arrive as two fused bf16 tensors, ``experts.gate_up_proj``
[E, 2N, K] and ``experts.down_proj`` [E, K, N].  ``expand_fused_experts`` turns
them into per-expert GPTQ tensors (``experts.{e}.gate_proj.qweight`` ...) exactly
as gptqmodel writes them for the target (sym, ``checkpoint_format=gptq``:
``qzeros`` = 0x77777777, ``g_idx`` = arange(K)//gs), and the stock loaders do the
rest.  Quantization runs on the rank's GPU one expert at a time (~6.5 MB), so the
load-time cost is a few seconds per rank and the peak memory a few MB.
"""

import os

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_MODES = ("off", "w8", "w8moe", "w4")
_MODE: str | None = None


def mtp_quant_mode() -> str:
    """``VLLM_GFX908_MTP_QUANT`` normalised to one of ``_MODES`` (cached)."""
    global _MODE
    if _MODE is None:
        v = os.environ.get("VLLM_GFX908_MTP_QUANT", "off").strip().lower()
        if v in ("", "0", "none", "false"):
            v = "off"
        if v not in _MODES:
            logger.warning_once(
                "VLLM_GFX908_MTP_QUANT=%r is not one of %s; using 'off'", v, _MODES
            )
            v = "off"
        if v != "off":
            try:
                from vllm.platforms.rocm import on_gfx908

                if not on_gfx908():
                    logger.warning_once(
                        "VLLM_GFX908_MTP_QUANT=%s ignored: not running on gfx908", v
                    )
                    v = "off"
            except Exception:  # platform probe unavailable (CPU tests)
                pass
        _MODE = v
    return _MODE


def _reset_env_cache() -> None:  # tests
    global _MODE
    _MODE = None


def mtp_quant_dense_w8() -> bool:
    return mtp_quant_mode() in ("w8", "w8moe", "w4")


def mtp_quant_experts_bits() -> int:
    """0 = experts stay bf16; 4 / 8 = load-time RTN to that width."""
    return {"w4": 4, "w8moe": 8}.get(mtp_quant_mode(), 0)


def mtp_quant_shared_expert_w4() -> bool:
    return mtp_quant_mode() == "w4"


# Group sizes of the RTN experts.  W4 gs32 = the target checkpoint's layout (the
# HIP GEMV and the tuned MoE configs assume it); int8 gs128 = the W8A16 layout.
def mtp_quant_experts_gs(bits: int) -> int:
    return 32 if bits == 4 else int(os.environ.get("VLLM_GFX908_MTP_QUANT_GS8", "128"))


# ---------------------------------------------------------------------------
# RTN -> GPTQ-format packing
# ---------------------------------------------------------------------------
def rtn_gptq_pack(
    w: torch.Tensor, bits: int, gs: int, scales_dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """Symmetric RTN of a bf16/fp16 ``[N, K]`` weight to the GPTQ v1 checkpoint layout.

    Returns ``qweight`` int32 ``[K // pf, N]`` (``pf = 32 // bits`` values per word,
    value ``k = pf*j + i`` in bits ``[bits*i, bits*(i+1))``), ``scales``
    ``[K // gs, N]``, ``qzeros`` int32 ``[K // gs, N // pf]`` holding the GPTQ v1
    stored zero (``zp - 1``), and ``g_idx`` int32 ``[K]``.  Dequant is
    ``w ~= scale * (q - zp)`` with ``zp = 2**(bits-1)``, i.e. exactly what the
    MoeWNA16 / AutoGPTQ loaders and the gfx908 W4 kernels assume for ``sym=True``.

    4-bit scale rule ``q4_0``: the largest-magnitude element of each group lands
    exactly on ``-8`` (llama.cpp); measured 10-15% better than ``amax/7`` on every
    mtp.* tensor.  8-bit: ``amax/127``.
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    n, k = int(w.shape[0]), int(w.shape[1])
    pf = 32 // bits
    if k % gs or gs % 8 or k % pf:
        raise ValueError(f"rtn_gptq_pack: K={k} must be a multiple of gs={gs} and {pf}")
    if n % pf:
        raise ValueError(f"rtn_gptq_pack: N={n} must be a multiple of {pf} (qzeros packing)")
    zp = 1 << (bits - 1)
    qmax = (1 << bits) - 1
    wf = w.float().view(n, k // gs, gs)
    if bits == 4:
        amax = wf.abs().amax(-1)
        idx = wf.abs().argmax(-1, keepdim=True)
        sc = torch.gather(wf, -1, idx).squeeze(-1) / -8.0
        sc = torch.where(amax > 0, sc, torch.ones_like(sc))
        sc = torch.where(sc.abs() < 1e-12, torch.full_like(sc, 1e-12), sc)
    else:
        sc = wf.abs().amax(-1).clamp_min(1e-12) / 127.0
    q = (torch.round(wf / sc[..., None]) + zp).clamp_(0, qmax).to(torch.int32)
    q = q.view(n, k).t().contiguous()  # [K, N], K-major like GPTQ
    q = q.view(k // pf, pf, n)
    packed = q[:, 0, :].clone()
    for i in range(1, pf):
        packed |= q[:, i, :] << (bits * i)
    scales = sc.t().contiguous().to(scales_dtype)  # [K/gs, N]
    zword = 0
    for i in range(pf):
        zword |= (zp - 1) << (bits * i)
    if zword >= 1 << 31:  # int32 wrap (0x7f7f7f7f fits; keep generic)
        zword -= 1 << 32
    qzeros = torch.full((k // gs, n // pf), zword, dtype=torch.int32, device=w.device)
    g_idx = (torch.arange(k, device=w.device, dtype=torch.int32) // gs).to(torch.int32)
    return {"qweight": packed, "scales": scales, "qzeros": qzeros, "g_idx": g_idx}


def gptq_dequant(qweight: torch.Tensor, scales: torch.Tensor, bits: int, gs: int) -> torch.Tensor:
    """Reference unpack of ``rtn_gptq_pack`` output back to fp32 ``[N, K]`` (tests)."""
    pf = 32 // bits
    zp = 1 << (bits - 1)
    kq, n = qweight.shape
    k = kq * pf
    mask = (1 << bits) - 1
    q = torch.empty((k, n), dtype=torch.int32, device=qweight.device)
    for i in range(pf):
        q[i::pf] = (qweight >> (bits * i)) & mask
    s = scales.float().repeat_interleave(gs, dim=0)  # [K, N]
    return ((q.float() - zp) * s).t().contiguous()


# ---------------------------------------------------------------------------
# checkpoint stream expansion
# ---------------------------------------------------------------------------
_EXPERT_KINDS = ("qweight", "scales", "qzeros", "g_idx")


def _dev() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def expand_fused_experts(name: str, tensor: torch.Tensor, bits: int, gs: int):
    """``<p>.mlp.experts.gate_up_proj`` [E, 2N, K] / ``<p>.mlp.experts.down_proj``
    [E, K, N] (bf16) -> per-expert GPTQ tensors ``<p>.mlp.experts.{e}.{gate,up,down}_proj.{kind}``."""
    base, leaf = name.rsplit(".", 1)
    dev = _dev()
    e_count = int(tensor.shape[0])
    if leaf == "gate_up_proj":
        two_n = int(tensor.shape[1])
        half = two_n // 2
        for e in range(e_count):
            w = tensor[e].to(dev, non_blocking=False)
            for proj, sl in (("gate_proj", w[:half]), ("up_proj", w[half:])):
                out = rtn_gptq_pack(sl, bits, gs)
                for kind in _EXPERT_KINDS:
                    yield f"{base}.{e}.{proj}.{kind}", out[kind]
    elif leaf == "down_proj":
        for e in range(e_count):
            w = tensor[e].to(dev, non_blocking=False)  # [K_hidden, N_inter] = [N_out, K_in]
            out = rtn_gptq_pack(w, bits, gs)
            for kind in _EXPERT_KINDS:
                yield f"{base}.{e}.down_proj.{kind}", out[kind]
    else:
        raise ValueError(f"expand_fused_experts: unexpected tensor {name}")


def expand_shared_expert(name: str, tensor: torch.Tensor, bits: int, gs: int):
    """``<p>.mlp.shared_expert.{gate,up,down}_proj.weight`` [N, K] bf16 -> GPTQ tensors
    (fp16 scales, like gptqmodel writes them)."""
    base = name[: -len(".weight")]
    out = rtn_gptq_pack(tensor.to(_dev()), bits, gs, scales_dtype=torch.float16)
    for kind in _EXPERT_KINDS:
        yield f"{base}.{kind}", out[kind]


def is_fused_expert_weight(name: str) -> bool:
    return name.endswith(".mlp.experts.gate_up_proj") or name.endswith(".mlp.experts.down_proj")


def is_shared_expert_weight(name: str) -> bool:
    return (
        ".mlp.shared_expert." in name
        and name.endswith(".weight")
        and name.rsplit(".", 2)[-2] in ("gate_proj", "up_proj", "down_proj")
    )


def quantize_draft_weights(weights):
    """Wrap the draft checkpoint stream: replace the bf16 expert / shared-expert
    tensors by GPTQ-format RTN tensors according to ``VLLM_GFX908_MTP_QUANT``.
    Everything else passes through untouched."""
    bits = mtp_quant_experts_bits()
    shared_w4 = mtp_quant_shared_expert_w4()
    if bits == 0 and not shared_w4:
        yield from weights
        return
    gs = mtp_quant_experts_gs(bits) if bits else 32
    n_exp = 0
    for name, tensor in weights:
        if bits and is_fused_expert_weight(name):
            logger.info(
                "gfx908 MTP: RTN-quantizing %s %s bf16 -> W%d gs%d GPTQ layout on %s",
                name, tuple(tensor.shape), bits, gs, _dev(),
            )
            for item in expand_fused_experts(name, tensor, bits, gs):
                n_exp += 1
                yield item
            continue
        if shared_w4 and is_shared_expert_weight(name):
            logger.info("gfx908 MTP: RTN-quantizing %s %s bf16 -> W4 gs32", name, tuple(tensor.shape))
            yield from expand_shared_expert(name, tensor, 4, 32)
            continue
        yield name, tensor
    if n_exp:
        logger.info("gfx908 MTP: emitted %d per-expert GPTQ tensors", n_exp)


# ---------------------------------------------------------------------------
# W8A16 opt-in for the draft's dense bf16 projections
# ---------------------------------------------------------------------------
_DENSE_W8_NAMES = (
    "fc_embedding",
    "fc_hidden",
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "self_attn.indexer.index_qk_proj",
    "mlp.shared_expert.gate_up_proj",
    "mlp.shared_expert.down_proj",
)


def mark_draft_w8a16(model: nn.Module) -> list[str]:
    """Flag the draft's dense bf16 linears for the W8A16 load-time path.

    ``prepare_w8a16_layer`` (called from ``UnquantizedLinearMethod.
    process_weights_after_loading``) honours ``_gfx908_w8a16_force`` and quantizes
    the weight regardless of the shape whitelist; the bf16 master is kept
    (``_gfx908_w8a16_free=False``) so ``M > 4`` (batched decode: the draft runs
    ``num_reqs * (n+1)`` rows) stays on the stock bf16 path bit-for-bit.  The
    router ``mlp.gate`` is deliberately left bf16 (the fused router reads it), as
    are the hyper-connection mixes (they have their own ``VLLM_GFX908_HC_W8`` path)
    and lm_head / embed_tokens (shared with the target).
    """
    if not mtp_quant_dense_w8():
        return []
    marked = []
    for mod_name, mod in model.named_modules():
        if not any(mod_name == s or mod_name.endswith("." + s) for s in _DENSE_W8_NAMES):
            continue
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16:
            continue  # quantized (GPTQ) or absent: nothing to do
        mod._gfx908_w8a16_force = True
        mod._gfx908_w8a16_free = os.environ.get("VLLM_GFX908_MTP_QUANT_FREE", "0") == "1"
        marked.append(mod_name)
    logger.info("gfx908 MTP: W8A16 opt-in for %d draft projections: %s", len(marked), marked)
    return marked
