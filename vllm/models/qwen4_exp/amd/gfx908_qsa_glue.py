"""gfx908 fused QSA decode glue (env ``VLLM_GFX908_QSA_GLUE=1``, default off).

For uniform decode (one token per request, the shape of every captured decode
graph) the launches between ``qkv_proj`` and ``o_proj`` of a QSA layer go from
26 to 6 (measured on the rc5 tree: the 3 inductor norm/RoPE kernels, the
opaque op's 22 kernels and the gate multiply -> indexer GEMV, ``qsa_glue_pre``,
scorer, ``qsa_topk_expand``, split-K attention, merge-with-gate):

  qsa_glue_pre     main q/k Gemma-RMSNorm + interleaved MRoPE (bit-exact with the
                   inductor-compiled projection), K/V written straight into the
                   paged cache, indexer q norm + RoPE, raw key + int64 RoPE
                   positions into the compressor ring, group pooling + k norm +
                   RoPE into the compressed-key cache.  One HIP launch, one
                   workgroup per row (csrc/gfx908_qsa_glue.hip).
  qsa_topk_expand  deterministic top-512 (radix select, the stable tie rule of
                   ``qsa_stable_topk_``) + token expansion, one launch instead of
                   topKPerRowDecode + repair + expand.
  merge epilogue   ``sigmoid(gate)`` applied in the split-K merge (inductor's
                   fp32 formula), padded rows zeroed there (no ``output.zero_()``).

Rows with up to ``VLLM_GFX908_QSA_GLUE_MAX_Q`` query tokens per request (default
1; 4 covers MTP verify at n <= 3 and small multi-token decode requests) take the
same fast path: every kernel of it is per row except the compressor ring, whose
only cross-row hazard is a later row of the same request overwriting a ring slot
an earlier row still has to pool from.  Within a request the ring reads are at
positions below the request's first row and the stores at its own rows, so the
per-row grid is hazard-free whenever ``ring_size >= max_query_len + 3`` -- true
of every spec-verify batch, since the ring is sized ``4 * ceil((4 + n) / 4)``
for ``n + 1`` rows.  Narrower rings (no spec decode, ring 4) with multi-token
rows use the kernel's per-request grid (all rows of a request in one workgroup,
ring stores after a barrier).

Everything else (prefill, longer verify rows, ``skip_topk`` index reuse,
dense-short and tiled prefill) takes the fallback inside the same op: the main
projection still runs through ``qsa_glue_pre`` in main-only mode (no cross-row
hazard there) and the stock indexer / attention code follows, with the gate
applied by ``qsa_gate_mul`` (same formula as the compiled multiply).

Fixed per-rank shapes (Qwen3.8-Flash-Next TP4): 6 q heads / 1 kv head x 256,
rotary_dim 64, MRoPE section of 3 (interleaved), indexer 4 + 1 heads x 128,
budget 2048, compress_ratio 4.
"""

import functools
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "gfx908_qsa_glue.hip")
_FLAG: bool | None = None
_MAX_Q: int | None = None
# bit 2: norm output stays fp32 into the RoPE (what inductor's fused kernel does),
# bits 0-1: fp-contraction form of the rotation (1 = fma(x1, cos, ...)).  Modes
# 4/5/6 all reproduce the compiled projection bit for bit (agents/qsa_glue).
ROPE_MODE = 5
STATS = {"fused_calls": 0, "fallback_calls": 0}


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    build_dir = os.path.join(build_dir, "qsa_glue")
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading fused QSA glue extension in %s", build_dir)
    return load(
        name="gfx908_qsa_glue_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def qsa_glue_enabled() -> bool:
    """Env flag (default off) and a successful extension build."""
    global _FLAG
    if _FLAG is None:
        from vllm.platforms.rocm import on_gfx908

        _FLAG = on_gfx908() and os.environ.get("VLLM_GFX908_QSA_GLUE", "1") == "1"
        if _FLAG:
            try:
                _ext()
            except Exception as exc:
                logger.warning_once("gfx908: fused QSA glue extension unavailable (%s)", exc)
                _FLAG = False
    return _FLAG


def qsa_glue_max_q() -> int:
    """Largest per-request query length the fast path takes
    (``VLLM_GFX908_QSA_GLUE_MAX_Q``, default 1, clamped to 1..4)."""
    global _MAX_Q
    if _MAX_Q is None:
        try:
            value = int(os.environ.get("VLLM_GFX908_QSA_GLUE_MAX_Q", "1"))
        except ValueError:
            value = 1
        _MAX_Q = max(1, min(4, value))
    return _MAX_Q


def qsa_glue_layer_supported(layer, vllm_config) -> bool:
    """Static (init-time) eligibility of one Qwen4ExpQSAAttention owner."""
    if not qsa_glue_enabled():
        return False
    rope = layer.rotary_emb
    idx = layer.indexer
    section = getattr(rope, "mrope_section", None)
    ok = (
        layer.num_heads == 6
        and layer.num_kv_heads == 1
        and layer.head_dim == 256
        and layer.attn_output_gate
        and not layer.use_fused_qk_norm_rope_gate
        and int(rope.rotary_dim) == 64
        and int(rope.head_size) == 256
        and bool(getattr(rope, "is_neox_style", True))
        and section is not None
        and len(section) == 3
        and bool(getattr(rope, "mrope_interleaved", False))
        and idx.index_n_heads == 4
        and idx.index_kv_heads == 1
        and idx.index_head_dim == 128
        and idx.token_topk == 2048
        and idx.compress_ratio == 4
        and idx.rotary_emb is rope
        and float(layer.q_norm.variance_epsilon) == float(layer.k_norm.variance_epsilon)
        and float(idx.q_layernorm.variance_epsilon) == float(idx.k_layernorm.variance_epsilon)
        and getattr(vllm_config.model_config, "dtype", torch.bfloat16) == torch.bfloat16
    )
    logger.info_once(
        "gfx908: fused QSA decode glue %s for %s", "ENABLED" if ok else "not applicable", layer.layer_name
    )
    return ok


def glue_pre(layer, qkv, iqk, positions, query, iq, main_md, raw_md, cmp_md, num_rows, mode, max_query_len=1):
    """One launch.  mode bit 0: main q/k/v (query out + KV cache); bit 1: indexer
    q (``iq`` out) + compressor ring + compressed cache.  One workgroup per row,
    or -- when the indexer part runs on multi-token rows and the ring is narrower
    than ``max_query_len + 3`` -- one workgroup per request (rows per workgroup =
    ``max_query_len``, 2..4) so the ring stores follow every row's ring reads."""
    idx = layer.indexer
    rope = layer.rotary_emb
    cos_sin = rope._match_cos_sin_cache_dtype(qkv)  # bf16 [max_pos, 64]
    key_cache, value_cache = layer.kv_cache.transpose(1, 2).split(layer.head_dim, dim=-1)
    section = rope.mrope_section
    ring_size = int(idx.raw_key_cache.key_cache.shape[1])
    tq = 1
    if max_query_len > 1 and (mode & 2) and ring_size < max_query_len + 3:
        tq = int(max_query_len)
    _ext().qsa_glue_pre(
        qkv, iqk, positions, cos_sin,
        layer.q_norm.weight, layer.k_norm.weight, idx.q_layernorm.weight, idx.k_layernorm.weight,
        float(layer.q_norm.variance_epsilon), float(idx.q_layernorm.variance_epsilon),
        query, iq, key_cache, value_cache, main_md.slot_mapping, int(layer.kv_cache.shape[2]),
        idx.raw_key_cache.key_cache, raw_md.slot_mapping, idx.raw_key_cache.rope_position_cache,
        raw_md.block_table, cmp_md.token_to_req, cmp_md.query_start_loc, cmp_md.logical_positions,
        cmp_md.slot_mapping, idx.compressed_key_cache.kv_cache,
        int(section[1]), int(section[2]), ROPE_MODE, int(num_rows), int(mode), tq,
    )


def topk_expand(logits, visible, out, logical_positions, seq_lens, token_to_req, rows):
    """Deterministic top-512 + expansion of ``rows`` rows into ``out`` [rows, 2051]."""
    _ext().qsa_topk_expand(logits, visible, out, None, logical_positions, seq_lens, token_to_req, int(rows))


__all__ = ["glue_pre", "qsa_glue_enabled", "qsa_glue_layer_supported", "qsa_glue_max_q", "topk_expand", "STATS"]
