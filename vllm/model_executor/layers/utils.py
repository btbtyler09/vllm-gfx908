# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""

import os
from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MOE_LAYER_ROUTER_GATE_SUFFIXES = {
    "gate",
    "router",
    "router_gate",
    "shared_expert_gate",
    "expert_gate",
}


def is_layer_moe_router_gate(prefix: str) -> bool:
    if not prefix:
        return False
    return prefix.rsplit(".", 1)[-1] in MOE_LAYER_ROUTER_GATE_SUFFIXES


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)


def use_aiter_triton_gemm(n, m, k, dtype):
    if (
        not rocm_aiter_ops.is_triton_gemm_enabled()
        # MI300's - fp8nuz=True
        or current_platform.is_fp8_fnuz()
        or dtype not in [torch.float16, torch.bfloat16]
    ):
        return False

    # use hipblaslt for the larger GEMMs
    if n > 2048 and m > 512:
        return False
    return (
        (m == 5120 and k == 2880)
        or (m == 2880 and k == 4096)
        or (m == 128 and k == 2880)
        or (m == 640 and k == 2880)
        or (m == 2880 and k == 512)
        # Qwen3.6-35B-A3B lm_head: M=1, N=62080 — AITER 1.74x vs rocBLAS w/ BEST_CFG
        # (lm_head replicated across TP ranks, not column-split). Other unquantized
        # shapes in this model lose to rocBLAS at AITER's ~50μs floor — keep them off.
        or (m == 62080 and k == 2048)
    )


# gfx908 small-M tuning: AITER's default _get_config picks M_LEQ_64 for our shapes,
# which wastes blocks at M=1. This M=1 N≥1024 K=2048 config wins ~1.74x for lm_head.
_AITER_GEMM_M1_BEST_CFG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg",
    "NUM_KSPLIT": 1,
    "SPLITK_BLOCK_SIZE": 2048,
    "kpack": 1,
}


def rocm_unquantized_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950

    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]

    cu_count = num_compute_units()

    # Next ^2 of n
    N_p2 = 1 << (n - 1).bit_length()
    # With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
    # and each working on a 512-shard of K, how many CUs would we need?
    rndup_cus = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512)
    # How many of 4 waves in a group can work on same 16 Ms at same time?
    # This reduces the Ms each group works on, i.e. increasing the number of CUs needed.
    GrpsShrB = min(N_p2 // 16, 4)
    # Given the above, how many CUs would we need?
    CuNeeded = rndup_cus * GrpsShrB
    # candidate for atomic reduce count splitk?
    fits_wvsplitkrc = (
        N_p2 * m * ((k + 512 - 1) // 512)
    ) <= 128 * 1024 * 12  # deterministic
    fits_wvsplitkrc &= CuNeeded <= cu_count

    use_skinny_reduce_counting = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and on_gfx950()
        and x.dtype in [torch.float16, torch.bfloat16]
        and (
            10 <= n <= 128
            and k % 8 == 0
            and k > 512
            and m % 16 == 0
            and fits_wvsplitkrc
            and weight.is_contiguous()
        )
    )
    if use_skinny_reduce_counting:
        return ops.wvSplitKrc(x, weight, cu_count, bias)

    if use_aiter_triton_gemm(n, m, k, x.dtype):
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        # gfx908: pass M=1 small-M config for lm_head shape (1.74x vs default config)
        cfg = _AITER_GEMM_M1_BEST_CFG if n == 1 and m == 62080 and k == 2048 else None
        if cfg is not None:
            return gemm_a16w16(x, weight, bias, config=cfg)
        return gemm_a16w16(x, weight, bias)

    use_skinny = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and (on_gfx9() or on_gfx1x())
        and x.dtype in [torch.float16, torch.bfloat16]
        and k % 8 == 0
    )

    if not use_skinny:
        return torch.nn.functional.linear(x, weight, bias)

    x_view = x.reshape(-1, x.size(-1))
    if m > 8 and 0 < n <= 4:
        cu_count = num_compute_units()
        out = ops.wvSplitK(weight, x_view, cu_count, bias)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    elif m % 4 == 0 and n == 1 and k <= 8192 and bias is None:
        out = ops.LLMM1(weight, x_view, 4)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return weight.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm",
    op_func=rocm_unquantized_gemm_impl,
    fake_impl=rocm_unquantized_gemm_fake,
)


def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    return (
        torch.cpu._is_amx_tile_supported()
        and (dtype in (torch.bfloat16, torch.int8))
        and k % 32 == 0
        and n % 16 == 0
    )


def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module,
    remove_weight: bool,
) -> None:
    # skip for missing layers
    if layer.weight.is_meta:
        layer.cpu_linear = torch.nn.functional.linear
        return

    N, K = layer.weight.size()
    dtype = layer.weight.dtype

    # Zen CPU path: zentorch_linear_unary with optional eager weight prepacking.
    if current_platform.is_zen_cpu() and hasattr(
        torch.ops.zentorch, "zentorch_linear_unary"
    ):
        zen_weight = layer.weight.detach()
        is_prepacked = False

        if envs.VLLM_ZENTORCH_WEIGHT_PREPACK and hasattr(
            torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
        ):
            zen_weight = torch.ops.zentorch.zentorch_weight_prepack_for_linear(
                zen_weight
            )
            is_prepacked = True

        layer.cpu_linear = lambda x, weight, bias, _p=is_prepacked: (
            torch.ops.zentorch.zentorch_linear_unary(
                x, zen_weight, bias, is_weight_prepacked=_p
            )
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        return

    if envs.VLLM_CPU_SGL_KERNEL and check_cpu_sgl_kernel(N, K, dtype):
        packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
        if getattr(layer, "bias", None) is not None:
            bias_f32 = layer.bias.to(torch.float32)
        else:
            bias_f32 = None
        layer.cpu_linear = lambda x, weight, bias: torch.ops._C.weight_packed_linear(
            x, packed_weight, bias_f32 if bias is not None else None, True
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        return
    elif (
        ops._supports_onednn
        and current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC
    ):
        try:
            origin_weight = layer.weight
            handler = ops.create_onednn_mm(origin_weight.t(), 32)
            layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(handler, x, bias)
            if remove_weight:
                layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            return
        except RuntimeError as e:
            logger.warning_once(
                "Failed to create oneDNN linear, fallback to torch linear."
                f" Exception: {e}"
            )

    # fallback case
    layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
        x, weight, bias
    )


def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x, weight, bias)


def rocm_unquantized_gemm_gfx908_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """gfx908 dispatch implementation (registered as torch custom op below).

    Routes through a `direct_register_custom_op` wrapper so torch.compile /
    inductor sees one opaque graph node and does NOT inline this Python down
    to `aten::mm` (rocBLAS). That ensures QKV/QKVZ/o_proj inside compiled model
    forwards still hit our LLMM1/wvSplitK dispatch, not just the eager-path
    MoE block / lm_head.

    Dispatch priority on gfx908 for fp16/bf16 weight (k % 8 == 0):
      1. LLMM1   for n==1, m % 4 == 0, k <= 8192, bias is None (fastest at M=1)
      2. wvSplitK for m > 8 and 0 < n <= 4 (skinny-M decode)
      3. AITER gemm_a16w16 for whitelisted lm_head shape (Stage 2)
      4. F.linear fallback (rocBLAS for M >= 8)

    Microbench (gfx908 single-GPU, 2026-04-24):
      - LLMM1   2.1-2.7x faster than rocBLAS for our M=1 hot shapes
      - wvSplitK 2.1-6.7x faster than rocBLAS (wins on (1, 1, 2048))
      - LLMM1 also beats AITER for (1, 62080, 2048) lm_head: 244us vs 267us
    """
    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]
    debug = os.environ.get("VLLM_GFX908_DEBUG_DISPATCH") == "1"

    # Skinny GEMM dispatch (PR adapted from larkinwc/vllm-gfx908#4 microbench).
    # Required conditions for both: weight.is_contiguous() (LLMM1/wvSplitK
    # assume contiguous), fp16/bf16 dtype, k % 8 == 0 (vectorized loads).
    skinny_ok = (
        x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype in (torch.float16, torch.bfloat16)
        and k % 8 == 0
        and weight.is_contiguous()
    )
    if skinny_ok:
        x_view = x.reshape(-1, x.size(-1))
        if n == 1 and m % 4 == 0 and k <= 8192 and bias is None:
            if debug:
                import sys as _sys
                print(f"[LLMM1] n={n} m={m} k={k}",
                      file=_sys.stderr, flush=True)
            if x.dtype != weight.dtype:
                x_view = x_view.to(weight.dtype)
            out = ops.LLMM1(weight, x_view, 4)
            return out.reshape(*x.shape[:-1], weight.shape[0])
        if m > 8 and 0 < n <= 4:
            if debug:
                import sys as _sys
                print(f"[wvSplitK] n={n} m={m} k={k}",
                      file=_sys.stderr, flush=True)
            cu_count = num_compute_units()
            if x.dtype != weight.dtype:
                x_view = x_view.to(weight.dtype)
            out = ops.wvSplitK(weight, x_view, cu_count, bias)
            return out.reshape(*x.shape[:-1], weight.shape[0])

    if use_aiter_triton_gemm(n, m, k, x.dtype):
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
        if debug:
            import sys as _sys
            print(f"[AITER_DISPATCH] n={n} m={m} k={k} dtype={x.dtype}",
                  file=_sys.stderr, flush=True)
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        cfg = _AITER_GEMM_M1_BEST_CFG if n == 1 and m == 62080 and k == 2048 else None
        if cfg is not None:
            return gemm_a16w16(x, weight, bias, config=cfg)
        return gemm_a16w16(x, weight, bias)
    # rocBLAS fallback for M >= 8 (no skinny-GEMM applies). Now reachable
    # from inside compiled forwards too (Stage 5h custom-op wrapper below).
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_gfx908(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """gfx908 dispatch — torch custom op wrapper.

    Routes through `torch.ops.vllm.rocm_unquantized_gemm_gfx908` so inductor
    treats it as a single opaque graph node. This is what makes the LLMM1 /
    wvSplitK dispatch fire for QKV/QKVZ/o_proj inside compiled model forwards
    (round-3 Stage 5h). Without this wrapping, inductor inlines our Python and
    lowers the trailing `F.linear` straight to `aten::mm` → rocBLAS, bypassing
    the dispatch.
    """
    return torch.ops.vllm.rocm_unquantized_gemm_gfx908(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm_gfx908",
    op_func=rocm_unquantized_gemm_gfx908_impl,
    fake_impl=rocm_unquantized_gemm_fake,  # output shape == weight.new_empty((*x.shape[:-1], weight.shape[0]))
)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx908
        if on_gfx908():
            return rocm_unquantized_gemm_gfx908
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm
