# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 fused hyper-connection projections (M <= 3).

A copy of vLLM's wvSplitK small-M skinny GEMM (csrc/rocm/skinny_gemms.cu) with
two fused epilogues, JIT-built with torch.utils.cpp_extension:

  epi 1: mix_down (+ block inject) GEMV with silu(v / HC) applied to the lora
         columns (< lora_rank) — replaces hc_silu.
  epi 2: mix_up GEMV over a row-permuted weight (row i*HC + s = original row
         s*HD + i) with YTILE = HC so one wave holds the HC stream values of an
         output column; the epilogue writes mean_s sigmoid(g_s) * xn[s*HD + i]
         — replaces hc_gate_mix.

Graph-timed on MI100 (bit-exact vs the stock chain): mix_down+silu 7.2 vs 8.3
us, mix_up+gate_mix 8.3 vs 11.6 us at M=1. HC chain 5 -> 3 launches.
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

HC_FUSED_MAX_M = 3
_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "gfx908_wv_fused.hip")
_FLAG: bool | None = None


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading fused HC extension in %s", build_dir)
    return load(
        name="gfx908_wv_fused_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def hc_fused_enabled() -> bool:
    global _FLAG
    if _FLAG is None:
        from vllm.platforms.rocm import on_gfx908

        _FLAG = on_gfx908() and os.environ.get("VLLM_GFX908_HC_FUSED", "1") == "1"
        if _FLAG:
            try:
                _ext()
            except Exception as exc:
                logger.warning_once("gfx908: fused HC extension unavailable (%s)", exc)
                _FLAG = False
    return _FLAG


@functools.cache
def _cu_count() -> int:
    from vllm.utils.platform_utils import num_compute_units

    return int(num_compute_units())


def permute_up_weight(w_up: torch.Tensor, hc_count: int, hidden: int) -> torch.Tensor:
    """[HC*HD, R] -> rows reordered so row i*HC + s = original row s*HD + i."""
    perm = torch.arange(hc_count * hidden, device=w_up.device).view(hc_count, hidden).t().reshape(-1)
    return w_up[perm].contiguous()


_PERM_CACHE: dict[tuple, torch.Tensor] = {}


def _w_up_perm(w_up: torch.Tensor, hc_count: int, hidden: int) -> torch.Tensor:
    key = (w_up.data_ptr(), tuple(w_up.shape), str(w_up.device))
    t = _PERM_CACHE.get(key)
    if t is None:
        t = permute_up_weight(w_up, hc_count, hidden)
        _PERM_CACHE[key] = t
    return t


def _hc_mix_impl(
    xn: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor,
    hc_count: int, lora_rank: int, hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque op: fused wvSplitK epilogues for M <= 3, stock math otherwise."""
    M = xn.shape[0]
    if M <= HC_FUSED_MAX_M and xn.is_contiguous():
        ext = _ext()
        cu = _cu_count()
        w_up_perm = _w_up_perm(w_up, hc_count, hidden)
        down = torch.empty((M, w_down.shape[0]), dtype=xn.dtype, device=xn.device)
        ext.wv_fused(w_down, xn, down, xn, 1, hc_count, lora_rank, cu)
        lora = down[:, :lora_rank]
        injection = down[:, lora_rank : lora_rank + hc_count].contiguous()
        y = torch.empty((M, hidden), dtype=xn.dtype, device=xn.device)
        ext.wv_fused(w_up_perm, lora, y, xn, 2, hc_count, lora_rank, cu)
        return y, injection
    # stock chain (same dispatch as the ReplicatedLinear layers would take)
    from vllm.models.qwen4_exp.amd.ops.hc import hc_gate_mix, hc_silu

    down = torch.ops.vllm.rocm_unquantized_gemm_gfx908(xn, w_down, None)
    lora = hc_silu(down[:, :lora_rank].contiguous(), hc_count)
    injection = down[:, lora_rank : lora_rank + hc_count].contiguous()
    gate = torch.ops.vllm.rocm_unquantized_gemm_gfx908(lora, w_up, None)
    return hc_gate_mix(xn, gate, hc_count), injection


def _hc_mix_fake(xn, w_down, w_up, hc_count, lora_rank, hidden):
    return (
        xn.new_empty((xn.shape[0], hidden)),
        xn.new_empty((xn.shape[0], hc_count)),
    )


direct_register_custom_op(
    op_name="gfx908_hc_fused_mix",
    op_func=_hc_mix_impl,
    fake_impl=_hc_mix_fake,
)


def hc_fused_mix(xn, w_down, w_up, hc_count, lora_rank, hidden):
    return torch.ops.vllm.gfx908_hc_fused_mix(xn, w_down, w_up, hc_count, lora_rank, hidden)
