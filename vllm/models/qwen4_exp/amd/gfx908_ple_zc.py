# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 zero-copy PLE embedding (Qwen4Exp n-gram table in host page cache).

Stock mmap path per decode step: the worker computes n-gram ids on the GPU,
blocks on a D2H copy of them (i.e. on the previous step's whole graph),
gathers rows from the memmap'd table on the host, stages them pinned, copies
H2D, then builds attention metadata and launches the graph. On 4xMI100 that
serial chain leaves the GPU idle ~2.5 ms per token (15% of a 16.7 ms step).

This path registers the checkpoint's PLE shard files (already resident in
the host page cache) with the GPU via hipHostRegister and gathers rows on
the GPU, inside the captured graph. The kernel driver caps pinned system
memory per node, so each TP rank registers only the shards it owns
(``shard_idx % tp_size == rank``), writes zeros for the others, and one
TP all-reduce (sum) of the ~5 KB/token result reassembles every row
exactly. The worker never waits on the GPU, so consecutive steps queue
back to back.

Enable with ``VLLM_PLE_ZEROCOPY=1``.
"""

from __future__ import annotations

import functools
import os

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_CSRC = os.path.join(os.path.dirname(__file__), "csrc", "gfx908_ple_zc.hip")
_PAGE = 4096


def zerocopy_enabled() -> bool:
    return os.environ.get("VLLM_PLE_ZEROCOPY", "0") == "1"


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading HIP PLE zero-copy extension in %s", build_dir)
    return load(
        name="gfx908_ple_zc_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


class ZeroCopyTable:
    """GPU-visible view of the rank-owned shards of an ``MmapPleTable``."""

    def __init__(self, table, rank: int, world: int, device: torch.device) -> None:
        ext = _ext()
        n_slots = len(table.mm)
        ptrs = [0] * n_slots
        self._registered: list[int] = []
        self.bytes = 0
        align = 16
        for idx, mm in enumerate(table.mm):
            if mm is None or idx % world != rank:
                continue
            data = mm.ctypes.data
            base = data & ~(_PAGE - 1)
            size = mm.nbytes + (data - base)
            dp = ext.host_register(base, size)
            if dp == 0:
                self.close()
                raise RuntimeError(
                    f"hipHostRegister failed for shard {idx} ({size / 2**30:.1f} GiB) "
                    f"after {self.bytes / 2**30:.1f} GiB"
                )
            self._registered.append(base)
            ptrs[idx] = dp + (data - base)
            self.bytes += size
            while align > 1 and (ptrs[idx] % align or table.row_bytes % align):
                align //= 2
        self.shard_ptr = torch.tensor(ptrs, dtype=torch.int64, device=device)
        self.shard_size = int(table.shard_size)
        self.row_bytes = int(table.row_bytes)
        self.align = align
        self.owned = len(self._registered)

    def gather(self, ids: torch.Tensor, out_bytes: torch.Tensor) -> None:
        _ext().gather(ids, self.shard_ptr, self.shard_size, self.row_bytes, out_bytes, self.align)

    def close(self) -> None:
        ext = _ext()
        for base in self._registered:
            ext.host_unregister(base)
        self._registered = []


def attach(embedding, layer_idx: int) -> None:
    """Register this rank's shards of ``embedding.table`` (called by build_tables)."""
    table = embedding.table
    if table is None:
        return
    rank = get_tensor_model_parallel_rank()
    world = get_tensor_model_parallel_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        zc = ZeroCopyTable(table, rank, world, device)
    except Exception as exc:  # fall back to the host gather path
        logger.warning("PLE zero-copy: layer %d attach failed (%s); using host gather", layer_idx, exc)
        embedding._gfx908_zc = None
        return
    embedding._gfx908_zc = zc
    logger.info(
        "PLE zero-copy: layer %d rank %d/%d registered %d/%d shards (%.1f GiB), align %d",
        layer_idx, rank, world, zc.owned, len(table.mm), zc.bytes / 2**30, zc.align,
    )


def zerocopy_table(ngram_embedding) -> ZeroCopyTable | None:
    return getattr(ngram_embedding, "_gfx908_zc", None)


def _ple_zc_embed(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    emb = layer.ple_embedding
    zc = zerocopy_table(emb.ngram_embedding)
    # The padded tail of query_start_loc may hold stale entries from an earlier
    # batch; searchsorted needs a monotonic array, and a running max turns any
    # stale tail into valid padding without touching the real prefix.
    query_start_loc = torch.cummax(query_start_loc, dim=0).values
    # The padded tail of query_start_loc may hold stale entries from an earlier
    # batch; searchsorted needs a monotonic array, and a running max turns any
    # stale tail into valid padding without touching the real prefix.
    query_start_loc = torch.cummax(query_start_loc, dim=0).values
    ids = emb.compute_ngram_ids(input_ids, query_start_loc, ngram_context)  # [T, H]
    num_tokens, heads = ids.shape
    staging = emb._mmap_staging[:num_tokens]  # [T, H, head_dim]
    rows = staging.view(torch.uint8).reshape(num_tokens * heads, zc.row_bytes)
    zc.gather(ids.reshape(-1), rows)
    dbg = getattr(emb, "_zc_debug_ids", None)
    logger.info_once("PLE zc op: emb id %d, debug buffer %s, capturing %s", id(emb), dbg is not None, torch.cuda.is_current_stream_capturing())
    if dbg is not None:
        dbg[: ids.numel()].copy_(ids.reshape(-1))
        dbg[-1].fill_(ids.numel())
        din = emb._zc_debug_in
        t = input_ids.numel(); r = query_start_loc.numel() - 1; w = ngram_context.shape[1]
        din[0].fill_(t); din[1].fill_(r); din[2].fill_(w)
        din[3 : 3 + t].copy_(input_ids.reshape(-1))
        din[3 + t : 4 + t + r].copy_(query_start_loc.reshape(-1))
        din[4 + t + r : 4 + t + r + r * w].copy_(ngram_context[:r].reshape(-1))
    reduced = tensor_model_parallel_all_reduce(staging)
    output.copy_(reduced.flatten(-2))
    if _CHECK and not torch.cuda.is_current_stream_capturing():
        _self_check(emb, ids, staging, reduced)


_CHECK = os.environ.get("VLLM_PLE_ZEROCOPY_CHECK", "0") == "1"


def _self_check(emb, ids, staging, reduced) -> None:
    ref = torch.empty_like(staging)
    emb.ngram_embedding.gather_into(ids, ref)
    torch.cuda.synchronize()
    rows = staging.shape[0] * staging.shape[1]
    local_nz = (staging.reshape(rows, -1) != 0).any(-1).float().mean().item()
    red_nz = (reduced.reshape(rows, -1) != 0).any(-1).float().mean().item()
    ref_nz = (ref.reshape(rows, -1) != 0).any(-1).float().mean().item()
    bad = (reduced.reshape(rows, -1) != ref.reshape(rows, -1)).any(-1)
    logger.info(
        "PLE zc check: rows=%d local_nz=%.3f reduced_nz=%.3f ref_nz=%.3f mismatch_rows=%d "
        "maxdiff=%.4g ids[min,max]=[%d,%d]",
        rows, local_nz, red_nz, ref_nz, int(bad.sum()),
        (reduced.float() - ref.float()).abs().max().item(), int(ids.min()), int(ids.max()),
    )


def _ple_zc_embed_fake(input_ids, query_start_loc, ngram_context, output, layer_name) -> None:
    return


direct_register_custom_op(
    op_name="gfx908_ple_zc_embed",
    op_func=_ple_zc_embed,
    mutates_args=["output"],
    fake_impl=_ple_zc_embed_fake,
)
