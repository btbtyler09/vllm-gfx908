# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908: capture ``compute_logits`` into the FULL decode cudagraphs.

``VLLM_GFX908_LOGITS_IN_GRAPH=1`` (default OFF).

Today the V2 runner replays the FULL decode graph (model forward only) and then
runs, eagerly, ``hidden_states[logits_indices]`` -> lm_head GEMV (int8 W8A16,
M = num_reqs) -> TP all-gather of the vocab-parallel logits -> sampler.  At
c=1 on 4x MI100 that eager tail is ~5 launches, an RCCL all-gather (119 us
eager vs 30 us captured, agents/ar_ship 4.1) and the rank-0 step-boundary skew
(agents/ar_track 1c).

For a uniform decode graph (one token per request, no draft tokens) the logits
rows are exactly ``hidden_states[:num_reqs]``: ``combine_sampled_and_draft_tokens``
writes ``logits_indices[i] = query_start_loc[i+1] - 1 = i``.  So the lm_head
projection and the gather are graph-static per capture size and can be recorded
into the FULL graph right after the model forward, writing into one static
``[max_num_reqs, vocab]`` buffer (bf16, what the head returns: 48 x 248320 x 2 B
= 23.8 MB per rank).  ``sample()`` then reads ``buf[:num_reqs]`` instead of
calling ``compute_logits``.

Padded rows: a FULL graph of size T serves every batch with num_reqs <= T, so
the lm_head runs on T rows.  Rows ``>= num_reqs`` are garbage and never read.
The W8A16 dispatch buckets M the same way for the padded T and the real
num_reqs (both land in the same MFMA config bucket for the default capture
sizes 1,2,4,8,16,24,32,40,48), and the kernels compute rows independently, so
the rows that are read are bit-identical to the eager path
(``agents/logits_graph/test_logits_graph.py``).

Anything else (PIECEWISE / eager prefill, spec decode, batch-sharded sampling,
LoRA, PP, PCP) keeps the eager path unchanged.
"""

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

_FLAG: bool | None = None


def logits_in_graph_requested() -> bool:
    """``VLLM_GFX908_LOGITS_IN_GRAPH=1`` and running on gfx908."""
    global _FLAG
    if _FLAG is None:
        _FLAG = False
        if os.environ.get("VLLM_GFX908_LOGITS_IN_GRAPH", "0") == "1":
            try:
                from vllm.platforms.rocm import on_gfx908

                _FLAG = bool(on_gfx908())
            except Exception:  # not ROCm
                _FLAG = False
            if not _FLAG:
                logger.warning_once(
                    "VLLM_GFX908_LOGITS_IN_GRAPH=1 ignored: not running on gfx908"
                )
    return _FLAG


def desc_eligible(desc: "BatchExecutionDescriptor") -> bool:
    """A FULL graph whose rows are one token per request (uniform decode).

    ``uniform_token_count == 1`` is the separate-decode-routine descriptor with
    ``decode_query_len == 1`` (no spec decode); ``num_reqs == num_tokens`` is
    the same fact for the mixed FULL descriptor.  ``num_active_loras`` must be
    0: with LoRA the logits go through ``LogitsProcessorWithLoRA``.
    """
    return (
        desc.cg_mode == CUDAGraphMode.FULL
        and desc.num_reqs is not None
        and desc.num_reqs == desc.num_tokens
        and (desc.uniform_token_count is None or desc.uniform_token_count == 1)
        and desc.max_query_len is None
        and desc.num_active_loras == 0
    )


def batch_eligible(input_batch: "InputBatch") -> bool:
    """The replayed batch is one token per request with one logit per request,
    i.e. ``logits_indices == arange(num_reqs)``."""
    return (
        input_batch.num_draft_tokens == 0
        and input_batch.num_tokens == input_batch.num_reqs
        and input_batch.logits_indices.shape[0] == input_batch.num_reqs
    )


class LogitsGraphState:
    """Static logits buffer + the set of FULL descriptors that recorded it.

    Owned by the ``ModelCudaGraphManager`` (recreated with it, so a profiling
    capture into a throwaway pool never leaves stale descriptors behind).
    """

    def __init__(self, max_num_reqs: int):
        self.max_num_reqs = max_num_reqs
        self.buf: torch.Tensor | None = None
        self.captured: set[Any] = set()
        # Counters for the gfx908 step-timer log.
        self.hits = 0
        self.misses = 0

    def nbytes(self) -> int:
        return 0 if self.buf is None else self.buf.numel() * self.buf.element_size()

    def record(
        self, model: torch.nn.Module, hidden_states: torch.Tensor, desc: Any
    ) -> None:
        """Run ``compute_logits`` on ``hidden_states`` (the graph's static
        hidden buffer, ``[desc.num_tokens, hidden]``) and copy the result into
        the static logits buffer.  Called from the capture ``forward_fn`` for
        both the warm-up pass (eager, not recorded) and the recorded pass."""
        logits = model.compute_logits(hidden_states)
        assert logits is not None, "logits in-graph needs all-gather logits"
        n, vocab = logits.shape
        if self.buf is None:
            self.buf = torch.empty(
                (self.max_num_reqs, vocab), dtype=logits.dtype, device=logits.device
            )
            logger.info(
                "gfx908: logits in-graph buffer [%d, %d] %s = %.1f MB",
                self.max_num_reqs,
                vocab,
                logits.dtype,
                self.nbytes() / 2**20,
            )
        assert n <= self.max_num_reqs and self.buf.shape[1] == vocab
        self.buf[:n].copy_(logits)
        if torch.cuda.is_current_stream_capturing():
            self.captured.add(desc)

    def logits_for(
        self, desc: Any, input_batch: "InputBatch"
    ) -> torch.Tensor | None:
        """The captured logits ``[num_reqs, vocab]`` for a step that replayed
        ``desc``, or None (caller runs the eager path)."""
        if desc is None or self.buf is None or desc not in self.captured:
            self.misses += 1
            return None
        if not batch_eligible(input_batch):
            self.misses += 1
            return None
        self.hits += 1
        return self.buf[: input_batch.num_reqs]

    def stats_line(self) -> str:
        return (
            f"logits in-graph: hits {self.hits}, misses {self.misses}, "
            f"graphs {len(self.captured)}, buffer {self.nbytes() / 2**20:.1f} MB"
        )
