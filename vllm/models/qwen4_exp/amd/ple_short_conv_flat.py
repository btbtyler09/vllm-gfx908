# SPDX-License-Identifier: Apache-2.0
"""Padding-free dilated causal depthwise conv for the PLE short-conv prefill path.

The batched prefill path used to pack every prefill request of a step into a
``[num_prefills, channels, max_len + state_len]`` rectangle and run
``F.conv1d`` on it. With chunked prefill one 7-8k-token request next to a few
short ones turns that rectangle into ``num_prefills x max_len``; at 10,240
channels each materialized copy is ~700 MiB and the function keeps several
alive (packed, transposed, history, conv output, silu output) plus the conv
workspace. The memory profiler's dummy batch splits ``max_num_batched_tokens``
evenly over ``max_num_seqs`` requests (48 x 171 tokens), so it never sees the
rectangle. See docs/mi100_decode_opt/ple_prefill_oom_2026_09_14.md.

This module computes the same convolution directly on the flat token stream:

    out[t] = act( sum_k w[c, k] * hist[c, t + k * dilation] )

where ``hist`` for a request is ``[initial_state (state_len entries), tokens]``.
Every tap is a masked gather on the flat ``[T, C]`` input (values before the
request start come from the initial state), accumulated in fp32 and cast back,
in chunks of ``chunk_tokens`` rows. Peak transient is bounded by
``chunk_tokens x C`` and does not depend on the number of prefills or on the
longest request. No vLLM imports so it can be unit-tested on CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dilated_causal_conv_flat(
    x_p: torch.Tensor,
    q_starts: torch.Tensor,
    req_indices: torch.Tensor,
    col_indices: torch.Tensor,
    lengths: torch.Tensor,
    initial_state: torch.Tensor,
    conv_weights: torch.Tensor,
    dilation: int,
    chunk_tokens: int = 2048,
    apply_silu: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise dilated causal conv over flat, per-request-packed tokens.

    Args:
        x_p: ``[T, C]`` prefill tokens of all requests, concatenated.
        q_starts: ``[P + 1]`` int64 cumulative token offsets (q_starts[0] == 0).
        req_indices: ``[T]`` int64 request id of each token.
        col_indices: ``[T]`` int64 position of each token inside its request.
        lengths: ``[P]`` int64 tokens per request.
        initial_state: ``[P, C, L]`` conv state preceding each request
            (oldest first); rows without a valid initial state must be zero.
        conv_weights: ``[C, K]`` depthwise taps, ``L == (K - 1) * dilation``.
        dilation: conv dilation.
        chunk_tokens: rows per accumulation chunk (bounds the transient).
        apply_silu: apply SiLU to the conv output.

    Returns:
        ``(out [T, C] in x_p.dtype, next_state [P, C, L])`` where
        ``next_state`` holds the last ``L`` history entries of each request
        (identical to the padded implementation's write-back value).
    """
    T, C = x_p.shape
    P = lengths.numel()
    K = conv_weights.shape[1]
    L = (K - 1) * dilation
    dev = x_p.device
    out = torch.empty_like(x_p)
    if T == 0:
        return out, initial_state.clone()
    # state indexed as [P, L, C] so a (req, pos) gather yields [n, C] rows
    state_plc = initial_state.transpose(1, 2).contiguous()  # [P, L, C]
    w = conv_weights.to(torch.float32)  # [C, K]

    for s in range(0, T, chunk_tokens):
        e = min(T, s + chunk_tokens)
        pos = torch.arange(s, e, device=dev, dtype=torch.int64)
        req = req_indices[s:e]
        col = col_indices[s:e]
        acc = torch.zeros((e - s, C), dtype=torch.float32, device=dev)
        for k in range(K):
            d = (K - 1 - k) * dilation  # how far back this tap looks
            if d == 0:
                tap = x_p[s:e]
            else:
                from_tokens = col >= d
                tok_idx = torch.where(from_tokens, pos - d, torch.zeros_like(pos))
                tap_tok = x_p.index_select(0, tok_idx)
                # history index inside the state: L + (col - d), valid when col < d
                st_pos = (col - d + L).clamp_(0, L - 1)
                tap_st = state_plc[req, st_pos]
                tap = torch.where(from_tokens.unsqueeze(1), tap_tok, tap_st)
            acc.addcmul_(tap.to(torch.float32), w[:, k].unsqueeze(0))
        y = acc.to(x_p.dtype)
        out[s:e] = F.silu(y) if apply_silu else y

    # next_state[p, :, j] = hist_p[lengths_p + j], hist = [state (L), tokens]
    j = torch.arange(L, device=dev, dtype=torch.int64).view(1, L)
    hpos = lengths.view(P, 1) + j  # position in hist
    from_tokens = hpos >= L
    tok_idx = (q_starts[:P].view(P, 1) + hpos - L).clamp_(0, max(T - 1, 0))
    tok_idx = torch.where(from_tokens, tok_idx, torch.zeros_like(tok_idx))
    ns_tok = x_p.index_select(0, tok_idx.reshape(-1)).view(P, L, C)
    ns_st = torch.gather(
        state_plc, 1, hpos.clamp(0, L - 1).unsqueeze(-1).expand(P, L, C)
    )
    next_state = torch.where(from_tokens.unsqueeze(-1), ns_tok, ns_st)
    return out, next_state.transpose(1, 2).contiguous()  # [P, C, L]
