# SPDX-License-Identifier: Apache-2.0
"""Equivalence of the padding-free PLE short-conv prefill path with the
padded F.conv1d formulation it replaces. Pure torch; runs on CPU."""

import importlib.util
import pathlib

import pytest
import torch
import torch.nn.functional as F

_HERE = pathlib.Path(__file__).resolve()
_MOD = _HERE.parents[3] / "vllm" / "models" / "qwen4_exp" / "amd" / "ple_short_conv_flat.py"
_spec = importlib.util.spec_from_file_location("ple_short_conv_flat", _MOD)
_flat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_flat)
dilated_causal_conv_flat = _flat.dilated_causal_conv_flat


def padded_reference(x_p, lengths, initial_state, w, dilation):
    """The original rectangle formulation (packed -> cat state -> conv1d -> silu)."""
    P = lengths.numel()
    C, K = w.shape
    L = (K - 1) * dilation
    q_starts = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    T = int(q_starts[-1])
    max_len = int(lengths.max())
    positions = torch.arange(T, dtype=torch.int64)
    req = torch.searchsorted(q_starts[1:], positions, right=True)
    col = positions - q_starts[req]
    packed = x_p.new_zeros((P, max_len, C))
    packed[req, col] = x_p
    packed = packed.transpose(1, 2).contiguous()  # [P, C, max_len]
    history = torch.cat((initial_state, packed), dim=-1)  # [P, C, L + max_len]
    y = F.conv1d(history, w.unsqueeze(1).contiguous(), groups=C, dilation=dilation)
    y = F.silu(y).transpose(1, 2).contiguous()  # [P, max_len, C]
    out = y[req, col]
    idx = (lengths.view(P, 1, 1) + torch.arange(L).view(1, 1, L)).expand(-1, C, -1)
    next_state = history.gather(2, idx)
    return out, next_state


def _inputs(lengths, C, K, dilation, dtype, with_state, seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    P = lengths.numel()
    L = (K - 1) * dilation
    T = int(lengths.sum())
    x = torch.randn(T, C, generator=g).to(dtype)
    w = (torch.randn(C, K, generator=g) * 0.3).to(dtype)
    state = torch.randn(P, C, L, generator=g).to(dtype)
    if with_state is not True:
        mask = torch.tensor(with_state if with_state else [False] * P).view(P, 1, 1)
        state = torch.where(mask, state, torch.zeros_like(state))
    q_starts = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    positions = torch.arange(T, dtype=torch.int64)
    req = torch.searchsorted(q_starts[1:], positions, right=True)
    col = positions - q_starts[req]
    return x, q_starts, req, col, lengths, state, w


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "lengths,with_state",
    [
        ([16, 16, 21, 384, 720], True),  # the 11:09:56 crash shape, scaled
        ([1, 3, 9, 10, 2], True),  # requests shorter than the state window
        ([5, 700, 6], [True, False, True]),  # mixed initial states
        ([200], False),  # single request, no state
    ],
)
@pytest.mark.parametrize("chunk", [2048, 37])
def test_flat_matches_padded(dtype, lengths, with_state, chunk):
    C, K, dilation = 64, 4, 3
    x, q_starts, req, col, lens, state, w = _inputs(lengths, C, K, dilation, dtype, with_state)
    ref_out, ref_state = padded_reference(x, lens, state, w, dilation)
    out, next_state = dilated_causal_conv_flat(
        x, q_starts, req, col, lens, state, w, dilation, chunk_tokens=chunk
    )
    assert out.dtype == x.dtype and out.shape == x.shape
    assert next_state.shape == ref_state.shape
    if dtype == torch.float32:
        torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)
    else:
        # fp32 accumulate in both; only summation order can differ
        torch.testing.assert_close(out.float(), ref_out.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(next_state, ref_state, rtol=0, atol=0)


def test_empty():
    C, K, dilation = 8, 4, 3
    x = torch.zeros(0, C)
    lengths = torch.zeros(0, dtype=torch.int64)
    state = torch.zeros(0, C, 9)
    out, ns = dilated_causal_conv_flat(
        x, torch.zeros(1, dtype=torch.int64), torch.zeros(0, dtype=torch.int64),
        torch.zeros(0, dtype=torch.int64), lengths, state, torch.zeros(C, K), dilation,
    )
    assert out.shape == (0, C) and ns.shape == (0, C, 9)
