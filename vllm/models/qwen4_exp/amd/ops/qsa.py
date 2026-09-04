# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the Qwen4Exp weight-free QSA path."""

from __future__ import annotations

import math
import os

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_TOPK_WORKSPACE_BYTES = 1024 * 1024


@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (row < num_rows)
        & (columns < num_columns)
        & (columns < visible)
        & (request >= 0)
        & (request < num_requests)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_req
        + safe_logical_page * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_pages)
    # physical_page * block stride can overflow int32 for large caches.
    safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)

    score /= score_divisor
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(valid, score, -float("inf")),
        mask=(row < num_rows) & (columns < num_columns),
    )


@triton.jit
def _expand_qsa_indices_tile(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    row,
    columns,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
) -> None:
    """Expand one tile of one row's selection into token ids.

    Factored out of ``_expand_qsa_indices_kernel`` so the fused
    stable-top-k-plus-expand kernel runs the *same* code rather than a copy
    that could drift.  ``row`` and ``columns`` are supplied by the caller
    instead of being read from ``program_id``.
    """

    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    _expand_qsa_indices_tile(
        block_indices_ptr,
        query_positions_ptr,
        sequence_lengths_ptr,
        token_to_req_ptr,
        output_ptr,
        stride_blocks_row,
        stride_blocks_column,
        stride_output_row,
        stride_output_column,
        rows,
        num_requests,
        tl.program_id(0),
        tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK),
        BLOCK_TOPK=BLOCK_TOPK,
        COMPRESS_RATIO=COMPRESS_RATIO,
        TOKEN_TOPK=TOKEN_TOPK,
        OUTPUT_WIDTH=OUTPUT_WIDTH,
    )


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores directly from a paged compressed-key cache."""

    _validate_mqa(q)
    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA scoring requires a GPU and Triton")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if q.shape[0] and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError("QSA paged scoring cache and page table must be nonempty")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if score_divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0:
        raise ValueError("QSA score width must be non-negative")
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    block_n = 32
    _qsa_mqa_paged_kernel[(q.shape[0], triton.cdiv(columns, block_n))](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
    )
    return logits, visible_blocks


def _stable_topk_enabled() -> bool:
    """``VLLM_GFX908_QSA_STABLE_TOPK=1`` opts into a reproducible selection.

    ``vllm::topKPerRowDecode`` writes each selected column at its wave-arrival
    slot (``atomicAdd(&smemFoundTopKValues[0], 1)``) and drains the threshold
    bin -- whose members are bit-identical floats -- with another atomic.  The
    *values* it selects are therefore always the true top-k, but neither the
    output order nor, when the k-th largest logit is duplicated, the selected
    set is reproducible.  QSA logits are a sum of ``index_n_heads`` ReLU'd dot
    products, so ~6% of them are exactly ``0.0``; whenever a query's budget cut
    lands in that atom the selection changes run to run.

    Read per call (one dict lookup); the default is on (set 0 to disable).
    """

    return os.environ.get("VLLM_GFX908_QSA_STABLE_TOPK", "1").strip() in (
        "1",
        "true",
        "True",
    )


@triton.jit
def _qsa_stable_topk_row(
    logits_ptr,
    visible_ptr,
    blocks_ptr,
    stride_logits_row,
    stride_blocks_row,
    num_columns,
    row,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Rewrite one row of ``blocks`` as the deterministic top-k of ``logits``.

    ``blocks`` arrives holding *a* correct top-k: the multiset of selected
    logit values is unique, so its smallest member is the k-th largest logit of
    the row no matter which tied column the atomic race handed out.  That value
    is the only thing this kernel takes from the input; it then rebuilds the
    row from scratch as

        {c : logit[c] > t}  U  the first (k - |{c : logit[c] > t}|)
                                columns with logit[c] == t

    written in ascending column order.  For a row whose k-th largest logit is
    unique the second set is a single column and the result is exactly today's
    selection; otherwise ties resolve to the lowest block index.  The rebuild
    is a plain per-row scan with no atomics, so both the set and the order are
    reproducible.
    """

    visible = tl.load(visible_ptr + row)
    visible = tl.minimum(tl.maximum(visible, 0), num_columns)

    logits_row = logits_ptr + row * stride_logits_row
    blocks_row = blocks_ptr + row * stride_blocks_row
    ranks = tl.arange(0, BLOCK_K)

    # A row with no more visible blocks than budget selects *every* visible
    # block, so the answer is ``0..visible-1`` ascending then -1 padding with
    # no reference to the logits at all -- one store, and out.  This is the
    # whole batch for any context at or below the token budget
    # (``TOPK * compress_ratio`` = 2048 tokens), where the selection is the
    # identity, and it is also every prefill row below that position.  It is
    # the case short-context decode spends its whole life in, so it must not
    # touch the 8192-column capture-time logits buffer at all.  Writing the
    # returning early keeps the kernel independent of whether the top-k
    # backend's own short-row shortcut happens to be ordered.
    if visible <= TOPK:
        tl.store(
            blocks_row + ranks,
            tl.where(ranks < visible, ranks, -1).to(tl.int32),
            mask=ranks < TOPK,
        )
        return

    # Phase 0: the threshold, read back from the incoming selection.
    keep = ranks < TOPK
    picked = tl.load(blocks_row + ranks, mask=keep, other=0)
    picked = tl.minimum(tl.maximum(picked, 0), num_columns - 1)
    values = tl.load(logits_row + picked, mask=keep, other=float("inf"))
    threshold = tl.min(tl.where(keep, values, float("inf")), axis=0)

    num_tiles = (visible + BLOCK_N - 1) // BLOCK_N
    offsets = tl.arange(0, BLOCK_N)

    # Phase 1: how many logits are strictly above the threshold.
    greater = 0
    for tile in range(0, num_tiles):
        columns = tile * BLOCK_N + offsets
        in_range = columns < visible
        value = tl.load(logits_row + columns, mask=in_range, other=-float("inf"))
        greater += tl.sum(tl.where(in_range & (value > threshold), 1, 0), axis=0)
    needed = tl.maximum(TOPK - greater, 0)

    # Phase 2: ascending-column placement of the strict set plus the first
    # ``needed`` members of the tie set.
    ties_seen = 0
    written = 0
    for tile in range(0, num_tiles):
        columns = tile * BLOCK_N + offsets
        in_range = columns < visible
        value = tl.load(logits_row + columns, mask=in_range, other=-float("inf"))
        above = in_range & (value > threshold)
        tied = tl.where(in_range & (value == threshold), 1, 0)
        tie_rank = ties_seen + tl.cumsum(tied, axis=0) - tied
        take = tl.where(above | ((tied == 1) & (tie_rank < needed)), 1, 0)
        position = written + tl.cumsum(take, axis=0) - take
        tl.store(
            blocks_row + position,
            columns.to(tl.int32),
            mask=(take == 1) & (position < TOPK),
        )
        ties_seen += tl.sum(tied, axis=0)
        written += tl.sum(take, axis=0)

    # Phase 3: pad a row that could not fill its budget.  ``visible > TOPK``
    # here, so this only fires if the incoming selection was malformed.
    if written < TOPK:
        tl.store(
            blocks_row + ranks,
            tl.full((BLOCK_K,), -1, tl.int32),
            mask=(ranks < TOPK) & (ranks >= written),
        )


@triton.jit
def _qsa_stable_topk_kernel(
    logits_ptr,
    visible_ptr,
    blocks_ptr,
    stride_logits_row,
    stride_blocks_row,
    num_columns,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    _qsa_stable_topk_row(
        logits_ptr,
        visible_ptr,
        blocks_ptr,
        stride_logits_row,
        stride_blocks_row,
        num_columns,
        tl.program_id(0).to(tl.int64),
        TOPK=TOPK,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


@triton.jit
def _qsa_stable_topk_expand_kernel(
    logits_ptr,
    visible_ptr,
    blocks_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_logits_row,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    num_columns,
    rows,
    num_requests,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
    NUM_COLUMN_TILES: tl.constexpr,
) -> None:
    """Repair one row's selection and expand it, in a single launch.

    Decode is launch-bound: the repair is ~1.8 us of work but a separate graph
    node between ``top_k_per_row_decode`` and the expand costs several times
    that once it is a real dependency in a real graph.  Folding the repair into
    the front of the expand means the flag adds *zero* nodes to the captured
    decode graph.

    The row's own program writes ``blocks`` and then reads it back, so a
    barrier is needed between the two -- which is exactly why this shape is
    decode-only: it forces one program per row, and prefill wants the expand
    spread over ``OUTPUT_WIDTH / COLUMN_BLOCK`` programs per row instead.
    """

    row = tl.program_id(0).to(tl.int64)
    _qsa_stable_topk_row(
        logits_ptr,
        visible_ptr,
        blocks_ptr,
        stride_logits_row,
        stride_blocks_row,
        num_columns,
        row,
        TOPK=TOPK,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    # Order this program's stores to ``blocks`` before its own reloads below.
    tl.debug_barrier()
    for tile in tl.static_range(0, NUM_COLUMN_TILES):
        _expand_qsa_indices_tile(
            blocks_ptr,
            query_positions_ptr,
            sequence_lengths_ptr,
            token_to_req_ptr,
            output_ptr,
            stride_blocks_row,
            stride_blocks_column,
            stride_output_row,
            stride_output_column,
            rows,
            num_requests,
            row,
            tile * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK),
            BLOCK_TOPK=TOPK,
            COMPRESS_RATIO=COMPRESS_RATIO,
            TOKEN_TOPK=TOKEN_TOPK,
            OUTPUT_WIDTH=OUTPUT_WIDTH,
        )


def qsa_stable_topk_(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    blocks: torch.Tensor,
    block_n: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Make an already-computed compressed-block top-k reproducible, in place.

    ``block_n`` / ``num_warps`` exist for the microbenchmark; the defaults are
    the measured ones.  Decode launches only a handful of workgroups, so it is
    latency- rather than throughput-bound and wants a wider block; prefill has
    thousands of rows in flight and wants the narrow, high-occupancy one.
    """

    if blocks.ndim != 2 or logits.ndim != 2:
        raise ValueError("QSA stable top-k needs 2-D logits and selections")
    if blocks.shape[0] != logits.shape[0]:
        raise ValueError("QSA stable top-k row counts must match")
    if visible_blocks.shape[0] < blocks.shape[0]:
        raise ValueError("QSA stable top-k needs one visible count per row")
    if blocks.stride(1) != 1 or logits.stride(1) != 1:
        raise ValueError("QSA stable top-k needs row-contiguous inputs")
    rows, topk = blocks.shape
    if not rows or not topk:
        return blocks
    if block_n is None:
        block_n = 2048 if rows <= 64 else 1024
    if num_warps is None:
        num_warps = 8 if rows <= 64 else 4
    _qsa_stable_topk_kernel[(rows,)](
        logits,
        visible_blocks,
        blocks,
        logits.stride(0),
        blocks.stride(0),
        logits.shape[1],
        TOPK=topk,
        BLOCK_N=block_n,
        BLOCK_K=triton.next_power_of_2(topk),
        num_warps=num_warps,
    )
    return blocks


# Row count at or below which the repair is fused into the expand.  Decode
# batches are launch-bound and want one kernel; prefill wants the expand spread
# over many programs per row (measured: fusing costs prefill far more than the
# launch it saves).
_STABLE_TOPK_FUSED_MAX_ROWS = 64


def qsa_stable_topk_expand(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    blocks: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor,
    column_block: int = 1024,
) -> torch.Tensor:
    """Deterministic repair + index expansion in one launch (decode shapes)."""

    rows, topk = blocks.shape
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    if topk != token_topk // compress_ratio:
        raise ValueError("QSA fused stable top-k has an invalid selection width")
    output_width = token_topk + compress_ratio - 1
    if out.shape != (rows, output_width):
        raise ValueError("QSA fused stable expansion output has an invalid shape")
    if blocks.stride(1) != 1 or logits.stride(1) != 1:
        raise ValueError("QSA fused stable top-k needs row-contiguous inputs")
    if not rows or not topk:
        return out
    _qsa_stable_topk_expand_kernel[(rows,)](
        logits,
        visible_blocks,
        blocks,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        logits.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        out.stride(0),
        out.stride(1),
        logits.shape[1],
        rows,
        sequence_lengths.shape[0],
        TOPK=topk,
        BLOCK_N=2048,
        BLOCK_K=triton.next_power_of_2(topk),
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        NUM_COLUMN_TILES=triton.cdiv(output_width, column_block),
        num_warps=8,
    )
    return out


def expand_qsa_block_indices_cuda(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand compressed blocks and compact the causal tail of the open group."""

    if not block_indices.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA index expansion requires a GPU and Triton")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA expansion output has an invalid shape")
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )
    return out


def qsa_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    num_columns: int | None = None,
) -> torch.Tensor:
    """Score, select, and expand QSA indices without host synchronization."""

    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out

    columns = page_table.shape[1] * k_cache.shape[1]
    block_topk = token_topk // compress_ratio
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    chunk_rows = min(rows, rows_per_chunk)
    blocks_buffer = torch.empty(
        (chunk_rows, block_topk), dtype=torch.int32, device=q.device
    )
    topk_workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
    )
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, visible_blocks = qsa_mqa_paged(
            q[row_slice],
            k_cache,
            page_table,
            token_to_req[row_slice],
            query_positions[row_slice],
            sequence_lengths,
            compress_ratio,
            num_columns=num_columns,
        )
        blocks = blocks_buffer[: row_end - row_start]
        use_cooperative_topk = (
            current_platform.is_cuda()
            and blocks.shape[0] <= 32
            and logits.stride(0) % 4 == 0
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        if use_cooperative_topk:
            torch.ops._C.cooperative_topk(
                logits,
                visible_blocks,
                blocks,
                topk_workspace,
                block_topk,
                columns,
            )
        elif current_platform.is_cuda():
            torch.ops._C.persistent_topk(
                logits,
                visible_blocks,
                blocks,
                topk_workspace,
                block_topk,
                columns,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                1,
                visible_blocks,
                blocks,
                blocks.shape[0],
                logits.stride(0),
                logits.stride(1),
                block_topk,
            )
        if _stable_topk_enabled():
            # Two ways to pay for determinism, and which is cheaper depends on
            # whether the caller is building a graph or launching from Python.
            #
            #   captured  an extra node between the top-k and the expand costs
            #             its GPU time only (+1.1 us/layer measured); folding
            #             the expand into one program per row to avoid that
            #             node costs *more* (+2.1 us), because the expand loses
            #             its OUTPUT_WIDTH/COLUMN_BLOCK-way parallelism.
            #   eager     an extra Triton launch costs the Python launcher
            #             (+31 us/layer measured, ~25x its GPU time), so
            #             saving the launch is well worth losing the
            #             parallelism (+12 us).
            #
            # ``_qsa_num_columns`` already branches on the same predicate.
            if (
                blocks.shape[0] <= _STABLE_TOPK_FUSED_MAX_ROWS
                and not torch.cuda.is_current_stream_capturing()
            ):
                qsa_stable_topk_expand(
                    logits,
                    visible_blocks,
                    blocks,
                    query_positions[row_slice],
                    sequence_lengths,
                    token_to_req[row_slice],
                    compress_ratio,
                    token_topk,
                    out[row_slice],
                )
                continue
            qsa_stable_topk_(logits, visible_blocks, blocks)
        expand_qsa_block_indices_cuda(
            blocks,
            query_positions[row_slice],
            sequence_lengths,
            token_to_req[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
        )
    return out


def qsa_dense_causal_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """Dense causal GQA over the paged BF16 K/V cache.

    Used by the ``VLLM_GFX908_QSA_DENSE_SHORT`` fast path: when every request
    in the batch has a context no longer than ``indexer_budget`` the QSA
    selection is the identity (it selects the whole causal prefix), so the
    sparse kernel computes exactly dense causal attention over a materialised
    index list.  This runs the same math through vLLM's Triton unified
    attention, which tiles the query dimension (BLOCK_M=16 shared by
    ``BLOCK_Q`` consecutive tokens x ``num_queries_per_kv`` heads) instead of
    launching one program per token, and walks the KV once per tile instead of
    re-gathering a 2051-wide index per token.

    ``k_cache``/``v_cache`` are the ``[pages, page_size, kv_heads, head_dim]``
    views the QSA owner already builds from the merged ``(B, H, N, 2 * D)``
    cache; only their strides are passed to the kernel, so the K/V interleaving
    on the last axis is irrelevant.
    """

    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA dense attention requires a GPU and Triton")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA dense attention received invalid Q/K/V shapes")
    if out.shape != q.shape:
        raise ValueError("QSA dense output must match its query")
    if block_table.ndim != 2 or query_start_loc.ndim != 1 or seq_lens.ndim != 1:
        raise ValueError("QSA dense attention metadata has invalid shapes")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA dense attention requires valid grouped-query heads")
    assert q.dtype == k_cache.dtype == v_cache.dtype == out.dtype
    assert q.stride(2) == 1 and out.stride(2) == 1
    assert k_cache.stride(3) == 1 and v_cache.stride(3) == 1
    if not q.shape[0]:
        return out

    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    # ``num_seqs`` is taken from ``len(seqused_k)``; the two metadata tensors
    # must therefore agree on the request count (they can be padded).
    num_requests = query_start_loc.shape[0] - 1
    if seq_lens.shape[0] < num_requests:
        raise ValueError("QSA dense attention seq_lens is shorter than the batch")

    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_query_len,
        seqused_k=seq_lens[:num_requests],
        max_seqlen_k=max_seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    return out


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 K/V caches."""

    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA sparse attention requires a GPU and Triton")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    if current_platform.is_rocm():
        # gfx908: BLOCK_M of 4/8 miscompiles this kernel's online-softmax
        # dot chain (garbage output for GROUP_SIZE <= 8, i.e. TP4 with 2 KV
        # heads). The head mask already handles GROUP_SIZE < BLOCK_M, so a
        # 16-row minimum is correctness-neutral.
        block_m = max(block_m, 16)
    base_programs = q.shape[0] * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2
    # gfx942 and gfx950 have a 64 KiB LDS limit. One software-pipelining
    # stage keeps the wide TP4 tile within that shared-memory budget.
    partial_stages = 1 if current_platform.is_rocm() else 2

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    # Avoid empty splits when the selection width is smaller than the profile.
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=partial_stages,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA cache stores require a GPU and Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""

    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA compression requires a GPU and Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


@triton.jit
def _qsa_prefill_selection_words_kernel(
    indices_ptr,
    words_ptr,
    stride_indices_row,
    stride_words_row,
    num_rows,
    num_entries,
    num_words,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_E: tl.constexpr,
) -> None:
    """Pack the expanded QSA selection into a per-row compressed-block bitmap.

    ``logical_indices`` lists ``token_topk + compress_ratio - 1`` token ids per
    query: ``compress_ratio`` consecutive columns per selected compressed block
    followed by the ragged tail of the open group.  Both runs start on a column
    that is a multiple of ``compress_ratio`` (``expanded_count`` is
    ``complete_blocks * compress_ratio``), so reading every ``compress_ratio``-th
    column visits each selected block exactly once.

    A block is marked; the attention kernel re-applies the causal bound, which
    is what clips the open block back to the ragged tail.  The scorer only ever
    makes complete causal blocks visible (``columns < visible`` in
    ``_qsa_mqa_paged_kernel``), so block-granular marking plus a causal mask
    reproduces the expanded token set exactly.
    """

    row = tl.program_id(0)
    entries = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    valid = (row < num_rows) & (entries < num_entries)
    token = tl.load(
        indices_ptr + row * stride_indices_row + entries * COMPRESS_RATIO,
        mask=valid,
        other=-1,
    )
    valid &= token >= 0
    block = token // COMPRESS_RATIO
    word = block // 32
    bit = block % 32
    valid &= word < num_words
    ones = tl.full(bit.shape, 1, dtype=tl.int32)
    tl.atomic_or(
        words_ptr + row * stride_words_row + word,
        ones << bit,
        mask=valid,
    )


@triton.jit
def _qsa_prefill_block_mask_kernel(
    indices_ptr,
    mask_ptr,
    stride_indices_row,
    stride_mask_row,
    num_rows,
    num_entries,
    num_blocks,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_E: tl.constexpr,
) -> None:
    """Scatter the selected compressed blocks of each row into a byte mask.

    The ``compress_ratio`` block ids a row selects are distinct, so every store
    lands on its own byte and no atomic is needed -- the atomic-``or`` variant
    of this serialises ~8-way inside a wave because one program's lanes all
    target the same row's handful of bitmap words.
    """

    row = tl.program_id(0)
    entries = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    valid = (row < num_rows) & (entries < num_entries)
    token = tl.load(
        indices_ptr + row * stride_indices_row + entries * COMPRESS_RATIO,
        mask=valid,
        other=-1,
    )
    block = token // COMPRESS_RATIO
    valid &= (token >= 0) & (block < num_blocks)
    tl.store(
        mask_ptr + row * stride_mask_row + block,
        tl.full(block.shape, 1, dtype=tl.int8),
        mask=valid,
    )


@triton.jit
def _qsa_prefill_pack_mask_kernel(
    mask_ptr,
    words_ptr,
    stride_mask_row,
    stride_words_row,
    num_rows,
    num_blocks,
    num_words,
    BLOCK_W: tl.constexpr,
) -> None:
    """Pack the byte mask into 32-blocks-per-word bitmap words.

    The attention kernel then needs one ``int32`` per query row per key tile
    instead of a ``BLOCK_M x BLOCK_N`` byte gather.
    """

    row = tl.program_id(0)
    words = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    lanes = tl.arange(0, 32)
    block = words[:, None] * 32 + lanes[None, :]
    valid = (row < num_rows) & (words[:, None] < num_words) & (block < num_blocks)
    bits = tl.load(mask_ptr + row * stride_mask_row + block, mask=valid, other=0)
    ones = tl.full(block.shape, 1, dtype=tl.int32)
    packed = tl.sum(tl.where(bits != 0, ones << lanes[None, :], 0), axis=1)
    tl.store(
        words_ptr + row * stride_words_row + words,
        packed,
        mask=(row < num_rows) & (words < num_words),
    )


@triton.jit
def _qsa_prefill_tile_starts_kernel(
    query_start_loc_ptr,
    tile_starts_ptr,
    num_requests,
    BLOCK_Q: tl.constexpr,
    NUM_REQ_POW2: tl.constexpr,
) -> None:
    """Exclusive prefix sum of per-request query-tile counts (one program).

    ``tile_starts[r]`` is the first tile id of request ``r`` and
    ``tile_starts[num_requests]`` the total tile count, so the attention kernel
    can map a flat tile id back to its request without a host sync.
    """

    request = tl.arange(0, NUM_REQ_POW2)
    in_range = request < num_requests
    start = tl.load(query_start_loc_ptr + request, mask=in_range, other=0)
    end = tl.load(query_start_loc_ptr + request + 1, mask=in_range, other=0)
    query_len = tl.where(in_range, end - start, 0)
    tiles = (query_len + BLOCK_Q - 1) // BLOCK_Q
    exclusive = tl.cumsum(tiles, axis=0) - tiles
    tl.store(tile_starts_ptr + request, exclusive, mask=in_range)
    tl.store(tile_starts_ptr + num_requests, tl.sum(tiles, axis=0))


@triton.jit
def _qsa_prefill_tiled_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    sel_words_ptr,
    query_positions_ptr,
    query_start_loc_ptr,
    tile_starts_ptr,
    block_table_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_words_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_cache_blocks,
    num_requests,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NUM_REQ_POW2: tl.constexpr,
) -> None:
    """Query-tiled sparse GQA for prefill-shaped batches.

    One program owns ``BLOCK_Q`` consecutive queries of one request across all
    ``GROUP_SIZE`` query heads of one KV head, so the MFMA M dimension is
    ``BLOCK_Q * GROUP_SIZE`` instead of the 6-of-16 padding the per-token
    kernel feeds it.  The tile walks its shared causal key range once (a
    contiguous token run, so the page gather is two block-table entries rather
    than a 2051-wide scatter) and applies the per-row selection as a bitmap
    test; a key tile no row selects is skipped entirely.
    """

    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    total_tiles = tl.load(tile_starts_ptr + num_requests)
    if tile >= total_tiles:
        return

    requests = tl.arange(0, NUM_REQ_POW2)
    in_range = requests < num_requests
    starts = tl.load(tile_starts_ptr + requests, mask=in_range, other=0x7FFFFFFF)
    request = tl.sum(((starts <= tile) & in_range).to(tl.int32), axis=0) - 1
    request_first_tile = tl.load(tile_starts_ptr + request)
    row_begin = tl.load(query_start_loc_ptr + request)
    row_end = tl.load(query_start_loc_ptr + request + 1)
    row_base = row_begin + (tile - request_first_tile) * BLOCK_Q

    lanes = tl.arange(0, BLOCK_M)
    query_index = lanes // GROUP_SIZE
    head_index = lanes % GROUP_SIZE
    row = row_base + query_index
    row_valid = (query_index < BLOCK_Q) & (row < row_end)
    safe_row = tl.where(row_valid, row, row_begin)
    positions = tl.load(query_positions_ptr + safe_row, mask=row_valid, other=-1)
    last_position = tl.max(positions, axis=0)

    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + safe_row[:, None] * stride_q_row
        + (first_head + head_index)[:, None] * stride_q_head
        + dim_offsets[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    num_tiles = (last_position + BLOCK_N) // BLOCK_N
    for tile_id in range(0, num_tiles):
        tokens = tile_id * BLOCK_N + column_offsets
        blocks = tokens // COMPRESS_RATIO
        # BLOCK_N // COMPRESS_RATIO divides 32, so one bitmap word per row
        # covers the whole key tile and the load stays wave-uniform in N.
        word = tile_id * (BLOCK_N // COMPRESS_RATIO) // 32
        bits = blocks - word * 32
        words = tl.load(
            sel_words_ptr + safe_row * stride_words_row + word,
            mask=row_valid,
            other=0,
        )
        selected = ((words[:, None] >> bits[None, :]) & 1) != 0
        valid = row_valid[:, None] & (tokens[None, :] <= positions[:, None]) & selected
        if tl.max(valid.to(tl.int32)) > 0:
            logical_page = tokens // PAGE_SIZE
            page_offset = tokens % PAGE_SIZE
            page_valid = logical_page < PAGE_TABLE_WIDTH
            physical_page = tl.load(
                block_table_ptr
                + request * stride_table_req
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
                mask=page_valid,
                other=-1,
            )
            page_valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
            # physical_page * block stride can overflow int32 for large caches.
            safe_page = tl.maximum(physical_page, 0).to(tl.int64)
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[:, None],
                mask=page_valid[None, :],
                other=0.0,
            )
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=page_valid[:, None],
                other=0.0,
            )
            valid &= page_valid[None, :]
            scores = tl.dot(query, keys)
            # Scaling scores avoids re-quantizing a scaled query to BF16.
            scores *= softmax_scale_log2
            scores = tl.where(valid, scores, -1.0e20)
            next_max = tl.maximum(max_value, tl.max(scores, axis=1))
            alpha = tl.math.exp2(max_value - next_max)
            probabilities = tl.where(
                valid, tl.math.exp2(scores - next_max[:, None]), 0.0
            )
            accumulator = tl.dot(
                probabilities.to(values.dtype),
                values,
                acc=accumulator * alpha[:, None],
            )
            normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
            max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    tl.store(
        output_ptr
        + safe_row[:, None] * stride_output_row
        + (first_head + head_index)[:, None] * stride_output_head
        + dim_offsets[None, :],
        normalized_output,
        mask=row_valid[:, None],
    )

def _qsa_tiled_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def qsa_prefill_tiled_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_seq_len: int,
    compress_ratio: int,
    out: torch.Tensor,
    selection_words: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query-tiled sparse GQA over the paged BF16 K/V cache (prefill shapes).

    Computes exactly what ``qsa_sparse_paged_attention`` computes for the same
    ``logical_indices`` (up to fp32 summation order), but tiles the query
    dimension: ``BLOCK_Q`` consecutive queries of one request share one pass
    over their common causal key range, with the per-query selection applied as
    a compressed-block bitmap.  That removes the per-token 2051-wide index
    gather and the ``6``-of-``16`` MFMA row padding of the decode-shaped kernel.

    Requires a prefill-shaped batch: ``query_positions`` must be the logical
    position of every row and ``query_start_loc`` the per-request row offsets.
    ``max_seq_len`` bounds the batch's contexts and sizes the bitmap.
    """

    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA tiled attention requires a GPU and Triton")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA tiled attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if query_positions.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA tiled attention metadata has invalid shapes")
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA tiled attention needs a query_start_loc")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA tiled attention cache and block table must be nonempty")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA tiled attention requires valid grouped-query heads")
    if compress_ratio < 1 or logical_indices.shape[1] < compress_ratio:
        raise ValueError("QSA tiled attention requires a valid compression ratio")
    if max_seq_len <= 0:
        raise ValueError("QSA tiled attention requires a positive max_seq_len")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    if out.shape != q.shape:
        raise ValueError("QSA tiled output must match its query")
    assert out.dtype == q.dtype and out.stride(2) == 1
    num_rows = q.shape[0]
    if not num_rows:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = _qsa_tiled_env_int("VLLM_GFX908_QSA_TILED_BM", 128)
    block_m = max(block_m, triton.next_power_of_2(group_size), 16)
    block_n = _qsa_tiled_env_int("VLLM_GFX908_QSA_TILED_BN", 32)
    num_warps = _qsa_tiled_env_int("VLLM_GFX908_QSA_TILED_WARPS", 4)
    blocks_per_tile = block_n // compress_ratio
    if block_n % compress_ratio or blocks_per_tile < 1 or 32 % blocks_per_tile:
        raise ValueError("QSA tiled BLOCK_N must cover 1..32 compressed blocks")
    block_q = max(block_m // group_size, 1)

    if query_positions.dtype != torch.int32:
        query_positions = query_positions.to(torch.int32)
    if query_start_loc.dtype != torch.int32:
        query_start_loc = query_start_loc.to(torch.int32)
    query_positions = query_positions.contiguous()
    query_start_loc = query_start_loc.contiguous()

    num_requests = query_start_loc.shape[0] - 1
    num_words = triton.cdiv(triton.cdiv(max_seq_len, compress_ratio), 32)
    if selection_words is None:
        selection_words = torch.zeros(
            (num_rows, num_words), dtype=torch.int32, device=q.device
        )
    else:
        if selection_words.shape[0] < num_rows or selection_words.shape[1] < num_words:
            raise ValueError("QSA tiled selection workspace is too small")
        selection_words = selection_words[:num_rows, :num_words]
        selection_words.zero_()

    num_entries = triton.cdiv(logical_indices.shape[1], compress_ratio)
    entry_block = 256
    num_blocks = triton.cdiv(max_seq_len, compress_ratio)
    # The scatter+pack build is ~6x faster than the atomic one (measured 94 us
    # vs 604 us at 4096 rows: one program's lanes all target the same row's
    # handful of bitmap words, so atomic_or serialises ~8-way inside a wave),
    # but it needs a [rows, blocks] int8 scratch.  Fall back to atomics when
    # that scratch would be large.
    mask_mode = os.environ.get("VLLM_GFX908_QSA_TILED_MASK", "auto")
    if mask_mode == "auto":
        mask_mode = "atomic" if num_rows * num_blocks > (64 << 20) else "scatter"
    if mask_mode == "atomic":
        _qsa_prefill_selection_words_kernel[
            (num_rows, triton.cdiv(num_entries, entry_block))
        ](
            logical_indices,
            selection_words,
            logical_indices.stride(0),
            selection_words.stride(0),
            num_rows,
            num_entries,
            num_words,
            COMPRESS_RATIO=compress_ratio,
            BLOCK_E=entry_block,
            num_warps=4,
        )
    else:
        block_mask = torch.zeros(
            (num_rows, num_blocks), dtype=torch.int8, device=q.device
        )
        _qsa_prefill_block_mask_kernel[
            (num_rows, triton.cdiv(num_entries, entry_block))
        ](
            logical_indices,
            block_mask,
            logical_indices.stride(0),
            block_mask.stride(0),
            num_rows,
            num_entries,
            num_blocks,
            COMPRESS_RATIO=compress_ratio,
            BLOCK_E=entry_block,
            num_warps=4,
        )
        word_block = 8
        _qsa_prefill_pack_mask_kernel[
            (num_rows, triton.cdiv(num_words, word_block))
        ](
            block_mask,
            selection_words,
            block_mask.stride(0),
            selection_words.stride(0),
            num_rows,
            num_blocks,
            num_words,
            BLOCK_W=word_block,
            num_warps=4,
        )

    num_req_pow2 = triton.next_power_of_2(max(num_requests, 1))
    tile_starts = torch.empty(
        (num_requests + 1,), dtype=torch.int32, device=q.device
    )
    _qsa_prefill_tile_starts_kernel[(1,)](
        query_start_loc,
        tile_starts,
        num_requests,
        BLOCK_Q=block_q,
        NUM_REQ_POW2=num_req_pow2,
        num_warps=1,
    )

    max_tiles = triton.cdiv(num_rows, block_q) + num_requests
    _qsa_prefill_tiled_kernel[(max_tiles, k_cache.shape[2])](
        q,
        k_cache,
        v_cache,
        selection_words,
        query_positions,
        query_start_loc,
        tile_starts,
        block_table,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        selection_words.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        k_cache.shape[0],
        num_requests,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_Q=block_q,
        NUM_REQ_POW2=num_req_pow2,
        num_warps=num_warps,
        num_stages=1,
    )
    return out



__all__ = [
    "expand_qsa_block_indices_cuda",
    "qsa_compress_groups_with_ratio",
    "qsa_dense_causal_paged_attention",
    "qsa_mqa_paged",
    "qsa_prefill_tiled_attention",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
]
