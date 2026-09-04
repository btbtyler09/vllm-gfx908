# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp weight-free QSA indexer."""

from __future__ import annotations

from typing import cast

import os

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)

from ..common.qsa_cache import (
    QSACompressedKeyCache,
    QSAForwardMetadata,
    QSAKeyStateCache,
    canonical_qsa_rope_positions,
)


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's exact 1D/MRoPE composition to QSA heads."""

    num_tokens, _, head_dim = tensor.shape
    rotary_dim = rotary_emb.rotary_dim
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        shape = tensor.shape
        tensor, _ = triton_mrope(
            tensor.reshape(num_tokens, -1),
            tensor.new_empty((num_tokens, head_dim)),
            cos,
            sin,
            rotary_emb.mrope_section,
            head_dim,
            rotary_dim,
            rotary_emb.mrope_interleaved,
            rotary_emb.is_neox_style,
        )
        return tensor.reshape(shape)

    rotated = rotary_emb.apply_rotary_emb(
        tensor[..., :rotary_dim],
        cos,
        sin,
    )
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


def apply_qsa_rmsnorm(
    norm: GemmaRMSNorm,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Gemma RMSNorm over the last dim as one Triton launch.

    The portable GemmaRMSNorm runs as ~7 eager elementwise/reduce kernels
    inside the QSA custom op (no inductor fusion there); the grouped Gemma
    kernel from the hyper-connection ops does the same (1 + w) affine in a
    single launch with a shared [head_dim] weight.
    """

    if tensor.dim() == 2 and tensor.stride(1) == 1 and tensor.is_cuda:
        from .ops.hc import grouped_gemma_rmsnorm

        return grouped_gemma_rmsnorm(
            tensor, norm.weight, float(norm.variance_epsilon), 1
        )
    return cast(torch.Tensor, norm(tensor))


class QSAIndexer(nn.Module):
    """Replicated Q/K projection plus paged, weight-free QSA selection.

    ``prefix`` must be the checkpoint's indexer prefix, normally
    ``model.layers.N.self_attn.indexer``.  Consequently the trainable names are
    ``index_qk_proj``, ``q_layernorm`` and ``k_layernorm`` under that prefix.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        rotary_emb: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.cache_config is None:
            raise ValueError("QSA requires a paged KV cache")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")

        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.rotary_emb = rotary_emb
        self.prefix = prefix
        # MTP step 0 selects the target-aligned rows; later steps reuse them
        # while continuing to update the QSA side cache.
        self.skip_topk = False

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )

        cache_config = vllm_config.cache_config
        cache_prefix = f"{prefix}." if prefix else ""
        self.raw_key_cache = QSAKeyStateCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            cache_rope_positions=vllm_config.model_config.uses_mrope,
            prefix=f"{cache_prefix}raw_key_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
            vllm_config=vllm_config,
        )
        self.compressed_key_cache = QSACompressedKeyCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.compress_ratio,
            prefix=f"{cache_prefix}compressed_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )

    @property
    def output_width(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        with_q: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Project replicated Q/K, normalize+rotate Q, and preserve raw K.

        ``with_q=False`` keeps the fused projection (its K half feeds the
        compressed-key state that decode still needs) but drops the query
        norm + RoPE, which only exist to feed the scorer.
        """

        qk, _ = self.index_qk_proj(hidden_states)
        q_raw, token_k = qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        token_k = token_k.reshape(-1, 1, self.index_head_dim)
        if not with_q:
            return None, token_k
        q = q_raw.reshape(-1, self.index_n_heads, self.index_head_dim)
        q = apply_qsa_rmsnorm(
            self.q_layernorm,
            q.reshape(-1, self.index_head_dim),
        ).reshape_as(q)
        q = apply_qsa_rope(self.rotary_emb, positions, q)
        return q, token_k

    def normalize_compressed_keys(
        self,
        compressed_keys: torch.Tensor,
        first_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize pooled K and apply the first token's exact group position."""

        keys = compressed_keys.reshape(-1, self.index_head_dim)
        keys = apply_qsa_rmsnorm(self.k_layernorm, keys).reshape(
            -1, 1, self.index_head_dim
        )
        if getattr(self.rotary_emb, "mrope_section", None):
            positions = first_rope_positions.transpose(0, 1)
        else:
            positions = first_rope_positions[:, 0]
        return apply_qsa_rope(self.rotary_emb, positions, keys)

    def _metadata(
        self,
    ) -> tuple[QSAForwardMetadata, QSAForwardMetadata] | None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            return None
        raw = cast(QSAForwardMetadata, metadata[self.raw_key_cache.prefix])
        compressed = cast(
            QSAForwardMetadata, metadata[self.compressed_key_cache.prefix]
        )
        if raw.num_actual_tokens != compressed.num_actual_tokens:
            raise RuntimeError("QSA side-cache metadata token counts disagree")
        if not raw.logical_positions.is_cuda and (
            not torch.equal(raw.logical_positions, compressed.logical_positions)
        ):
            raise RuntimeError("QSA side-cache metadata positions disagree")
        return raw, compressed

    def _update_and_compress(
        self,
        token_k: torch.Tensor,
        positions: torch.Tensor,
        raw_metadata: QSAForwardMetadata,
        compressed_metadata: QSAForwardMetadata,
    ) -> None:
        num_tokens = raw_metadata.num_actual_tokens
        raw_key_cache = self.raw_key_cache.key_cache
        rope_position_cache = self.raw_key_cache.rope_position_cache
        from .ops.qsa import qsa_compress_groups_with_ratio, qsa_store_cache_rows

        if rope_position_cache is None:
            position_rows = raw_metadata.logical_positions.view(-1, 1, 1).expand(
                -1, 1, 3
            )
        else:
            position_rows = canonical_qsa_rope_positions(positions)[:num_tokens].to(
                device=raw_key_cache.device
            )
        pooled, first_positions = qsa_compress_groups_with_ratio(
            token_k[:num_tokens],
            position_rows,
            raw_key_cache,
            raw_metadata.block_table,
            raw_metadata.token_to_req,
            raw_metadata.query_start_loc,
            raw_metadata.logical_positions,
            compressed_metadata.slot_mapping,
            self.compress_ratio,
            rope_position_cache,
        )
        normalized = self.normalize_compressed_keys(pooled, first_positions)
        qsa_store_cache_rows(
            self.compressed_key_cache.kv_cache,
            compressed_metadata.slot_mapping,
            normalized,
        )
        qsa_store_cache_rows(
            raw_key_cache,
            raw_metadata.slot_mapping,
            token_k[:num_tokens],
        )
        if rope_position_cache is not None:
            qsa_store_cache_rows(
                rope_position_cache,
                raw_metadata.slot_mapping,
                position_rows,
            )

    def _select(
        self,
        q: torch.Tensor,
        metadata: QSAForwardMetadata,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        from .ops.qsa import qsa_select_paged_tokens

        return qsa_select_paged_tokens(
            q,
            self.compressed_key_cache.kv_cache,
            metadata.block_table,
            metadata.token_to_req,
            metadata.logical_positions,
            metadata.seq_lens,
            self.token_topk,
            self.compress_ratio,
            out,
            num_columns=self._qsa_num_columns(metadata),
        )

    def _qsa_num_columns(self, metadata: QSAForwardMetadata) -> int | None:
        """gfx908: bound the indexer scorer's columns to the batch's real
        context (VLLM_GFX908_QSA_NUM_COLUMNS=1) instead of max_model_len/4.
        Under graph capture max_seq_len is the capture's dummy value, so the
        captured decode graphs keep the full width; eager prefill shrinks."""
        if os.environ.get("VLLM_GFX908_QSA_NUM_COLUMNS", "0") != "1":
            return None
        max_seq_len = int(getattr(metadata, "max_seq_len", 0) or 0)
        if max_seq_len <= 0 or torch.cuda.is_current_stream_capturing():
            return None
        capacity = metadata.block_table.shape[1] * self.compressed_key_cache.kv_cache.shape[1]
        cols = (max_seq_len + self.compress_ratio - 1) // self.compress_ratio
        cols = ((cols + 63) // 64) * 64
        return min(capacity, max(cols, 64))

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        skip_select: bool = False,
        force_select: bool = False,
    ) -> torch.Tensor:
        """Return fixed-width request-relative token indices padded with ``-1``.

        ``skip_select`` (the ``VLLM_GFX908_QSA_DENSE_SHORT`` fast path) keeps
        every cache-state update - raw keys, the compressor-state ring, the
        pooled/normalized compressed keys and the packed RoPE positions - and
        drops only the scorer / top-k / expand chain, whose result the caller
        does not consume because the selection would be the identity.  The
        returned buffer is then stale; the caller must not read it.
        ``force_select`` overrides ``skip_topk`` when the buffer went stale
        that way.
        """

        metadata = self._metadata()
        if metadata is None:
            # Preserve step-0 indices when later MTP steps reuse the buffer.
            if self.skip_topk and out is not None:
                return out
            result = torch.full(
                (hidden_states.shape[0], self.output_width),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            if out is not None:
                out.copy_(result)
                return out
            return result
        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        reuse = self.skip_topk and not force_select
        q, token_k = self.project_qk(
            hidden_states[:num_tokens],
            positions[..., :num_tokens],
            with_q=not (skip_select or reuse),
        )
        self._update_and_compress(
            token_k,
            positions[..., :num_tokens],
            raw_metadata,
            compressed_metadata,
        )
        if skip_select or reuse:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            return out
        return self._select(q, compressed_metadata, out)


__all__ = ["QSAIndexer", "apply_qsa_rope"]
