# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD ROCm QSA owner with Triton kernels."""

from __future__ import annotations

import os
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import gfx908_qsa_glue
from . import model
from .indexer_qsa import QSAIndexer


def _dense_short_enabled() -> bool:
    """``VLLM_GFX908_QSA_DENSE_SHORT=1`` opts into the dense-causal fast path.

    Read per call (one dict lookup) so a process can flip it between forward
    passes; the default is off.
    """

    return os.environ.get("VLLM_GFX908_QSA_DENSE_SHORT", "0").strip() in (
        "1",
        "true",
        "True",
    )


def _prefill_tiled_enabled() -> bool:
    """``VLLM_GFX908_QSA_PREFILL_TILED=1`` opts into the tiled prefill kernel.

    Covers the range the dense-short path cannot: contexts *above* the indexer
    budget, where the selection is genuinely sparse and still has to be built.
    Read per call (one dict lookup); the default is off.
    """

    return os.environ.get("VLLM_GFX908_QSA_PREFILL_TILED", "0").strip() in (
        "1",
        "true",
        "True",
    )


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")
        self.supports_quant_query_input = False

    def forward_dense_causal(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Dense causal attention over the paged KV, bypassing the selection.

        Only valid when every request's context fits the indexer budget, in
        which case the QSA selection is the whole causal prefix and this is the
        same math (see ``Qwen4ExpQSAAttention._dense_short_eligible``).
        """

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")

        from .ops.qsa import qsa_dense_causal_paged_attention

        qsa_dense_causal_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len,
            self.scale,
            output[:num_tokens],
        )
        return output

    def forward_qsa_tiled(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        query_positions: torch.Tensor,
        compress_ratio: int,
    ) -> torch.Tensor:
        """Sparse QSA with the query dimension tiled (prefill-shaped batches).

        Same selection and same math as :meth:`forward_qsa`; the kernel tiles
        ``BLOCK_Q`` consecutive queries of a request into the MFMA M dimension
        and walks their shared causal key range once instead of re-gathering a
        2051-wide index list per token.  Only valid when every row's logical
        position is known (prefill), which is why the gate requires
        ``max_query_len > 1``.
        """

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output
        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")

        from .ops.qsa import qsa_prefill_tiled_attention

        qsa_prefill_tiled_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            topk_buffer[:num_tokens],
            attn_metadata.block_table,
            query_positions[:num_tokens],
            attn_metadata.query_start_loc,
            int(attn_metadata.max_seq_len),
            compress_ratio,
            output[:num_tokens],
        )
        return output

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and text_only
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 cache storage")
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        # Set when a dense-causal step skipped the selection, so the buffer
        # holds the previous step's rows.
        self._topk_buffer_stale = False
        # gfx908 fused decode glue (VLLM_GFX908_QSA_GLUE=1): static, so the
        # compiled forward never branches on it at runtime.
        self._gfx908_qsa_glue = gfx908_qsa_glue.qsa_glue_layer_supported(self, vllm_config)

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _dense_short_eligible(self, main_metadata: FlashAttentionMetadata) -> bool:
        """True when the QSA selection is provably the identity for this batch.

        The indexer selects ``indexer_budget`` tokens as ``budget /
        compress_ratio`` compressed blocks plus the ragged tail, and a query at
        logical position ``p`` has ``(p + 1) // compress_ratio`` complete blocks
        visible.  Once ``p + 1 <= indexer_budget`` for every query, top-k has
        fewer candidates than slots, so it returns *every* visible block and the
        expansion covers the whole causal prefix ``[0, p]`` - exactly what dense
        causal attention computes.  ``max_seq_len`` (a host-side upper bound on
        the batch's context, cached prefix included) bounds every ``p + 1``.

        Guards: cudagraph capture/replay must never see a data-dependent branch,
        so only eager launches qualify; ``max_query_len > 1`` keeps this to
        prefill-shaped batches (pure decode keeps the tuned sparse decode
        kernel); and MTP index reuse (``skip_topk``) needs a live top-k buffer.
        """

        if not _dense_short_enabled():
            return False
        if self.indexer.skip_topk:
            return False
        if int(getattr(main_metadata, "max_query_len", 1) or 1) <= 1:
            return False
        max_seq_len = int(getattr(main_metadata, "max_seq_len", 0) or 0)
        if max_seq_len <= 0 or max_seq_len > self.indexer.token_topk:
            return False
        return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE

    def _prefill_tiled_eligible(self, main_metadata: FlashAttentionMetadata) -> bool:
        """True when the tiled prefill attention kernel may replace the sparse one.

        Complement of ``_dense_short_eligible``: the batch is prefill-shaped
        (``max_query_len > 1``) but its context is *above* the indexer budget,
        so the selection is real and the sparse kernel is the one running.  The
        tiled kernel needs a logical position per row, which only a
        prefill-shaped batch's side metadata provides, and it is an eager-only
        path for the same reason the dense-short one is.
        """

        if not _prefill_tiled_enabled():
            return False
        if self.indexer.skip_topk:
            return False
        if int(getattr(main_metadata, "max_query_len", 1) or 1) <= 1:
            return False
        if int(getattr(main_metadata, "max_seq_len", 0) or 0) <= self.indexer.token_topk:
            return False
        return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        dense_short = self._dense_short_eligible(main_metadata)
        # A skipped selection leaves the top-k buffer stale, so an MTP step
        # that would have reused it recomputes instead of reading garbage.
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
            skip_select=dense_short,
            force_select=getattr(self, "_topk_buffer_stale", False)
            and self.indexer.skip_topk,
        )
        self._topk_buffer_stale = dense_short
        if not dense_short and selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        if dense_short:
            impl.forward_dense_causal(
                query,
                self.kv_cache,
                main_metadata,
                output,
            )
            return
        if self._prefill_tiled_eligible(main_metadata):
            impl.forward_qsa_tiled(
                self,
                query,
                self.kv_cache,
                main_metadata,
                output,
                side_metadata.logical_positions,
                self.indexer.compress_ratio,
            )
            return
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
        )

    def _run_qsa_glue(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        qkv: torch.Tensor,
        query: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """gfx908 fused decode glue: the whole q/k/v-projection-to-attention
        transaction from the raw ``qkv`` GEMV output (see gfx908_qsa_glue.py).

        ``query`` [tokens, heads, head_dim] and ``output`` are written here.
        """

        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            query.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")
        indexer = self.indexer
        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(QSAForwardMetadata, metadata[indexer.raw_key_cache.prefix])
        cmp_metadata = cast(QSAForwardMetadata, metadata[indexer.compressed_key_cache.prefix])
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        gate = qkv[:, : self.q_size * 2].view(-1, self.num_heads, 2 * self.head_dim)[
            :, :, self.head_dim :
        ]
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        from .ops.qsa import qsa_gate_mul_, qsa_mqa_paged, qsa_sparse_paged_attention

        max_query_len = int(getattr(main_metadata, "max_query_len", 1) or 1)
        fast_path = (
            num_tokens > 0
            and 1 <= max_query_len <= gfx908_qsa_glue.qsa_glue_max_q()
            and not indexer.skip_topk
        )
        if fast_path and max_query_len > 1:
            # keep the eager multi-token batches that the fallback would hand to the
            # dense-short / tiled-prefill kernels on those kernels (bit-for-bit today)
            fast_path = not (
                self._dense_short_eligible(main_metadata)
                or self._prefill_tiled_eligible(main_metadata)
            )
        if fast_path:
            # one row per request, or up to max_q rows per request: the compressor
            # ring hazard is handled inside qsa_glue_pre (see gfx908_qsa_glue.glue_pre)
            gfx908_qsa_glue.STATS["fused_calls"] += 1
            iqk, _ = indexer.index_qk_proj(hidden_states[:num_tokens])
            iq = torch.empty(
                (num_tokens, indexer.index_n_heads, indexer.index_head_dim),
                dtype=qkv.dtype,
                device=qkv.device,
            )
            gfx908_qsa_glue.glue_pre(
                self, qkv, iqk, positions, query, iq, main_metadata, side_metadata,
                cmp_metadata, num_tokens, mode=3, max_query_len=max_query_len,
            )
            logits, visible = qsa_mqa_paged(
                iq,
                indexer.compressed_key_cache.kv_cache,
                cmp_metadata.block_table,
                cmp_metadata.token_to_req[:num_tokens],
                cmp_metadata.logical_positions[:num_tokens],
                cmp_metadata.seq_lens,
                indexer.compress_ratio,
                num_columns=indexer._qsa_num_columns(cmp_metadata),
            )
            selected = self.topk_indices_buffer[:num_tokens]
            gfx908_qsa_glue.topk_expand(
                logits, visible, selected, cmp_metadata.logical_positions,
                cmp_metadata.seq_lens, cmp_metadata.token_to_req, num_tokens,
            )
            self._topk_buffer_stale = False
            key_cache, value_cache = self.kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
            key_cache = canonicalize_singleton_dim_strides(key_cache)
            value_cache = canonicalize_singleton_dim_strides(value_cache)
            qsa_sparse_paged_attention(
                query[:num_tokens],
                key_cache,
                value_cache,
                selected,
                main_metadata.block_table,
                side_metadata.token_to_req[:num_tokens],
                output,
                gate=gate[:num_tokens],
                out_rows=num_tokens,
            )
            if not qsa_sparse_paged_attention.last_epilogue:
                # NUM_SPLITS == 1 shape: the kernel wrote output[:num_tokens] directly
                output[num_tokens:].zero_()
                qsa_gate_mul_(output[:num_tokens], gate[:num_tokens])
            return

        # Fallback (prefill, verify rows beyond max_q, index reuse): the main
        # projection + KV write still run as one launch, then the stock chain.
        gfx908_qsa_glue.STATS["fallback_calls"] += 1
        if num_tokens > 0:
            gfx908_qsa_glue.glue_pre(
                self, qkv, None, positions, query, None, main_metadata, side_metadata,
                cmp_metadata, num_tokens, mode=1,
            )
        dense_short = self._dense_short_eligible(main_metadata)
        selected = indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
            skip_select=dense_short,
            force_select=getattr(self, "_topk_buffer_stale", False) and indexer.skip_topk,
        )
        self._topk_buffer_stale = dense_short
        if not dense_short and selected.shape != (num_tokens, indexer.output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        if dense_short:
            impl.forward_dense_causal(query, self.kv_cache, main_metadata, output)
        elif self._prefill_tiled_eligible(main_metadata):
            impl.forward_qsa_tiled(
                self, query, self.kv_cache, main_metadata, output,
                side_metadata.logical_positions, indexer.compress_ratio,
            )
        else:
            impl.forward_qsa(
                self, query, None, None, self.kv_cache, main_metadata, output,
                token_to_req=side_metadata.token_to_req,
            )
        if num_tokens > 0:
            qsa_gate_mul_(output[:num_tokens], gate[:num_tokens])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self._gfx908_qsa_glue:
            num_tokens = hidden_states.shape[0]
            query = torch.empty(
                (num_tokens, self.num_heads, self.head_dim),
                dtype=qkv.dtype,
                device=qkv.device,
            )
            attn_output = torch.empty_like(query)
            torch.ops.vllm.qwen4_exp_qsa_glue_with_output(
                hidden_states,
                positions,
                qkv,
                query,
                attn_output,
                _encode_layer_name(self.layer_name),
            )
            output, _ = self.o_proj(attn_output.view(num_tokens, -1))
            return output
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def qwen4_exp_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )


def qwen4_exp_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, layer_name


direct_register_custom_op(
    op_name="qwen4_exp_qsa_with_output",
    op_func=qwen4_exp_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen4_exp_qsa_with_output_fake,
)


def qwen4_exp_qsa_glue_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    qkv: torch.Tensor,
    query: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """gfx908 fused decode glue: projection glue + state update + attention."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa_glue(hidden_states, positions, qkv, query, output)


def qwen4_exp_qsa_glue_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    qkv: torch.Tensor,
    query: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, qkv, query, output, layer_name


direct_register_custom_op(
    op_name="qwen4_exp_qsa_glue_with_output",
    op_func=qwen4_exp_qsa_glue_with_output,
    mutates_args=["query", "output"],
    fake_impl=qwen4_exp_qsa_glue_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "qwen4_exp_qsa_with_output",
]
