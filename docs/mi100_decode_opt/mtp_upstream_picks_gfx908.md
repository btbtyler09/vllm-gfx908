# Upstream MTP / spec-decode PRs vs the gfx908 fork (2026-09-30)

Branch `gfx908-mtp-upstream-picks`, based on `gfx908-mtp-verify` @ `bb9c016913`
(rc10 tree `ple-prefill-flat-conv` plus the QSA eager-gate verify fix). Target: Qwen3.8-Flash-Next
(`qwen4_exp`, AMD path `vllm/models/qwen4_exp/amd`), 4x MI100 TP=4, K1 = MTP with
`num_speculative_tokens=1`.

Code-only work. Nothing here ran on a GPU, so every speed number is an **estimate**.
Checks run: `py_compile` on every touched file and
`pytest --noconftest tests/gfx908/test_qsa_eager_gate.py` (14 passed) after each pick. The
upstream builder test that came with #58114 is included but was not run, because `vllm` does not
import on this host.

Fork base: upstream `main` merge-base `810bc3250c` (2026-08-31). Upstream was fetched on
2026-09-30 (`3627a6a124`).

## Commits on this branch

| SHA | upstream | what |
|---|---|---|
| `c3ab10d533` | #55404 (`6cbb3c154e`) | clean pick |
| `9ed05c0a96` | #55054 (`edc0fb7e03`) | clean pick; **prerequisite** for #58114 (not on the requested list) |
| `a26712f6ff` | #58114 (`20b52e9f5b`) | pick with conflicts resolved in `nvidia/ple_layer.py` and `tests/models/qwen4_exp/test_ple.py` (see the commit message) |

## Verdicts

| PR | state | portable to gfx908? | files | notes / interaction with our fixes | K1 step effect (ESTIMATE) |
|---|---|---|---|---|---|
| #58114 PLE metadata −38% on pure-MTP batches (merged 09-28) | **APPLIED** (`a26712f6ff`, after #55054 `9ed05c0a96`) | yes. Shared builder code; the AMD consumer change is in upstream too | `v1/attention/backends/short_conv_attn.py`, `mamba_attn.py`, `models/qwen4_exp/{amd,nvidia}/ple_layer.py`, test | **Supersedes** the builder change on `gfx908-mtp-draft-capture` (`VLLM_GFX908_SPEC_META_ASYNC`). Details below. It will **conflict** with that branch in `short_conv_attn.py`, so drop that branch's builder hunk (and `test_spec_meta_async.py`, which tests its helpers) and keep its doc and trace tool | Same as the draft-capture estimate, not additive: **−0.5 to −1.2 ms/step** (removes the 1.97/step blocking `hipMemcpyWithStream`, 2.72 ms median GPU idle in the rc10 trace). #58114's increment over #55054 is about 3–5 fewer tiny eager launches plus some host µs, under 0.05 ms and hidden once the host runs ahead |
| #55404 no device sync in GDN/short-conv capture metadata (merged 09-05) | **APPLIED** (`c3ab10d533`) | yes | `v1/attention/backends/gdn_attn.py`, `short_conv_attn.py`, kimi_k3 test | Changes only `build_for_cudagraph_capture` (`.cpu()` → derive from `query_start_loc_cpu`). That path runs at capture and on DP idle dummy steps. We serve TP4 with no DP | **~0 ms/step** at steady state; slightly faster graph capture |
| #54612 GDN decode-path opts, sync 9.4→0 (OPEN) | **ABSENT, CONFLICTS** | not as a pick | 43 files / +7k lines, incl. `qwen_gdn_linear_attn.py`, `gdn_attn.py`, FLA `fused_sigmoid_gating.py`/`fused_recurrent.py`, `gpu_model_runner.py`, NVFP4 MoE | Stacked on open #50172 (`mamba_cache_mode='all'` + MTP). `916557c49d` and `67853c0c42` conflict in qwen_gdn_linear_attn/FLA ops/gdn_attn. The 9.4→0 syncs are in the all-mode metadata path; we run `mamba_cache_mode=none`, and the rc10 K1 trace shows no GDN blocking call. The packed-qkv spec opt targets the stock spec path, which `VLLM_GFX908_GDN_FUSED_SPEC=1` already replaces with one kernel | **~0 ms** with our config; unknown with GDN_FUSED_SPEC off |
| #58449 in-place QSA draft-metadata update (OPEN) | **PRESENT (independent equivalent)**. The pick CONFLICTS (qsa_cache.py 3 hunks, test 1) | n/a | `models/qwen4_exp/common/qsa_cache.py`, test | Our `c24162b281` already sets `QSAMetadataBuilder.supports_draft_decode_metadata_update = True` and has `update_draft_decode_metadata` (replays `build_qsa_metadata(..., mapped_from_device=True)`). The in-graph variant is behind `VLLM_GFX908_DRAFT_CAPTURED_METADATA` (measured then: bit-exact, neutral c=1 / −4.6% c=4 at n=2, so off). Upstream's version sits on the #54513 metadata kernel (`visible_blocks`) | **0 ms at K1**. There are no draft decode steps after the first, so it only matters for K≥2 |
| #54513 QSA uniform decode/spec-decode indexer kernel (merged 09-02) | **ABSENT, CONFLICTS** (`common/qsa_cache.py` 2 hunks, `warmup/kernel_warmup.py` 1) | **no** as a pick: new kernels are in `nvidia/ops/qsa_indexer.py` and `nvidia/indexer_qsa.py`; upstream never ported them to `amd/` | `common/qsa_cache.py`, `nvidia/{indexer_qsa,ops/qsa,ops/qsa_indexer}.py`, warmup | Reference for our verify fix and consistent with it. Upstream classifies QSA batches with `split_decodes_and_prefills(require_uniform=True)` + `supports_spec_as_decode`, so q_len ≤ K+1 uniform is decode. Our `_eager_fast_path_min_qlen` = K+2 draws the same line on the AMD path. Picking it would add `visible_blocks` + a reorder threshold to the common builder that no AMD consumer reads, and our in-graph draft refresh would also have to refresh `visible_blocks`. Porting `_qsa_mqa_paged_uniform_kernel` to `amd/ops/qsa.py` is a separate kernel task (GB300-tuned; the 1.15–1.32x wins are at DQL 4, ≥50k ctx) | **0 ms** as a pick. A gfx908 port of the kernel: unknown |
| #56148 multi-query split-K verify attention (OPEN) | **PARTLY PRESENT, CONFLICTS** | Triton, so the kernel is portable, but not relevant to Flash-Next | `v1/attention/backends/triton_attn.py`, `ops/triton_unified_attention.py`, `ops/triton_attention_helpers.py`, tests | Our `c76fb82c8f` already admits q_len>1 into 3D split-KV behind `VLLM_ATTN_3D_MAX_QLEN` (default 1) + `VLLM_ATTN_3D_QLEN_MIN_CTX`. The 3-way apply conflicts on that `use_3d` gate (`triton_unified_attention.py` 1, `triton_attn.py` 2, tests 2). New upstream: `is_uniform_decode` metadata, per-token scratch sizing, scratch validation, and an empty-tile `softmax_step` fix plus a reducer fix. The helper fix applies cleanly, but only together with the reducer change in the conflicting file, so it was not split out. Flash-Next never reaches `triton_attn`: all full-attention layers, including the MTP layer, are QSA | **0 ms** for Flash-Next K1. Relevant to TRITON_ATTN models (Qwen3.8-27B MTP); the hardening is worth porting onto our gate later |

## #58114 vs `VLLM_GFX908_SPEC_META_ASYNC` (gfx908-mtp-draft-capture)

Both remove the blocking `.to(device)` uploads from the spec branch of
`PleShortConvAttentionMetadataBuilder.build`. Both return identical metadata values.

| | ours (flag on) | upstream #55054 + #58114 |
|---|---|---|
| default | off (reference `.to()`) | always on |
| pure-spec batch (every K1 decode step) | spec rows detected as prefix → `[:n]` slice, no upload | `[:num_spec_decodes]` slice, no upload, no `nonzero`/`cat`/`arange`/`empty`, token maps `None` |
| mixed batch | async H2D; prefix slice if spec rows are a prefix; `repeat_interleave(output_size=...)` | async H2D + gather; `index_fill_`; `repeat_interleave(output_size=...)`; skips the unused `compute_causal_conv1d_metadata` |
| non-spec PLE builds | unchanged | skip the generic causal-conv1d metadata (`needs_causal_conv1d_metadata = False`); the PLE consumers read `query_start_loc_p` |
| eager strided state column | `.contiguous()` | left strided. The AMD spec path does `.to(int64)` (a copy), and the full-graph path copies into the persistent buffer, so this is safe |

Verdict: upstream **supersedes** ours for the builder. It covers the same stall, does a bit
more work removal, and will come back on every upstream sync. Ours adds only the default-off
gate and the boot banner row.

Env gate: I did not add one. The upstream change can't be gated without keeping a second copy of
the ~200-line builder, and the metadata values are identical. For the A/B, use the pre-pick files
as arm A, as in the draft-capture recipe:

```
# arm B (picks): bind-mount this branch's files over the rc10 image
#   vllm/v1/attention/backends/{short_conv_attn,mamba_attn,gdn_attn}.py
#   vllm/models/qwen4_exp/amd/ple_layer.py   (+ amd/qsa.py for the verify fix)
# arm A (reference): the same mounts, but short_conv_attn.py and mamba_attn.py taken from
#   gfx908-mtp-verify (the old builder still fills query_start_loc_p, so the new ple_layer.py is fine)
```

Gate on acceptance and tokens/step matching arm A. A drop means a race that the removed sync
was hiding. Expected result: the `hipMemcpyWithStream` rows disappear from the K1 trace
(`tools/spec_step_anatomy_gfx908.py` on the draft-capture branch), and step time drops by
~0.5–1.2 ms (ESTIMATE). With ~0.3–0.4 ms boot-to-boot scatter, run ≥2 boots per arm.

## Merge notes

- Merges cleanly with `gfx908-mtp-smallm`. It conflicts with `gfx908-mtp-draft-capture` in
  `short_conv_attn.py` only; resolve by taking this branch's version.
- The fork's `short_conv_attn.py` was byte-identical to upstream before #54517. #54517 (fused NVIDIA PLE
  kernels) is not in the fork, so the unused `self.spec_token_indx`/`self.non_spec_token_indx`
  buffers are still allocated. That is harmless.
