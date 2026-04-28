# Round-3 validation on Qwen3.6-35B-A3B-GPTQ-4bit + ship

## Context

Round-3 (Stage 5h `direct_register_custom_op` for `rocm_unquantized_gemm_gfx908` + Stage 5j `NCCL_ALGO=Tree NCCL_PROTO=LL` defaults) shipped on `mi100-optimized@2ae323c98` and gave **+25.7% throughput** at c=1 on Qwen3.6-35B-A3B-GPTQ-**8bit** (TPOT 13.82 → 11.00 ms; 86 → 91 tok/s).

The patches are not 35B-specific — they fire on any gfx908 workload that hits the unquantized GEMM dispatch. **Qwen3.6-35B-A3B-GPTQ-4bit has IDENTICAL `dynamic` GPTQ exclusions** to the 8bit (verified):

```json
"dynamic": {
  "-:.*linear_attn\\.in_proj_qkv": {},
  "-:.*linear_attn\\.in_proj_z": {},
  "-:.*linear_attn\\.out_proj": {},
  "-:.*shared_expert\\.down_proj": {},
  "-:.*shared_expert\\.gate_proj": {},
  "-:.*shared_expert\\.up_proj": {}
}
```

→ Same projections route through `rocm_unquantized_gemm_gfx908` → round-3 patches fire on the same hot layers → predicted similar throughput win.

Goal: confirm the 4bit variant loads cleanly, dispatches as expected, produces coherent output, and lands a comparable win. Then `git push origin mi100-optimized` and document the actual scope (any gfx908 model with `dynamic` GPTQ exclusions — currently both 35B-A3B variants; 27B variants will become eligible after user re-quantizes them with the `dynamic` field).

(27B-4bit is being re-quantized with the `dynamic` field — out of scope for this round.)

## TL;DR

| Stage | Goal | Wallclock | Ship gate |
|---|---|---|---|
| 0 *(opt)* | Pre-round-3 baseline on 35B-4bit (worktree at `fb72c78b8`) — measures actual delta | ~1 hr | informational |
| 1 | Round-3 HEAD bench on 35B-4bit (3-run TPOT @ c=1 + 12-scenario BenchAndReport) | ~1 hr | coherence pre+post PASS, c=1 TPOT in sane range, no startup errors |
| 2 | `VERIFY_DISPATCH=1` priming probe — confirm `wvSplitK`/`LLMM1`/`AITER_DISPATCH` fire on 4bit shapes | ~10 min | dispatch fires for the expected shapes |
| 3 | Comparison + write-up | ~15 min | gate to push |
| 4 | `git push origin mi100-optimized` + memory update + summary report | ~10 min | only if Stage 1+2 pass; if Stage 0 ran, win must be ≥ +15% (loose target vs 8bit's +25.7%) |

**Core path:** ~1.5 hr (skip Stage 0). **With pre-round-3 baseline:** ~2.5 hr.

## Hard rules (carry from round-3)

1. Coherence pre+post on every container start.
2. 3-run TPOT median (spread should be ≤ ±0.05 ms; re-run if not).
3. Container teardown between stages (`docker rm -f decode_opt`).
4. Quiet prod runs — no `VLLM_GFX908_DEBUG_DISPATCH=1` in the bench script body. Use `VERIFY_DISPATCH=1` opt-in only for Stage 2.
5. Mount overlays read-only (`-v ...:ro`).
6. **Do NOT push** if coherence post-FAIL or if c=1 TPOT regresses ≥ 5% vs 8bit's published numbers (sanity check — would suggest a load-time issue).

## Stop conditions

- Model fails to load → STOP, triage, do not push.
- Coherence post-FAIL → STOP, do not push.
- Stage 0 (if run) shows post < pre on c=1 → STOP, do not push, investigate.
- Wallclock 3 hr → STOP, write what we have, do not push untested code.

---

## Stage 0 *(optional — recommended)*: Pre-round-3 baseline (~1 hr)

**Why optional:** Round-3 patches are already proven on 8bit; 4bit `dynamic` field is identical so the win is high-confidence predictable. But measuring the actual delta on 4bit is the right thing to put in release notes / memory.

**Method:**
1. `git worktree add /tmp/vllm-gfx908-prer3 fb72c78b8` — checkout the commit immediately before round-3 (Stage 5 round-2 summary, with Stage 5g LLMM1+wvSplitK already in tree).
2. Fork `test_stage5_baseline.sh` → `test_4bit_baseline.sh` with overrides:
   - `SERVED="qwen3.6-35b-a3b-4bit"`
   - Model path `/models/Qwen3.6-35B-A3B-GPTQ-4bit`
   - `OVERLAY_ROOT` env var: `/home/tyler/vllm-gfx908` (HEAD, default) or `/tmp/vllm-gfx908-prer3` (pre-round-3). Mount block uses `$OVERLAY_ROOT/vllm/...` for the three overlay files.
   - Drop the hard-coded `lm_head` AITER dispatch hit count (vocab unchanged across 4bit/8bit so per-rank lm_head shape is the same `m=62080 k=2048`, but log it rather than fail on count).
3. Run with `OVERLAY_ROOT=/tmp/vllm-gfx908-prer3 ./test_4bit_baseline.sh`:
   - Coherence pre on `qwen3.6-35b-a3b-4bit`.
   - 3-run TPOT @ c=1 (256-tok decode).
   - Full 12-scenario `BenchAndReport.py --served qwen3.6-35b-a3b-4bit`. Save raw JSON to `/tmp/decode_opt/4bit_pre_round3.json`.
   - Coherence post.
4. `docker rm -f decode_opt`.

**Deliverable:** `/tmp/decode_opt/4bit_pre_round3.json`, `~/mi100-llm-testing/Model_Reports/round3_4bit_pre_qwen3.6-35b-a3b-4bit_2026-04-25.md`.

**Gate:** if model fails to load at `fb72c78b8`, STOP Stage 0 and proceed to Stage 1 only (informational stage, not blocking).

---

## Stage 1: Round-3 HEAD bench on 35B-4bit (~1 hr)

**Method:**
1. Run `test_4bit_baseline.sh` (default `OVERLAY_ROOT=/home/tyler/vllm-gfx908` = current HEAD = round-3).
2. Coherence pre on `qwen3.6-35b-a3b-4bit`.
3. 3-run TPOT @ c=1.
4. Full 12-scenario BenchAndReport. JSON → `/tmp/decode_opt/4bit_post_round3.json`. Markdown → `~/mi100-llm-testing/Model_Reports/round3_4bit_post_qwen3.6-35b-a3b-4bit_2026-04-25.md`.
5. Coherence post.
6. Leave container running for Stage 2.

**Decision gate:**
- Coherence pre + post PASS.
- c=1 TPOT 3-run spread ≤ ±0.05 ms.
- Absolute c=1 throughput in a sane range (somewhere in 80-110 tok/s based on 8bit's 91 tok/s; 4bit may be a touch faster from less weight bandwidth).

---

## Stage 2: Dispatch verification (~10 min)

**Method:**
1. Restart container with `VERIFY_DISPATCH=1 ./test_4bit_baseline.sh` (turns on `VLLM_GFX908_DEBUG_DISPATCH=1` + `VLLM_GFX908_PROBE_SHAPES=3000`).
2. Send the priming chat request (script already does this).
3. Check `/tmp/decode_opt/serve_baseline.log` for non-zero counts:
   - `[wvSplitK]` — should fire for 35B-A3B linear_attn projections (per-rank n=1, the unquantized QKV/o_proj shapes from the `dynamic` exclusions).
   - `[LLMM1]` — should fire for lm_head per-rank (n=1, m=62080, k=2048).
   - `[AITER_DISPATCH]` — should fire for whitelisted shapes from round-3 surgical AITER work.
4. Tear down container.

**Decision gate:** at least one of `wvSplitK` / `LLMM1` / `AITER_DISPATCH` fires with non-trivial counts. If ZERO fire → patches not active in 4bit's hot path → STOP, do not push, investigate.

---

## Stage 3: Comparison + write-up (~15 min)

**Method:**
1. If Stage 0 ran: compute deltas (reuse round-3 comparison helper if it exists at `/tmp/decode_opt/compare_bench.py`; otherwise quick Python script comparing `4bit_pre_round3.json` vs `4bit_post_round3.json` per scenario). Output → `/tmp/decode_opt/round3_4bit_validation.md`.
2. If Stage 0 skipped: write a single-page validation summary with absolute Stage 1 numbers + dispatch verification from Stage 2.
3. Compose summary that documents the actual scope: "round-3 wins land on any gfx908 workload with `dynamic` GPTQ exclusions in the model config — confirmed on 35B-A3B-{8bit, 4bit}; null on 27B-{8bit, 4bit} until re-quantized with `dynamic` field."

**Decision gate (if Stage 0 ran):** c=1 throughput delta ≥ +15% → ship as expected. Smaller positive delta with clean coherence → still ship but document the smaller gain. Negative delta → STOP, do not push, investigate.

---

## Stage 4: Push + memory update (~10 min)

**Method:**
1. Confirm with the user before pushing — `git push` is a remote-affecting action. (Plan mode + push gate.)
2. `git -C /home/tyler/vllm-gfx908 push origin mi100-optimized` — pushes the round-3 ship commit `2ae323c98` (no new vLLM source commits required for validation).
3. Single follow-up commit on `mi100-optimized` with the new docs/scripts:
   - `docs/mi100_decode_opt/scripts/test_4bit_baseline.sh`
   - `docs/mi100_decode_opt/round3_4bit_validation.md`
   - Subject: `[MI100] round-3 validation on 35B-A3B-GPTQ-4bit`
   - Push that follow-up commit too.
4. Update memory:
   - New file `~/.claude/projects/-home-tyler-aiter/memory/project_decode_opt_round3_4bit_validation_2026_04_25.md` — short entry, baseline numbers, ship decision, scope clarification.
   - Update `MEMORY.md` with one-line reference.
5. Cleanup: `git worktree remove /tmp/vllm-gfx908-prer3` if Stage 0 ran.
6. Container teardown.

**Image release deferred** — if user wants a tagged Docker image, that's a separate follow-up after the branch push lands.

---

## Critical files

### Will be created
- `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/test_4bit_baseline.sh` (forked from `test_stage5_baseline.sh`)
- `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/round3_4bit_validation.md` (Stage 3 summary)
- `/tmp/decode_opt/4bit_post_round3.json`, `4bit_pre_round3.json` (latter only if Stage 0 ran)
- `~/mi100-llm-testing/Model_Reports/round3_4bit_{pre,post}_qwen3.6-35b-a3b-4bit_2026-04-25.md`
- `~/.claude/projects/-home-tyler-aiter/memory/project_decode_opt_round3_4bit_validation_2026_04_25.md`

### Will be modified
- `~/.claude/projects/-home-tyler-aiter/memory/MEMORY.md` — index entry for the new memory file
- (No vLLM source files modified — round-3 patches already in tree at `2ae323c98`)

### Read-only references
- `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/test_stage5_baseline.sh` — fork source
- `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh` — coherence harness (multi-line regex fix from round-27B is in tree)
- `~/mi100-llm-testing/BenchAndReport.py` — 12-scenario bench
- `/home/tyler/quantize/quant/Qwen3.6-35B-A3B-GPTQ-4bit/config.json` — verified `dynamic` exclusions match 8bit
- `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/00_FINAL_SUMMARY_round3.md` — round-3 8bit baseline numbers for cross-reference

## Verification

End-to-end checks before declaring success:
- [ ] Stage 1: 35B-4bit boots cleanly, no startup errors in `/tmp/decode_opt/serve_baseline.log`
- [ ] Stage 1: Coherence pre + post PASS
- [ ] Stage 1: 3-run TPOT @ c=1 spread ≤ ±0.05 ms
- [ ] Stage 1: Full 12-scenario BenchAndReport completes without coherence drift
- [ ] Stage 2: `wvSplitK` / `LLMM1` / `AITER_DISPATCH` fire with non-zero counts in dispatch log
- [ ] Stage 0 (if run): pre vs post delta positive on c=1 throughput; target ≥ +15%
- [ ] Stage 4: `git push` lands; remote `mi100-optimized` is at `2ae323c98` (+ optional follow-up commit for new docs/scripts)
- [ ] Stage 4: Memory entry written, `MEMORY.md` updated
- [ ] Container torn down at end
- [ ] No `VLLM_GFX908_DEBUG_DISPATCH=1` in any shipped script
