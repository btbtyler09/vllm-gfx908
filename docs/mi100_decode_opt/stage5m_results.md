# Stage 5m — MoE block torch.compile wrap (NO-OP — already done upstream)

**Date:** 2026-04-25
**Result:** SKIPPED — the lever doesn't exist for this model.
**Decision: NO SHIP, NO CODE CHANGE.**

## What we found

`docs/mi100_decode_opt/round4_candidates.md` item D claimed the MoE block runs in eager Python and would benefit from a `@support_torch_compile` wrap. Tracing the Qwen3.6 model class hierarchy showed this is **incorrect** for the specific model we're optimizing.

Both relevant model classes already carry the decorator:

```python
# vllm/model_executor/models/qwen3_next.py:458
@support_torch_compile
class Qwen3NextModel(nn.Module, EagleModelMixin):
    ...

# vllm/model_executor/models/qwen3_5.py:197-207
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    ...
```

The `Qwen3NextSparseMoeBlock.forward` (`qwen3_next.py:165`) is called from inside the compiled `Qwen3_5Model.forward`, so the entire MoE block — router, gate, shared_expert, experts — runs inside the inductor graph already.

Stage 5g notes ("the MoE block is NOT in graph") that the round4 candidates doc cited were either misread or referred to the SharedFusedMoE *internal kernel calls* being extern Triton calls (which is true and not fixable by another @support_torch_compile decorator — it's how inductor handles ops backed by hand-written kernels).

## What this means for round-4

- **Item D in `round4_candidates.md` is invalidated** for Qwen3.6-35B-A3B specifically. Any future model where `@support_torch_compile` *isn't* on the model class would still benefit from the lever as originally described.
- The remaining MoE-block opportunity is at the *kernel-fusion* level (Stage 5k audit will tell us if the surrounding norm/add are fused into our custom-op call), not at the graph-mode wrapping level.
- Per-step launch overhead from the MoE block is therefore already mostly hidden by cudagraph capture.

## No code change required

Saved 2-3 hours of patch + risk for zero expected gain.
