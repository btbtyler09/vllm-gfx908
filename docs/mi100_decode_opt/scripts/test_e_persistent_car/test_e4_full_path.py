"""Test E.4 — exercise the full ca.custom_all_reduce() entry point under
capture, with the gfx908 routing fix mounted. Proves the high-level path
that vLLM actually calls produces drift-free results.

Without the fix: should_custom_ar() returned False on gfx908 captures →
custom_all_reduce() returned None → caller fell through to RCCL.
With the fix: should_custom_ar() returns True → custom_all_reduce() routes
to all_reduce(input, registered=False) → uncached buffer_ptrs path → clean.

Run via:
  torchrun --nproc_per_node=4 test_e4_full_path.py
"""
import os
import sys

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    gloo_group = dist.group.WORLD
    nccl_group = dist.new_group(backend="nccl")

    ca = CustomAllreduce(group=gloo_group, device=local_rank)
    if ca.disabled:
        print(f"[rank {rank}] CustomAllreduce DISABLED — bailing", flush=True)
        dist.destroy_process_group()
        return 1

    size = 8192
    torch.manual_seed(42 + rank)
    base_inp = torch.randn(size, dtype=torch.float16, device="cuda")
    inp = base_inp.clone()

    # Confirm the fix path is reachable: should_custom_ar should now return
    # True on gfx908 even when the stream is capturing.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            should = ca.should_custom_ar(inp)
            if not should:
                print(
                    f"[rank {rank}] should_custom_ar=False under capture — "
                    f"the gfx908 bypass was NOT removed",
                    flush=True,
                )
                dist.destroy_process_group()
                return 3
    torch.cuda.current_stream().wait_stream(s)

    # Warmup eager
    out = ca.custom_all_reduce(inp)
    assert out is not None, "eager custom_all_reduce returned None"
    torch.cuda.synchronize()

    # Capture and replay
    g = torch.cuda.CUDAGraph()
    out_buf: torch.Tensor | None = None
    with ca.capture():
        with torch.cuda.graph(g):
            out_buf = ca.custom_all_reduce(inp)
            assert out_buf is not None, "captured custom_all_reduce returned None"

    perturb_gen = torch.Generator(device="cuda").manual_seed(7 + rank)
    failures: list[str] = []
    REPLAYS = 500
    for i in range(REPLAYS):
        delta = torch.randn(size, dtype=torch.float16, device="cuda",
                            generator=perturb_gen) * 0.01
        inp.copy_(base_inp + delta)

        g.replay()
        torch.cuda.synchronize()

        if torch.isnan(out_buf).any().item():
            failures.append(f"replay {i}: NaN detected")
            if len(failures) <= 3:
                print(f"[rank {rank}] {failures[-1]}", flush=True)

        if i % 25 == 0:
            ref = inp.clone()
            dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=nccl_group)
            diff = (out_buf.float() - ref.float()).abs().max().item()
            if diff > 2e-2 and not torch.isnan(out_buf).any():
                msg = f"replay {i}: drift max_diff={diff:.6f}"
                failures.append(msg)
                if len(failures) <= 6:
                    print(f"[rank {rank}] {msg}", flush=True)

    dist.barrier(group=gloo_group)
    dist.destroy_process_group()

    if failures:
        print(
            f"[rank {rank}] test_e4 full path: FAILED {len(failures)} replays",
            flush=True,
        )
        return 2
    print(
        f"[rank {rank}] test_e4 full path: PASS — {REPLAYS} perturbed replays clean",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
