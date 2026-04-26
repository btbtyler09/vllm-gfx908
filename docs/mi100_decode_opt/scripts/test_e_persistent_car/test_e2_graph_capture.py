"""Test E.2 — CAR under cudagraph capture: NaN reproducer + fix verifier.

This is the round-2 failure-mode reproducer. On UNFIXED code, calling CAR
inside a torch.cuda.graph() capture on gfx908 produces NaN in the output
on replay because the deferred-handle peer pointers go stale.

We bypass `should_custom_ar` (which would normally short-circuit on gfx908)
by calling `ca.all_reduce()` directly. Two probe modes:

  --mode=registered_true  — input pointer baked into captured kernel args
                            (the CURRENT broken path on gfx908)
  --mode=registered_false — stage input through pre-registered buffer_ptrs
                            (the Mori-pattern fix path; should be stable)

Run via:
  torchrun --nproc_per_node=4 test_e2_graph_capture.py [--mode=...]
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=("registered_true", "registered_false"),
        default="registered_false",
        help="registered_true reproduces the bug; registered_false is the fix",
    )
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--replays", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()

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

    size = args.size
    torch.manual_seed(42 + rank)
    inp = torch.randn(size, dtype=torch.float16, device="cuda")
    out_buf = torch.empty_like(inp)

    ref = inp.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=nccl_group)

    # Warmup must use registered=False — `inp` is not in the IPC buffer map,
    # so eager-mode registered=True would throw "buffer not registered".
    _ = ca.all_reduce(inp, out=out_buf, registered=False)
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(g):
            ca.all_reduce(inp, out=out_buf,
                          registered=(args.mode == "registered_true"))

    # Replay
    failures: list[str] = []
    for i in range(args.replays):
        out_buf.fill_(float("nan"))  # poison the output to detect kernel skip
        g.replay()
        torch.cuda.synchronize()

        nan_count = int(torch.isnan(out_buf).sum().item())
        diff = (out_buf.float() - ref.float()).abs().max().item()
        if nan_count > 0 or diff > 2e-2:
            failures.append(
                f"replay {i}: nans={nan_count}, max_diff={diff:.6f}"
            )
            if len(failures) <= 3:
                print(f"[rank {rank}] {failures[-1]}", flush=True)

    dist.barrier(group=gloo_group)
    dist.destroy_process_group()

    if failures:
        print(
            f"[rank {rank}] test_e2 mode={args.mode}: FAILED "
            f"{len(failures)}/{args.replays} replays",
            flush=True,
        )
        return 2
    print(
        f"[rank {rank}] test_e2 mode={args.mode}: PASS — "
        f"{args.replays} replays clean",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
