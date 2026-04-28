"""Test E.3 — 1000-replay soak with input perturbation between replays.

Catches slow-corruption patterns analogous to the AITER UA bug, where
output stays correct for the first ~200 calls and then degenerates.
Perturbs the input tensor between each replay so we are not just
re-replaying the same arguments (which can hide write-after-read hazards).

Run via:
  torchrun --nproc_per_node=4 test_e3_graph_replay_soak.py [--mode=...]
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
    )
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--replays", type=int, default=1000)
    p.add_argument("--check-every", type=int, default=10,
                   help="full reference comparison cadence (cheap NaN check on every replay)")
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
    base_inp = torch.randn(size, dtype=torch.float16, device="cuda")
    inp = base_inp.clone()
    out_buf = torch.empty_like(inp)

    # Warmup must use registered=False — see test_e2 comment.
    _ = ca.all_reduce(inp, out=out_buf, registered=False)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(g):
            ca.all_reduce(inp, out=out_buf,
                          registered=(args.mode == "registered_true"))

    perturb_gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    failures: list[str] = []
    first_nan_iter: int | None = None

    for i in range(args.replays):
        # Perturb input between replays so we exercise different values.
        # In-place writes to `inp` propagate to the captured cudaMemcpyAsync.
        delta = torch.randn(size, dtype=torch.float16, device="cuda",
                            generator=perturb_gen) * 0.01
        inp.copy_(base_inp + delta)

        out_buf.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()

        # Cheap NaN check on every replay
        if torch.isnan(out_buf).any().item():
            nan_count = int(torch.isnan(out_buf).sum().item())
            if first_nan_iter is None:
                first_nan_iter = i
            failures.append(f"replay {i}: nans={nan_count}")
            if len(failures) <= 3:
                print(f"[rank {rank}] {failures[-1]}", flush=True)

        # Full reference comparison every check-every iterations
        if i % args.check_every == 0:
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
        first = first_nan_iter if first_nan_iter is not None else "—"
        print(
            f"[rank {rank}] test_e3 mode={args.mode}: FAILED "
            f"{len(failures)} bad replays (first NaN at iter {first})",
            flush=True,
        )
        return 2
    print(
        f"[rank {rank}] test_e3 mode={args.mode}: PASS — "
        f"{args.replays} replays clean (perturbed)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
