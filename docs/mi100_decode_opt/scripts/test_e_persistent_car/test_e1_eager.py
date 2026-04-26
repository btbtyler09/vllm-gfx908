"""Test E.1 — eager-mode CAR bit-exactness vs NCCL reference.

Sanity baseline: prove that the existing eager CAR path (which uses the
init-time-registered buffer_ptrs) produces correct results on gfx908.
This test should PASS on unmodified code — if it fails, something is
broken in the baseline before we touch anything.

Run via:
  torchrun --nproc_per_node=4 test_e1_eager.py
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

    # fp16 SUM across 4 ranks accumulates ~sqrt(N)*eps_fp16 ULP error.
    # CAR sums in fp32 then downcasts; NCCL ring-reduces in fp16 with a
    # different accumulation order. ULP-level differences are normal.
    ATOL = 2e-2

    failures: list[str] = []
    for size in (1024, 8192, 65536, 262144):
        torch.manual_seed(42 + rank * 17 + size)
        inp = torch.randn(size, dtype=torch.float16, device="cuda")

        ref = inp.clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=nccl_group)

        out = ca.custom_all_reduce(inp)
        if out is None:
            failures.append(f"size={size}: CAR returned None (disabled?)")
            continue

        diff = (out.float() - ref.float()).abs().max().item()
        nan_count = int(torch.isnan(out).sum().item())

        ok = diff < ATOL and nan_count == 0
        msg = (
            f"[rank {rank}] size={size:>7d}: "
            f"max_diff={diff:.6f}, nans={nan_count}  "
            f"[{'PASS' if ok else 'FAIL'}]"
        )
        print(msg, flush=True)
        if not ok:
            failures.append(msg)

    dist.barrier(group=gloo_group)
    dist.destroy_process_group()

    if failures:
        print(f"[rank {rank}] test_e1_eager: FAILED — {len(failures)} errors",
              flush=True)
        return 2
    print(f"[rank {rank}] test_e1_eager: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
