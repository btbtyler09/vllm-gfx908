# SPDX-License-Identifier: Apache-2.0
"""ONE-GPU correctness + graph-timed microbench: shipping W8 HC chain vs the small-M kernels.

Needs one MI100 (gfx908).  Run inside the rc10 q38fn image with this worktree mounted at /src
(see docs/mi100_decode_opt/mtp_smallm_gemv_gfx908.md, "GPU validation recipe"):

    python3 /src/tests/gfx908/bench_hc_w8_smallm.py [--out /work/hcsm_bench.json]

What it does, at the production shapes (mix_down 336 x 10240 int8 gs128 with 320 lora + 4 inject
columns, mix_up 10240 x 320 int8 gs64 row-permuted, HC = 4, hidden 2560) for M = 1..4:

1. correctness
   a. small-M chain vs the shipping chain (``hc_w8_gemv`` x 2 + the injection ``.contiguous()``,
      exactly what ``_hc_mix_local`` runs today): max bf16-ulp distance and fraction of differing
      elements of ``block_input`` and ``injection``;
   b. small-M kernels vs the CPU emulator of their own tiling/summation order
      (tests/gfx908/test_hc_w8_smallm_ref.py): the injection columns (no transcendental in the
      epilogue) must be BIT-EXACT; silu / sigmoid outputs may differ by <= 1 ulp (v_exp_f32 vs
      np.exp).  This is the check that the device index math is the emulated one;
   c. both vs an fp64 reference of the (q, s) math, with the tolerance of agents/hc_w8.
2. timing: torch.cuda.CUDAGraph with ITERS chains captured, weights rotated over NROT copies
   (L2-cold, ~85 MB), best of REPS replays, us per call: shipping down / up / injection copy /
   chain vs small-M down (ytile 2, 3, 4) / up / chain.

Exit code 0 = all correctness gates passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HC, HD, R, N_DOWN = 4, 2560, 320, 336
K_DOWN = HC * HD
GS_D, GS_U = 128, 64
NROT, ITERS, REPS = 12, 48, 20

SRC = os.environ.get("HCSM_SRC", "/src")
sys.path.insert(0, os.path.join(SRC, "tests", "gfx908"))


def load_exts(build_dir: str):
    """Shipping W8 extensions from the installed tree (prebuilt in the image) + the new one
    built from the mounted worktree source."""
    import vllm.models.qwen4_exp.amd.gfx908_hc_fused as hcm
    from torch.utils.cpp_extension import load

    os.makedirs(build_dir, exist_ok=True)
    new = load(
        name="gfx908_hc_w8_smallm_ext",
        sources=[os.path.join(SRC, "vllm/models/qwen4_exp/amd/csrc/gfx908_hc_w8_smallm.hip")],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )
    return hcm, hcm._ext_w8(), hcm._ext_w8_big(), new


def graph_time(fn, iters=ITERS, reps=REPS) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    del g
    return best * 1e6


def ulp(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float]:
    import test_hc_w8_smallm_ref as T

    x = a.float().cpu().numpy()
    y = b.float().cpu().numpy()
    return int(T.ulp_dist(x, y).max()), float(np.mean(x != y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/hcsm_bench.json")
    ap.add_argument("--build", default="/work/hcsm_build")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()

    import test_hc_w8_smallm_ref as T

    dev = "cuda"
    hcm, ext_w8, ext_w8b, ext_new = load_exts(args.build)
    cu = hcm._cu_count()
    print(f"device {torch.cuda.get_device_name()} cu={cu}  new ext {ext_new.__file__}", flush=True)

    # weights: the same generator as the CPU test (outliers included), NROT copies for L2-cold
    q8 = []
    for i in range(NROT):
        qd, sd, qu, su, _ = T.make_case(1, seed=100 + i)
        q8.append(tuple(torch.from_numpy(a).to(dev) for a in (qd, sd, qu, su)))
    ok = True
    res = {"device": torch.cuda.get_device_name(), "cu": cu, "rows": []}

    for M in (1, 2, 3, 4):
        _, _, _, _, xn_np = T.make_case(M, seed=M)
        xn = torch.from_numpy(xn_np).to(dev).to(torch.bfloat16)
        ship = ext_w8 if M <= 3 else ext_w8b
        cfg_d, cfg_u = hcm._CFG_DOWN[M], hcm._CFG_UP[M]

        def ship_chain(i=0):
            qd, sd, qu, su = q8[i]
            d = torch.empty((M, N_DOWN), dtype=torch.bfloat16, device=dev)
            assert ship.hc_w8_gemv(qd, sd, xn, d, xn, 1, GS_D, *cfg_d, HC, R, cu)
            inj = d[:, R:R + HC].contiguous()
            y = torch.empty((M, HD), dtype=torch.bfloat16, device=dev)
            assert ship.hc_w8_gemv(qu, su, d[:, :R], y, xn, 2, GS_U, *cfg_u, HC, R, cu)
            return y, inj, d

        def new_chain(i=0, yt=3):
            qd, sd, qu, su = q8[i]
            d = torch.empty((M, N_DOWN), dtype=torch.bfloat16, device=dev)
            inj = torch.empty((M, HC), dtype=torch.bfloat16, device=dev)
            y = torch.empty((M, HD), dtype=torch.bfloat16, device=dev)
            assert ext_new.hcsm_down(qd, sd, xn, d, inj, GS_D, yt, 1, HC, R)
            assert ext_new.hcsm_up(qu, su, d[:, :R], xn, y, GS_U, HC)
            return y, inj, d

        # ---------------- correctness (weight copy 0)
        ys, injs, ds = ship_chain()
        row = {"M": M}
        for yt in (2, 3, 4):
            yn, injn, dn = new_chain(0, yt)
            torch.cuda.synchronize()
            row[f"yt{yt}_y_vs_ship"] = ulp(yn, ys)
            row[f"yt{yt}_inj_vs_ship"] = ulp(injn, injs)
            row[f"yt{yt}_down_vs_ship"] = ulp(dn, ds)
        yn, injn, dn = new_chain(0, 3)
        qd, sd, qu, su = (t.cpu().numpy() for t in q8[0])
        e_down, e_inj = T.down_new(qd, sd, xn_np, GS_D, R, HC)
        e_y = T.up_new(qu, su, dn[:, :R].float().cpu().numpy(), xn_np, GS_U, HD)
        row["inj_vs_emulator"] = ulp(injn, torch.from_numpy(e_inj))
        row["down_vs_emulator"] = ulp(dn, torch.from_numpy(e_down))
        row["y_vs_emulator"] = ulp(yn, torch.from_numpy(e_y))
        row["inj_equals_down_cols"] = bool(torch.equal(injn, dn[:, R:R + HC]))
        gate = (
            row["inj_vs_emulator"][0] == 0
            and row["down_vs_emulator"][0] <= 1
            and row["y_vs_emulator"][0] <= 1
            and row["inj_equals_down_cols"]
            and all(row[f"yt{yt}_y_vs_ship"][0] <= 1 and row[f"yt{yt}_inj_vs_ship"][0] <= 1
                    and row[f"yt{yt}_y_vs_ship"][1] <= 0.01 for yt in (2, 3, 4))
        )
        row["gate"] = gate
        ok &= gate

        # ---------------- timing (L2-cold rotation)
        if not args.no_timing:
            ctr = {"i": 0}

            def nxt():
                ctr["i"] += 1
                return ctr["i"] % NROT

            lora_buf = torch.zeros((M, N_DOWN), dtype=torch.bfloat16, device=dev)
            inj_buf = torch.empty((M, HC), dtype=torch.bfloat16, device=dev)

            def s_down():
                qd_, sd_, _, _ = q8[nxt()]
                d = torch.empty((M, N_DOWN), dtype=torch.bfloat16, device=dev)
                ship.hc_w8_gemv(qd_, sd_, xn, d, xn, 1, GS_D, *cfg_d, HC, R, cu)

            def s_up():
                _, _, qu_, su_ = q8[nxt()]
                y = torch.empty((M, HD), dtype=torch.bfloat16, device=dev)
                ship.hc_w8_gemv(qu_, su_, lora_buf[:, :R], y, xn, 2, GS_U, *cfg_u, HC, R, cu)

            def s_copy():
                inj_buf.copy_(lora_buf[:, R:R + HC])

            def n_down(yt):
                def f():
                    qd_, sd_, _, _ = q8[nxt()]
                    d = torch.empty((M, N_DOWN), dtype=torch.bfloat16, device=dev)
                    ext_new.hcsm_down(qd_, sd_, xn, d, inj_buf, GS_D, yt, 1, HC, R)
                return f

            def n_up():
                _, _, qu_, su_ = q8[nxt()]
                y = torch.empty((M, HD), dtype=torch.bfloat16, device=dev)
                ext_new.hcsm_up(qu_, su_, lora_buf[:, :R], xn, y, GS_U, HC)

            t = {
                "ship_down": graph_time(s_down),
                "ship_up": graph_time(s_up),
                "ship_inj_copy": graph_time(s_copy) if M > 1 else 0.0,
                "ship_chain": graph_time(lambda: ship_chain(nxt())),
                "new_down_yt2": graph_time(n_down(2)),
                "new_down_yt3": graph_time(n_down(3)),
                "new_down_yt4": graph_time(n_down(4)),
                "new_up": graph_time(n_up),
                "new_chain_yt3": graph_time(lambda: new_chain(nxt(), 3)),
            }
            row["us"] = t
        res["rows"].append(row)
        print(json.dumps(row), flush=True)

    # ---------------- summary
    print("\n| M | ship down | ship up | inj copy | ship chain | new down yt2/3/4 | new up | new chain | "
          "chain saving | y vs ship (ulp, frac) | inj/down/y vs emulator (max ulp) | gate |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        t = r.get("us", {})
        g = lambda k: f"{t[k]:.2f}" if k in t else "-"  # noqa: E731
        sav = f"{t['ship_chain'] - t['new_chain_yt3']:+.2f}" if t else "-"
        print(f"| {r['M']} | {g('ship_down')} | {g('ship_up')} | {g('ship_inj_copy')} | {g('ship_chain')} | "
              f"{g('new_down_yt2')}/{g('new_down_yt3')}/{g('new_down_yt4')} | {g('new_up')} | "
              f"{g('new_chain_yt3')} | {sav} | {r['yt3_y_vs_ship'][0]}, {r['yt3_y_vs_ship'][1]:.4f} | "
              f"{r['inj_vs_emulator'][0]}/{r['down_vs_emulator'][0]}/{r['y_vs_emulator'][0]} | "
              f"{'PASS' if r['gate'] else 'FAIL'} |")
    res["pass"] = bool(ok)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n{'PASS' if ok else 'FAIL'}  (wrote {args.out})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
