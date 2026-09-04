#!/usr/bin/env python3
"""Prebuild every gfx908 JIT HIP extension into $VLLM_GFX908_HIP_BUILD_DIR.

Run inside the serving image at build time (no GPU needed: hipcc only), so the
workers load the .so files instead of compiling them at boot.
"""
import importlib
import os
import sys

MODULES = [
    "vllm.model_executor.layers.fused_moe.gfx908_moe_hip",
    "vllm.model_executor.layers.fused_moe.gfx908_w4a8",
    "vllm.model_executor.layers.fused_moe.gfx908_router_topk",
    "vllm.model_executor.layers.gfx908_w8a16",
    "vllm.models.qwen4_exp.amd.gfx908_ple_zc",
    "vllm.models.qwen4_exp.amd.gfx908_hc_fused",
]

build_dir = os.environ.get("VLLM_GFX908_HIP_BUILD_DIR")
if not build_dir:
    sys.exit("set VLLM_GFX908_HIP_BUILD_DIR")
os.makedirs(build_dir, exist_ok=True)
failed = []
for name in MODULES:
    try:
        mod = importlib.import_module(name)
        for fn in ("_ext", "_ext_f16"):
            if hasattr(mod, fn):
                ext = getattr(mod, fn)()
                print(f"built {name}.{fn}: {getattr(ext, '__file__', ext)}")
    except Exception as exc:  # keep going; report at the end
        print(f"FAILED {name}: {exc}")
        failed.append(name)
if failed:
    sys.exit(f"failed: {failed}")
