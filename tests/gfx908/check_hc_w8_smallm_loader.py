# SPDX-License-Identifier: Apache-2.0
# CPU-only (no GPU) check of the VLLM_GFX908_SMALLM_HC_W8 loader / flag / decline logic.
# Builds the real extensions with hipcc through the module's own loader functions, so it doubles as
# the "compiles cleanly for gfx908" gate.  Run in the rc10 q38fn image WITHOUT any --device flag:
#
#   docker run --rm --network=none --user $(id -u):$(id -g) -e HOME=/tmp -e USER=$USER \
#     -e TORCHINDUCTOR_CACHE_DIR=/tmp/ind -e GPU_ARCHS=gfx908 -e PYTHONPATH=/src \
#     -v <rocminfo stub printing "Name: gfx908">:/opt/rocm-7.2.4/bin/rocminfo:ro \
#     -v <worktree>:/src:ro -v <scratch>:/work btbtyler09/vllm-rocm-gfx908:v0.28.0rc10.dev-q38fn \
#     python3 /src/tests/gfx908/check_hc_w8_smallm_loader.py
#
# (torch.cuda device properties are stubbed below; nothing is launched.)
"""CPU-only loader/flag check of the small-M path (no GPU): builds the extensions via the real
loader functions, checks the flag logic and the failure modes."""

import importlib
import os

os.environ.update(
    VLLM_GFX908_HIP_BUILD_DIR=os.environ.get("HCSM_JIT", "/work/jit"),
    VLLM_GFX908_HC_W8="1",
    VLLM_GFX908_SMALLM_HC_W8="1",
)
import types

import torch

torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(
    gcnArchName="gfx908:sramecc+:xnack-",
    name="AMD Instinct MI100",
    total_memory=32 << 30,
    multi_processor_count=120,
    major=9,
    minor=0,
)
import vllm.platforms.rocm as R

R.on_gfx908 = lambda: True
m = importlib.import_module("vllm.models.qwen4_exp.amd.gfx908_hc_fused")
print("hc_w8_enabled", m.hc_w8_enabled())
print(
    "smallm enabled",
    m.hc_w8_smallm_enabled(),
    "range",
    m.hc_w8_smallm_range(),
    "yt",
    m.hc_w8_smallm_down_ytile(),
)
ext = m._ext_w8_smallm()
print("ext", ext.__file__, hasattr(ext, "hcsm_down"), hasattr(ext, "hcsm_up"))
# flag off
m._reset_env_cache()
os.environ["VLLM_GFX908_SMALLM_HC_W8"] = "0"
print("flag off ->", m.hc_w8_smallm_enabled())
# flag on but W8 off -> loud failure under STRICT
m._reset_env_cache()
os.environ["VLLM_GFX908_SMALLM_HC_W8"] = "1"
os.environ["VLLM_GFX908_HC_W8"] = "0"
try:
    m.hc_w8_smallm_enabled()
    print("ERROR: expected raise")
except RuntimeError as e:
    print("strict raise OK:", str(e)[:90])
os.environ["VLLM_GFX908_STRICT_EXT"] = "0"
m._reset_env_cache()
print("non-strict ->", m.hc_w8_smallm_enabled())
# range env
os.environ.update(
    VLLM_GFX908_HC_W8="1",
    VLLM_GFX908_STRICT_EXT="1",
    VLLM_GFX908_SMALLM_HC_W8_MIN_M="1",
    VLLM_GFX908_SMALLM_HC_W8_MAX_M="9",
    VLLM_GFX908_SMALLM_HC_W8_DOWN_YT="5",
)
m._reset_env_cache()
print("range", m.hc_w8_smallm_range(), "yt(bad->default)", m.hc_w8_smallm_down_ytile())
# CPU tensors: the binding's argument checks run before any launch; a declined shape returns False
import torch

q = torch.zeros(336, 10240, dtype=torch.int8)
s = torch.ones(336, 80)
x = torch.zeros(5, 10240, dtype=torch.bfloat16)
out = torch.empty(5, 336, dtype=torch.bfloat16)
inj = torch.empty(5, 4, dtype=torch.bfloat16)
print("M=5 declined ->", ext.hcsm_down(q, s, x, out, inj, 128, 3, 1, 4, 320))
x2 = torch.zeros(2, 10240, dtype=torch.bfloat16)
out2 = torch.empty(2, 336, dtype=torch.bfloat16)
print(
    "bad ytile declined ->",
    ext.hcsm_down(q, s, x2, out2, inj[:2].contiguous(), 128, 5, 1, 4, 320),
)
print(
    "spw 2 declined ->",
    ext.hcsm_down(q, s, x2, out2, inj[:2].contiguous(), 128, 3, 2, 4, 320),
)
print(
    "gs=0 declined ->",
    ext.hcsm_down(q, torch.ones(336), x2, out2, inj[:2].contiguous(), 0, 3, 1, 4, 320),
)
qu = torch.zeros(10240, 320, dtype=torch.int8)
su = torch.ones(10240, 5)
y = torch.empty(2, 2560, dtype=torch.bfloat16)
xn = torch.zeros(2, 10240, dtype=torch.bfloat16)
try:
    ext.hcsm_up(qu, su, out2[:, :320], xn, y, 64, 3)
    print("ERROR: hc=3 accepted")
except RuntimeError as e:
    print("up hc=3 rejected ->", str(e)[:60])
print(
    "up K%16 declined ->",
    ext.hcsm_up(
        torch.zeros(10240, 312, dtype=torch.int8),
        torch.ones(10240, 312 // 8),
        torch.zeros(2, 312, dtype=torch.bfloat16),
        xn,
        y,
        8,
        4,
    ),
)
print("LOADER CHECK DONE")
