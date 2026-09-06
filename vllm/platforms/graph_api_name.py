"""Vendor-correct name for the device graph API in user-facing log lines.

PyTorch keeps the ``torch.cuda.graph`` API name on ROCm because the ROCm build
is HIPified from the CUDA sources, so upstream log lines say "CUDA graph" on
AMD GPUs.  The mechanism there is ``hipGraph``; say so in what we print.
API names, config keys (``cudagraph_mode``) and identifiers are unchanged.
"""

from __future__ import annotations

import functools


@functools.cache
def graph_api_name() -> str:
    try:
        import torch

        if getattr(torch.version, "hip", None):
            return "HIP"
    except Exception:  # noqa: BLE001
        pass
    return "CUDA"
