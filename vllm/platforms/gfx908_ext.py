"""Content-addressed build directories for the gfx908 JIT HIP extensions.

torch.utils.cpp_extension.load() keys its rebuild decision on file mtimes and
can hand back a stale .so when the overlaid source changed (seen 2026-09-06: a
compile error in a new kernel left the previous image's extension loaded, with
no error in the log).  Hashing the sources into the build path makes a changed
source either build fresh or fail loudly; the image bake uses the same function
so the pure image still finds its prebuilt extensions.
"""

from __future__ import annotations

import glob
import hashlib
import os


def hashed_build_dir(base: str, subdir: str, sources: list[str], extra_include_paths=()) -> str:
    h = hashlib.sha1()
    seen: list[str] = []
    for src in sources:
        seen.append(src)
        seen.extend(sorted(glob.glob(os.path.join(os.path.dirname(src), "*.cuh"))))
    for inc in extra_include_paths:
        seen.extend(sorted(glob.glob(os.path.join(inc, "*.cuh"))))
    for f in dict.fromkeys(seen):
        h.update(os.path.basename(f).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    d = os.path.join(base, f"{subdir}-{h.hexdigest()[:10]}")
    os.makedirs(d, exist_ok=True)
    return d
