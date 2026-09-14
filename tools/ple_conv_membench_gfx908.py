# GPU microbench: peak transient of the padded PLE short-conv prefill vs the flat path.
# Run in the rc9 image with one MI100:  python3 ple_conv_membench.py [C=10240]
import sys, time, importlib.util, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'tests', 'models', 'qwen4_exp'))
from test_ple_short_conv_flat import padded_reference, _inputs, dilated_causal_conv_flat
C = int(sys.argv[1]) if len(sys.argv) > 1 else 10240
K, dil = 4, 3
dev = 'cuda'
shapes = {
    'crash 11:09 (5 prefills, max 7200)': [16, 16, 21, 384, 7200],
    'one 8192 chunk': [8192],
    '12 x 600 (agents, even)': [600] * 12,
    'worst: 8000 + 47 x 4 (48 seqs)': [8000] + [4] * 47,
}
def peak(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t0 = time.time(); r = fn(); torch.cuda.synchronize(); dt = (time.time() - t0) * 1e3
    return (torch.cuda.max_memory_allocated() - base) / 2**20, dt, r
for name, lengths in shapes.items():
    x, q, req, col, lens, state, w = [t.to(dev) if torch.is_tensor(t) else t for t in _inputs(lengths, C, K, dil, torch.bfloat16, True)]
    for label, fn in (('padded', lambda: padded_reference(x, lens, state, w, dil)),
                      ('flat', lambda: dilated_causal_conv_flat(x, q, req, col, lens, state, w, dil))):
        try:
            fn(); m, dt, r = peak(fn)  # warm once, then measure
            print(f"{name:38s} {label:6s} peak transient {m:8.1f} MiB  {dt:7.1f} ms")
        except (torch.OutOfMemoryError, RuntimeError) as e:
            print(f"{name:38s} {label:6s} FAILED: {type(e).__name__}: {str(e)[:70]}")
        torch.cuda.empty_cache()
    try:
        o1, s1 = padded_reference(x, lens, state, w, dil); o2, s2 = dilated_causal_conv_flat(x, q, req, col, lens, state, w, dil)
    except RuntimeError:
        print(f"{'':38s} (no diff: padded path cannot run this shape)"); torch.cuda.empty_cache(); continue
    print(f"{'':38s} max |diff| out {(o1.float()-o2.float()).abs().max().item():.3e}  state exact {torch.equal(s1, s2)}")
