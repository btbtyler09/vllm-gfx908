# GPTQ Kernel Memory Access Bug

## Issue Description

The GPTQ kernels in `q_gemm.cu` have a bounds checking bug that can cause GPU memory access violations.

## Root Cause

In both 4-bit and 8-bit GPTQ kernels, the code performs `item4()` calls that access 4 consecutive columns BEFORE checking if the thread's column index is within bounds.

### Example from 4-bit kernel:

```cpp
int n = offset_n + t * 4;  // Each thread handles 4 columns

// ... other code ...

// These calls access columns n, n+1, n+2, n+3
b_gptq_qzeros_.item4(zeros, group, n);    // LINE 259
b_gptq_scales_.item4_f(scales, group, n); // LINE 260

// The bounds check comes AFTER, but it's too late!
if (n >= size_n) return;  // LINE 233
```

## Problem Scenario

With n=264 columns:
- Thread 65: n = 0 + 65 * 4 = 260
- Tries to access columns 260, 261, 262, 263 ✓
- Thread 66: n = 0 + 66 * 4 = 264  
- Tries to access columns 264, 265, 266, 267 ✗ (out of bounds!)

## Workaround

Until the kernel is fixed, ensure test dimensions satisfy:
- n must be divisible by 8 (4-bit) or 4 (8-bit) for GPTQ packing
- n should be a multiple of 128 or 256 to avoid the bounds issue
- Avoid dimensions where threads near the boundary might access out of bounds

## Proper Fix

The bounds check should be:
```cpp
if (n + 3 >= size_n) return;  // Check all 4 columns will be in bounds
```

And it should come BEFORE any `item4()` calls that access those columns.

## Impact

This bug can cause:
- GPU memory access violations
- Kernel launch failures
- Inconsistent behavior across different tensor sizes
- Test failures on certain dimension combinations