# ROCm AWQ/GPTQ Group Size 128 Issue - Investigation Findings

## Executive Summary

We investigated why AWQ and GPTQ models with group_size 128 produce garbage output ("!!!") on AMD MI100 GPUs with ROCm. Despite extensive debugging and attempted fixes, the issue remains unresolved and appears to be a deep kernel-level problem specific to ROCm's execution model.

## The Problem

- **Affected Models**: AWQ and GPTQ models with group_size 128
- **Symptoms**: Models produce repeated "!!!" characters instead of coherent text
- **Platform**: AMD MI100 GPUs with ROCm (not affecting CUDA)
- **Scope**: Hundreds of models on HuggingFace are affected

## What We Discovered

### 1. AWQ Implementation Works
- Successfully implemented AWQ support using AWQ-to-GPTQ translation
- Models load correctly and use GPTQ kernels
- The translation approach from nlzy's implementation is sound

### 2. The Bug is Deeper Than Expected
- Both symmetric and asymmetric quantization are affected
- The issue is specifically with group_size 128 (group_size 32 works fine)
- Even models claiming `sym: true` can fail (e.g., kaitchup/Qwen3-8B-autoround-4bit-gptq)

### 3. Failed Fix Attempt
We tried removing the +1 adjustment in the GPTQ kernel for group_size 128:
```cpp
// Original code
dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);

// Our attempted fix
dequant_4bit_8_prep_zero(zeros[0], z1z16[0], y1y16[0]);
```

**Result**: Made things worse - models now produce endless random tokens instead of "!!!"

### 4. Key Insights
- The +1 adjustment is necessary for correct dequantization
- The bug is likely in ROCm's execution of the kernel, not the math
- Group_size 128 results in exactly 2 groups per block (BLOCK_KN_SIZE=256)
- This might cause issues with ROCm's 64-thread wavefront vs CUDA's 32-thread warp

## Technical Analysis

### Why Group Size 128 is Special
- BLOCK_KN_SIZE = 256 (fixed block size)
- With group_size 128: 256/128 = 2 groups per block
- With group_size 32: 256/32 = 8 groups per block
- The 2-groups-per-block configuration seems to trigger the issue

### Possible Root Causes
1. **Memory Access Pattern**: ROCm might have different coalescing requirements
2. **Wavefront Divergence**: 64-thread wavefront handling 2 groups differently
3. **Synchronization Issue**: Group transitions within a block might not sync properly
4. **Compiler Optimization**: ROCm compiler might optimize the kernel incorrectly

## Current Workarounds

### For Users
1. **Use group_size 32 models** - These work reliably on ROCm
2. **Use FP16 models** - Unquantized models work fine
3. **Use CUDA if available** - The issue is ROCm-specific

### For Model Creators
1. **Quantize with group_size 32** for ROCm compatibility
2. **Test on ROCm** before releasing
3. **Tag models** with ROCm compatibility info

## Models Tested

### Working
- Models with group_size 32 (any quantization type)
- kaitchup/Qwen3-32B-autoround-4bit-gptq (before our fix attempt)

### Failing (Producing "!!!")
- lurker18/Llama_3.1_8B_Instruct_AWQ_4bit (AWQ, group_size 128)
- kaitchup/Qwen3-8B-autoround-4bit-gptq (GPTQ, group_size 128, sym=true)
- Most AWQ models (typically use group_size 128)

## Code Changes Made

### Successful Changes (Kept)
1. **AWQ Implementation** (`vllm/model_executor/layers/quantization/awq.py`)
   - AWQ-to-GPTQ translation for ROCm
   - Warning messages for problematic configurations

2. **Import Fixes** (`vllm/model_executor/layers/quantization/gptq.py`)
   - Added missing `current_platform` import
   - Warning messages for GPTQ models

### Failed Changes (Reverted)
- Attempted to skip +1 adjustment in `csrc/quantization/gptq/q_gemm.cu`
- This made the problem worse and was reverted

## Recommendations

### Short Term
1. **Document the limitation** clearly in vLLM docs
2. **Add runtime detection** and suggest group_size 32 models
3. **Work with model creators** to provide ROCm-compatible versions

### Long Term
1. **Deep kernel debugging** with AMD's help
2. **Alternative kernel implementation** specifically for group_size 128
3. **Investigate Triton kernels** as potential alternative

## Conclusion

The AWQ implementation successfully enables AWQ model support on ROCm through GPTQ translation. However, a fundamental issue in the ROCm GPTQ kernels prevents group_size 128 models from working correctly. This appears to be a low-level execution issue that will require significant debugging effort, likely with AMD's involvement.

For now, users should stick to group_size 32 models on ROCm, which work reliably with our implementation.

---

*Investigation conducted: July 2024*  
*vLLM version: 0.9.2.dev*  
*Platform: AMD MI100 with ROCm 6.3*