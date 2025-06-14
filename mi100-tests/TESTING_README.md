# GPTQ Testing Suite for MI100 Optimization

This testing suite provides three levels of testing for GPTQ kernel optimizations:

## 🎯 Level 1: Baseline Capture & Comparison (Python)

**Use for:** Establishing ground truth and comprehensive output validation

### Capture Baseline (Run Once on Clean vLLM)
```bash
# Start vLLM server first
vllm serve kaitchup/Qwen3-32B-autoround-4bit-gptq --tensor-parallel-size 4

# In another terminal, capture baseline
python3 test_gptq_baseline.py --capture-baseline --model kaitchup/Qwen3-32B-autoround-4bit-gptq
```

### Test Against Baseline (Run After Each Change)
```bash
# Start vLLM server with your optimized version
vllm serve kaitchup/Qwen3-32B-autoround-4bit-gptq --tensor-parallel-size 4

# Test against baseline
python3 test_gptq_baseline.py --test --model kaitchup/Qwen3-32B-autoround-4bit-gptq
```

**What it checks:**
- ✅ Exact token-by-token output matching
- ✅ Performance comparison (tokens/second)  
- ✅ Regression detection ("!!!!" strings)
- ✅ Multiple prompt types (short, long, creative, technical)

---

## 🔧 Level 2: Direct Kernel Testing (Python)

**Use for:** Quick kernel validation during development

### Capture Kernel Baseline
```bash
python3 test_gptq_kernels_direct.py --save-baseline gptq_kernel_baseline.pt
```

### Test Kernel Changes
```bash
python3 test_gptq_kernels_direct.py
```

**What it checks:**
- ✅ 4-bit and 8-bit GPTQ kernel execution
- ✅ Output numerical accuracy vs baseline
- ✅ Kernel timing and performance
- ✅ NaN/Inf detection

---

## 🚀 Level 3: Quick Manual Testing (Shell)

**Use for:** Fast regression checking during container testing

```bash
# Start vLLM server
vllm serve kaitchup/Qwen3-32B-autoround-4bit-gptq --tensor-parallel-size 4

# Run quick test
./test_gptq_manual.sh
```

**What it checks:**
- ✅ Server responsiveness
- ✅ Basic inference functionality
- ✅ "!!!!" regression detection
- ✅ Multiple prompt types
- ✅ Response timing

---

## 📋 Recommended Testing Workflow

### Initial Setup (Once)
1. **Clean vLLM baseline:**
   ```bash
   # Build clean container, start server
   python3 test_gptq_baseline.py --capture-baseline
   python3 test_gptq_kernels_direct.py --save-baseline gptq_kernel_baseline.pt
   ```

### During Development (Each Change)
1. **Kernel-level testing** (fast iteration):
   ```bash
   python3 test_gptq_kernels_direct.py
   ```

2. **Container testing** (after builds):
   ```bash
   ./test_gptq_manual.sh  # Quick check
   python3 test_gptq_baseline.py --test  # Full validation
   ```

### Success Criteria
- ✅ **No regressions:** All tests pass, no "!!!!" output
- ✅ **Output accuracy:** Token-by-token matching with baseline
- ✅ **Performance:** Equal or better throughput vs baseline

---

## 📊 Interpreting Results

### Good Results
```
✅ ALL TESTS PASSED
🚀 Performance improved by +15.2%
Output accuracy: 6/6 (100.0%)
Average performance change: +15.2%
```

### Regression Detected
```
❌ TESTS FAILED
⚠️ CRITICAL: '!!!!' REGRESSION DETECTED
Output accuracy: 3/6 (50.0%)
```

### Kernel Issues
```
❌ 4-bit kernel: Outputs differ beyond tolerance
   Max absolute diff: 1.23e-02 (threshold: 1.00e-06)
```

---

## 🐛 Troubleshooting

### "Server not responding"
- Check vLLM server is running: `curl http://localhost:8000/health`
- Verify model loaded successfully in server logs

### "vLLM GPTQ ops not available"
- Normal for kernel tests - vLLM not built yet
- Only affects direct kernel testing

### "No baseline found"
- Run baseline capture first on clean vLLM
- Check baseline files exist: `ls gptq_baseline_*.json`

### High numerical differences
- May indicate kernel implementation issue
- Check FP32 vs FP16 accumulation
- Verify scale loading functions (item4 vs item4_f)

---

## 📁 Generated Files

- `gptq_baseline_*.json` - Full inference baseline results
- `gptq_comparison_*.json` - Detailed comparison reports  
- `gptq_kernel_baseline.pt` - Direct kernel baseline results

Keep these files to track progress across optimization attempts.