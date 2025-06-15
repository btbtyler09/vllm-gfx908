#!/bin/bash
echo "Running quick test to compare to baseline accuracy..."
python test_gptq_kernels_direct.py --baseline baseline_large.pt
python test_gptq_comprehensive.py --baseline comprehensive_baseline.pt --skip-edge-cases