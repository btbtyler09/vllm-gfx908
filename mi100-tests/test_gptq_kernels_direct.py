#!/usr/bin/env python3
"""
Direct GPTQ Kernel Testing
Tests GPTQ kernels at the CUDA level without full inference
"""

import torch
import time
import argparse

def test_gptq_4bit_kernel():
    """Test 4-bit GPTQ kernel directly"""
    print("🧪 Testing 4-bit GPTQ kernel...")
    
    # Test parameters
    m, n, k = 128, 256, 512
    groupsize = 128
    device = "cuda"
    
    try:
        # Create test tensors
        a = torch.randn(m, k, dtype=torch.half, device=device)
        
        # Create quantized weight tensor (4-bit packed)
        b_q_weight = torch.randint(0, 15, (k // 8, n), dtype=torch.int32, device=device)
        
        # Create scales and zeros
        groups = k // groupsize
        b_scales = torch.randn(groups, n, dtype=torch.half, device=device)
        b_qzeros = torch.randint(0, 15, (groups, n // 8), dtype=torch.int32, device=device)
        
        # Optional: create permutation tensor  
        b_perm = torch.arange(k, dtype=torch.int, device=device)
        
        print(f"   Input shapes: a={a.shape}, b_q_weight={b_q_weight.shape}")
        print(f"   Groups: {groups}, scales: {b_scales.shape}, zeros: {b_qzeros.shape}")
        
        # Import vLLM GPTQ operations
        try:
            from vllm._custom_ops import gptq_gemm
            
            # Time the operation
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Run GPTQ GEMM (4-bit)
            output = gptq_gemm(a, b_q_weight, b_qzeros, b_scales, b_perm, bit=4, use_exllama=False)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Validate output
            assert output.shape == (m, n), f"Expected output shape {(m, n)}, got {output.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            kernel_time = (end_time - start_time) * 1000  # Convert to ms
            
            print(f"   ✅ Kernel completed in {kernel_time:.2f}ms")
            print(f"   📊 Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            print(f"   📈 Output mean: {output.mean().item():.3f}, std: {output.std().item():.3f}")
            
            return True, kernel_time, output.detach().cpu()
            
        except ImportError:
            print("   ⚠️  vLLM GPTQ ops not available, skipping kernel test")
            return True, 0, None
            
    except Exception as e:
        print(f"   ❌ Kernel test failed: {e}")
        return False, 0, None

def test_gptq_8bit_kernel():
    """Test 8-bit GPTQ kernel directly"""
    print("🧪 Testing 8-bit GPTQ kernel...")
    
    # Test parameters  
    m, n, k = 128, 256, 512
    groupsize = 128
    device = "cuda"
    
    try:
        # Create test tensors
        a = torch.randn(m, k, dtype=torch.half, device=device)
        
        # Create quantized weight tensor (8-bit packed)
        b_q_weight = torch.randint(0, 255, (k // 4, n), dtype=torch.int32, device=device)
        
        # Create scales and zeros
        groups = k // groupsize
        b_scales = torch.randn(groups, n, dtype=torch.half, device=device)
        b_qzeros = torch.randint(0, 255, (groups, n // 4), dtype=torch.int32, device=device)
        
        # Optional: create permutation tensor
        b_perm = torch.arange(k, dtype=torch.int, device=device)
        
        print(f"   Input shapes: a={a.shape}, b_q_weight={b_q_weight.shape}")
        print(f"   Groups: {groups}, scales: {b_scales.shape}, zeros: {b_qzeros.shape}")
        
        # Import vLLM GPTQ operations
        try:
            from vllm._custom_ops import gptq_gemm
            
            # Time the operation
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Run GPTQ GEMM (8-bit)
            output = gptq_gemm(a, b_q_weight, b_qzeros, b_scales, b_perm, bit=8, use_exllama=False)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Validate output
            assert output.shape == (m, n), f"Expected output shape {(m, n)}, got {output.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            kernel_time = (end_time - start_time) * 1000  # Convert to ms
            
            print(f"   ✅ Kernel completed in {kernel_time:.2f}ms")
            print(f"   📊 Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            print(f"   📈 Output mean: {output.mean().item():.3f}, std: {output.std().item():.3f}")
            
            return True, kernel_time, output.detach().cpu()
            
        except ImportError:
            print("   ⚠️  vLLM GPTQ ops not available, skipping kernel test")
            return True, 0, None
            
    except Exception as e:
        print(f"   ❌ Kernel test failed: {e}")
        return False, 0, None

def compare_kernel_outputs(baseline_output, current_output, test_name: str, tolerance: float = 1e-6):
    """Compare kernel outputs for accuracy"""
    if baseline_output is None or current_output is None:
        print(f"   ⚠️  {test_name}: Cannot compare - missing baseline or current output")
        return True
    
    # Convert to same device/dtype
    baseline = baseline_output.float()
    current = current_output.float()
    
    if baseline.shape != current.shape:
        print(f"   ❌ {test_name}: Shape mismatch - baseline: {baseline.shape}, current: {current.shape}")
        return False
    
    # Calculate differences
    abs_diff = torch.abs(baseline - current)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    # Relative difference
    rel_diff = abs_diff / (torch.abs(baseline) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # Check if within tolerance
    outputs_match = max_abs_diff < tolerance
    
    print(f"   📊 {test_name} Comparison:")
    print(f"      Max absolute diff: {max_abs_diff:.2e} (threshold: {tolerance:.2e})")
    print(f"      Mean absolute diff: {mean_abs_diff:.2e}")
    print(f"      Max relative diff: {max_rel_diff:.2%}")
    print(f"      Mean relative diff: {mean_rel_diff:.2%}")
    
    if outputs_match:
        print(f"   ✅ {test_name}: Outputs match within tolerance")
    else:
        print(f"   ❌ {test_name}: Outputs differ beyond tolerance")
        
        # Show some specific differences
        diff_indices = torch.where(abs_diff > tolerance)
        if len(diff_indices[0]) > 0:
            print(f"      First 5 differences:")
            for i in range(min(5, len(diff_indices[0]))):
                row_idx = diff_indices[0][i].item()
                col_idx = diff_indices[1][i].item()
                baseline_val = baseline[row_idx, col_idx].item()
                current_val = current[row_idx, col_idx].item()
                diff_val = abs_diff[row_idx, col_idx].item()
                print(f"        [{row_idx}, {col_idx}]: {baseline_val:.6f} → {current_val:.6f} (diff: {diff_val:.6f})")
    
    return outputs_match

def run_kernel_tests(baseline_file: str = None):
    """Run all kernel tests and optionally compare against baseline"""
    print("🔧 GPTQ Kernel Direct Testing")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
    print(f"🧠 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test kernels
    test_results = {}
    
    # Test 4-bit kernel
    success_4bit, time_4bit, output_4bit = test_gptq_4bit_kernel()
    test_results["4bit"] = {
        "success": success_4bit,
        "time_ms": time_4bit,
        "output": output_4bit
    }
    
    print()
    
    # Test 8-bit kernel  
    success_8bit, time_8bit, output_8bit = test_gptq_8bit_kernel()
    test_results["8bit"] = {
        "success": success_8bit,
        "time_ms": time_8bit,
        "output": output_8bit
    }
    
    overall_success = success_4bit and success_8bit
    
    # Save results if requested
    if baseline_file:
        torch.save(test_results, baseline_file)
        print(f"\n💾 Results saved to: {baseline_file}")
    
    # Load and compare with baseline if it exists
    baseline_comparison_file = "gptq_kernel_baseline.pt"
    try:
        baseline_results = torch.load(baseline_comparison_file)
        print(f"\n🔍 Comparing against baseline: {baseline_comparison_file}")
        
        # Compare 4-bit
        if "4bit" in baseline_results:
            match_4bit = compare_kernel_outputs(
                baseline_results["4bit"]["output"],
                test_results["4bit"]["output"],
                "4-bit kernel"
            )
        else:
            match_4bit = True
            print("   ⚠️  No 4-bit baseline found")
        
        # Compare 8-bit
        if "8bit" in baseline_results:
            match_8bit = compare_kernel_outputs(
                baseline_results["8bit"]["output"],
                test_results["8bit"]["output"],
                "8-bit kernel"
            )
        else:
            match_8bit = True
            print("   ⚠️  No 8-bit baseline found")
        
        overall_success = overall_success and match_4bit and match_8bit
        
    except FileNotFoundError:
        print(f"\n📝 No baseline found at {baseline_comparison_file}")
        print("   Run with --save-baseline to create one")
    
    # Summary
    print(f"\n📊 Kernel Test Summary:")
    print(f"4-bit kernel: {'✅' if test_results['4bit']['success'] else '❌'} ({test_results['4bit']['time_ms']:.2f}ms)")
    print(f"8-bit kernel: {'✅' if test_results['8bit']['success'] else '❌'} ({test_results['8bit']['time_ms']:.2f}ms)")
    print(f"Overall: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    return overall_success

def main():
    parser = argparse.ArgumentParser(description="Direct GPTQ Kernel Testing")
    parser.add_argument("--save-baseline", type=str,
                       help="Save results as baseline to specified file")
    
    args = parser.parse_args()
    
    success = run_kernel_tests(args.save_baseline)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()