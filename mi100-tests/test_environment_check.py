#!/usr/bin/env python3
"""
Environment Check for GPTQ Testing
Validates the environment without requiring full vLLM build
"""

import torch
from pathlib import Path

def check_gpu():
    """Check GPU availability and specs"""
    print("🖥️  GPU Environment Check")
    print("=" * 30)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA available with {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        # Check if it's MI100
        if "mi100" in props.name.lower() or "gfx908" in str(props):
            print(f"   ✅ MI100 detected")
        else:
            print(f"   ⚠️  Not MI100: {props.name}")
    
    return True

def check_vllm():
    """Check vLLM installation and CUDA extensions"""
    print("\n🔧 vLLM Environment Check")
    print("=" * 30)
    
    try:
        import vllm
        print(f"✅ vLLM imported: {vllm.__version__}")
    except ImportError as e:
        print(f"❌ vLLM import failed: {e}")
        return False
    
    # Check CUDA extensions
    try:
        from vllm import _custom_ops
        print("✅ vLLM _custom_ops available")
        
        # Try to access GPTQ function
        if hasattr(_custom_ops, 'gptq_gemm'):
            print("✅ gptq_gemm function available")
            
            # Check function signature
            import inspect
            sig = inspect.signature(_custom_ops.gptq_gemm)
            params = list(sig.parameters.keys())
            print(f"   Parameters: {params}")
            
            return True
        else:
            print("❌ gptq_gemm function not found")
            return False
            
    except ImportError as e:
        print(f"❌ vLLM CUDA extensions not available: {e}")
        print("   This is normal outside of container - extensions need to be built")
        return False

def check_test_files():
    """Check if test files exist and are valid"""
    print("\n📁 Test File Check")
    print("=" * 20)
    
    test_files = [
        "test_gptq_baseline.py",
        "test_gptq_kernels_direct.py", 
        "test_gptq_manual.sh",
        "TESTING_README.md"
    ]
    
    all_exist = True
    for file in test_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def check_baseline_files():
    """Check for existing baseline files"""
    print("\n💾 Baseline File Check")
    print("=" * 25)
    
    baseline_files = [
        ("gptq_baseline_*.json", "Full inference baselines"),
        ("gptq_kernel_baseline.pt", "Kernel-level baselines"),
        ("gptq_comparison_*.json", "Comparison reports")
    ]
    
    import glob
    for pattern, description in baseline_files:
        files = glob.glob(pattern)
        if files:
            print(f"✅ {description}: {len(files)} file(s)")
            for f in files[:3]:  # Show first 3
                print(f"   - {f}")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more")
        else:
            print(f"📝 {description}: None found (will create on first run)")

def recommend_next_steps():
    """Recommend what to do next based on environment"""
    print("\n🎯 Recommended Next Steps")
    print("=" * 30)
    
    # Check if we're in container or local environment
    try:
        from vllm import _custom_ops
        in_container = True
    except ImportError:
        in_container = False
    
    if in_container:
        print("🐳 Container Environment Detected")
        print("You can run all testing levels:")
        print("1. ✅ Level 1: python3 test_gptq_baseline.py --capture-baseline")
        print("2. ✅ Level 2: python3 test_gptq_kernels_direct.py --save-baseline baseline.pt")
        print("3. ✅ Level 3: ./test_gptq_manual.sh")
    else:
        print("💻 Local Environment Detected")
        print("Limited testing available:")
        print("1. ❌ Level 1: Needs container (full vLLM inference)")
        print("2. ❌ Level 2: Needs container (CUDA extensions)")
        print("3. ✅ Level 3: ./test_gptq_manual.sh (if server running)")
        print()
        print("To enable full testing:")
        print("1. Build container with current clean vLLM")
        print("2. Start container and run baseline capture")
        print("3. Make incremental changes and test each step")

def main():
    print("🧪 GPTQ Testing Environment Check")
    print("=" * 50)
    
    gpu_ok = check_gpu()
    vllm_ok = check_vllm()
    files_ok = check_test_files()
    check_baseline_files()
    recommend_next_steps()
    
    print(f"\n📊 Environment Summary")
    print("=" * 25)
    print(f"GPU: {'✅' if gpu_ok else '❌'}")
    print(f"vLLM: {'✅' if vllm_ok else '❌'}")
    print(f"Test Files: {'✅' if files_ok else '❌'}")
    
    if gpu_ok and files_ok:
        if vllm_ok:
            print("🎉 Ready for full testing suite!")
        else:
            print("🔧 Ready for container-based testing")
    else:
        print("⚠️  Environment issues detected")

if __name__ == "__main__":
    main()