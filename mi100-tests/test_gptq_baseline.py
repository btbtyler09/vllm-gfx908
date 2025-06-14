#!/usr/bin/env python3
"""
GPTQ Baseline Testing Suite
Captures baseline outputs and compares against optimized versions
"""

import json
import time
import hashlib
import torch
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import argparse
from vllm import LLM, SamplingParams

class GPTQBaselineTester:
    def __init__(self, model_name: str, tensor_parallel_size: int = 4):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.baseline_file = Path(f"gptq_baseline_{self._get_model_hash()}.json")
        
    def _get_model_hash(self) -> str:
        """Create short hash from model name for file naming"""
        return hashlib.md5(self.model_name.encode()).hexdigest()[:8]
    
    def get_test_prompts(self) -> List[Dict[str, Any]]:
        """Define test prompts with different characteristics"""
        return [
            {
                "id": "short_greeting",
                "prompt": "Hello, how are you today?",
                "max_tokens": 50,
                "description": "Short greeting - tests basic token generation"
            },
            {
                "id": "math_simple", 
                "prompt": "What is 2 + 2?",
                "max_tokens": 20,
                "description": "Simple math - tests factual accuracy"
            },
            {
                "id": "story_prompt",
                "prompt": "Write a short story about a robot discovering emotions.",
                "max_tokens": 150,
                "description": "Creative writing - tests coherent longer generation"
            },
            {
                "id": "technical_explanation",
                "prompt": "Explain how a neural network learns, in simple terms.",
                "max_tokens": 200,
                "description": "Technical content - tests domain knowledge"
            },
            {
                "id": "conversation_start",
                "prompt": "User: I'm feeling stressed about work. Can you help me?\nAssistant:",
                "max_tokens": 100,
                "description": "Conversation - tests contextual understanding"
            },
            {
                "id": "edge_case_short",
                "prompt": "Hi",
                "max_tokens": 10,
                "description": "Very short prompt - edge case that often triggers issues"
            }
        ]
    
    def run_inference(self, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run inference and capture detailed results"""
        print(f"Loading model: {self.model_name}")
        print(f"Tensor parallel size: {self.tensor_parallel_size}")
        
        # Initialize LLM
        llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.98,
            max_model_len=8192
        )
        
        results = {
            "model_name": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "timestamp": time.time(),
            "test_results": []
        }
        
        print(f"\nRunning {len(prompts)} test prompts...")
        
        for i, test_case in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Testing: {test_case['description']}")
            print(f"Prompt: '{test_case['prompt'][:50]}{'...' if len(test_case['prompt']) > 50 else ''}'")
            
            # Prepare sampling params
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for consistency
                max_tokens=test_case["max_tokens"],
                top_p=1.0
            )
            
            # Time the inference
            start_time = time.time()
            
            try:
                outputs = llm.generate([test_case["prompt"]], sampling_params)
                inference_time = time.time() - start_time
                
                output = outputs[0]
                generated_text = output.outputs[0].text
                token_ids = output.outputs[0].token_ids
                
                # Check for regression indicators
                has_exclamation_regression = "!!!!" in generated_text
                
                result = {
                    "test_id": test_case["id"],
                    "prompt": test_case["prompt"],
                    "max_tokens": test_case["max_tokens"],
                    "description": test_case["description"],
                    "generated_text": generated_text,
                    "token_ids": token_ids,
                    "num_tokens": len(token_ids),
                    "inference_time": inference_time,
                    "tokens_per_second": len(token_ids) / inference_time if inference_time > 0 else 0,
                    "has_exclamation_regression": has_exclamation_regression,
                    "text_length": len(generated_text)
                }
                
                # Print summary
                print(f"✅ Generated {len(token_ids)} tokens in {inference_time:.2f}s ({result['tokens_per_second']:.1f} tok/s)")
                if has_exclamation_regression:
                    print(f"⚠️  WARNING: '!!!!' detected in output")
                print(f"Output preview: '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'")
                
            except Exception as e:
                print(f"❌ Error during inference: {e}")
                result = {
                    "test_id": test_case["id"],
                    "prompt": test_case["prompt"],
                    "error": str(e),
                    "inference_time": None,
                    "tokens_per_second": 0
                }
            
            results["test_results"].append(result)
        
        # Calculate overall statistics
        successful_tests = [r for r in results["test_results"] if "error" not in r]
        if successful_tests:
            total_tokens = sum(r["num_tokens"] for r in successful_tests)
            total_time = sum(r["inference_time"] for r in successful_tests)
            avg_throughput = total_tokens / total_time if total_time > 0 else 0
            
            results["summary"] = {
                "total_tests": len(prompts),
                "successful_tests": len(successful_tests),
                "failed_tests": len(prompts) - len(successful_tests),
                "total_tokens": total_tokens,
                "total_inference_time": total_time,
                "average_throughput": avg_throughput,
                "has_any_regressions": any(r.get("has_exclamation_regression", False) for r in successful_tests)
            }
            
            print(f"\n📊 Overall Results:")
            print(f"Successful tests: {len(successful_tests)}/{len(prompts)}")
            print(f"Total tokens generated: {total_tokens}")
            print(f"Average throughput: {avg_throughput:.1f} tokens/second")
            if results["summary"]["has_any_regressions"]:
                print(f"⚠️  REGRESSION DETECTED: '!!!!' found in outputs")
        
        return results
    
    def save_baseline(self, results: Dict[str, Any]) -> None:
        """Save baseline results to file"""
        with open(self.baseline_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Baseline saved to: {self.baseline_file}")
    
    def load_baseline(self) -> Dict[str, Any]:
        """Load baseline results from file"""
        if not self.baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {self.baseline_file}")
        
        with open(self.baseline_file, 'r') as f:
            return json.load(f)
    
    def compare_results(self, current_results: Dict[str, Any], baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results against baseline"""
        comparison = {
            "timestamp": time.time(),
            "baseline_timestamp": baseline_results["timestamp"],
            "model_name": current_results["model_name"],
            "test_comparisons": [],
            "summary": {}
        }
        
        print(f"\n🔍 Comparing against baseline from {time.ctime(baseline_results['timestamp'])}")
        
        # Compare each test
        baseline_tests = {t["test_id"]: t for t in baseline_results["test_results"]}
        current_tests = {t["test_id"]: t for t in current_results["test_results"]}
        
        output_matches = []
        performance_changes = []
        
        for test_id in baseline_tests:
            if test_id not in current_tests:
                print(f"⚠️  Test {test_id} missing in current results")
                continue
                
            baseline_test = baseline_tests[test_id]
            current_test = current_tests[test_id]
            
            # Compare outputs
            baseline_tokens = baseline_test.get("token_ids", [])
            current_tokens = current_test.get("token_ids", [])
            tokens_match = baseline_tokens == current_tokens
            
            baseline_text = baseline_test.get("generated_text", "")
            current_text = current_test.get("generated_text", "")
            text_matches = baseline_text == current_text
            
            # Performance comparison
            baseline_tps = baseline_test.get("tokens_per_second", 0)
            current_tps = current_test.get("tokens_per_second", 0)
            performance_change = (current_tps - baseline_tps) / baseline_tps * 100 if baseline_tps > 0 else 0
            
            # Regression check
            has_regression = current_test.get("has_exclamation_regression", False)
            
            test_comparison = {
                "test_id": test_id,
                "description": current_test.get("description", ""),
                "tokens_match": tokens_match,
                "text_matches": text_matches,
                "baseline_tokens_per_second": baseline_tps,
                "current_tokens_per_second": current_tps,
                "performance_change_percent": performance_change,
                "has_regression": has_regression,
                "baseline_tokens": len(baseline_tokens),
                "current_tokens": len(current_tokens)
            }
            
            comparison["test_comparisons"].append(test_comparison)
            output_matches.append(tokens_match)
            performance_changes.append(performance_change)
            
            # Print result
            status = "✅" if tokens_match and not has_regression else "❌"
            perf_indicator = "📈" if performance_change > 0 else "📉" if performance_change < 0 else "➡️"
            print(f"{status} {test_id}: Output {'matches' if tokens_match else 'DIFFERS'}, "
                  f"{perf_indicator} {performance_change:+.1f}% performance")
            
            if has_regression:
                print(f"   ⚠️  REGRESSION: '!!!!' detected in output")
            if not tokens_match:
                print(f"   📝 Token count: {len(baseline_tokens)} → {len(current_tokens)}")
        
        # Overall summary
        total_tests = len(output_matches)
        matching_outputs = sum(output_matches)
        avg_performance_change = np.mean(performance_changes) if performance_changes else 0
        any_regressions = any(t["has_regression"] for t in comparison["test_comparisons"])
        
        comparison["summary"] = {
            "total_tests": total_tests,
            "matching_outputs": matching_outputs,
            "output_accuracy": matching_outputs / total_tests * 100 if total_tests > 0 else 0,
            "average_performance_change": avg_performance_change,
            "any_regressions": any_regressions,
            "all_outputs_match": matching_outputs == total_tests,
            "performance_improved": avg_performance_change > 0
        }
        
        print(f"\n📊 Comparison Summary:")
        print(f"Output accuracy: {matching_outputs}/{total_tests} ({comparison['summary']['output_accuracy']:.1f}%)")
        print(f"Average performance change: {avg_performance_change:+.1f}%")
        if any_regressions:
            print(f"⚠️  REGRESSIONS DETECTED")
        
        return comparison
    
    def capture_baseline(self) -> None:
        """Capture baseline results"""
        print("🎯 Capturing GPTQ Baseline Results")
        print("=" * 50)
        
        prompts = self.get_test_prompts()
        results = self.run_inference(prompts)
        self.save_baseline(results)
        
        print(f"\n✅ Baseline capture complete!")
        print(f"📁 Results saved to: {self.baseline_file}")
    
    def test_against_baseline(self) -> bool:
        """Test current implementation against baseline"""
        print("🔍 Testing Against Baseline")
        print("=" * 50)
        
        # Load baseline
        try:
            baseline_results = self.load_baseline()
        except FileNotFoundError:
            print(f"❌ No baseline found. Run with --capture-baseline first.")
            return False
        
        # Run current tests
        prompts = self.get_test_prompts()
        current_results = self.run_inference(prompts)
        
        # Compare
        comparison = self.compare_results(current_results, baseline_results)
        
        # Save comparison
        comparison_file = f"gptq_comparison_{int(time.time())}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n💾 Comparison saved to: {comparison_file}")
        
        # Return success/failure
        success = (comparison["summary"]["all_outputs_match"] and 
                  not comparison["summary"]["any_regressions"])
        
        if success:
            print(f"\n✅ ALL TESTS PASSED")
            if comparison["summary"]["performance_improved"]:
                print(f"🚀 Performance improved by {comparison['summary']['average_performance_change']:+.1f}%")
        else:
            print(f"\n❌ TESTS FAILED")
            if not comparison["summary"]["all_outputs_match"]:
                print(f"   Output mismatch detected")
            if comparison["summary"]["any_regressions"]:
                print(f"   Regression ('!!!!') detected")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="GPTQ Baseline Testing Suite")
    parser.add_argument("--model", default="kaitchup/Qwen3-32B-autoround-4bit-gptq",
                       help="Model name to test")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                       help="Tensor parallel size")
    parser.add_argument("--capture-baseline", action="store_true",
                       help="Capture baseline results")
    parser.add_argument("--test", action="store_true",
                       help="Test against baseline")
    
    args = parser.parse_args()
    
    if not args.capture_baseline and not args.test:
        parser.error("Must specify either --capture-baseline or --test")
    
    tester = GPTQBaselineTester(args.model, args.tensor_parallel_size)
    
    if args.capture_baseline:
        tester.capture_baseline()
    
    if args.test:
        success = tester.test_against_baseline()
        exit(0 if success else 1)

if __name__ == "__main__":
    main()