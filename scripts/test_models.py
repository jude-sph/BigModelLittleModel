#!/usr/bin/env python3
"""Test that models load and run correctly on MLX."""

import argparse
import sys
import time

import mlx.core as mx


def test_model_loading(model_path: str, max_tokens: int = 50) -> dict:
    """Test loading and running a model.

    Args:
        model_path: HuggingFace model path
        max_tokens: Number of tokens to generate (higher = better throughput measurement)

    Returns:
        Dict with test results
    """
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print("=" * 60)

    results = {
        "model": model_path,
        "load_success": False,
        "generate_success": False,
        "load_time_s": 0,
        "tokens_per_second": 0,
        "memory_mb": 0,
    }

    # Test loading
    print("\n1. Loading model...")
    start = time.perf_counter()
    try:
        model, tokenizer = load(model_path)
        load_time = time.perf_counter() - start
        results["load_success"] = True
        results["load_time_s"] = round(load_time, 2)
        print(f"   ✓ Loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        return results

    # Check memory usage
    try:
        memory_info = mx.get_active_memory()
        memory_mb = memory_info / (1024 * 1024)
        results["memory_mb"] = round(memory_mb, 2)
        print(f"   Memory used: {memory_mb:.0f} MB")
    except Exception:
        pass

    # Test generation
    print(f"\n2. Testing generation ({max_tokens} tokens max)...")
    # Use a prompt that encourages longer generation for sustained throughput tests
    if max_tokens > 100:
        test_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Write a detailed explanation of how computers work, covering CPU, memory, and storage.<|im_end|>
<|im_start|>assistant
"""
    else:
        test_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
List 3 colors.<|im_end|>
<|im_start|>assistant
"""

    start = time.perf_counter()
    try:
        sampler = make_sampler(temp=0.7)
        response = generate(
            model,
            tokenizer,
            prompt=test_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        gen_time = time.perf_counter() - start
        tokens = len(tokenizer.encode(response))
        tps = tokens / gen_time if gen_time > 0 else 0

        results["generate_success"] = True
        results["tokens_per_second"] = round(tps, 2)

        print(f"   ✓ Generated {tokens} tokens in {gen_time:.2f}s ({tps:.1f} tok/s)")
        print(f"   Response: {response[:100]}...")
    except Exception as e:
        print(f"   ✗ Failed to generate: {e}")

    # Cleanup
    del model
    del tokenizer
    mx.clear_cache()

    return results


def main():
    """Test all configured models."""
    parser = argparse.ArgumentParser(description="Test MLX models for BMLM")
    parser.add_argument(
        "models",
        nargs="*",
        help="Model paths to test (default: test all configured models)",
    )
    parser.add_argument(
        "--sustained",
        action="store_true",
        help="Test sustained throughput with 256 tokens instead of 50",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Custom max tokens to generate (overrides --sustained)",
    )
    args = parser.parse_args()

    # Models to test (you can modify this list)
    models_to_test = args.models or [
        # Big model candidates
        "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        # Small model candidates
        "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    ]

    # Determine max tokens
    if args.max_tokens:
        max_tokens = args.max_tokens
    elif args.sustained:
        max_tokens = 256
    else:
        max_tokens = 50

    print("BMLM Model Test Suite")
    print("=" * 60)
    print(f"Testing {len(models_to_test)} model(s)")
    print(f"Max tokens: {max_tokens}" + (" (sustained)" if args.sustained else ""))

    all_results = []
    for model_path in models_to_test:
        try:
            results = test_model_loading(model_path, max_tokens=max_tokens)
            all_results.append(results)
        except Exception as e:
            print(f"\n✗ Error testing {model_path}: {e}")
            all_results.append({"model": model_path, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<45} {'Load':<8} {'Gen':<8} {'tok/s':<10}")
    print("-" * 60)

    for r in all_results:
        model_name = r["model"].split("/")[-1][:40]
        load_ok = "✓" if r.get("load_success") else "✗"
        gen_ok = "✓" if r.get("generate_success") else "✗"
        tps = r.get("tokens_per_second", "N/A")
        print(f"{model_name:<45} {load_ok:<8} {gen_ok:<8} {tps:<10}")

    # Check if all passed
    all_passed = all(
        r.get("load_success") and r.get("generate_success")
        for r in all_results
    )

    if all_passed:
        print("\n✓ All models working correctly!")
        return 0
    else:
        print("\n✗ Some models failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
