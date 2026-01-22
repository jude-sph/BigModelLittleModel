#!/usr/bin/env python3
"""Profile latency of model inference for different configurations."""

import statistics
import sys
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


@dataclass
class LatencyProfile:
    """Latency statistics for a model configuration."""

    model: str
    max_tokens: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    tokens_per_second: float


def profile_model(
    model_path: str,
    max_tokens_list: list[int],
    n_runs: int = 10,
    warmup_runs: int = 2,
) -> list[LatencyProfile]:
    """Profile a model at different token counts.

    Args:
        model_path: Path to MLX model
        max_tokens_list: List of max_tokens values to test
        n_runs: Number of runs per configuration
        warmup_runs: Number of warmup runs (not counted)

    Returns:
        List of LatencyProfile results
    """
    print(f"\nLoading {model_path}...")
    model, tokenizer = load(model_path)

    # Sample prompts that simulate our use case
    planning_prompt = """<|im_start|>system
You are an Android GUI automation agent. Output JSON plans.<|im_end|>
<|im_start|>user
Task: Open the settings app and enable dark mode.
UI Elements: [{"id": "settings_icon", "text": "Settings", "type": "ImageView"}]
Create a plan.<|im_end|>
<|im_start|>assistant
"""

    execution_prompt = """<|im_start|>system
You are an Android executor agent. Output JSON actions.<|im_end|>
<|im_start|>user
Step: Tap the settings icon
UI Elements: [{"id": "settings_icon", "text": "Settings"}]
Execute this step.<|im_end|>
<|im_start|>assistant
"""

    results = []
    sampler = make_sampler(temp=0.7)

    for max_tokens in max_tokens_list:
        # Use appropriate prompt based on token count
        prompt = planning_prompt if max_tokens > 100 else execution_prompt

        print(f"\n  Testing max_tokens={max_tokens}...")

        # Warmup
        for _ in range(warmup_runs):
            generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)
            mx.eval()  # Ensure computation is complete

        # Timed runs
        latencies = []
        total_tokens = 0

        for i in range(n_runs):
            start = time.perf_counter()
            response = generate(
                model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler
            )
            mx.eval()
            elapsed_ms = (time.perf_counter() - start) * 1000

            tokens = len(tokenizer.encode(response))
            total_tokens += tokens
            latencies.append(elapsed_ms)

            print(f"    Run {i+1}: {elapsed_ms:.1f}ms ({tokens} tokens)")

        # Calculate statistics
        latencies_sorted = sorted(latencies)
        avg_tokens = total_tokens / n_runs
        total_time_s = sum(latencies) / 1000

        profile = LatencyProfile(
            model=model_path.split("/")[-1],
            max_tokens=max_tokens,
            mean_ms=statistics.mean(latencies),
            std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            min_ms=min(latencies),
            max_ms=max(latencies),
            p50_ms=latencies_sorted[len(latencies) // 2],
            p95_ms=latencies_sorted[int(len(latencies) * 0.95)],
            tokens_per_second=total_tokens / total_time_s if total_time_s > 0 else 0,
        )
        results.append(profile)

    # Cleanup
    del model
    del tokenizer
    mx.clear_cache()

    return results


def print_results(results: list[LatencyProfile]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("LATENCY PROFILE RESULTS")
    print("=" * 80)

    headers = ["Model", "MaxTok", "Mean(ms)", "Std", "Min", "Max", "P50", "P95", "tok/s"]
    widths = [25, 8, 10, 8, 8, 8, 8, 8, 8]

    # Print header
    header_line = " ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header_line)
    print("-" * 80)

    # Print rows
    for r in results:
        row = [
            r.model[:24],
            str(r.max_tokens),
            f"{r.mean_ms:.1f}",
            f"{r.std_ms:.1f}",
            f"{r.min_ms:.1f}",
            f"{r.max_ms:.1f}",
            f"{r.p50_ms:.1f}",
            f"{r.p95_ms:.1f}",
            f"{r.tokens_per_second:.1f}",
        ]
        print(" ".join(f"{v:<{w}}" for v, w in zip(row, widths)))


def main():
    """Run latency profiling."""
    # Default configuration
    models = [
        "mlx-community/Qwen2.5-3B-Instruct-4bit",  # Small model
    ]
    max_tokens_list = [64, 128, 256]
    n_runs = 5

    # Parse arguments
    if "--help" in sys.argv:
        print("Usage: profile_latency.py [model_path] [--tokens N,N,N] [--runs N]")
        return 0

    # Allow model override
    for i, arg in enumerate(sys.argv[1:], 1):
        if not arg.startswith("--"):
            models = [arg]
        elif arg == "--tokens" and i + 1 < len(sys.argv):
            max_tokens_list = [int(x) for x in sys.argv[i + 1].split(",")]
        elif arg == "--runs" and i + 1 < len(sys.argv):
            n_runs = int(sys.argv[i + 1])

    print("BMLM Latency Profiler")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Token counts: {max_tokens_list}")
    print(f"Runs per config: {n_runs}")

    all_results = []
    for model_path in models:
        try:
            results = profile_model(
                model_path=model_path,
                max_tokens_list=max_tokens_list,
                n_runs=n_runs,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\n✗ Error profiling {model_path}: {e}")

    if all_results:
        print_results(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
