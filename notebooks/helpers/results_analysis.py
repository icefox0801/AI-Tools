"""
ASR Benchmark Results Analysis

Helper functions for analyzing and visualizing benchmark results.
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jiwer import wer
from typing import List, Dict


def create_results_dataframe(benchmark_results: List[Dict]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from benchmark results.

    Args:
        benchmark_results: List of benchmark result dicts

    Returns:
        DataFrame with benchmark metrics
    """
    successful_results = [r for r in benchmark_results if r.get("success", False)]

    if not successful_results:
        return pd.DataFrame()

    rows = []
    for r in successful_results:
        # Create a readable config description
        config = r.get("config", {})
        config_str = ", ".join([f"{k}={v}" for k, v in sorted(config.items())]) if config else "default"
        
        rows.append({
            "Backend": r["backend"],
            "Config": config_str,
            "Avg Latency (ms)": r["avg_latency_ms"],
            "P95 Latency (ms)": r["p95_latency_ms"],
            "Max Latency (ms)": r["max_latency_ms"],
            "Final Transcript": r["final_transcript"],
        })

    df = pd.DataFrame(rows)
    return df


def print_results_summary(df: pd.DataFrame):
    """Print formatted results summary table."""
    if df.empty:
        print("No successful benchmark results to display.")
        return

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


def plot_latency_comparison(successful_results: List[Dict]):
    """
    Create latency comparison visualizations.

    Args:
        successful_results: List of successful benchmark results
    """
    if not successful_results:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Create labels with backend and config
    labels = []
    for r in successful_results:
        config = r.get("config", {})
        if config:
            config_str = ", ".join([f"{k}={v}" for k, v in sorted(config.items())])
            labels.append(f"{r['backend']}\n({config_str})")
        else:
            labels.append(r["backend"])
    
    avg_latencies = [r["avg_latency_ms"] for r in successful_results]

    axes[0].bar(
        labels, avg_latencies, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"][: len(labels)]
    )
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Average Latency by Backend")
    axes[0].tick_params(axis="x", rotation=45)

    # Latency Distribution (Box Plot)
    latency_data = [r["latencies"] for r in successful_results]
    axes[1].boxplot([np.array(l) * 1000 for l in latency_data], labels=labels)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Latency Distribution")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_latency_over_time(successful_results: List[Dict]):
    """
    Plot latency over time for each backend.

    Args:
        successful_results: List of successful benchmark results
    """
    if not successful_results:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 6))

    for result in successful_results:
        latencies_ms = np.array(result["latencies"]) * 1000
        chunks = range(1, len(latencies_ms) + 1)
        
        # Create label with config
        config = result.get("config", {})
        if config:
            config_str = ", ".join([f"{k}={v}" for k, v in sorted(config.items())])
            label = f"{result['backend']} ({config_str})"
        else:
            label = result["backend"]
        
        plt.plot(chunks, latencies_ms, marker="o", label=label)

    plt.xlabel("Chunk Number")
    plt.ylabel("Latency (ms)")
    plt.title("Latency Over Time for Each Backend")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_wer_metrics(successful_results: List[Dict], reference_transcript: str):
    """
    Calculate and print Word Error Rate for each backend.

    Args:
        successful_results: List of successful benchmark results
        reference_transcript: Ground truth transcript
    """
    if not successful_results or not reference_transcript:
        print("Skipping WER calculation (no reference transcript or successful results)")
        return

    print("\nWord Error Rate (WER) Analysis:")
    print("=" * 80)

    for result in successful_results:
        hypothesis = result["final_transcript"].lower().strip()
        reference = reference_transcript.lower().strip()

        if hypothesis:
            error_rate = wer(reference, hypothesis)
            print(f"\n{result['backend'].upper()}:")
            print(f"  Reference:  '{reference[:100]}...'")
            print(f"  Hypothesis: '{hypothesis[:100]}...'")
            print(f"  WER: {error_rate*100:.2f}%")
        else:
            print(f"\n{result['backend'].upper()}: No transcript generated")

    print("=" * 80)


def export_results_json(
    benchmark_results: List[Dict],
    test_audio: np.ndarray,
    test_config: Dict,
    output_file: str = "benchmark_results.json",
) -> str:
    """
    Export benchmark results to JSON file.

    Args:
        benchmark_results: List of benchmark results
        test_audio: Audio data used for testing
        test_config: Test configuration dict
        output_file: Output filename

    Returns:
        Path to output file
    """
    export_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "chunk_duration": test_config["chunk_duration"],
            "sample_rate": test_config["sample_rate"],
            "audio_duration": len(test_audio) / test_config["sample_rate"],
        },
        "results": [
            {
                "backend": r["backend"],
                "avg_latency_ms": r.get("avg_latency_ms"),
                "p95_latency_ms": r.get("p95_latency_ms"),
                "max_latency_ms": r.get("max_latency_ms"),
                "final_transcript": r.get("final_transcript"),
                "success": r.get("success", False),
            }
            for r in benchmark_results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    return output_file
