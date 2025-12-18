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


def create_short_label(backend: str, config: Dict) -> str:
    """
    Create a short label for plotting.

    Args:
        backend: Backend name
        config: Configuration dict

    Returns:
        Short label string (e.g., "Whisper", "Parakeet-greedy", "FastConf-rnnt")
    """
    if not config:
        return backend.capitalize()

    # Extract key distinguishing parameters
    key_params = []
    important_keys = ["DECODING_STRATEGY", "DECODER_TYPE", "BEAM_SIZE"]

    for key in important_keys:
        if key in config and config[key]:
            value = config[key]
            # Shorten common values
            if isinstance(value, str):
                key_params.append(value[:6])  # First 6 chars
            elif isinstance(value, (int, float)) and value != 1:
                key_params.append(str(value))

    # Create backend abbreviation
    backend_abbr = backend.capitalize()[:7]  # Max 7 chars

    if key_params:
        return f"{backend_abbr}-{'-'.join(key_params)}"
    return backend_abbr


def print_backend_configurations(backends) -> None:
    """
    Print backend configurations in a readable format.

    Args:
        backends: Dict or list of backend configuration dicts
    """
    print("=" * 80)
    print("CONFIGURED BACKENDS")
    print("=" * 80)

    # Handle both dict (from get_backends()) and list formats
    if isinstance(backends, dict):
        backends_list = list(backends.values())
    else:
        backends_list = backends

    for i, backend in enumerate(backends_list, 1):
        name = backend.get("name", "Unknown")
        url = backend.get("url", "Unknown")
        config = backend.get("config", {})

        # Create a short identifier
        short_label = create_short_label(name, config)

        print(f"\n{i}. {name}")
        print(f"   URL: {url}")
        if config:
            print(f"   Config:")
            for key, value in sorted(config.items()):
                print(f"      {key}: {value}")
        else:
            print(f"   Config: Default settings")
        print(f"   Chart Label: {short_label}")

    print("\n" + "=" * 80 + "\n")


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
        rows.append(
            {
                "Backend": r["backend"],
                "Avg Latency (ms)": r["avg_latency_ms"],
                "P95 Latency (ms)": r["p95_latency_ms"],
                "Max Latency (ms)": r["max_latency_ms"],
                "Final Transcript": r["final_transcript"],
            }
        )

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

    # Create short labels for plots
    labels = [create_short_label(r["backend"], r.get("config", {})) for r in successful_results]
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

        # Create short label for plot
        label = create_short_label(result["backend"], result.get("config", {}))

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

    # Normalize reference transcript
    import re

    ref_normalized = reference_transcript.lower().strip()
    # Remove extra whitespace and punctuation for better comparison
    ref_normalized = re.sub(r"[^\w\s]", "", ref_normalized)
    ref_normalized = " ".join(ref_normalized.split())

    for result in successful_results:
        hypothesis = result["final_transcript"].strip()

        if hypothesis:
            # Normalize hypothesis the same way
            hyp_normalized = hypothesis.lower().strip()
            hyp_normalized = re.sub(r"[^\w\s]", "", hyp_normalized)
            hyp_normalized = " ".join(hyp_normalized.split())

            # Calculate WER
            error_rate = wer(ref_normalized, hyp_normalized)

            # Count words for context
            ref_words = len(ref_normalized.split())
            hyp_words = len(hyp_normalized.split())

            print(f"\n{result['backend'].upper()}:")
            print(f"  Reference:  '{reference_transcript[:80]}...' ({ref_words} words)")
            print(f"  Hypothesis: '{hypothesis[:80]}...' ({hyp_words} words)")
            print(f"  WER: {error_rate*100:.2f}% (lower is better)")
            print(f"  Accuracy: {(1-error_rate)*100:.2f}%")
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
