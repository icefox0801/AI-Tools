"""Helper modules for Jupyter notebooks."""

from .asr_config import (
    create_config_widget,
    render_variants,
    add_variant,
    get_active_backends,
    get_test_config,
)

from .audio_generation import (
    generate_text_with_ollama,
    generate_fallback_text,
    text_to_speech_gtts,
)

from .benchmark import (
    StreamingASRClient,
    benchmark_backend,
    run_all_benchmarks,
)

from .results_analysis import (
    create_results_dataframe,
    print_results_summary,
    plot_latency_comparison,
    plot_latency_over_time,
    calculate_wer_metrics,
    export_results_json,
)

from .config_ui import (
    create_configuration_ui,
)

__all__ = [
    # Configuration
    "create_config_widget",
    "render_variants",
    "add_variant",
    "get_active_backends",
    "get_test_config",
    # Audio generation
    "generate_text_with_ollama",
    "generate_fallback_text",
    "text_to_speech_gtts",
    # Benchmarking
    "StreamingASRClient",
    "benchmark_backend",
    "run_all_benchmarks",
    # Results analysis
    "create_results_dataframe",
    "print_results_summary",
    "plot_latency_comparison",
    "plot_latency_over_time",
    "calculate_wer_metrics",
    "export_results_json",
    # Gradio UIs
    "create_configuration_ui",
    "create_audio_generation_ui",
    "get_active_backends_gradio",
    "get_test_config_gradio",
    "get_test_audio_gradio",
]
