"""Helper modules for Jupyter notebooks."""

# Import only from existing modules
from .audio_generation import (
    generate_text_with_ollama,
    generate_fallback_text,
    text_to_speech_gtts,
)

from .audio_upload import (
    display_upload_ui,
    process_uploaded_file,
    resample_audio,
    transcribe_with_whisper,
    upload_and_process_audio,
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
    # Audio generation
    "generate_text_with_ollama",
    "generate_fallback_text",
    "text_to_speech_gtts",
    # Audio upload
    "display_upload_ui",
    "process_uploaded_file",
    "resample_audio",
    "transcribe_with_whisper",
    "upload_and_process_audio",
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
    # Configuration UI
    "create_configuration_ui",
]
