# Notebook Helpers

This directory contains helper modules to keep Jupyter notebooks clean and focused on configuration and analysis, rather than implementation details.

## Modules

### `asr_config.py`
Configuration UI helpers for ASR benchmarking.

**Functions:**
- `create_config_widget(opt_name, opt_config)` - Create ipywidgets based on configuration spec
- `render_variants(variants_output)` - Render all configured model variants
- `add_variant(backend_name, variant_name, backend_urls, backend_options, variants_output)` - Add a new variant
- `get_active_backends()` - Get list of configured variants with their settings
- `get_test_config(chunk_duration_widget, sample_rate_widget)` - Get test configuration
- `reset_variants()` - Clear all variants

**Usage in notebooks:**
```python
from asr_config import create_config_widget, render_variants, add_variant, get_active_backends, get_test_config

# Create widgets and UI
variants_output = widgets.Output()
backend_selector = widgets.Dropdown(options=['whisper', 'parakeet', 'vosk', 'fastconformer'])

# Add variants
add_variant('whisper', 'Fast Whisper', BACKEND_URLS, BACKEND_OPTIONS, variants_output)

# Get configured backends
active_backends = get_active_backends()
```

### `audio_generation.py`
Audio generation utilities for creating test audio.

**Functions:**
- `generate_text_with_ollama(word_count, models, base_urls)` - Generate text using Ollama LLM
- `generate_fallback_text(word_count)` - Generate fallback text if Ollama unavailable
- `text_to_speech_gtts(text, sample_rate, speed)` - Convert text to speech using gTTS

**Usage in notebooks:**
```python
from audio_generation import generate_text_with_ollama, text_to_speech_gtts

# Generate text
text = generate_text_with_ollama(word_count=500)

# Convert to speech
audio = text_to_speech_gtts(text, sample_rate=16000, speed=1.5)
```

### `benchmark.py`
WebSocket-based ASR benchmarking utilities.

**Classes:**
- `StreamingASRClient` - WebSocket client for streaming ASR services

**Functions:**
- `benchmark_backend(backend_name, base_url, audio, chunk_size, config)` - Benchmark a single backend
- `run_all_benchmarks(active_backends, test_audio, test_config)` - Run benchmarks for all configured backends

**Usage in notebooks:**
```python
from benchmark import StreamingASRClient, benchmark_backend, run_all_benchmarks

# Run all benchmarks
results = await run_all_benchmarks(
    get_active_backends(),
    test_audio,
    get_test_config()
)
```

### `results_analysis.py`
Results analysis and visualization utilities.

**Functions:**
- `create_results_dataframe(benchmark_results)` - Create pandas DataFrame from results
- `print_results_summary(df)` - Print formatted results table
- `plot_latency_comparison(successful_results)` - Plot average and distribution latency
- `plot_latency_over_time(successful_results)` - Plot latency progression over chunks
- `calculate_wer_metrics(successful_results, reference_transcript)` - Calculate Word Error Rate
- `export_results_json(benchmark_results, test_audio, test_config, output_file)` - Export to JSON

**Usage in notebooks:**
```python
from results_analysis import (
    create_results_dataframe, 
    print_results_summary,
    plot_latency_comparison,
    calculate_wer_metrics
)

# Analyze results
df = create_results_dataframe(benchmark_results)
print_results_summary(df)

# Visualize
plot_latency_comparison(successful_results)

# Calculate WER
calculate_wer_metrics(successful_results, reference_transcript)
```

## Design Philosophy

The helpers are designed to:

1. **Separate concerns** - Keep UI code separate from notebook logic
2. **Reduce clutter** - Notebooks focus on variables and configuration
3. **Improve maintainability** - Helper functions can be tested and updated independently
4. **Enable reuse** - Same helpers can be used across multiple notebooks

## Adding New Helpers

When creating new helper modules:

1. Create a new `.py` file in this directory
2. Add clear docstrings for all functions
3. Import and expose in `__init__.py`
4. Update this README with usage examples
5. Keep helpers focused on a single responsibility

## Testing

Helper modules should be pure Python functions that can be tested independently. Consider adding unit tests in `tests/` directory for complex logic.
