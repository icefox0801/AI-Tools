"""
ASR Benchmark Helper Functions

This module contains WebSocket-based ASR client and benchmark functions
for testing streaming ASR backends.
"""

import asyncio
import json
import time
import numpy as np
import websockets
from typing import Dict, Tuple, List


class StreamingASRClient:
    """WebSocket-based streaming ASR client for all backends."""

    def __init__(self, backend_name: str, base_url: str):
        self.backend_name = backend_name
        self.base_url = base_url
        self.ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/stream"

    async def transcribe_stream(self, audio: np.ndarray, chunk_size: int) -> Tuple[str, list, list]:
        """
        Stream audio to ASR service via WebSocket and collect results.

        Args:
            audio: Audio data as float32 numpy array (-1.0 to 1.0)
            chunk_size: Number of samples per chunk

        Returns:
            Tuple of (final_transcript, list of partial transcripts, list of latencies)

        Note:
            Model settings (VAD, beam size, language, etc.) are configured via
            environment variables in docker-compose.yaml and cannot be changed at runtime.
        """
        transcripts = []
        latencies = []
        final_transcript = ""

        try:
            # Increase timeouts significantly for slower backends like VOSK
            # VOSK can be very slow, especially on first requests
            async with websockets.connect(
                self.ws_url,
                ping_interval=30,  # Send ping every 30s
                ping_timeout=60,  # Wait 60s for pong response (2x ping_interval)
                close_timeout=15,  # Wait 15s for close handshake
                max_size=10 * 1024 * 1024,  # 10MB max message size
            ) as ws:
                print(f"[{self.backend_name}] WebSocket connected to {self.ws_url}")

                # Convert audio to int16 PCM bytes
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                # Send audio in chunks
                num_chunks = len(audio_bytes) // (chunk_size * 2)  # *2 because int16 = 2 bytes
                bytes_per_chunk = chunk_size * 2

                for i in range(num_chunks):
                    start_idx = i * bytes_per_chunk
                    end_idx = start_idx + bytes_per_chunk
                    chunk = audio_bytes[start_idx:end_idx]

                    # Measure latency
                    start_time = time.time()
                    await ws.send(chunk)

                    # Try to receive response with longer timeout for slow backends
                    # VOSK can be much slower than other backends
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        latency = time.time() - start_time

                        data = json.loads(response)
                        text = data.get("text", "")

                        transcripts.append(text)
                        latencies.append(latency)

                        if text:
                            print(
                                f"[{self.backend_name}] Chunk {i+1}/{num_chunks}: {latency*1000:.2f}ms - '{text}'"
                            )

                        if data.get("final"):
                            final_transcript = text
                    except asyncio.TimeoutError:
                        # No response yet, that's okay
                        pass

                # Send end-of-stream signal (empty bytes)
                await ws.send(b"")
                print(f"[{self.backend_name}] Sent end-of-stream signal")

                # Wait for final response with extended timeout
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    data = json.loads(response)
                    final_text = data.get("text", "")
                    if final_text:
                        final_transcript = final_text
                    print(f"[{self.backend_name}] Final response: '{final_text}'")
                except asyncio.TimeoutError:
                    print(f"[{self.backend_name}] No final response (timeout)")

                # Use last non-empty transcript if no final transcript
                if not final_transcript and transcripts:
                    final_transcript = next((t for t in reversed(transcripts) if t), "")

                return final_transcript, transcripts, latencies

        except Exception as e:
            print(f"[{self.backend_name}] WebSocket error: {e}")
            raise


async def benchmark_backend(
    backend_name: str, base_url: str, audio: np.ndarray, chunk_size: int
) -> Dict:
    """
    Benchmark a single ASR backend with streaming audio via WebSocket.

    Args:
        backend_name: Name of the backend
        base_url: Base URL of the ASR service
        audio: Audio data as numpy array
        chunk_size: Size of each audio chunk (in samples)

    Returns:
        Dict containing benchmark results with keys:
        - backend: backend name
        - config: configuration used
        - avg_latency_ms: average latency
        - p95_latency_ms: 95th percentile latency
        - max_latency_ms: maximum latency
        - final_transcript: final transcription
        - latencies: list of latencies
        - transcripts: list of partial transcripts
        - num_responses: number of responses received
        - success: True if successful
        - error: error message if failed
    """
    client = StreamingASRClient(backend_name, base_url)

    try:
        print(f"[{backend_name}] Starting transcription...")

        final_transcript, transcripts, latencies = await client.transcribe_stream(audio, chunk_size)

        print(f"[{backend_name}] Transcription complete")
        print(f"[{backend_name}] Final transcript: '{final_transcript}'")

        # Calculate metrics
        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            max_latency = np.max(latencies)
        else:
            avg_latency = p95_latency = max_latency = 0.0

        return {
            "backend": backend_name,
            "config": {},  # Config now comes from environment variables
            "avg_latency_ms": avg_latency * 1000,
            "p95_latency_ms": p95_latency * 1000,
            "max_latency_ms": max_latency * 1000,
            "final_transcript": final_transcript,
            "latencies": latencies,
            "transcripts": transcripts,
            "num_responses": len(transcripts),
            "success": True,
        }

    except Exception as e:
        print(f"[{backend_name}] Error: {e}")
        import traceback

        traceback.print_exc()
        return {"backend": backend_name, "success": False, "error": str(e)}


async def run_all_benchmarks(
    active_backends: Dict, test_audio: np.ndarray, test_config: Dict
) -> List[Dict]:
    """
    Run benchmarks for all configured backends.

    Args:
        active_backends: Dict of backend configurations from get_active_backends()
        test_audio: Audio data as numpy array
        test_config: Test configuration from get_test_config()

    Returns:
        List of benchmark result dicts
    """
    results = []

    print(f"\n🎯 Running benchmarks with:")
    print(f"   • Chunk Duration: {test_config['chunk_duration']}s")
    print(f"   • Sample Rate: {test_config['sample_rate']}Hz")
    print(f"   • Active Variants: {list(active_backends.keys())}")
    print()

    for variant_key, backend_info in active_backends.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {backend_info['name']}")
        print(f"Backend: {backend_info['backend'].upper()}")
        print(f"{'='*60}")

        result = await benchmark_backend(
            backend_info["name"],  # Use variant name for display
            backend_info["url"],
            test_audio,
            test_config["chunk_size"],
        )
        results.append(result)

    return results
