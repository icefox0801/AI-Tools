#!/usr/bin/env python3
"""
Audio Notes - Transcription & AI Summarization App

A Gradio web UI that:
1. Lists available recordings from the shared recordings directory
2. Receives audio files (from Live Captions or direct upload)
3. Transcribes using Whisper/Parakeet ASR service
4. Summarizes using Ollama LLM
5. Provides interactive chat for Q&A about the content

API Endpoints:
- POST /api/upload-audio: Upload audio chunks from Live Captions
- GET /api/recordings: List available recordings
- GET /api/health: Health check

Usage:
  python audio_notes.py                    # Start web UI
  python audio_notes.py --port 7860        # Custom port
"""

import argparse
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import uvicorn
from api.routes import setup_api_routes
from config import RECORDINGS_DIR, logger
from fastapi import FastAPI
from ui import create_ui
from ui.styles import CUSTOM_CSS

# ==============================================================================
# Version
# ==============================================================================

__version__ = "1.2"


def create_app(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """Create FastAPI app with Gradio mounted."""

    # Create FastAPI app
    app = FastAPI(title="Audio Notes API", version=__version__)

    # Set up custom API routes FIRST (before Gradio takes over)
    setup_api_routes(app)
    logger.info("API routes registered: /api/upload-audio, /api/recordings, /api/health")

    # Create Gradio UI
    demo = create_ui()

    # Custom theme to fix radio/checkbox hover visibility
    custom_theme = gr.themes.Default().set(
        checkbox_background_color_hover="#e5e7eb",  # Light gray instead of white
        checkbox_background_color_hover_dark="#374151",  # Dark mode
    )

    # Mount Gradio app at root with custom theme and CSS
    app = gr.mount_gradio_app(app, demo, path="/", theme=custom_theme, css=CUSTOM_CSS)

    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audio Notes - Transcription & AI Summarization")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    print("=" * 50)
    print("📝 Audio Notes")
    print("=" * 50)
    print(f"Recordings directory: {RECORDINGS_DIR}")
    print(f"Server: http://{args.host}:{args.port}")
    print("API endpoints:")
    print("  POST /api/upload-audio - Upload audio from Live Captions")
    print("  GET  /api/recordings   - List recordings")
    print("  GET  /api/health       - Health check")
    print("=" * 50)

    # Docker mode: auto-detect via /.dockerenv
    host = args.host
    if os.path.exists("/.dockerenv"):
        host = "0.0.0.0"

    # Create and run app
    app = create_app(host=host, port=args.port, share=args.share)

    uvicorn.run(app, host=host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
