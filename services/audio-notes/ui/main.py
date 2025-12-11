#!/usr/bin/env python3
"""
Main Gradio UI assembly for Audio Notes.

This module brings together all UI sections and tabs,
and wires up the event handlers.
"""

from datetime import datetime
from pathlib import Path

import gradio as gr
from config import RECORDINGS_DIR, logger
from services.langchain_chat import chat_streaming as langchain_chat_streaming
from services.llm import get_summary_prompt_for_length

from services import (
    check_ollama_health,
    check_parakeet_health,
    check_whisper_health,
    clean_transcribed_recordings,
    delete_selected_recordings,
    generate_chat_title,
    list_recordings,
    summarize_streaming,
    transcribe_audio,
    unload_asr_model,
)
from ui.sections import (
    create_recordings_section,
    create_status_section,
    create_summarize_section,
    create_transcription_section,
    create_upload_section,
)
from ui.styles import CUSTOM_JS
from ui.tabs import create_chat_tab, create_summary_tab, create_transcript_tab


def format_recording_title(filename: str) -> str:
    """Format a recording filename into a human-readable title.

    Examples:
        recording_20251203_205252.wav -> Recording - Dec 3, 2025 8:52 PM
        uploaded_20251201_143000.mp3 -> Uploaded - Dec 1, 2025 2:30 PM
        my_audio.wav -> my_audio
    """
    import re
    from datetime import datetime

    name = Path(filename).stem

    # Try to parse recording_YYYYMMDD_HHMMSS or uploaded_YYYYMMDD_HHMMSS pattern
    match = re.match(r"(recording|uploaded)_(\d{8})_(\d{6})", name, re.IGNORECASE)
    if match:
        prefix = match.group(1).capitalize()
        date_str = match.group(2)
        time_str = match.group(3)
        try:
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            formatted_date = dt.strftime("%b %-d, %Y %-I:%M %p").replace(" 0", " ")
            # Windows compatibility - use %#d and %#I instead of %-d and %-I
            try:
                formatted_date = dt.strftime("%b %d, %Y %I:%M %p").lstrip("0").replace(" 0", " ")
            except:
                pass
            return f"{prefix} - {formatted_date}"
        except ValueError:
            pass

    # Fallback: clean up underscores and return the name
    return name.replace("_", " ").title()


def batch_transcribe_streaming(selected_files: list[str], backend: str = "parakeet"):
    """Batch transcribe multiple audio files with streaming progress.

    Yields:
        tuple: (status_text, transcript, state) - intermediate and final results
    """
    if not selected_files:
        yield "⚠️ No files selected for batch transcription", "", ""
        return

    backend_lower = backend.lower() if backend else "parakeet"

    # Optimize GPU memory: unload the other ASR backend before starting
    yield "⏳ Preparing GPU memory...", "", ""
    if backend_lower == "whisper":
        # Unload Parakeet to free memory for Whisper
        unload_asr_model("Parakeet")
        service_ok, service_msg = check_whisper_health()
    else:
        # Unload Whisper to free memory for Parakeet
        unload_asr_model("Whisper")
        service_ok, service_msg = check_parakeet_health()

    if not service_ok:
        yield f"❌ {service_msg}", "", ""
        return

    results = []
    all_transcripts = []
    total_files = len(selected_files)

    for idx, file_path in enumerate(selected_files, 1):
        file_name = Path(file_path).name

        # Show progress for current file
        progress = f"⏳ **Transcribing** ({idx}/{total_files}): `{file_name}`..."
        if results:
            progress = "\n\n".join(results) + "\n\n" + progress
        yield progress, "", ""

        try:
            transcript, duration = transcribe_audio(file_path, backend=backend_lower)

            if transcript.startswith("Error"):
                results.append(f"❌ {file_name}: {transcript}")
                continue

            txt_path = Path(file_path).with_suffix(".txt")
            txt_path.write_text(transcript, encoding="utf-8")

            results.append(
                f"✅ {file_name} ➜ {txt_path.name} ({duration:.1f}s, {len(transcript)} chars)"
            )
            # Use markdown heading for better readability in transcript
            all_transcripts.append(f"## 🎙️ {file_name}\n\n{transcript}")
            logger.info(f"Batch transcribed: {file_name} -> {txt_path}")

        except Exception as e:
            results.append(f"❌ {file_name}: {e}")
            logger.error(f"Batch transcription error for {file_name}: {e}")

    combined_transcript = "\n\n---\n\n".join(all_transcripts) if all_transcripts else ""

    unload_ok, unload_msg = unload_asr_model(backend)
    if unload_ok:
        logger.info(f"ASR model unloaded: {unload_msg}")

    status = "\n\n".join(results)
    logger.info(f"batch_transcribe returning transcript of length: {len(combined_transcript)}")
    yield status, combined_transcript, combined_transcript


def generate_summary(transcript: str, custom_prompt: str = "", model: str = ""):
    """Generate summary with streaming."""
    if not transcript:
        yield "⚠️ No transcript available. Please transcribe first."
        return

    ollama_ok, ollama_msg = check_ollama_health()
    if not ollama_ok:
        yield f"❌ {ollama_msg}"
        return

    system_prompt = custom_prompt.strip() if custom_prompt.strip() else None
    use_model = model if model else None

    yield from summarize_streaming(transcript, system_prompt, use_model)


def create_ui(initial_audio: str | None = None, auto_transcribe: bool = False):
    """Create Gradio interface."""

    with gr.Blocks(title="Audio Notes") as demo:

        # State
        transcript_state = gr.State("")
        summary_state = gr.State("")

        gr.Markdown(
            """
        # 📝 Audio Notes

        Transform audio recordings into searchable notes with AI-powered summarization.
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # ===== LEFT PANEL SECTIONS =====

                # Recordings section
                recordings = create_recordings_section()
                recordings_accordion = recordings["recordings_accordion"]
                new_recordings_accordion = recordings["new_recordings_accordion"]
                transcribed_accordion = recordings["transcribed_accordion"]
                new_recordings_checkboxes = recordings["new_recordings_checkboxes"]
                transcribed_checkboxes = recordings["transcribed_checkboxes"]
                select_all_btn = recordings["select_all_btn"]
                delete_selected_btn = recordings["delete_selected_btn"]
                clean_transcribed_btn = recordings["clean_transcribed_btn"]
                no_recordings_msg = recordings["no_recordings_msg"]
                audio_player = recordings["audio_player"]
                select_all_state = gr.State(False)

                # Transcription section
                transcription = create_transcription_section()
                backend_radio = transcription["backend_radio"]
                batch_transcribe_btn = transcription["batch_transcribe_btn"]
                load_transcript_btn = transcription["load_transcript_btn"]
                batch_status = transcription["batch_status"]

                # Summarize section
                summarize = create_summarize_section()
                summarize_accordion = summarize["summarize_accordion"]
                llm_model_dropdown = summarize["llm_model_dropdown"]
                summary_prompt = summarize["summary_prompt"]
                summarize_btn = summarize["summarize_btn"]

                gr.Markdown("---")

                # Upload section
                upload = create_upload_section()
                file_input = upload["file_input"]
                upload_status = upload["upload_status"]

                # Status section
                status_section = create_status_section()
                refresh_btn = status_section["refresh_btn"]
                service_status = status_section["service_status"]

                reset_btn = gr.Button("🔄 Reset All", variant="secondary", size="lg")

            with gr.Column(scale=2):
                # ===== RIGHT PANEL TABS =====
                with gr.Tabs(selected=0) as result_tabs:
                    # Transcript tab (first)
                    transcript_components = create_transcript_tab()
                    transcript_components["transcript_tab"]
                    transcript_output = transcript_components["transcript_output"]

                    # Summary tab (second)
                    summary_components = create_summary_tab()
                    summary_tab = summary_components["summary_tab"]
                    summary_output = summary_components["summary_output"]

                    # Chat tab
                    chat_components = create_chat_tab()
                    chat_tab = chat_components["chat_tab"]
                    chat_title = chat_components["chat_title"]
                    chatbot = chat_components["chatbot"]
                    chat_input = chat_components["chat_input"]
                    chat_model_dropdown = chat_components["chat_model_dropdown"]
                    chat_btn = chat_components["chat_btn"]

        # ===== EVENT HANDLERS =====

        def refresh_recordings():
            """Refresh the recordings list."""
            recordings = list_recordings()
            new_recordings = [r for r in recordings if not r["has_transcript"]]
            transcribed_recordings = [r for r in recordings if r["has_transcript"]]

            new_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in new_recordings
            ]
            transcribed_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in transcribed_recordings
            ]

            has_any_recordings = len(new_choices) > 0 or len(transcribed_choices) > 0

            return (
                gr.update(choices=new_choices, value=[], interactive=True),
                gr.update(choices=transcribed_choices, value=[], interactive=True),
                "*Select recordings and click Transcribe.*",
                gr.update(visible=len(new_choices) > 0),
                gr.update(visible=len(transcribed_choices) > 0),
                gr.update(visible=not has_any_recordings),
            )

        def toggle_select_all(current_state):
            """Toggle select all state."""
            new_state = not current_state
            if new_state:
                recordings = list_recordings()
                new_paths = [r["path"] for r in recordings if not r["has_transcript"]]
                transcribed_paths = [r["path"] for r in recordings if r["has_transcript"]]
                return new_paths, transcribed_paths, True, "☐ Unselect All"
            else:
                return [], [], False, "☑ Select All"

        def update_audio_player(new_selected, transcribed_selected):
            """Update audio player when exactly 1 recording is selected."""
            all_selected = []
            if new_selected:
                all_selected.extend(new_selected)
            if transcribed_selected:
                all_selected.extend(transcribed_selected)

            # Only show audio when exactly 1 file is selected
            if len(all_selected) == 1:
                return gr.update(value=all_selected[0])
            else:
                return gr.update(value=None)

        def get_combined_selections(new_selected, transcribed_selected):
            """Combine selections from both groups."""
            all_selected = []
            if new_selected:
                all_selected.extend(new_selected)
            if transcribed_selected:
                all_selected.extend(transcribed_selected)
            return all_selected

        def on_batch_transcribe(new_selected, transcribed_selected, backend):
            """Handle batch transcription with streaming progress."""
            selected_files = get_combined_selections(new_selected, transcribed_selected)
            if not selected_files:
                yield (
                    "⚠️ No files selected. Check the boxes next to recordings to select them.",
                    "",
                    "",
                    gr.update(open=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),  # summary_prompt
                )
                return

            # Stream transcription progress
            # NOTE: gr.State doesn't work with gr.update() in generators,
            # so we must yield actual values for transcript_state at each step
            final_status, final_transcript, final_state = "", "", ""
            for status, transcript, state in batch_transcribe_streaming(
                selected_files, backend=backend
            ):
                final_status, final_transcript, final_state = status, transcript, state
                # During progress, update status and keep state in sync
                # Important: gr.State needs actual value, not gr.update()
                yield (
                    status,
                    gr.update(),  # transcript_output - don't update yet
                    state,  # transcript_state - must be actual value for gr.State!
                    gr.update(),  # recordings_accordion
                    gr.update(),  # batch_transcribe_btn
                    gr.update(),  # summarize_btn
                    gr.update(),  # reset_btn
                    gr.update(),
                    gr.update(),  # checkboxes
                    gr.update(),
                    gr.update(),  # accordions
                    gr.update(),  # summary_prompt
                )

            logger.info(
                f"on_batch_transcribe got transcript of length: {len(final_transcript)}, state length: {len(final_state)}"
            )

            # Refresh recordings list at the end
            recordings = list_recordings()
            new_recordings = [r for r in recordings if not r["has_transcript"]]
            transcribed_recordings = [r for r in recordings if r["has_transcript"]]
            new_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in new_recordings
            ]
            transcribed_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in transcribed_recordings
            ]

            # Calculate word count and get appropriate prompt
            word_count = len(final_transcript.split())
            auto_prompt = get_summary_prompt_for_length(word_count)

            yield (
                final_status,
                final_transcript,
                final_state,
                gr.update(open=False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(choices=new_choices, value=[], interactive=True),
                gr.update(choices=transcribed_choices, value=[], interactive=True),
                gr.update(visible=len(new_choices) > 0),
                gr.update(visible=len(transcribed_choices) > 0),
                gr.update(value=auto_prompt, interactive=True),  # summary_prompt
            )

        def on_load_transcript(transcribed_selected):
            """Load existing transcript from selected transcribed recordings."""
            if not transcribed_selected:
                return (
                    "⚠️ No transcribed recordings selected. Check transcribed recordings to load their transcripts.",
                    "",
                    "",
                    gr.update(open=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),  # summary_prompt
                )

            results = []
            all_transcripts = []

            for file_path in transcribed_selected:
                file_name = Path(file_path).name
                txt_path = Path(file_path).with_suffix(".txt")

                try:
                    if not txt_path.exists():
                        results.append(f"❌ {file_name}: Transcript file not found")
                        continue

                    transcript = txt_path.read_text(encoding="utf-8")
                    results.append(f"✅ {file_name} ➜ Loaded ({len(transcript)} chars)")
                    all_transcripts.append(f"## 🎙️ {file_name}\n\n{transcript}")
                    logger.info(f"Loaded transcript: {txt_path}")

                except Exception as e:
                    results.append(f"❌ {file_name}: {e}")
                    logger.error(f"Error loading transcript for {file_name}: {e}")

            combined_transcript = "\n\n---\n\n".join(all_transcripts) if all_transcripts else ""

            # Refresh recordings list
            recordings = list_recordings()
            new_recordings = [r for r in recordings if not r["has_transcript"]]
            transcribed_recordings = [r for r in recordings if r["has_transcript"]]
            new_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in new_recordings
            ]
            transcribed_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in transcribed_recordings
            ]

            # Calculate word count and get appropriate prompt
            word_count = len(combined_transcript.split())
            auto_prompt = get_summary_prompt_for_length(word_count)

            status = "\n\n".join(results)
            return (
                status,
                combined_transcript,
                combined_transcript,
                gr.update(open=False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(choices=new_choices, value=[], interactive=True),
                gr.update(choices=transcribed_choices, value=[], interactive=True),
                gr.update(visible=len(new_choices) > 0),
                gr.update(visible=len(transcribed_choices) > 0),
                gr.update(value=auto_prompt, interactive=True),  # summary_prompt
            )

        def on_summarize(transcript, prompt, model):
            """Handle summarization with streaming."""
            if not transcript:
                yield "⚠️ No transcript available. Please transcribe first."
                return

            try:
                first_chunk = True
                for chunk in generate_summary(transcript, prompt, model):
                    if first_chunk:
                        first_chunk = False
                    # Compact status indicator
                    yield f"⏳ Summarizing...\n\n{chunk}"
            except Exception as e:
                logger.error(f"Summarization error: {e}")
                yield f"❌ Error during summarization: {e!s}"

        def on_file_upload(file):
            """Handle file upload."""
            if file is None:
                return "", gr.update(), gr.update(), gr.update(), gr.update()

            try:
                import shutil

                src_path = Path(file.name if hasattr(file, "name") else file)
                ext = src_path.suffix.lower() or ".wav"

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"uploaded_{timestamp}{ext}"
                dest_path = RECORDINGS_DIR / new_name

                shutil.copy2(src_path, dest_path)
                logger.info(f"Uploaded file saved as: {dest_path}")

                # Refresh
                recordings = list_recordings()
                new_recordings = [r for r in recordings if not r["has_transcript"]]
                transcribed_recordings = [r for r in recordings if r["has_transcript"]]
                new_choices = [
                    (
                        f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                        r["path"],
                    )
                    for r in new_recordings
                ]
                transcribed_choices = [
                    (
                        f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                        r["path"],
                    )
                    for r in transcribed_recordings
                ]

                return (
                    f"✅ Saved as **{new_name}**",
                    gr.update(choices=new_choices, value=[]),
                    gr.update(choices=transcribed_choices, value=[]),
                    gr.update(visible=len(new_choices) > 0),
                    gr.update(visible=len(transcribed_choices) > 0),
                )
            except Exception as e:
                logger.error(f"Upload error: {e}")
                return (
                    f"❌ Upload failed: {e}",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

        def on_chat(message, history, transcript, summary, model):
            """Handle chat message with streaming response."""
            logger.info(
                f"on_chat called - transcript len: {len(transcript) if transcript else 0}, summary len: {len(summary) if summary else 0}"
            )
            if transcript:
                logger.info(f"on_chat - transcript preview: {transcript[:200]}...")
            else:
                logger.warning("on_chat - transcript is EMPTY!")
            if summary:
                logger.info(f"on_chat - summary preview: {summary[:200]}...")

            if not message.strip():
                yield history, "", gr.update(), gr.update(interactive=True)
                return

            if not transcript:
                error_history = [
                    *history,
                    {"role": "user", "content": message},
                    {
                        "role": "assistant",
                        "content": "⚠️ Please transcribe audio first. The transcript was not found in context.",
                    },
                ]
                yield error_history, "", gr.update(), gr.update(interactive=True)
                return

            # Add user message and placeholder for assistant
            new_history = [
                *history,
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]

            # Generate title on first message
            title_update = gr.update()
            if len(history) == 0:
                title_update = generate_chat_title(message, model)

            # Immediately disable button and show user message
            yield new_history, "", title_update, gr.update(interactive=False)

            # Stream the response using LangChain OpenAI-compatible API
            for response_chunk in langchain_chat_streaming(
                message, history, transcript, summary, model
            ):
                new_history[-1]["content"] = response_chunk
                yield new_history, "", title_update, gr.update(interactive=False)

            # Final yield with cleared input and re-enabled button
            yield new_history, "", title_update, gr.update(interactive=True)

        def on_refresh():
            parakeet_ok, parakeet_msg = check_parakeet_health()
            whisper_ok, whisper_msg = check_whisper_health()
            ollama_ok, ollama_msg = check_ollama_health()

            return f"""
**Status:**
- Parakeet: {"✅" if parakeet_ok else "❌"} {parakeet_msg}
- Whisper: {"✅" if whisper_ok else "❌"} {whisper_msg}
- Ollama: {"✅" if ollama_ok else "❌"} {ollama_msg}
            """

        def update_transcribe_button_state(new_selected, transcribed_selected):
            """Enable/disable transcribe and load transcript buttons based on selection."""
            has_new = len(new_selected) > 0 if new_selected else False
            has_transcribed = len(transcribed_selected) > 0 if transcribed_selected else False
            # Transcribe button: enabled if any selection
            # Load Transcript button: enabled only if transcribed recordings selected
            return gr.update(interactive=has_new or has_transcribed), gr.update(
                interactive=has_transcribed
            )

        def on_delete_selected(new_selected, transcribed_selected):
            """Delete selected recordings and refresh list."""
            all_selected = list(new_selected or []) + list(transcribed_selected or [])
            if not all_selected:
                return (
                    gr.update(),
                    gr.update(),
                    "No recordings selected.",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            deleted_count = delete_selected_recordings(all_selected)
            logger.info(f"Deleted {deleted_count} selected recordings")

            # Refresh recordings list
            recordings_list = list_recordings()
            new_recordings = [r for r in recordings_list if not r["has_transcript"]]
            transcribed_recordings = [r for r in recordings_list if r["has_transcript"]]
            new_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in new_recordings
            ]
            transcribed_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in transcribed_recordings
            ]
            has_any_recordings = len(new_choices) > 0 or len(transcribed_choices) > 0

            status_msg = f"🗑️ Deleted {deleted_count} recording(s)."

            return (
                gr.update(choices=new_choices, value=[]),
                gr.update(choices=transcribed_choices, value=[]),
                status_msg,
                gr.update(visible=len(new_choices) > 0),
                gr.update(visible=len(transcribed_choices) > 0),
                gr.update(visible=not has_any_recordings),
            )

        # ===== WIRE UP EVENTS =====

        delete_selected_btn.click(
            on_delete_selected,
            inputs=[new_recordings_checkboxes, transcribed_checkboxes],
            outputs=[
                new_recordings_checkboxes,
                transcribed_checkboxes,
                batch_status,
                new_recordings_accordion,
                transcribed_accordion,
                no_recordings_msg,
            ],
        )

        select_all_btn.click(
            toggle_select_all,
            inputs=[select_all_state],
            outputs=[
                new_recordings_checkboxes,
                transcribed_checkboxes,
                select_all_state,
                select_all_btn,
            ],
        )

        def on_clean_transcribed():
            """Delete all transcribed recordings and refresh list."""
            deleted_count = clean_transcribed_recordings()
            logger.info(f"Cleaned {deleted_count} transcribed recordings")

            # Refresh recordings list
            recordings_list = list_recordings()
            new_recordings = [r for r in recordings_list if not r["has_transcript"]]
            transcribed_recordings = [r for r in recordings_list if r["has_transcript"]]
            new_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in new_recordings
            ]
            transcribed_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in transcribed_recordings
            ]
            has_any_recordings = len(new_choices) > 0 or len(transcribed_choices) > 0

            status_msg = (
                f"🗑️ Deleted {deleted_count} transcribed recording(s)."
                if deleted_count > 0
                else "No transcribed recordings to delete."
            )

            return (
                gr.update(choices=new_choices, value=[]),
                gr.update(choices=transcribed_choices, value=[]),
                status_msg,
                gr.update(visible=len(new_choices) > 0),
                gr.update(visible=len(transcribed_choices) > 0),
                gr.update(visible=not has_any_recordings),
            )

        clean_transcribed_btn.click(
            on_clean_transcribed,
            outputs=[
                new_recordings_checkboxes,
                transcribed_checkboxes,
                batch_status,
                new_recordings_accordion,
                transcribed_accordion,
                no_recordings_msg,
            ],
        )

        new_recordings_checkboxes.change(
            update_transcribe_button_state,
            inputs=[new_recordings_checkboxes, transcribed_checkboxes],
            outputs=[batch_transcribe_btn, load_transcript_btn],
        )

        new_recordings_checkboxes.change(
            update_audio_player,
            inputs=[new_recordings_checkboxes, transcribed_checkboxes],
            outputs=[audio_player],
        )

        transcribed_checkboxes.change(
            update_transcribe_button_state,
            inputs=[new_recordings_checkboxes, transcribed_checkboxes],
            outputs=[batch_transcribe_btn, load_transcript_btn],
        )

        transcribed_checkboxes.change(
            update_audio_player,
            inputs=[new_recordings_checkboxes, transcribed_checkboxes],
            outputs=[audio_player],
        )

        # Lock UI before transcription, then run
        batch_transcribe_btn.click(
            lambda: (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value="⏳ Transcribing..."),
            ),
            outputs=[batch_transcribe_btn, summarize_btn, reset_btn, batch_status],
        ).then(
            on_batch_transcribe,
            inputs=[new_recordings_checkboxes, transcribed_checkboxes, backend_radio],
            outputs=[
                batch_status,
                transcript_output,
                transcript_state,
                recordings_accordion,
                batch_transcribe_btn,
                summarize_btn,
                reset_btn,
                new_recordings_checkboxes,
                transcribed_checkboxes,
                new_recordings_accordion,
                transcribed_accordion,
                summary_prompt,
            ],
        )

        # Load Transcript button
        load_transcript_btn.click(
            lambda: (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value="⏳ Loading transcripts..."),
            ),
            outputs=[
                batch_transcribe_btn,
                load_transcript_btn,
                summarize_btn,
                reset_btn,
                batch_status,
            ],
        ).then(
            on_load_transcript,
            inputs=[transcribed_checkboxes],
            outputs=[
                batch_status,
                transcript_output,
                transcript_state,
                recordings_accordion,
                batch_transcribe_btn,
                load_transcript_btn,
                summarize_btn,
                reset_btn,
                new_recordings_checkboxes,
                transcribed_checkboxes,
                new_recordings_accordion,
                transcribed_accordion,
                summary_prompt,
            ],
        )

        # Summarize button
        def start_summarize():
            return (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(selected=1),
                gr.update(interactive=True),
                "⏳ Connecting to LLM...",
            )

        def finish_summarize(summary_text):
            # Remove the "Summarizing..." prefix if present
            clean_summary = summary_text
            if summary_text and summary_text.startswith("⏳ Summarizing..."):
                clean_summary = summary_text.replace("⏳ Summarizing...\n\n", "")
            return (
                clean_summary,  # summary_output - show clean version
                clean_summary,  # summary_state - store clean version
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(open=False),
            )

        summarize_btn.click(
            start_summarize,
            outputs=[
                batch_transcribe_btn,
                summarize_btn,
                reset_btn,
                result_tabs,
                summary_tab,
                summary_output,
            ],
        ).then(
            on_summarize,
            inputs=[transcript_state, summary_prompt, llm_model_dropdown],
            outputs=[summary_output],
        ).then(
            finish_summarize,
            inputs=[summary_output],
            outputs=[
                summary_output,
                summary_state,
                batch_transcribe_btn,
                summarize_btn,
                reset_btn,
                chat_tab,
                summarize_accordion,
            ],
        )

        file_input.change(
            on_file_upload,
            inputs=[file_input],
            outputs=[
                upload_status,
                new_recordings_checkboxes,
                transcribed_checkboxes,
                new_recordings_accordion,
                transcribed_accordion,
            ],
        )

        # Reset button
        def on_reset():
            recordings = list_recordings()
            new_recordings = [r for r in recordings if not r["has_transcript"]]
            transcribed_recordings = [r for r in recordings if r["has_transcript"]]
            new_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in new_recordings
            ]
            transcribed_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in transcribed_recordings
            ]
            return (
                gr.update(choices=new_choices, value=[]),
                gr.update(choices=transcribed_choices, value=[]),
                False,
                gr.update(value="☑ Select All"),
                "*Select recordings and click Transcribe.*",
                "",
                "",
                "",
                "",
                [],
                "*Start a conversation to generate a title...*",
                gr.update(selected=0),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(open=True),
                gr.update(visible=len(new_choices) > 0),
                gr.update(visible=len(transcribed_choices) > 0),
                gr.update(open=True),
            )

        reset_btn.click(
            on_reset,
            outputs=[
                new_recordings_checkboxes,
                transcribed_checkboxes,
                select_all_state,
                select_all_btn,
                batch_status,
                transcript_output,
                summary_output,
                transcript_state,
                summary_state,
                chatbot,
                chat_title,
                result_tabs,
                summarize_btn,
                batch_transcribe_btn,
                load_transcript_btn,
                summary_tab,
                chat_tab,
                recordings_accordion,
                new_recordings_accordion,
                transcribed_accordion,
                summarize_accordion,
            ],
        )

        chat_btn.click(
            on_chat,
            inputs=[
                chat_input,
                chatbot,
                transcript_state,
                summary_state,
                chat_model_dropdown,
            ],
            outputs=[chatbot, chat_input, chat_title, chat_btn],
        )

        chat_input.submit(
            on_chat,
            inputs=[
                chat_input,
                chatbot,
                transcript_state,
                summary_state,
                chat_model_dropdown,
            ],
            outputs=[chatbot, chat_input, chat_title, chat_btn],
        )

        refresh_btn.click(on_refresh, None, service_status)

        # Initial load
        def on_load():
            recordings = list_recordings()
            new_recordings = [r for r in recordings if not r["has_transcript"]]
            transcribed_recordings = [r for r in recordings if r["has_transcript"]]
            new_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in new_recordings
            ]
            transcribed_choices = [
                (
                    f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)",
                    r["path"],
                )
                for r in transcribed_recordings
            ]
            has_any_recordings = len(new_choices) > 0 or len(transcribed_choices) > 0
            return (
                gr.update(choices=new_choices, value=[], interactive=True),
                gr.update(choices=transcribed_choices, value=[], interactive=True),
                gr.update(visible=len(new_choices) > 0),
                gr.update(visible=len(transcribed_choices) > 0),
                gr.update(visible=not has_any_recordings),
            )

        demo.load(
            on_load,
            outputs=[
                new_recordings_checkboxes,
                transcribed_checkboxes,
                new_recordings_accordion,
                transcribed_accordion,
                no_recordings_msg,
            ],
            js=CUSTOM_JS,
        )

    return demo
