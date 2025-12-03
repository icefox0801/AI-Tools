#!/usr/bin/env python3
"""
Main Gradio UI assembly for Audio Notes.

This module brings together all UI sections and tabs,
and wires up the event handlers.
"""

import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import gradio as gr

from config import RECORDINGS_DIR, logger
from services import (
    check_whisper_health, check_parakeet_health, check_ollama_health,
    list_recordings, get_audio_duration,
    transcribe_audio, unload_asr_model,
    summarize_streaming, chat_with_context, generate_chat_title
)
from ui.styles import CUSTOM_CSS, CUSTOM_JS
from ui.sections import (
    create_recordings_section, 
    create_transcription_section,
    create_summarize_section,
    create_upload_section,
    create_status_section
)
from ui.tabs import (
    create_summary_tab,
    create_transcript_tab,
    create_chat_tab
)


def batch_transcribe_streaming(selected_files: List[str], backend: str = "parakeet"):
    """Batch transcribe multiple audio files with streaming progress.
    
    Yields:
        tuple: (status_text, transcript, state) - intermediate and final results
    """
    if not selected_files:
        yield "⚠️ No files selected for batch transcription", "", ""
        return
    
    backend_lower = backend.lower() if backend else "parakeet"
    
    if backend_lower == "whisper":
        service_ok, service_msg = check_whisper_health()
    else:
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
            
            txt_path = Path(file_path).with_suffix('.txt')
            txt_path.write_text(transcript, encoding='utf-8')
            
            results.append(f"✅ {file_name} ➜ {txt_path.name} ({duration:.1f}s, {len(transcript)} chars)")
            all_transcripts.append(f"### {file_name}\n\n{transcript}")
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
    
    for chunk in summarize_streaming(transcript, system_prompt, use_model):
        yield chunk


def create_ui(initial_audio: Optional[str] = None, auto_transcribe: bool = False):
    """Create Gradio interface."""
    
    with gr.Blocks(title="Audio Notes") as demo:
        # Inject CSS
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        
        # State
        transcript_state = gr.State("")
        summary_state = gr.State("")
        
        gr.Markdown("""
        # 📝 Audio Notes
        
        Transform audio recordings into searchable notes with AI-powered summarization.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # ===== LEFT PANEL SECTIONS =====
                
                # Recordings section
                recordings = create_recordings_section()
                recordings_accordion = recordings['recordings_accordion']
                new_recordings_accordion = recordings['new_recordings_accordion']
                transcribed_accordion = recordings['transcribed_accordion']
                new_recordings_checkboxes = recordings['new_recordings_checkboxes']
                transcribed_checkboxes = recordings['transcribed_checkboxes']
                select_all_btn = recordings['select_all_btn']
                refresh_trigger_btn = recordings['refresh_trigger_btn']
                select_all_state = gr.State(False)
                
                # Transcription section
                transcription = create_transcription_section()
                backend_radio = transcription['backend_radio']
                batch_transcribe_btn = transcription['batch_transcribe_btn']
                batch_status = transcription['batch_status']
                
                # Summarize section
                summarize = create_summarize_section()
                summarize_accordion = summarize['summarize_accordion']
                llm_model_dropdown = summarize['llm_model_dropdown']
                summary_prompt = summarize['summary_prompt']
                summarize_btn = summarize['summarize_btn']
                
                gr.Markdown("---")
                
                # Upload section
                upload = create_upload_section()
                file_input = upload['file_input']
                upload_status = upload['upload_status']
                
                # Status section
                status_section = create_status_section()
                refresh_btn = status_section['refresh_btn']
                service_status = status_section['service_status']
                
                reset_btn = gr.Button("🔄 Reset All", variant="secondary", size="lg")
            
            with gr.Column(scale=2):
                # ===== RIGHT PANEL TABS =====
                with gr.Tabs(selected=0) as result_tabs:
                    # Transcript tab (first)
                    transcript_components = create_transcript_tab()
                    transcript_tab = transcript_components['transcript_tab']
                    transcript_output = transcript_components['transcript_output']
                    
                    # Summary tab (second)
                    summary_components = create_summary_tab()
                    summary_tab = summary_components['summary_tab']
                    summary_output = summary_components['summary_output']
                    
                    # Chat tab
                    chat_components = create_chat_tab()
                    chat_tab = chat_components['chat_tab']
                    chat_title = chat_components['chat_title']
                    chatbot = chat_components['chatbot']
                    chat_input = chat_components['chat_input']
                    chat_model_dropdown = chat_components['chat_model_dropdown']
                    chat_btn = chat_components['chat_btn']
        
        # ===== EVENT HANDLERS =====
        
        def refresh_recordings():
            """Refresh the recordings list."""
            recordings = list_recordings()
            new_recordings = [r for r in recordings if not r['has_transcript']]
            transcribed_recordings = [r for r in recordings if r['has_transcript']]
            
            new_choices = [(f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in new_recordings]
            transcribed_choices = [(f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in transcribed_recordings]
            
            return (
                gr.update(choices=new_choices, value=[], interactive=True),
                gr.update(choices=transcribed_choices, value=[], interactive=True),
                "*Select recordings and click Transcribe.*",
                gr.update(visible=len(new_choices) > 0),
                gr.update(visible=len(transcribed_choices) > 0),
            )
        
        def toggle_select_all(current_state):
            """Toggle select all state."""
            new_state = not current_state
            if new_state:
                recordings = list_recordings()
                new_paths = [r['path'] for r in recordings if not r['has_transcript']]
                transcribed_paths = [r['path'] for r in recordings if r['has_transcript']]
                return new_paths, transcribed_paths
            else:
                return [], []
        
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
                    "", "",
                    gr.update(open=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(), gr.update(),
                    gr.update(), gr.update(),
                )
                return
            
            # Stream transcription progress
            final_status, final_transcript, final_state = "", "", ""
            for status, transcript, state in batch_transcribe_streaming(selected_files, backend=backend):
                final_status, final_transcript, final_state = status, transcript, state
                # During progress, just update status, keep other outputs unchanged
                yield (
                    status, 
                    gr.update(),  # transcript_output - don't update yet
                    gr.update(),  # transcript_state - don't update yet
                    gr.update(),  # recordings_accordion
                    gr.update(),  # batch_transcribe_btn
                    gr.update(),  # summarize_btn
                    gr.update(),  # reset_btn
                    gr.update(), gr.update(),  # checkboxes
                    gr.update(), gr.update(),  # accordions
                )
            
            logger.info(f"on_batch_transcribe got transcript of length: {len(final_transcript)}, state length: {len(final_state)}")
            
            # Refresh recordings list at the end
            recordings = list_recordings()
            new_recordings = [r for r in recordings if not r['has_transcript']]
            transcribed_recordings = [r for r in recordings if r['has_transcript']]
            new_choices = [(f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in new_recordings]
            transcribed_choices = [(f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in transcribed_recordings]
            
            yield (
                final_status, final_transcript, final_state,
                gr.update(open=False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(choices=new_choices, value=[], interactive=True),
                gr.update(choices=transcribed_choices, value=[], interactive=True),
                gr.update(visible=len(new_choices) > 0),
                gr.update(visible=len(transcribed_choices) > 0),
            )
        
        def on_summarize(transcript, prompt, model):
            """Handle summarization with streaming."""
            if not transcript:
                yield "⚠️ No transcript available. Please transcribe first."
                return
            
            try:
                for chunk in generate_summary(transcript, prompt, model):
                    yield chunk
            except Exception as e:
                logger.error(f"Summarization error: {e}")
                yield f"❌ Error during summarization: {str(e)}"
        
        def on_file_upload(file):
            """Handle file upload."""
            if file is None:
                return "", gr.update(), gr.update(), gr.update(), gr.update()
            
            try:
                import shutil
                src_path = Path(file.name if hasattr(file, 'name') else file)
                ext = src_path.suffix.lower() or '.wav'
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"uploaded_{timestamp}{ext}"
                dest_path = RECORDINGS_DIR / new_name
                
                shutil.copy2(src_path, dest_path)
                logger.info(f"Uploaded file saved as: {dest_path}")
                
                # Refresh
                recordings = list_recordings()
                new_recordings = [r for r in recordings if not r['has_transcript']]
                transcribed_recordings = [r for r in recordings if r['has_transcript']]
                new_choices = [(f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in new_recordings]
                transcribed_choices = [(f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in transcribed_recordings]
                
                return (
                    f"✅ Saved as **{new_name}**",
                    gr.update(choices=new_choices, value=[]),
                    gr.update(choices=transcribed_choices, value=[]),
                    gr.update(visible=len(new_choices) > 0),
                    gr.update(visible=len(transcribed_choices) > 0),
                )
            except Exception as e:
                logger.error(f"Upload error: {e}")
                return f"❌ Upload failed: {e}", gr.update(), gr.update(), gr.update(), gr.update()
        
        def on_chat(message, history, transcript, summary, model):
            """Handle chat message."""
            logger.info(f"on_chat called - transcript len: {len(transcript) if transcript else 0}, summary len: {len(summary) if summary else 0}")
            logger.info(f"on_chat - transcript preview: {transcript[:200] if transcript else 'EMPTY'}...")
            logger.info(f"on_chat - summary preview: {summary[:200] if summary else 'EMPTY'}...")
            if not message.strip():
                return history, "", gr.update()
            if not transcript:
                error_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "⚠️ Please transcribe audio first"}
                ]
                return error_history, "", gr.update()
            
            response = chat_with_context(message, history, transcript, summary, model)
            new_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            
            if len(history) == 0:
                title = generate_chat_title(message, model)
                return new_history, "", title
            
            return new_history, "", gr.update()
        
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
            """Enable/disable transcribe button based on selection."""
            has_new = len(new_selected) > 0 if new_selected else False
            has_transcribed = len(transcribed_selected) > 0 if transcribed_selected else False
            return gr.update(interactive=has_new or has_transcribed)
        
        # ===== WIRE UP EVENTS =====
        
        refresh_trigger_btn.click(
            refresh_recordings,
            outputs=[new_recordings_checkboxes, transcribed_checkboxes, batch_status, new_recordings_accordion, transcribed_accordion]
        )
        
        select_all_btn.click(
            toggle_select_all,
            inputs=[select_all_state],
            outputs=[new_recordings_checkboxes, transcribed_checkboxes]
        )
        
        new_recordings_checkboxes.change(
            update_transcribe_button_state,
            inputs=[new_recordings_checkboxes, transcribed_checkboxes],
            outputs=[batch_transcribe_btn]
        )
        
        transcribed_checkboxes.change(
            update_transcribe_button_state,
            inputs=[new_recordings_checkboxes, transcribed_checkboxes],
            outputs=[batch_transcribe_btn]
        )
        
        # Lock UI before transcription, then run
        batch_transcribe_btn.click(
            lambda: (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value="⏳ Transcribing..."),
            ),
            outputs=[batch_transcribe_btn, summarize_btn, reset_btn, batch_status]
        ).then(
            on_batch_transcribe,
            inputs=[new_recordings_checkboxes, transcribed_checkboxes, backend_radio],
            outputs=[batch_status, transcript_output, transcript_state, recordings_accordion, 
                     batch_transcribe_btn, summarize_btn, reset_btn, 
                     new_recordings_checkboxes, transcribed_checkboxes, 
                     new_recordings_accordion, transcribed_accordion]
        )
        
        # Summarize button
        def start_summarize():
            return (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(selected=1),
                gr.update(interactive=True),
                "⏳ *Generating summary...*",
            )
        
        def finish_summarize(summary_text):
            return (
                summary_text,
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(open=False),
            )
        
        summarize_btn.click(
            start_summarize,
            outputs=[batch_transcribe_btn, summarize_btn, reset_btn, result_tabs, summary_tab, summary_output]
        ).then(
            on_summarize,
            inputs=[transcript_state, summary_prompt, llm_model_dropdown],
            outputs=[summary_output]
        ).then(
            finish_summarize,
            inputs=[summary_output],
            outputs=[summary_state, batch_transcribe_btn, summarize_btn, reset_btn, chat_tab, summarize_accordion]
        )
        
        file_input.change(
            on_file_upload,
            inputs=[file_input],
            outputs=[upload_status, new_recordings_checkboxes, transcribed_checkboxes, new_recordings_accordion, transcribed_accordion]
        )
        
        # Reset button
        def on_reset():
            recordings = list_recordings()
            new_recordings = [r for r in recordings if not r['has_transcript']]
            transcribed_recordings = [r for r in recordings if r['has_transcript']]
            new_choices = [(f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in new_recordings]
            transcribed_choices = [(f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in transcribed_recordings]
            return (
                gr.update(choices=new_choices, value=[]),
                gr.update(choices=transcribed_choices, value=[]),
                False,
                gr.update(value="☑️ Select All"),
                "*Select recordings and click Transcribe.*",
                "", "", "", "",
                [],
                "*Start a conversation to generate a title...*",
                gr.update(selected=0),
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
            outputs=[new_recordings_checkboxes, transcribed_checkboxes, select_all_state, select_all_btn, 
                     batch_status, transcript_output, summary_output, transcript_state, summary_state, 
                     chatbot, chat_title, result_tabs, summarize_btn, batch_transcribe_btn, 
                     summary_tab, chat_tab, recordings_accordion, new_recordings_accordion, 
                     transcribed_accordion, summarize_accordion]
        )
        
        chat_btn.click(
            on_chat,
            inputs=[chat_input, chatbot, transcript_state, summary_state, chat_model_dropdown],
            outputs=[chatbot, chat_input, chat_title]
        )
        
        chat_input.submit(
            on_chat,
            inputs=[chat_input, chatbot, transcript_state, summary_state, chat_model_dropdown],
            outputs=[chatbot, chat_input, chat_title]
        )
        
        refresh_btn.click(on_refresh, None, service_status)
        
        # Initial load
        def on_load():
            recordings = list_recordings()
            new_recordings = [r for r in recordings if not r['has_transcript']]
            transcribed_recordings = [r for r in recordings if r['has_transcript']]
            new_choices = [(f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in new_recordings]
            transcribed_choices = [(f"🔊 {r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in transcribed_recordings]
            return (
                gr.update(choices=new_choices, value=[], interactive=True),
                gr.update(choices=transcribed_choices, value=[], interactive=True),
                gr.update(visible=len(new_choices) > 0),
                gr.update(visible=len(transcribed_choices) > 0),
            )
        
        demo.load(
            on_load,
            outputs=[new_recordings_checkboxes, transcribed_checkboxes, new_recordings_accordion, transcribed_accordion],
            js=CUSTOM_JS
        )
    
    return demo
