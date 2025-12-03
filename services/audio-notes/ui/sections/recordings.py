"""Recordings section UI component."""

import gradio as gr


def create_recordings_section():
    """Create the recordings section with nested accordions.
    
    Returns:
        Tuple of UI components for recordings
    """
    with gr.Accordion("📁 Recordings", open=True) as recordings_accordion:
        select_all_state = gr.State(False)
        
        with gr.Accordion("🎵 New Recordings", open=True) as new_recordings_accordion:
            new_recordings_checkboxes = gr.CheckboxGroup(
                label="",
                choices=[],
                value=[],
                interactive=True,
                show_label=False,
                container=False
            )
        
        with gr.Accordion("📄 Already Transcribed (re-transcribe)", open=False) as transcribed_accordion:
            transcribed_checkboxes = gr.CheckboxGroup(
                label="",
                choices=[],
                value=[],
                interactive=True,
                show_label=False,
                container=False
            )
        
        with gr.Row():
            select_all_btn = gr.Button("☑️ Select All", size="sm", scale=1)
            refresh_trigger_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
    
    return {
        'recordings_accordion': recordings_accordion,
        'select_all_state': select_all_state,
        'new_recordings_accordion': new_recordings_accordion,
        'new_recordings_checkboxes': new_recordings_checkboxes,
        'transcribed_accordion': transcribed_accordion,
        'transcribed_checkboxes': transcribed_checkboxes,
        'select_all_btn': select_all_btn,
        'refresh_trigger_btn': refresh_trigger_btn,
    }
