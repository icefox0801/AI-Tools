"""Recordings section UI component."""

import gradio as gr


def create_recordings_section():
    """Create the recordings section with nested accordions.

    Returns:
        Tuple of UI components for recordings
    """
    with gr.Accordion("📁 Recordings", open=True) as recordings_accordion:
        select_all_state = gr.State(False)

        no_recordings_msg = gr.Markdown(
            "*No recordings found. Upload audio files to the recordings folder.*",
            visible=False,
        )

        with gr.Accordion("🎵 New Recordings", open=True, visible=True) as new_recordings_accordion:
            new_recordings_checkboxes = gr.CheckboxGroup(
                label="",
                choices=[],
                value=[],
                interactive=True,
                show_label=False,
                container=False,
                elem_id="new-recordings-list",
            )

        with gr.Accordion(
            "📄 Already Transcribed (re-transcribe)", open=False, visible=True
        ) as transcribed_accordion:
            with gr.Column(elem_id="transcribed-recordings-list"):
                transcribed_checkboxes = gr.CheckboxGroup(
                    label="",
                    choices=[],
                    value=[],
                    interactive=True,
                    show_label=False,
                    container=False,
                )

        with gr.Row():
            select_all_btn = gr.Button("☑ Select All", size="sm", scale=1)
            delete_selected_btn = gr.Button("🗑️ Delete Selected", size="sm", scale=1)
            clean_transcribed_btn = gr.Button("🗑️ Clean Transcribed", size="sm", scale=1)

        # Audio player for previewing recordings
        # Always visible but empty initially - avoids Gradio first-render issues
        audio_player = gr.Audio(
            label="▶️ Preview",
            type="filepath",
            interactive=False,
            show_label=True,
            visible=True,
        )

    return {
        "recordings_accordion": recordings_accordion,
        "select_all_state": select_all_state,
        "no_recordings_msg": no_recordings_msg,
        "new_recordings_accordion": new_recordings_accordion,
        "new_recordings_checkboxes": new_recordings_checkboxes,
        "transcribed_accordion": transcribed_accordion,
        "transcribed_checkboxes": transcribed_checkboxes,
        "select_all_btn": select_all_btn,
        "delete_selected_btn": delete_selected_btn,
        "clean_transcribed_btn": clean_transcribed_btn,
        "audio_player": audio_player,
    }
