"""
ASR Backend Configuration UI with ipywidgets.
Provides a fancy interface for configuring ASR backend variants.
"""

import ipywidgets as widgets
from IPython.display import display, HTML


def create_configuration_ui(variants, test_config, backend_urls, backend_options):
    """
    Create a fancy configuration UI for ASR backends.

    Args:
        variants: List to store variant configurations
        test_config: Dict to store test configuration
        backend_urls: Dict mapping backend names to URLs
        backend_options: Dict mapping backend names to their config options

    Returns:
        tuple: (ui_widget, helper_functions_dict)
    """

    # Custom CSS for styling
    display(
        HTML(
            """
    <style>
    .widget-label { font-weight: 600; color: #2c3e50; }
    .widget-button { font-weight: 600 !important; }
    </style>
    """
        )
    )

    # Header
    header = widgets.HTML(
        value="""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0; text-align: center;'>
                🎛️ ASR Backend Configuration
            </h2>
            <p style='color: #e0e0e0; margin: 5px 0 0 0; text-align: center;'>
                Configure and test multiple ASR backends
            </p>
        </div>
        """,
        layout=widgets.Layout(width="100%"),
    )

    # Footer
    footer = widgets.HTML(
        value="""
        <div style='background: #f1f1f1; padding: 10px; border-radius: 5px; margin-top: 20px; text-align: center;'>
            <small style='color: #7f8c8d;'>
                💡 Configure your ASR backends above and run benchmarks to compare performance!
        </small>
        </div>
        """,
        layout=widgets.Layout(width="100%"),
    )

    # Backend Selection
    backend_dropdown = widgets.Dropdown(
        options=list(backend_urls.keys()),
        value=list(backend_urls.keys())[0],
        description="🔧 Backend:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="350px"),
    )

    # Variant Name
    name_input = widgets.Text(
        placeholder="Leave empty for default name",
        description="📝 Custom Name:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="350px"),
    )

    # Backend-specific config container (dynamic)
    backend_config_container = widgets.VBox(
        [], layout=widgets.Layout(width="350px", margin="10px 0")
    )
    backend_config_widgets = {}

    # Audio Generation Settings
    word_count_slider = widgets.IntSlider(
        value=500,
        min=10,
        max=1000,
        step=10,
        description="📝 Words:",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="280px"),
        readout_format="d",
    )

    speech_speed_slider = widgets.FloatSlider(
        value=1.5,
        min=0.5,
        max=2.0,
        step=0.1,
        description="🗣️ Speed:",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="280px"),
        readout_format=".1f",
    )

    # Audio Settings (moved from backend-specific to global)
    chunk_duration_slider = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=5.0,
        step=0.1,
        description="⏱️ Chunk (s):",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="280px"),
        readout_format=".1f",
    )

    sample_rate_dropdown = widgets.Dropdown(
        options=[8000, 16000, 22050, 44100, 48000],
        value=16000,
        description="🎵 Sample Rate:",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="280px"),
    )

    # HTML widget for variants display (on the right)
    variants_html = widgets.HTML(
        value="<div style='color: #7f8c8d; font-style: italic;'>📋 No variants configured yet.<br/>Click 'Add Variant' to begin!</div>",
        layout=widgets.Layout(
            padding="5px",
            width="100%",
            max_height="none",
            overflow="hidden",
        ),
    )

    # Action Buttons (only Add and Clear)
    add_button = widgets.Button(
        description="Add Variant",
        button_style="success",
        tooltip="Add the selected backend configuration",
        icon="plus",
        layout=widgets.Layout(width="45%", height="30px"),
    )

    clear_button = widgets.Button(
        description="Clear All",
        button_style="danger",
        tooltip="Remove all configured variants",
        icon="trash",
        layout=widgets.Layout(width="45%", height="30px"),
    )

    # Functions
    def update_backend_config(change=None):
        """Update backend-specific config options when backend changes."""
        backend = backend_dropdown.value
        config_options = backend_options.get(backend, {})

        # Clear previous widgets
        backend_config_widgets.clear()

        config_widgets_list = [
            widgets.Label("⚙️ Model Settings:"),
        ]

        # Add backend-specific options
        if config_options:
            for key, default_value in config_options.items():
                if isinstance(default_value, bool):
                    widget = widgets.Checkbox(
                        value=default_value,
                        description=key,
                        style={"description_width": "150px"},
                        layout=widgets.Layout(width="340px"),
                    )
                elif isinstance(default_value, (int, float)):
                    widget = widgets.FloatText(
                        value=float(default_value),
                        description=f"{key}:",
                        style={"description_width": "150px"},
                        layout=widgets.Layout(width="340px"),
                    )
                else:  # String
                    widget = widgets.Text(
                        value=str(default_value),
                        description=f"{key}:",
                        style={"description_width": "150px"},
                        layout=widgets.Layout(width="340px"),
                    )

                backend_config_widgets[key] = widget
                config_widgets_list.append(widget)

        backend_config_container.children = config_widgets_list

    def update_display():
        """Update the variants display area using HTML."""
        if not variants:
            variants_html.value = "<div style='color: #7f8c8d; font-style: italic;'>📋 No variants configured yet.<br/>Click 'Add Variant' to begin!</div>"
        else:
            html_parts = []
            for i, v in enumerate(variants, 1):
                config_str = ", ".join([f"{k}={val}" for k, val in v["config"].items()])
                config_html = f" | Config: {config_str}" if config_str else ""

                variant_html = f"""
                <div style='margin-bottom: 6px; padding: 3px 6px; background: #f8f9fa; border-left: 3px solid #3498db; border-radius: 2px; line-height: 1.1;'>
                    <div style='font-weight: 600; color: #2c3e50; font-size: 0.88em; line-height: 1.2;'>{i}. {v['name']} <span style='font-weight: 400; color: #555; font-size: 1em;'>(Backend: {v['backend']} | URL: {v['url']})</span></div>
                    <div style='font-size: 0.9em; color: #666; margin-left: 12px; margin-top: 1px;'>{config_str if config_str else '<i>No custom config</i>'}</div>
                </div>
                """
                html_parts.append(variant_html)

            footer = f"<div style='border-top: 1px solid #3498db; padding-top: 2px; margin-top: 2px; font-weight: 600; color: #27ae60; font-size: 0.8em;'>💡 {len(variants)} variant(s) ready</div>"
            variants_html.value = "".join(html_parts) + footer

    def add_variant_clicked(b):
        """Handle Add Variant button click."""
        backend = backend_dropdown.value
        custom_name = name_input.value.strip()

        # Generate unique variant name if custom name not provided
        if not custom_name:
            # Count existing variants of this backend
            backend_count = sum(1 for v in variants if v["backend"] == backend)
            variant_name = f"{backend.title()}-{backend_count}"
        else:
            variant_name = custom_name

        # Get config from widgets (user-configured values)
        config = {}
        for key, widget in backend_config_widgets.items():
            value = widget.value
            # Round floats to avoid precision issues (1.0000000000000002 -> 1.0)
            if isinstance(value, float):
                value = round(value, 2)
            config[key] = value

        variant = {
            "backend": backend,
            "name": variant_name,
            "url": backend_urls[backend],
            "config": config,
        }

        variants.append(variant)

        # Save audio generation settings
        test_config["word_count"] = word_count_slider.value
        test_config["speech_speed"] = speech_speed_slider.value

        # Save audio settings (from left section)
        test_config["chunk_duration"] = round(chunk_duration_slider.value, 2)
        test_config["sample_rate"] = sample_rate_dropdown.value

        # Clear name input
        name_input.value = ""

        # Update display
        update_display()

    def clear_all_clicked(b):
        """Handle Clear All button click."""
        variants.clear()
        update_display()

    # Attach event handlers
    backend_dropdown.observe(update_backend_config, names="value")
    add_button.on_click(add_variant_clicked)
    clear_button.on_click(clear_all_clicked)

    # Initialize backend config display
    update_backend_config()

    # Layout - Left column: Audio Settings + Audio Generation + Benchmark Settings
    audio_settings_accordion = widgets.Accordion(
        children=[
            widgets.VBox([chunk_duration_slider, sample_rate_dropdown]),
        ],
        titles=("🎚️ Audio Settings",),
        layout=widgets.Layout(width="400px"),
    )
    audio_settings_accordion.selected_index = 0  # Open by default

    audio_generation_accordion = widgets.Accordion(
        children=[
            widgets.VBox([word_count_slider, speech_speed_slider]),
        ],
        titles=("🎙️ Audio Generation",),
        layout=widgets.Layout(width="400px"),
    )
    audio_generation_accordion.selected_index = 0  # Open by default

    left_section = widgets.VBox(
        [
            audio_settings_accordion,
            audio_generation_accordion,
        ],
        layout=widgets.Layout(width="400px", overflow="visible"),
    )

    # Layout - Middle column: Backend Configuration
    backend_config_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    backend_dropdown,
                    name_input,
                    backend_config_container,
                    widgets.HBox(
                        [add_button, clear_button],
                        layout=widgets.Layout(
                            justify_content="space-between", width="350px", margin="10px 0"
                        ),
                    ),
                ]
            ),
        ],
        titles=("⚙️ Backend Configuration",),
        layout=widgets.Layout(width="400px"),
    )
    backend_config_accordion.selected_index = 0  # Open by default

    # Layout - Right side: Variants display
    variants_accordion = widgets.Accordion(
        children=[
            widgets.VBox([variants_html], layout=widgets.Layout(overflow="visible")),
        ],
        titles=("📋 Configured Variants",),
        layout=widgets.Layout(width="600px"),
    )
    variants_accordion.selected_index = 0  # Open by default

    # Main layout - 3 columns grid with 50px gap
    main_content = widgets.GridBox(
        [left_section, backend_config_accordion, variants_accordion],
        layout=widgets.Layout(
            width="max-content",
            grid_template_columns="1fr 1fr 1.5fr",
            grid_gap="10px",
            margin="0 auto",
        ),
    )

    # Complete UI with padding container
    ui = widgets.VBox(
        [header, main_content, footer],
        layout=widgets.Layout(
            padding="0",
            width="100%",
            overflow="hidden",
            align_items="center",
        ),
    )

    # Helper functions for external access
    def get_backends():
        """Get configured backends as dict."""
        return {f"{v['backend']}_{i}": v for i, v in enumerate(variants)}

    def get_config():
        """Get test configuration."""
        return test_config.copy()

    def show_variants():
        """Show configured variants."""
        update_display()

    def clear_all():
        """Clear all variants."""
        variants.clear()
        update_display()

    helpers = {
        "get_backends": get_backends,
        "get_config": get_config,
        "show_variants": show_variants,
        "clear_all": clear_all,
        "update_display": update_display,
    }

    return ui, helpers
