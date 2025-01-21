# Audio_Generation_Playground.py
# Description: Gradio UI for text-to-speech audio generation
#
import logging
import os
import time
import gradio as gr
#
# Local Imports
from App_Function_Libraries.TTS.TTS_Providers import generate_audio
from App_Function_Libraries.DB.DB_Manager import add_media_to_database
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
#
########################################################################################################################
# Constants and Configuration

TTS_PROVIDERS = ["openai", "elevenlabs", "alltalk", "kokoro"]
DEFAULT_VOICES = {
    "openai": ["alloy", "echo", "fable", "nova", "onyx", "shimmer"],
    "elevenlabs": ["21m00", "AZnzlk1XvdvUeBnXmlld"],
    "alltalk": ["default"],
    "kokoro": ["af", "bf", "cm", "af_sarah"]
}


########################################################################################################################
# Core Functions

def generate_tts_audio(
        text,
        provider,
        voice,
        model,
        speed,
        output_format,
        save_to_db,
        custom_title
):
    """Wrapper function for TTS generation with logging and error handling"""
    log_counter("tts_generation_attempt", labels={"provider": provider})
    start_time = time.time()

    try:
        # Generate audio using selected provider
        audio_file = generate_audio(
            api_key=None,  # Will use config credentials
            text=text,
            provider=provider,
            voice=voice,
            model=model,
            response_format=output_format,
            streaming=False
        )

        # Validate output
        if not os.path.exists(audio_file):
            raise ValueError(f"Generated file not found at {audio_file}")

        # Save to database if requested
        db_result = ""
        if save_to_db:
            db_result = save_audio_to_db(audio_file, custom_title, provider)

        log_counter("tts_generation_success", labels={"provider": provider})
        log_histogram("tts_generation_duration", time.time() - start_time)

        return audio_file, db_result

    except Exception as e:
        log_counter("tts_generation_error", labels={"provider": provider, "error": str(e)})
        return None, f"Error generating audio: {str(e)}"


def save_audio_to_db(audio_path, title, provider):
    """Save generated audio to media database"""
    try:
        if not title.strip():
            title = f"{provider} TTS Generation - {time.strftime('%Y-%m-%d %H:%M:%S')}"

        info_dict = {
            "title": title,
            "uploader": provider,
            "webpage_url": "generated"
        }

        return add_media_to_database(
            url=audio_path,
            info_dict=info_dict,
            segments=[],
            summary="TTS Generated Audio",
            keywords=["tts", provider],
            media_type="audio"
        )

    except Exception as e:
        return f"Error saving to database: {str(e)}"


########################################################################
# UI Components

def create_audio_generation_tab():
    """Create the Gradio UI tab for audio generation"""
    with gr.Tab("Audio Generation Playground"):
        gr.Markdown("# Text-to-Speech Audio Generation")

        with gr.Row():
            with gr.Column(scale=2):
                # Input Controls
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=5,
                    placeholder="Enter text to convert to speech..."
                )

                provider_dropdown = gr.Dropdown(
                    label="TTS Provider",
                    choices=TTS_PROVIDERS,
                    value="openai"
                )

                voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=DEFAULT_VOICES["openai"],
                    value="alloy"
                )

                model_dropdown = gr.Dropdown(
                    label="Model",
                    visible=False,  # Only show when relevant
                    interactive=True
                )

                with gr.Row():
                    speed_slider = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )

                    format_dropdown = gr.Dropdown(
                        label="Output Format",
                        choices=["mp3", "wav"],
                        value="mp3"
                    )

                with gr.Accordion("Database Options", open=False):
                    save_checkbox = gr.Checkbox(
                        label="Save to Database",
                        value=False
                    )

                    title_input = gr.Textbox(
                        label="Custom Title",
                        visible=False,
                        placeholder="Enter custom title for database entry"
                    )

                generate_btn = gr.Button("Generate Audio", variant="primary")

            with gr.Column(scale=1):
                # Output Components
                audio_output = gr.Audio(label="Generated Audio")
                download_button = gr.File(label="Download File")
                db_status = gr.Textbox(label="Database Status", interactive=False)

        # Dynamic UI Updates
        provider_dropdown.change(
            fn=lambda p: gr.Dropdown(choices=DEFAULT_VOICES.get(p, [])),
            inputs=[provider_dropdown],
            outputs=[voice_dropdown]
        )

        save_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[save_checkbox],
            outputs=[title_input]
        )

        # Generation Handler
        generate_btn.click(
            fn=generate_tts_audio,
            inputs=[
                text_input,
                provider_dropdown,
                voice_dropdown,
                model_dropdown,
                speed_slider,
                format_dropdown,
                save_checkbox,
                title_input
            ],
            outputs=[audio_output, db_status]
        )

        # Auto-populate download button
        audio_output.change(
            fn=lambda x: x,
            inputs=[audio_output],
            outputs=[download_button]
        )

    return gr.update()

#
# End of TTS_Playground.py
######################################################################################################################
