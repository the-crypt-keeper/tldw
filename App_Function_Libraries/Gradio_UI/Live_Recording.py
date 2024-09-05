# Live_Recording.py
# Description: Gradio UI for live audio recording and transcription.
#
# Import necessary modules and functions
import os
# External Imports
import gradio as gr
# Local Imports
from App_Function_Libraries.Audio_Transcription_Lib import (record_audio, speech_to_text, save_audio_temp)
from App_Function_Libraries.DB.DB_Manager import add_media_to_database
#
#######################################################################################################################
#
# Functions:

whisper_models = ["small", "medium", "small.en", "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3",
                  "distil-large-v2", "distil-medium.en", "distil-small.en"]

def create_live_recording_tab():
    with gr.Tab("Live Recording and Transcription"):
        gr.Markdown("# Live Audio Recording and Transcription")
        with gr.Row():
            with gr.Column():
                duration = gr.Slider(minimum=1, maximum=8000, value=15, label="Recording Duration (seconds)")
                whisper_models_input = gr.Dropdown(choices=whisper_models, value="medium", label="Whisper Model")
                vad_filter = gr.Checkbox(label="Use VAD Filter")
                save_recording = gr.Checkbox(label="Save Recording")
                save_to_db = gr.Checkbox(label="Save Transcription to Database")
                custom_title = gr.Textbox(label="Custom Title (for database)", visible=False)
                record_button = gr.Button("Record and Transcribe")
            with gr.Column():
                output = gr.Textbox(label="Transcription", lines=10)
                audio_output = gr.Audio(label="Recorded Audio", visible=False)

        def record_and_transcribe(duration, whisper_model, vad_filter, save_recording):
            audio_data = record_audio(duration)
            temp_file = save_audio_temp(audio_data)
            segments = speech_to_text(temp_file, whisper_model=whisper_model, vad_filter=vad_filter)
            transcription = "\n".join([segment["Text"] for segment in segments])

            if save_recording:
                return transcription, temp_file
            else:
                os.remove(temp_file)
                return transcription, None

        def save_transcription_to_db(transcription, custom_title):
            if custom_title.strip() == "":
                custom_title = "Self-recorded Audio"

            add_media_to_database(
                url="self_recorded",
                info_dict={"title": custom_title, "uploader": "self-recorded"},
                segments=[{"Text": transcription}],
                summary="",
                keywords="self-recorded,audio",
                custom_prompt_input="",
                whisper_model="self-recorded"
            )
            return "Transcription saved to database successfully."

        def update_custom_title_visibility(save_to_db):
            return gr.update(visible=save_to_db)

        record_button.click(
            fn=record_and_transcribe,
            inputs=[duration, whisper_models_input, vad_filter, save_recording, save_to_db, custom_title],
            outputs=[output, audio_output]
        )

        save_to_db.change(
            fn=update_custom_title_visibility,
            inputs=[save_to_db],
            outputs=[custom_title]
        )

        gr.Button("Save to Database").click(
            fn=save_transcription_to_db,
            inputs=[output, custom_title],
            outputs=gr.Textbox(label="Database Save Status")
        )

#
# End of Functions
########################################################################################################################
