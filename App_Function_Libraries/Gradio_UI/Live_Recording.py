# Live_Recording.py
# Description: Gradio UI for live audio recording and transcription.
#
# Import necessary modules and functions
import logging
import os
import queue
import threading
import time

# External Imports
import gradio as gr
import numpy as np
import pyaudio

# Local Imports
from App_Function_Libraries.Audio.Audio_Transcription_Lib import (record_audio, speech_to_text, save_audio_temp,
                                                                  stop_recording, transcribe_audio)
from App_Function_Libraries.DB.DB_Manager import add_media_to_database
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name

#
#######################################################################################################################
#
# Functions:

whisper_models = ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
        "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-medium.en", "distil-small.en",
        "distil-large-v3", "deepdml/faster-whisper-large-v3-turbo-ct2", "nyrahealth/faster_CrisperWhisper"]


########################################################################
# 1. Recording set-up (indefinite until user stops)

def record_audio_indef(sample_rate=16000, chunk_size=1024):
    """
    Indefinite recording until a stop event is triggered.
    Returns (pyaudio_instance, stream, audio_queue, stop_event, audio_thread).
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )
    stop_event = threading.Event()
    audio_queue = queue.Queue()

    def audio_callback():
        while not stop_event.is_set():
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_queue.put(data)
            except Exception as e:
                logging.error(f"Error in audio callback: {str(e)}")
                break

    audio_thread = threading.Thread(target=audio_callback, daemon=True)
    audio_thread.start()
    return p, stream, audio_queue, stop_event, audio_thread


########################################################################
# 2. Background thread for partial transcription

class PartialTranscriptionThread(threading.Thread):
    """
    Periodically merges all audio so far, calls `transcribe_audio`,
    and updates a shared 'partial_text' variable.
    """

    def __init__(self, audio_queue, stop_event, transcription_method, partial_text_state, lock, sample_rate=16000,
                 update_interval=2.0, whisper_model="distil-large-v3", speaker_lang="en"):
        """
        :param audio_queue: queue of PCM chunks from the mic
        :param stop_event: threading.Event to stop this tclshread
        :param transcription_method: "faster-whisper" or "qwen2audio"
        :param partial_text_state: reference to a dictionary or list storing partial text
        :param lock: threading.Lock to coordinate read/write of partial_text
        :param sample_rate: 16k default
        :param update_interval: how often (seconds) we do a partial transcription
        """
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.transcription_method = transcription_method
        self.partial_text_state = partial_text_state
        self.lock = lock
        self.sample_rate = sample_rate
        self.update_interval = update_interval
        self.audio_buffer = []
        self.last_transcription_time = time.time()
        self.whisper_model = whisper_model
        self.speaker_lang = speaker_lang

    def run(self):
        while not self.stop_event.is_set():
            current_time = time.time()

            # Only process if enough time has passed
            if current_time - self.last_transcription_time < self.update_interval:
                time.sleep(0.1)  # Short sleep to prevent CPU spinning
                continue

            # Collect all available audio data
            while not self.audio_queue.empty():
                try:
                    chunk = self.audio_queue.get_nowait()
                    self.audio_buffer.append(chunk)
                except queue.Empty:
                    break

            if not self.audio_buffer:
                continue

            # Convert accumulated audio to numpy array
            try:
                combined_data = b''.join(self.audio_buffer)
                audio_np = np.frombuffer(combined_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Only transcribe if we have enough audio data
                if len(audio_np) > self.sample_rate:  # At least 1 second of audio
                    partial_result = transcribe_audio(
                        audio_np,
                        self.transcription_method,
                        sample_rate=self.sample_rate,
                        whisper_model=self.whisper_model,
                        speaker_lang=self.speaker_lang
                    )

                    with self.lock:
                        self.partial_text_state["text"] = partial_result

                    # Keep last 5 seconds of audio for context
                    max_buffer_size = 5 * self.sample_rate * 2  # 5 seconds * sample_rate * 2 bytes per sample
                    if len(combined_data) > max_buffer_size:
                        excess_bytes = len(combined_data) - max_buffer_size
                        self.audio_buffer = [combined_data[excess_bytes:]]

                    self.last_transcription_time = current_time

            except Exception as e:
                logging.error(f"Error in partial transcription: {str(e)}")
                time.sleep(0.1)  # Prevent rapid error loops



########################################################################
# 3. Single toggle function: start or stop + final transcription

def toggle_recording(
        is_recording,
        recording_state,
        transcription_method,
        live_update,
        save_recording,
        partial_text_state,
        whisper_model,
        speaker_lang="en"
):
    """
    Single function to handle both 'start' and 'stop' logic.

    If not recording yet, start indefinite recording, and if live_update is True,
    also spawn partial transcription thread.

    If already recording, stop + final transcription, kill partial thread if any,
    and return final text.
    """
    if not is_recording:
        try:
            # ================== START RECORDING ======================
            log_counter("live_recording_start_attempt")
            p, stream, audio_queue, stop_event, audio_thread = record_audio_indef()
            log_counter("live_recording_start_success")

            # Add a small delay after starting recording to avoid cutting off the start
            time.sleep(0.5)

            # If user wants real-time partial updates, start the partial thread
            partial_thread = None
            lock = threading.Lock()  # to protect partial_text_state
            if live_update:
                partial_thread = PartialTranscriptionThread(
                    audio_queue=audio_queue,
                    stop_event=stop_event,
                    transcription_method=transcription_method,
                    partial_text_state=partial_text_state,
                    lock=lock,
                    sample_rate=16000,
                    update_interval=2.0,
                    whisper_model=whisper_model,
                    speaker_lang=speaker_lang
                )
                partial_thread.start()

            # recording_state: store everything we might need
            new_state = {
                "p": p,
                "stream": stream,
                "audio_queue": audio_queue,
                "stop_event": stop_event,
                "audio_thread": audio_thread,
                "partial_thread": partial_thread,
                "lock": lock,
                "start_time": time.time(),
                "all_audio_chunks": [],
                "whisper_model": whisper_model
            }
            return new_state, True, "Stop Recording", "", None

        except Exception as e:
            logging.error(f"Error starting recording: {str(e)}")
            return None, False, "Start Recording", f"Error starting recording: {str(e)}", None

    # ================== STOP RECORDING ======================
    if not recording_state:
        # No active recording to stop
        return None, False, "Start Recording", "No active recording.", None

    try:
        p = recording_state["p"]
        stream = recording_state["stream"]
        audio_queue = recording_state["audio_queue"]
        stop_event = recording_state["stop_event"]
        audio_thread = recording_state["audio_thread"]
        partial_thread = recording_state["partial_thread"]
        whisper_model = recording_state.get("whisper_model", "distil-large-v3")
        lock = recording_state["lock"]

        # 1) Stop the partial transcription thread if it exists
        if partial_thread is not None:
            stop_event.set()  # tell thread to stop
            partial_thread.join(timeout=5)

        # 2) Stop final recording and get all audio
        raw_audio = stop_recording(p, stream, audio_queue, stop_event, audio_thread)

        # Process any remaining audio in queue
        while not audio_queue.empty():
            try:
                raw_audio += audio_queue.get_nowait()
            except queue.Empty:
                break

        # Check for valid audio
        if not raw_audio:
            return None, False, "Start Recording", "No audio recorded", None

        # FIXME - Make sure this works....
        # Convert raw_audio to a NumPy array if it's in byte format
        if isinstance(raw_audio, bytes):
            raw_audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            raw_audio_np = raw_audio

        # 3) Final transcription of entire audio
        temp_file = save_audio_temp(raw_audio_np)
        final_audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            final_result = transcribe_audio(
                final_audio_np,
                transcription_method,
                sample_rate=16000,
                whisper_model=whisper_model,
                speaker_lang=speaker_lang
            )

            # Update both partial and final text
            with lock:
                partial_text_state["text"] = final_result

        except Exception as e:
            error_message = f"[Error in final transcription: {str(e)}]"
            with lock:
                partial_text_state["text"] = error_message
            final_result = error_message

        # 4) Optionally save the WAV
        if not save_recording and os.path.exists(temp_file):
            os.remove(temp_file)
            audio_file_path = None
        else:
            audio_file_path = temp_file

        return None, False, "Start Recording", final_result, audio_file_path

    except Exception as e:
        logging.error(f"Error stopping recording: {str(e)}")
        error_message = f"Error stopping recording: {str(e)}"
        with recording_state["lock"]:
            partial_text_state["text"] = error_message
        return None, False, "Start Recording", error_message, None


########################################################################
# 4. Save Transcription to DB

def save_transcription_to_db(transcription, custom_title):
    """
    Same logic as your existing approach
    """
    log_counter("save_transcription_to_db_attempt")
    start_time = time.time()
    if custom_title.strip() == "":
        custom_title = "Self-recorded Audio"

    from App_Function_Libraries.DB.DB_Manager import add_media_to_database
    try:
        url = "self_recorded"
        info_dict = {
            "title": custom_title,
            "uploader": "self-recorded",
            "webpage_url": url
        }
        segments = [{"Text": transcription}]
        summary = ""
        keywords = ["self-recorded", "audio"]
        custom_prompt_input = ""
        whisper_model = "self-recorded"
        media_type = "audio"

        result = add_media_to_database(
            url=url,
            info_dict=info_dict,
            segments=segments,
            summary=summary,
            keywords=keywords,
            custom_prompt_input=custom_prompt_input,
            whisper_model=whisper_model,
            media_type=media_type
        )
        end_time = time.time() - start_time
        log_histogram("save_transcription_to_db_duration", end_time)
        log_counter("save_transcription_to_db_success")
        return f"Transcription saved to database successfully. {result}"
    except Exception as e:
        logging.error(f"Error saving transcription to database: {str(e)}")
        log_counter("save_transcription_to_db_error", labels={"error": str(e)})
        return f"Error saving transcription to database: {str(e)}"


def update_custom_title_visibility(save_to_db):
    return gr.update(visible=save_to_db)


def get_partial_transcript(partial_text_state):
    """
    Polled by a Gradio Timer or UI refresh.
    Return the partial_text_state["text"] (safely).
    """
    return partial_text_state.get("text", "")




########################################################################
# 5. The main Gradio tab
# Provide a couple transcription methods
TRANSCRIPTION_METHODS = ["faster-whisper", "qwen2audio", "parakeet"]

def create_live_recording_tab():
    try:
        default_value = None
        if default_api_endpoint and default_api_endpoint in global_api_endpoints:
            default_value = format_api_name(default_api_endpoint)
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None

    with gr.Tab("Live Recording and Transcription", visible=True):
        gr.Markdown("# Live Audio Recording with Real-time Partial Updates")

        with gr.Row():
            with gr.Column():
                transcription_method = gr.Dropdown(
                    label="Transcription Method",
                    choices=TRANSCRIPTION_METHODS,
                    value="faster-whisper"
                )
                # Add Whisper model selection
                whisper_model = gr.Dropdown(
                    label="Whisper Model",
                    choices=whisper_models,
                    value="distil-large-v3",
                    visible=True
                )
                live_update = gr.Checkbox(label="Enable Real-time Partial Transcription", value=False)
                save_recording = gr.Checkbox(label="Save WAV File After Stopping", value=False)
                save_to_db_checkbox = gr.Checkbox(label="Save Transcription to Database (after stopping)", value=False)
                custom_title = gr.Textbox(label="Custom Title (for DB)", visible=False)

                # Single toggle button
                record_btn = gr.Button("Start Recording")
                # Add recording status indicator
                recording_status = gr.Markdown("Not Recording", visible=True)

            with gr.Column():
                # Show partial transcription if "live_update" is on
                partial_txt = gr.Textbox(
                    label="Partial Transcription (refreshes every 2s if live_update enabled)",
                    lines=5,
                    interactive=False,
                    show_copy_button=True
                )
                final_txt = gr.Textbox(
                    label="Final Transcription (once stopped)",
                    lines=5,
                    interactive = False,
                    show_copy_button = True
                )
                audio_output = gr.Audio(label="Recorded Audio", visible=False)
                db_save_status = gr.Textbox(label="Database Save Status", lines=2)

        # States
        recording_state = gr.State(value=None)         # stores mic objects, threads, etc.
        is_recording = gr.State(value=False)           # bool
        partial_text_state = gr.State({"text": ""})    # store partial transcription

        # Modified toggle_recording to handle recording status and model selection
        def toggle_recording_wrapper(*args):
            result = toggle_recording(*args)
            # Update recording status based on is_recording state
            is_rec = result[1] if len(result) > 1 else False
            status_html = """<div style="padding: 10px; background-color: #ff4444; color: white; text-align: center; font-weight: bold; border-radius: 5px;">🔴 Recording in Progress</div>""" if is_rec else """<div style="padding: 10px; background-color: #444444; color: white; text-align: center; font-weight: bold; border-radius: 5px;">⚪ Not Recording</div>"""
            return result + (status_html,)

        # Toggle start/stop with updated inputs and outputs
        record_btn.click(
            fn=toggle_recording_wrapper,
            inputs=[
                is_recording,
                recording_state,
                transcription_method,
                live_update,
                save_recording,
                partial_text_state,
                whisper_model  # Add whisper_model to inputs
            ],
            outputs=[
                recording_state,
                is_recording,
                record_btn,
                final_txt,
                audio_output,
                recording_status
            ]
        )

        # Show/hide custom_title
        save_to_db_checkbox.change(
            fn=update_custom_title_visibility,
            inputs=[save_to_db_checkbox],
            outputs=[custom_title]
        )

        # Save to DB button
        gr.Button("Save to Database").click(
            fn=save_transcription_to_db,
            inputs=[final_txt, custom_title],
            outputs=[db_save_status]
        )

        # A small trick: we poll partial_text_state every 2s if live_update is True.
        # We'll do it with a Timer that runs regardless, but partial_txt will only
        # show something if the user checked live_update. The partial_text gets updated
        # by the background PartialTranscriptionThread.

        # Define your polling function
        def poll_partial_text(live, partial_text_state):
            """Return partial text only if user set live_update = True."""
            if live:
                # Access the text safely through the state object
                return partial_text_state["text"] if partial_text_state else "(No transcription yet)"
            else:
                return "(Live update disabled)"

        # Create the Timer with a 2-second interval
        my_timer = gr.Timer(value=2.0)

        # Attach the event listener with .tick()
        my_timer.tick(
            fn=poll_partial_text,
            inputs=[live_update, partial_text_state],
            outputs=partial_txt,
        )
#
# End of Functions
########################################################################################################################
