# Audio_Files.py
#########################################
# Audio Processing Library
# This library is used to download or load audio files from a local directory.
#
####
#
# Functions:
#
# download_audio_file(url, save_path)
# process_audio(
# process_audio_file(audio_url, audio_file, whisper_model="small.en", api_name=None, api_key=None)
#
#
#########################################
# Imports
import json
import logging
import requests
import os

import yt_dlp

from App_Function_Libraries.Audio_Transcription_Lib import speech_to_text
#
# Local Imports
from App_Function_Libraries.SQLite_DB import add_media_to_database, add_media_with_keywords
from App_Function_Libraries.Utils import extract_text_from_segments, download_file, create_download_directory
from App_Function_Libraries.Summarization_General_Lib import save_transcription_and_summary, perform_transcription, \
    perform_summarization
#
#######################################################################################################################
# Function Definitions
#


def download_audio_file(url, save_path):
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))
    if file_size > 500 * 1024 * 1024:  # 500 MB limit
        raise ValueError("File size exceeds the 500MB limit.")
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path


def save_segments_to_json(segments, file_name="transcription_segments.json"):
    """
    Save transcription segments to a JSON file.

    Parameters:
    segments (list): List of transcription segments
    file_name (str): Name of the JSON file to save (default: "transcription_segments.json")

    Returns:
    str: Path to the saved JSON file
    """
    # Ensure the Results directory exists
    os.makedirs("Results", exist_ok=True)

    # Full path for the JSON file
    json_file_path = os.path.join("Results", file_name)

    # Save segments to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(segments, json_file, ensure_ascii=False, indent=4)

    return json_file_path

def process_audio(
        audio_file_path,
        num_speakers=2,
        whisper_model="small.en",
        custom_prompt_input=None,
        offset=0,
        api_name=None,
        api_key=None,
        vad_filter=False,
        rolling_summarization=False,
        detail_level=0.01,
        keywords="default,no_keyword_set",
        chunk_text_by_words=False,
        max_words=0,
        chunk_text_by_sentences=False,
        max_sentences=0,
        chunk_text_by_paragraphs=False,
        max_paragraphs=0,
        chunk_text_by_tokens=False,
        max_tokens=0
):
    try:
        # Perform transcription
        audio_file_path, segments = perform_transcription(audio_file_path, offset, whisper_model, vad_filter)

        if audio_file_path is None or segments is None:
            logging.error("Process_Audio: Transcription failed or segments not available.")
            return "Process_Audio: Transcription failed.", None, None, None, None, None

        logging.debug(f"Process_Audio: Transcription audio_file: {audio_file_path}")
        logging.debug(f"Process_Audio: Transcription segments: {segments}")

        transcription_text = {'audio_file': audio_file_path, 'transcription': segments}
        logging.debug(f"Process_Audio: Transcription text: {transcription_text}")

        # Save segments to JSON
        segments_json_path = save_segments_to_json(segments)

        # Perform summarization
        summary_text = None
        if api_name:
            if rolling_summarization is not None:
                pass
                # FIXME rolling summarization
                # summary_text = rolling_summarize_function(
                #     transcription_text,
                #     detail=detail_level,
                #     api_name=api_name,
                #     api_key=api_key,
                #     custom_prompt=custom_prompt_input,
                #     chunk_by_words=chunk_text_by_words,
                #     max_words=max_words,
                #     chunk_by_sentences=chunk_text_by_sentences,
                #     max_sentences=max_sentences,
                #     chunk_by_paragraphs=chunk_text_by_paragraphs,
                #     max_paragraphs=max_paragraphs,
                #     chunk_by_tokens=chunk_text_by_tokens,
                #     max_tokens=max_tokens
                # )
            else:
                summary_text = perform_summarization(api_name, segments_json_path, custom_prompt_input, api_key)

            if summary_text is None:
                logging.error("Summary text is None. Check summarization function.")
                summary_file_path = None
        else:
            summary_text = 'Summary not available'
            summary_file_path = None

        # Save transcription and summary
        download_path = create_download_directory("Audio_Processing")
        json_file_path, summary_file_path = save_transcription_and_summary(transcription_text, summary_text,
                                                                           download_path)

        # Add to database
        add_media_to_database(None, {'title': 'Audio File', 'author': 'Unknown'}, segments, summary_text, keywords,
                              custom_prompt_input, whisper_model)

        return transcription_text, summary_text, json_file_path, summary_file_path, None, None

    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
        return str(e), None, None, None, None, None


def process_audio_file(audio_url, audio_file, whisper_model="small.en", api_name=None, api_key=None):
    progress = []
    transcriptions = []

    def update_progress(stage, message):
        progress.append(f"{stage}: {message}")
        return "\n".join(progress), "\n".join(transcriptions)

    try:
        if audio_url:
            # Process audio file from URL
            save_path = os.path.join(create_download_directory("Audio_Downloads"), "downloaded_audio_file.wav")
            try:
                download_file(audio_url, save_path)
                if os.path.getsize(save_path) > 500 * 1024 * 1024:  # 500 MB limit
                    return update_progress("Error", "Downloaded file size exceeds the 500MB limit.")
            except Exception as e:
                return update_progress("Error", f"Failed to download audio file: {str(e)}")
        elif audio_file:
            # Process uploaded audio file
            try:
                if os.path.getsize(audio_file.name) > 500 * 1024 * 1024:  # 500 MB limit
                    return update_progress("Error", "Uploaded file size exceeds the 500MB limit.")
                save_path = audio_file.name
            except Exception as e:
                return update_progress("Error", f"Failed to process uploaded file: {str(e)}")
        else:
            return update_progress("Error", "No audio file provided.")

        # Perform transcription and summarization using process_audio
        try:
            transcription, summary, json_file_path, summary_file_path, _, _ = process_audio(
                audio_file_path=save_path,
                whisper_model=whisper_model,
                api_name=api_name,
                api_key=api_key
            )

            if isinstance(transcription, dict) and 'transcription' in transcription:
                transcriptions.append(extract_text_from_segments(transcription['transcription']))
            else:
                transcriptions.append(str(transcription))

            progress.append("Processing complete.")
            if summary:
                progress.append(f"Summary: {summary}")
        except Exception as e:
            return update_progress("Error", f"Failed to process audio: {str(e)}")

    except Exception as e:
        progress.append(f"Unexpected error: {str(e)}")

    return "\n".join(progress), "\n".join(transcriptions)



def process_podcast(url, title, author, keywords, custom_prompt, api_name, api_key, whisper_model):
    progress = []
    def update_progress(message):
        progress.append(message)
        return "\n".join(progress)

    try:
        # Download podcast using yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': 'downloaded_podcast.%(ext)s'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_file = ydl.prepare_filename(info_dict).replace('.webm', '.wav')

        update_progress("Podcast downloaded successfully.")

        # Attempt to extract metadata
        detected_title = info_dict.get('title', '')
        detected_author = info_dict.get('uploader', '')
        detected_series = info_dict.get('series', '')

        # Use detected metadata if not provided by user
        title = title or detected_title or "Unknown Podcast"
        author = author or detected_author or "Unknown Author"

        # Add detected series to keywords if not already present
        if detected_series and detected_series.lower() not in keywords.lower():
            keywords = f"{keywords},series:{detected_series}" if keywords else f"series:{detected_series}"

        update_progress(f"Metadata detected/set - Title: {title}, Author: {author}, Keywords: {keywords}")

        # Transcribe the podcast
        segments = speech_to_text(audio_file, whisper_model=whisper_model)
        transcription = " ".join([segment['Text'] for segment in segments])
        update_progress("Podcast transcribed successfully.")

        # Summarize if API is provided
        summary = None
        if api_name and api_key:
            summary = perform_summarization(api_name, transcription, custom_prompt, api_key)
            update_progress("Podcast summarized successfully.")

        # Add to database
        add_media_with_keywords(
            url=url,
            title=title,
            media_type='podcast',
            content=transcription,
            keywords=keywords,
            prompt=custom_prompt,
            summary=summary or "No summary available",
            transcription_model=whisper_model,
            author=author,
            ingestion_date=None  # This will use the current date
        )
        update_progress("Podcast added to database successfully.")

        return (update_progress("Processing complete."), transcription, summary or "No summary generated.",
                title, author, keywords)

    except Exception as e:
        error_message = f"Error processing podcast: {str(e)}"
        logging.error(error_message)
        return update_progress(error_message), "", "", "", "", ""

#
#
#######################################################################################################################