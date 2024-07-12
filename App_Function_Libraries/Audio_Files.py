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
import tempfile
import uuid
from datetime import datetime

import requests
import os
from gradio import gradio
import yt_dlp

from App_Function_Libraries.Audio_Transcription_Lib import speech_to_text
#
# Local Imports
from App_Function_Libraries.SQLite_DB import add_media_to_database, add_media_with_keywords
from App_Function_Libraries.Utils import create_download_directory
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


MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB in bytes


def process_audio_files(audio_urls, audio_file, whisper_model, api_name, api_key, use_cookies, cookies, keep_original,
                        custom_keywords):
    progress = []
    temp_files = []
    all_transcriptions = []
    all_summaries = []

    def update_progress(message):
        progress.append(message)
        return gradio.update(value="\n".join(progress))

    def cleanup_files():
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    update_progress(f"Temporary file {file} removed.")
            except Exception as e:
                update_progress(f"Failed to remove temporary file {file}: {str(e)}")

    try:
        # Process multiple URLs
        urls = [url.strip() for url in audio_urls.split('\n') if url.strip()]

        for i, url in enumerate(urls):
            update_progress(f"Processing URL {i + 1}/{len(urls)}: {url}")

            # Create a unique directory for this audio file
            audio_dir = os.path.join('downloads', f'audio_{uuid.uuid4().hex[:8]}')
            os.makedirs(audio_dir, exist_ok=True)

            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': os.path.join(audio_dir, '%(title)s.%(ext)s')
            }

            # Add cookies if provided
            if use_cookies and cookies:
                try:
                    cookies_dict = json.loads(cookies)
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_cookie_file:
                        json.dump(cookies_dict, temp_cookie_file)
                        temp_cookie_file_path = temp_cookie_file.name
                    ydl_opts['cookiefile'] = temp_cookie_file_path
                    temp_files.append(temp_cookie_file_path)
                    update_progress("Cookies applied to audio download.")
                except json.JSONDecodeError:
                    update_progress("Invalid cookie format. Proceeding without cookies.")

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=True)
                    audio_file_path = ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + '.wav'
                temp_files.append(audio_file_path)
                update_progress("Audio file downloaded successfully.")
            except yt_dlp.DownloadError as e:
                update_progress(f"Failed to download audio file: {str(e)}")
                continue

            # Check file size
            if os.path.getsize(audio_file_path) > MAX_FILE_SIZE:
                update_progress(f"File size exceeds the maximum limit of 200MB. Skipping this file.")
                continue

            # Process the audio file
            yield from process_single_audio(audio_file_path, whisper_model, api_name, api_key, keep_original,
                                            custom_keywords, url)

        # Process uploaded file if provided
        if audio_file:
            if os.path.getsize(audio_file.name) > MAX_FILE_SIZE:
                update_progress(f"Uploaded file size exceeds the maximum limit of 200MB. Skipping this file.")
            else:
                yield from process_single_audio(audio_file.name, whisper_model, api_name, api_key, keep_original,
                                                custom_keywords, "Uploaded File")

        # Final cleanup
        if not keep_original:
            cleanup_files()

        return update_progress("All processing complete."), "\n\n".join(all_transcriptions), "\n\n".join(all_summaries)

    except Exception as e:
        logging.error(f"Error processing audio files: {str(e)}")
        cleanup_files()
        return update_progress(f"Processing failed: {str(e)}"), None, None


def process_single_audio(audio_file_path, whisper_model, api_name, api_key, keep_original, custom_keywords, source):
    progress = []
    progress = []
    all_transcriptions = []
    all_summaries = []
    def update_progress(message):
        progress.append(message)
        return "\n".join(progress)
    try:
        # Perform transcription
        yield update_progress("Starting transcription...")
        segments = speech_to_text(audio_file_path, whisper_model=whisper_model)
        transcription = " ".join([segment['Text'] for segment in segments])
        yield update_progress("Audio transcribed successfully.")
        all_transcriptions.append(f"Transcription for {source}:\n{transcription}")

        # Perform summarization if API is provided
        summary = None
        if api_name and api_key:
            yield update_progress("Starting summarization...")
            summary = perform_summarization(api_name, transcription, "Summarize the following audio transcript",
                                            api_key)
            yield update_progress("Audio summarized successfully.")
            all_summaries.append(f"Summary for {source}:\n{summary}")

        # Prepare keywords
        keywords = "audio,transcription"
        if custom_keywords:
            keywords += f",{custom_keywords}"

        # Add to database
        add_media_with_keywords(
            url=source,
            title=os.path.basename(audio_file_path),
            media_type='audio',
            content=transcription,
            keywords=keywords,
            prompt="Summarize the following audio transcript",
            summary=summary or "No summary available",
            transcription_model=whisper_model,
            author="Unknown",
            ingestion_date=None  # This will use the current date
        )
        yield update_progress("Audio file added to database successfully.")

        if not keep_original and source != "Uploaded File":
            os.remove(audio_file_path)
            yield update_progress(f"Temporary file {audio_file_path} removed.")
        elif keep_original and source != "Uploaded File":
            yield update_progress(f"Original audio file kept at: {audio_file_path}")

    except Exception as e:
        yield update_progress(f"Error processing {source}: {str(e)}")



def download_youtube_audio(url: str) -> str:
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': '%(title)s.%(ext)s'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return filename.rsplit('.', 1)[0] + '.wav'


def process_podcast(url, title, author, keywords, custom_prompt, api_name, api_key, whisper_model,
                    keep_original=False, enable_diarization=False, custom_vocabulary=None,
                    summary_type="short", summary_length=300, use_cookies=False, cookies=None):
    progress = []
    error_message = ""
    temp_files = []

    def update_progress(message):
        progress.append(message)
        return "\n".join(progress)

    def cleanup_files():
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    update_progress(f"Temporary file {file} removed.")
            except Exception as e:
                update_progress(f"Failed to remove temporary file {file}: {str(e)}")

    try:
        # Create a unique directory for this podcast
        podcast_dir = os.path.join('downloads', f'podcast_{uuid.uuid4().hex[:8]}')
        os.makedirs(podcast_dir, exist_ok=True)

        # Set up yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': os.path.join(podcast_dir, '%(title)s.%(ext)s')
        }

        # Add cookies if provided
        if use_cookies and cookies:
            try:
                cookies_dict = json.loads(cookies)
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_cookie_file:
                    json.dump(cookies_dict, temp_cookie_file)
                    temp_cookie_file_path = temp_cookie_file.name
                ydl_opts['cookiefile'] = temp_cookie_file_path
                temp_files.append(temp_cookie_file_path)
                update_progress("Cookies applied to yt-dlp.")
            except json.JSONDecodeError:
                update_progress("Invalid cookie format. Proceeding without cookies.")

        # Download podcast using yt-dlp
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                audio_file = ydl.prepare_filename(info_dict).replace('.webm', '.wav')
            temp_files.append(audio_file)
            update_progress("Podcast downloaded successfully.")
        except yt_dlp.DownloadError as e:
            error_message = f"Failed to download podcast: {str(e)}"
            raise

        # Expanded metadata extraction
        detected_title = info_dict.get('title', '')
        detected_author = info_dict.get('uploader', '')
        detected_series = info_dict.get('series', '')
        detected_description = info_dict.get('description', '')
        detected_upload_date = info_dict.get('upload_date', '')
        detected_duration = info_dict.get('duration', '')
        detected_episode = info_dict.get('episode', '')
        detected_season = info_dict.get('season', '')

        # Use detected metadata if not provided by user
        title = title or detected_title or "Unknown Podcast"
        author = author or detected_author or "Unknown Author"

        # Format metadata for storage
        metadata_text = f"""
Metadata:
Title: {title}
Author: {author}
Series: {detected_series}
Episode: {detected_episode}
Season: {detected_season}
Upload Date: {detected_upload_date}
Duration: {detected_duration} seconds
Description: {detected_description}

"""

        # Add detected series and other metadata to keywords
        new_keywords = []
        if detected_series:
            new_keywords.append(f"series:{detected_series}")
        if detected_episode:
            new_keywords.append(f"episode:{detected_episode}")
        if detected_season:
            new_keywords.append(f"season:{detected_season}")

        keywords = f"{keywords},{','.join(new_keywords)}" if keywords else ','.join(new_keywords)

        update_progress(f"Metadata extracted - Title: {title}, Author: {author}, Keywords: {keywords}")

        # Transcribe the podcast
        try:
            segments = speech_to_text(audio_file, whisper_model=whisper_model)
            transcription = " ".join([segment['Text'] for segment in segments])
            update_progress("Podcast transcribed successfully.")
        except Exception as e:
            error_message = f"Transcription failed: {str(e)}"
            raise

        # Combine metadata and transcription
        full_content = metadata_text + "\n\nTranscription:\n" + transcription

        # Summarize if API is provided
        summary = None
        if api_name and api_key:
            try:
                summary = perform_summarization(api_name, full_content, custom_prompt, api_key)
                update_progress("Podcast summarized successfully.")
            except requests.RequestException as e:
                error_message = f"API request failed during summarization: {str(e)}"
                raise
            except Exception as e:
                error_message = f"Summarization failed: {str(e)}"
                raise

        # Add to database
        try:
            add_media_with_keywords(
                url=url,
                title=title,
                media_type='podcast',
                content=full_content,
                keywords=keywords,
                prompt=custom_prompt,
                summary=summary or "No summary available",
                transcription_model=whisper_model,
                author=author,
                ingestion_date=datetime.now().strftime('%Y-%m-%d')
            )
            update_progress("Podcast added to database successfully.")
        except Exception as e:
            logging.error(f"Error processing podcast: {str(e)}")
            cleanup_files()
            return update_progress("Processing failed. See error message for details."), "", "", "", "", "", error_message

    finally:
        cleanup_files()

    return (update_progress("Processing complete."), full_content, summary or "No summary generated.",
            title, author, keywords, error_message)

#
#
#######################################################################################################################