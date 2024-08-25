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
import os
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import requests
import yt_dlp

from App_Function_Libraries.Audio_Transcription_Lib import speech_to_text
from App_Function_Libraries.Chunk_Lib import improved_chunking_process
#
# Local Imports
from App_Function_Libraries.DB_Manager import add_media_to_database, add_media_with_keywords, \
    check_media_and_whisper_model
from App_Function_Libraries.Summarization_General_Lib import save_transcription_and_summary, perform_transcription, \
    perform_summarization
from App_Function_Libraries.Utils import create_download_directory, save_segments_to_json, downloaded_files, \
    sanitize_filename
from App_Function_Libraries.Video_DL_Ingestion_Lib import extract_metadata

#
#######################################################################################################################
# Function Definitions
#

MAX_FILE_SIZE = 500 * 1024 * 1024


def download_audio_file(url, current_whisper_model="", use_cookies=False, cookies=None):
    try:
        # Check if media already exists in the database and compare whisper models
        should_download, reason = check_media_and_whisper_model(
            url=url,
            current_whisper_model=current_whisper_model
        )

        if not should_download:
            logging.info(f"Skipping audio download: {reason}")
            return None

        logging.info(f"Proceeding with audio download: {reason}")

        # Set up the request headers
        headers = {}
        if use_cookies and cookies:
            try:
                cookie_dict = json.loads(cookies)
                headers['Cookie'] = '; '.join([f'{k}={v}' for k, v in cookie_dict.items()])
            except json.JSONDecodeError:
                logging.warning("Invalid cookie format. Proceeding without cookies.")

        # Make the request
        response = requests.get(url, headers=headers, stream=True)
        # Raise an exception for bad status codes
        response.raise_for_status()

        # Get the file size
        file_size = int(response.headers.get('content-length', 0))
        if file_size > 500 * 1024 * 1024:  # 500 MB limit
            raise ValueError("File size exceeds the 500MB limit.")

        # Generate a unique filename
        file_name = f"audio_{uuid.uuid4().hex[:8]}.mp3"
        save_path = os.path.join('downloads', file_name)

        # Ensure the downloads directory exists
        os.makedirs('downloads', exist_ok=True)


        # Download the file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logging.info(f"Audio file downloaded successfully: {save_path}")
        return save_path

    except requests.RequestException as e:
        logging.error(f"Error downloading audio file: {str(e)}")
        raise
    except ValueError as e:
        logging.error(str(e))
        raise
    except Exception as e:
        logging.error(f"Unexpected error downloading audio file: {str(e)}")
        raise


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

        # Update function call to add_media_to_database so that it properly applies the title, author and file type
        # Add to database
        add_media_to_database(None, {'title': 'Audio File', 'author': 'Unknown'}, segments, summary_text, keywords,
                              custom_prompt_input, whisper_model)

        return transcription_text, summary_text, json_file_path, summary_file_path, None, None

    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
        return str(e), None, None, None, None, None


def process_single_audio(audio_file_path, whisper_model, api_name, api_key, keep_original,custom_keywords, source,
                         custom_prompt_input, chunk_method, max_chunk_size, chunk_overlap, use_adaptive_chunking,
                         use_multi_level_chunking, chunk_language):
    progress = []
    transcription = ""
    summary = ""

    def update_progress(message):
        progress.append(message)
        return "\n".join(progress)

    try:
        # Check file size before processing
        file_size = os.path.getsize(audio_file_path)
        if file_size > MAX_FILE_SIZE:
            update_progress(f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds the maximum limit of {MAX_FILE_SIZE / (1024 * 1024):.2f} MB. Skipping this file.")
            return "\n".join(progress), "", ""

        # Perform transcription
        update_progress("Starting transcription...")
        segments = speech_to_text(audio_file_path, whisper_model=whisper_model)
        transcription = " ".join([segment['Text'] for segment in segments])
        update_progress("Audio transcribed successfully.")

        # Perform summarization if API is provided
        if api_name and api_key:
            update_progress("Starting summarization...")
            summary = perform_summarization(api_name, transcription, "Summarize the following audio transcript",
                                            api_key)
            update_progress("Audio summarized successfully.")
        else:
            summary = "No summary available"

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
            summary=summary,
            transcription_model=whisper_model,
            author="Unknown",
            ingestion_date=None  # This will use the current date
        )
        update_progress("Audio file added to database successfully.")

        if not keep_original and source != "Uploaded File":
            os.remove(audio_file_path)
            update_progress(f"Temporary file {audio_file_path} removed.")
        elif keep_original and source != "Uploaded File":
            update_progress(f"Original audio file kept at: {audio_file_path}")

    except Exception as e:
        update_progress(f"Error processing {source}: {str(e)}")
        transcription = f"Error: {str(e)}"
        summary = "No summary due to error"

    return "\n".join(progress), transcription, summary


def process_audio_files(audio_urls, audio_file, whisper_model, api_name, api_key, use_cookies, cookies, keep_original,
                        custom_keywords, custom_prompt_input, chunk_method, max_chunk_size, chunk_overlap,
                        use_adaptive_chunking, use_multi_level_chunking, chunk_language, diarize):
    progress = []
    temp_files = []
    all_transcriptions = []
    all_summaries = []

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

    def reencode_mp3(mp3_file_path):
        try:
            reencoded_mp3_path = mp3_file_path.replace(".mp3", "_reencoded.mp3")
            subprocess.run([ffmpeg_cmd, '-i', mp3_file_path, '-codec:a', 'libmp3lame', reencoded_mp3_path], check=True)
            update_progress(f"Re-encoded {mp3_file_path} to {reencoded_mp3_path}.")
            return reencoded_mp3_path
        except subprocess.CalledProcessError as e:
            update_progress(f"Error re-encoding {mp3_file_path}: {str(e)}")
            raise

    def convert_mp3_to_wav(mp3_file_path):
        try:
            wav_file_path = mp3_file_path.replace(".mp3", ".wav")
            subprocess.run([ffmpeg_cmd, '-i', mp3_file_path, wav_file_path], check=True)
            update_progress(f"Converted {mp3_file_path} to {wav_file_path}.")
            return wav_file_path
        except subprocess.CalledProcessError as e:
            update_progress(f"Error converting {mp3_file_path} to WAV: {str(e)}")
            raise

    try:
        # Check and set the ffmpeg command
        global ffmpeg_cmd
        if os.name == "nt":
            logging.debug("Running on Windows")
            ffmpeg_cmd = os.path.join(os.getcwd(), "Bin", "ffmpeg.exe")
        else:
            ffmpeg_cmd = 'ffmpeg'  # Assume 'ffmpeg' is in PATH for non-Windows systems

        # Ensure ffmpeg is accessible
        if not os.path.exists(ffmpeg_cmd) and os.name == "nt":
            raise FileNotFoundError(f"ffmpeg executable not found at path: {ffmpeg_cmd}")

        # Define chunk options early to avoid undefined errors
        chunk_options = {
            'method': chunk_method,
            'max_size': max_chunk_size,
            'overlap': chunk_overlap,
            'adaptive': use_adaptive_chunking,
            'multi_level': use_multi_level_chunking,
            'language': chunk_language
        }

        # Process multiple URLs
        urls = [url.strip() for url in audio_urls.split('\n') if url.strip()]

        for i, url in enumerate(urls):
            update_progress(f"Processing URL {i + 1}/{len(urls)}: {url}")

            # Download and process audio file
            audio_file_path = download_audio_file(url, use_cookies, cookies)
            if not os.path.exists(audio_file_path):
                update_progress(f"Downloaded file not found: {audio_file_path}")
                continue

            temp_files.append(audio_file_path)
            update_progress("Audio file downloaded successfully.")

            # Re-encode MP3 to fix potential issues
            reencoded_mp3_path = reencode_mp3(audio_file_path)
            if not os.path.exists(reencoded_mp3_path):
                update_progress(f"Re-encoded file not found: {reencoded_mp3_path}")
                continue

            temp_files.append(reencoded_mp3_path)

            # Convert re-encoded MP3 to WAV
            wav_file_path = convert_mp3_to_wav(reencoded_mp3_path)
            if not os.path.exists(wav_file_path):
                update_progress(f"Converted WAV file not found: {wav_file_path}")
                continue

            temp_files.append(wav_file_path)

            # Initialize transcription
            transcription = ""

            # Transcribe audio
            if diarize:
                segments = speech_to_text(wav_file_path, whisper_model=whisper_model, diarize=True)
            else:
                segments = speech_to_text(wav_file_path, whisper_model=whisper_model)

            # Handle segments nested under 'segments' key
            if isinstance(segments, dict) and 'segments' in segments:
                segments = segments['segments']

            if isinstance(segments, list):
                transcription = " ".join([segment.get('Text', '') for segment in segments])
                update_progress("Audio transcribed successfully.")
            else:
                update_progress("Unexpected segments format received from speech_to_text.")
                logging.error(f"Unexpected segments format: {segments}")
                continue

            if not transcription.strip():
                update_progress("Transcription is empty.")
            else:
                # Apply chunking
                chunked_text = improved_chunking_process(transcription, chunk_options)

                # Summarize
                if api_name:
                    try:
                        summary = perform_summarization(api_name, chunked_text, custom_prompt_input, api_key)
                        update_progress("Audio summarized successfully.")
                    except Exception as e:
                        logging.error(f"Error during summarization: {str(e)}")
                        summary = "Summary generation failed"
                else:
                    summary = "No summary available (API not provided)"

                all_transcriptions.append(transcription)
                all_summaries.append(summary)

                # Add to database
                add_media_with_keywords(
                    url=url,
                    title=os.path.basename(wav_file_path),
                    media_type='audio',
                    content=transcription,
                    keywords=custom_keywords,
                    prompt=custom_prompt_input,
                    summary=summary,
                    transcription_model=whisper_model,
                    author="Unknown",
                    ingestion_date=datetime.now().strftime('%Y-%m-%d')
                )
                update_progress("Audio file processed and added to database.")

        # Process uploaded file if provided
        if audio_file:
            if os.path.getsize(audio_file.name) > MAX_FILE_SIZE:
                update_progress(
                    f"Uploaded file size exceeds the maximum limit of {MAX_FILE_SIZE / (1024 * 1024):.2f}MB. Skipping this file.")
            else:
                # Re-encode MP3 to fix potential issues
                reencoded_mp3_path = reencode_mp3(audio_file.name)
                if not os.path.exists(reencoded_mp3_path):
                    update_progress(f"Re-encoded file not found: {reencoded_mp3_path}")
                    return update_progress("Processing failed: Re-encoded file not found"), "", ""

                temp_files.append(reencoded_mp3_path)

                # Convert re-encoded MP3 to WAV
                wav_file_path = convert_mp3_to_wav(reencoded_mp3_path)
                if not os.path.exists(wav_file_path):
                    update_progress(f"Converted WAV file not found: {wav_file_path}")
                    return update_progress("Processing failed: Converted WAV file not found"), "", ""

                temp_files.append(wav_file_path)

                # Initialize transcription
                transcription = ""

                if diarize:
                    segments = speech_to_text(wav_file_path, whisper_model=whisper_model, diarize=True)
                else:
                    segments = speech_to_text(wav_file_path, whisper_model=whisper_model)

                # Handle segments nested under 'segments' key
                if isinstance(segments, dict) and 'segments' in segments:
                    segments = segments['segments']

                if isinstance(segments, list):
                    transcription = " ".join([segment.get('Text', '') for segment in segments])
                else:
                    update_progress("Unexpected segments format received from speech_to_text.")
                    logging.error(f"Unexpected segments format: {segments}")

                chunked_text = improved_chunking_process(transcription, chunk_options)

                if api_name and api_key:
                    try:
                        summary = perform_summarization(api_name, chunked_text, custom_prompt_input, api_key)
                        update_progress("Audio summarized successfully.")
                    except Exception as e:
                        logging.error(f"Error during summarization: {str(e)}")
                        summary = "Summary generation failed"
                else:
                    summary = "No summary available (API not provided)"

                all_transcriptions.append(transcription)
                all_summaries.append(summary)

                add_media_with_keywords(
                    url="Uploaded File",
                    title=os.path.basename(wav_file_path),
                    media_type='audio',
                    content=transcription,
                    keywords=custom_keywords,
                    prompt=custom_prompt_input,
                    summary=summary,
                    transcription_model=whisper_model,
                    author="Unknown",
                    ingestion_date=datetime.now().strftime('%Y-%m-%d')
                )
                update_progress("Uploaded file processed and added to database.")

        # Final cleanup
        if not keep_original:
            cleanup_files()

        final_progress = update_progress("All processing complete.")
        final_transcriptions = "\n\n".join(all_transcriptions)
        final_summaries = "\n\n".join(all_summaries)

        return final_progress, final_transcriptions, final_summaries

    except Exception as e:
        logging.error(f"Error processing audio files: {str(e)}")
        cleanup_files()
        return update_progress(f"Processing failed: {str(e)}"), "", ""


def download_youtube_audio(url):
    try:
        # Determine ffmpeg path based on the operating system.
        ffmpeg_path = './Bin/ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract information about the video
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                sanitized_title = sanitize_filename(info_dict['title'])

            # Setup the temporary filenames
            temp_video_path = Path(temp_dir) / f"{sanitized_title}_temp.mp4"
            temp_audio_path = Path(temp_dir) / f"{sanitized_title}.mp3"

            # Initialize yt-dlp with options for downloading
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/best[height<=480]',  # Prefer best audio, or video up to 480p
                'ffmpeg_location': ffmpeg_path,
                'outtmpl': str(temp_video_path),
                'noplaylist': True,
                'quiet': True
            }

            # Execute yt-dlp to download the video/audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Check if the file exists
            if not temp_video_path.exists():
                raise FileNotFoundError(f"Expected file was not found: {temp_video_path}")

            # Use ffmpeg to extract audio
            ffmpeg_command = [
                ffmpeg_path,
                '-i', str(temp_video_path),
                '-vn',  # No video
                '-acodec', 'libmp3lame',
                '-b:a', '192k',
                str(temp_audio_path)
            ]
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Check if the audio file was created
            if not temp_audio_path.exists():
                raise FileNotFoundError(f"Expected audio file was not found: {temp_audio_path}")

            # Create a persistent directory for the download if it doesn't exist
            persistent_dir = Path("downloads")
            persistent_dir.mkdir(exist_ok=True)

            # Move the file from the temporary directory to the persistent directory
            persistent_file_path = persistent_dir / f"{sanitized_title}.mp3"
            os.replace(str(temp_audio_path), str(persistent_file_path))

            # Add the file to the list of downloaded files
            downloaded_files.append(str(persistent_file_path))

            return str(persistent_file_path), f"Audio downloaded successfully: {sanitized_title}.mp3"
    except Exception as e:
        return None, f"Error downloading audio: {str(e)}"


def process_podcast(url, title, author, keywords, custom_prompt, api_name, api_key, whisper_model,
                    keep_original=False, enable_diarization=False, use_cookies=False, cookies=None,
                    chunk_method=None, max_chunk_size=300, chunk_overlap=0, use_adaptive_chunking=False,
                    use_multi_level_chunking=False, chunk_language='english'):
    progress = []
    error_message = ""
    temp_files = []

    def update_progress(message):
        progress.append(message)
        return "\n".join(progress)

    def cleanup_files():
        if not keep_original:
            for file in temp_files:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                        update_progress(f"Temporary file {file} removed.")
                except Exception as e:
                    update_progress(f"Failed to remove temporary file {file}: {str(e)}")

    try:
        # Download podcast
        audio_file = download_audio_file(url, use_cookies, cookies)
        temp_files.append(audio_file)
        update_progress("Podcast downloaded successfully.")

        # Extract metadata
        metadata = extract_metadata(url)
        title = title or metadata.get('title', 'Unknown Podcast')
        author = author or metadata.get('uploader', 'Unknown Author')

        # Format metadata for storage
        metadata_text = f"""
Metadata:
Title: {title}
Author: {author}
Series: {metadata.get('series', 'N/A')}
Episode: {metadata.get('episode', 'N/A')}
Season: {metadata.get('season', 'N/A')}
Upload Date: {metadata.get('upload_date', 'N/A')}
Duration: {metadata.get('duration', 'N/A')} seconds
Description: {metadata.get('description', 'N/A')}
"""

        # Update keywords
        new_keywords = []
        if metadata.get('series'):
            new_keywords.append(f"series:{metadata['series']}")
        if metadata.get('episode'):
            new_keywords.append(f"episode:{metadata['episode']}")
        if metadata.get('season'):
            new_keywords.append(f"season:{metadata['season']}")

        keywords = f"{keywords},{','.join(new_keywords)}" if keywords else ','.join(new_keywords)

        update_progress(f"Metadata extracted - Title: {title}, Author: {author}, Keywords: {keywords}")

        # Transcribe the podcast
        try:
            if enable_diarization:
                segments = speech_to_text(audio_file, whisper_model=whisper_model, diarize=True)
            else:
                segments = speech_to_text(audio_file, whisper_model=whisper_model)
            transcription = " ".join([segment['Text'] for segment in segments])
            update_progress("Podcast transcribed successfully.")
        except Exception as e:
            error_message = f"Transcription failed: {str(e)}"
            raise

        # Apply chunking
        chunk_options = {
            'method': chunk_method,
            'max_size': max_chunk_size,
            'overlap': chunk_overlap,
            'adaptive': use_adaptive_chunking,
            'multi_level': use_multi_level_chunking,
            'language': chunk_language
        }
        chunked_text = improved_chunking_process(transcription, chunk_options)

        # Combine metadata and transcription
        full_content = metadata_text + "\n\nTranscription:\n" + transcription

        # Summarize if API is provided
        summary = None
        if api_name and api_key:
            try:
                summary = perform_summarization(api_name, chunked_text, custom_prompt, api_key)
                update_progress("Podcast summarized successfully.")
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
            error_message = f"Error adding podcast to database: {str(e)}"
            raise

        # Cleanup
        cleanup_files()

        return (update_progress("Processing complete."), full_content, summary or "No summary generated.",
                title, author, keywords, error_message)

    except Exception as e:
        logging.error(f"Error processing podcast: {str(e)}")
        cleanup_files()
        return update_progress(f"Processing failed: {str(e)}"), "", "", "", "", "", str(e)


#
#
#######################################################################################################################