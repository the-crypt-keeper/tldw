# Video_DL_Ingestion_Lib.py
#########################################
# Video Downloader and Ingestion Library
# This library is used to handle downloading videos from YouTube and other platforms.
# It also handles the ingestion of the videos into the database.
# It uses yt-dlp to extract video information and download the videos.
####

####################
# Function List
#
# 1. get_video_info(url)
# 2. create_download_directory(title)
# 3. sanitize_filename(title)
# 4. normalize_title(title)
# 5. get_youtube(video_url)
# 6. get_playlist_videos(playlist_url)
# 7. download_video(video_url, download_path, info_dict, download_video_flag)
# 8. save_to_file(video_urls, filename)
# 9. save_summary_to_file(summary, file_path)
# 10. process_url(url, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video, download_audio, rolling_summarization, detail_level, question_box, keywords, chunk_summarization, chunk_duration_input, words_per_second_input)
#
#
####################


# Import necessary libraries to run solo for testing
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import unicodedata
import yt_dlp




#######################################################################################################################
# Function Definitions
#

def get_video_info(url: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict
        except Exception as e:
            logging.error(f"Error extracting video info: {e}")
            return None


def create_download_directory(title):
    base_dir = "Results"
    # Remove characters that are illegal in Windows filenames and normalize
    safe_title = normalize_title(title)
    logging.debug(f"{title} successfully normalized")
    session_path = os.path.join(base_dir, safe_title)
    if not os.path.exists(session_path):
        os.makedirs(session_path, exist_ok=True)
        logging.debug(f"Created directory for downloaded video: {session_path}")
    else:
        logging.debug(f"Directory already exists for downloaded video: {session_path}")
    return session_path


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def normalize_title(title):
    # Normalize the string to 'NFKD' form and encode to 'ascii' ignoring non-ascii characters
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
    title = title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('"', '').replace('*', '').replace('?',
                                                                                                                   '').replace(
        '<', '').replace('>', '').replace('|', '')
    return title


def get_youtube(video_url):
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'noplaylist': False,
        'quiet': True,
        'extract_flat': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logging.debug("About to extract youtube info")
        info_dict = ydl.extract_info(video_url, download=False)
        logging.debug("Youtube info successfully extracted")
    return info_dict


def get_playlist_videos(playlist_url):
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

        if 'entries' in info:
            video_urls = [entry['url'] for entry in info['entries']]
            playlist_title = info['title']
            return video_urls, playlist_title
        else:
            print("No videos found in the playlist.")
            return [], None


def download_video(video_url, download_path, info_dict, download_video_flag):
    logging.debug("About to normalize downloaded video title")
    title = normalize_title(info_dict['title'])

    if not download_video_flag:
        file_path = os.path.join(download_path, f"{title}.m4a")
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]',
            'outtmpl': file_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.debug("yt_dlp: About to download audio with youtube-dl")
            ydl.download([video_url])
            logging.debug("yt_dlp: Audio successfully downloaded with youtube-dl")
        return file_path
    else:
        video_file_path = os.path.join(download_path, f"{title}_video.mp4")
        audio_file_path = os.path.join(download_path, f"{title}_audio.m4a")
        ydl_opts_video = {
            'format': 'bestvideo[ext=mp4]',
            'outtmpl': video_file_path,
        }
        ydl_opts_audio = {
            'format': 'bestaudio[ext=m4a]',
            'outtmpl': audio_file_path,
        }

        with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
            logging.debug("yt_dlp: About to download video with youtube-dl")
            ydl.download([video_url])
            logging.debug("yt_dlp: Video successfully downloaded with youtube-dl")

        with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
            logging.debug("yt_dlp: About to download audio with youtube-dl")
            ydl.download([video_url])
            logging.debug("yt_dlp: Audio successfully downloaded with youtube-dl")

        output_file_path = os.path.join(download_path, f"{title}.mp4")

        if sys.platform.startswith('win'):
            logging.debug("Running ffmpeg on Windows...")
            ffmpeg_command = [
                '.\\Bin\\ffmpeg.exe',
                '-i', video_file_path,
                '-i', audio_file_path,
                '-c:v', 'copy',
                '-c:a', 'copy',
                output_file_path
            ]
            subprocess.run(ffmpeg_command, check=True)
        elif userOS == "Linux":
            logging.debug("Running ffmpeg on Linux...")
            ffmpeg_command = [
                'ffmpeg',
                '-i', video_file_path,
                '-i', audio_file_path,
                '-c:v', 'copy',
                '-c:a', 'copy',
                output_file_path
            ]
            subprocess.run(ffmpeg_command, check=True)
        else:
            logging.error("ffmpeg: Unsupported operating system for video download and merging.")
            raise RuntimeError("ffmpeg: Unsupported operating system for video download and merging.")
        os.remove(video_file_path)
        os.remove(audio_file_path)

        return output_file_path


def save_to_file(video_urls, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(video_urls))
    print(f"Video URLs saved to {filename}")


def save_summary_to_file(summary: str, file_path: str):
    """Save summary to a JSON file."""
    summary_data = {'summary': summary, 'generated_at': datetime.now().isoformat()}
    with open(file_path, 'w') as file:
        json.dump(summary_data, file, indent=4)


def process_url(url,
                num_speakers,
                whisper_model,
                custom_prompt,
                offset,
                api_name,
                api_key,
                vad_filter,
                download_video,
                download_audio,
                rolling_summarization,
                detail_level,
                question_box,
                keywords,
                chunk_summarization,
                chunk_duration_input,
                words_per_second_input,
                ):
    # Validate input
    if not url:
        return "No URL provided.", "No URL provided.", None, None, None, None, None, None

    if not is_valid_url(url):
        return "Invalid URL format.", "Invalid URL format.", None, None, None, None, None, None

    print("API Name received:", api_name)  # Debugging line

    logging.info(f"Processing URL: {url}")
    video_file_path = None

    try:
        # Instantiate the database, db as a instance of the Database class
        db = Database()
        media_url = url

        info_dict = get_youtube(url)  # Extract video information using yt_dlp
        media_title = info_dict['title'] if 'title' in info_dict else 'Untitled'

        results = main(url, api_name=api_name, api_key=api_key,
                       num_speakers=num_speakers,
                       whisper_model=whisper_model,
                       offset=offset,
                       vad_filter=vad_filter,
                       download_video_flag=download_video,
                       custom_prompt=custom_prompt,
                       overwrite=args.overwrite,
                       rolling_summarization=rolling_summarization,
                       detail=detail_level,
                       keywords=keywords,
                       chunk_summarization=chunk_summarization,
                       chunk_duration=chunk_duration_input,
                       words_per_second=words_per_second_input,
                       )

        if not results:
            return "No URL provided.", "No URL provided.", None, None, None, None, None, None

        transcription_result = results[0]
        transcription_text = json.dumps(transcription_result['transcription'], indent=2)
        summary_text = transcription_result.get('summary', 'Summary not available')

        # Prepare file paths for transcription and summary
        # Sanitize filenames
        audio_file_sanitized = sanitize_filename(transcription_result['audio_file'])
        json_file_path = audio_file_sanitized.replace('.wav', '.segments_pretty.json')
        summary_file_path = audio_file_sanitized.replace('.wav', '_summary.txt')

        logging.debug(f"Transcription result: {transcription_result}")
        logging.debug(f"Audio file path: {transcription_result['audio_file']}")

        # Write the transcription to the JSON File
        try:
            with open(json_file_path, 'w') as json_file:
                json.dump(transcription_result['transcription'], json_file, indent=2)
        except IOError as e:
            logging.error(f"Error writing transcription to JSON file: {e}")

        # Write the summary to the summary file
        with open(summary_file_path, 'w') as summary_file:
            summary_file.write(summary_text)

        if download_video:
            video_file_path = transcription_result['video_path'] if 'video_path' in transcription_result else None

        # Check if files exist before returning paths
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File not found: {json_file_path}")
        if not os.path.exists(summary_file_path):
            raise FileNotFoundError(f"File not found: {summary_file_path}")

        formatted_transcription = format_transcription(transcription_result)

        # Check for chunk summarization
        if chunk_summarization:
            chunk_duration = chunk_duration_input if chunk_duration_input else DEFAULT_CHUNK_DURATION
            words_per_second = words_per_second_input if words_per_second_input else WORDS_PER_SECOND
            summary_text = summarize_chunks(api_name, api_key, transcription_result['transcription'], chunk_duration,
                                            words_per_second)

        # FIXME - This is a mess
        # # Check for time-based chunking summarization
        # if time_based_summarization:
        #     logging.info("MAIN: Time-based Summarization")
        #
        #     # Set the json_file_path
        #     json_file_path = audio_file.replace('.wav', '.segments.json')
        #
        #     # Perform time-based summarization
        #     summary = time_chunk_summarize(api_name, api_key, json_file_path, time_chunk_duration, custom_prompt)
        #
        #     # Handle the summarized output
        #     if summary:
        #         transcription_result['summary'] = summary
        #         logging.info("MAIN: Time-based Summarization successful.")
        #         save_summary_to_file(summary, json_file_path)
        #     else:
        #         logging.warning("MAIN: Time-based Summarization failed.")

        # Add media to the database
        try:
            # Ensure these variables are correctly populated
            custom_prompt = args.custom_prompt if args.custom_prompt else ("\n\nabove is the transcript of a video "
                "Please read through the transcript carefully. Identify the main topics that are discussed over the "
                "course of the transcript. Then, summarize the key points about each main topic in a concise bullet "
                "point. The bullet points should cover the key information conveyed about each topic in the video, "
                "but should be much shorter than the full transcript. Please output your bullet point summary inside "
                "<bulletpoints> tags.")

            db = Database()
            create_tables()
            media_url = url
            # FIXME  - IDK?
            video_info = get_video_info(media_url)
            media_title = get_page_title(media_url)
            media_type = "video"
            media_content = transcription_text
            keyword_list = keywords.split(',') if keywords else ["default"]
            media_keywords = ', '.join(keyword_list)
            media_author = "auto_generated"
            media_ingestion_date = datetime.now().strftime('%Y-%m-%d')
            transcription_model = whisper_model  # Add the transcription model used

            # Log the values before calling the function
            logging.info(f"Media URL: {media_url}")
            logging.info(f"Media Title: {media_title}")
            logging.info(f"Media Type: {media_type}")
            logging.info(f"Media Content: {media_content}")
            logging.info(f"Media Keywords: {media_keywords}")
            logging.info(f"Media Author: {media_author}")
            logging.info(f"Ingestion Date: {media_ingestion_date}")
            logging.info(f"Custom Prompt: {custom_prompt}")
            logging.info(f"Summary Text: {summary_text}")
            logging.info(f"Transcription Model: {transcription_model}")

            # Check if any required field is empty
            if not media_url or not media_title or not media_type or not media_content or not media_keywords or not custom_prompt or not summary_text:
                raise InputError("Please provide all required fields.")

            add_media_with_keywords(
                url=media_url,
                title=media_title,
                media_type=media_type,
                content=media_content,
                keywords=media_keywords,
                prompt=custom_prompt,
                summary=summary_text,
                transcription_model=transcription_model,  # Pass the transcription model
                author=media_author,
                ingestion_date=media_ingestion_date
            )
        except Exception as e:
            logging.error(f"Failed to add media to the database: {e}")

        if summary_file_path and os.path.exists(summary_file_path):
            return transcription_text, summary_text, json_file_path, summary_file_path, video_file_path, None  # audio_file_path
        else:
            return transcription_text, summary_text, json_file_path, None, video_file_path, None  # audio_file_path
    except Exception as e:
        logging.error(f"Error processing URL: {e}")
        return str(e), 'Error processing the request.', None, None, None, None

#
#
#######################################################################################################################
