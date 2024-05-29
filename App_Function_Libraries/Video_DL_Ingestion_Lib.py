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
from datetime import datetime
import json
import logging
import os
import re
import subprocess
import sys
import unicodedata
# 3rd-Party Imports
import yt_dlp
# Import Local
import summarize
from Article_Summarization_Lib import *
from Article_Extractor_Lib import *
from Audio_Transcription_Lib import *
from Chunk_Lib import *
from Diarization_Lib import *
from Local_File_Processing_Lib import *
from Local_LLM_Inference_Engine_Lib import *
from Local_Summarization_Lib import *
from Old_Chunking_Lib import *
from SQLite_DB import *
from Summarization_General_Lib import *
from System_Checks_Lib import *
from Tokenization_Methods_Lib import *
#from Video_DL_Ingestion_Lib import *
#from Web_UI_Lib import *


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


def sanitize_filename(title, max_length=255):
    # Remove invalid path characters
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    # Truncate long titles to avoid filesystem errors
    return title[:max_length].rstrip()


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
    global video_file_path, ffmpeg_path
    global audio_file_path

    # Normalize Video Title name
    logging.debug("About to normalize downloaded video title")
    normalized_video_title = normalize_title(info_dict['title'])
    video_file_path = os.path.join(download_path, f"{normalized_video_title}.{info_dict['ext']}")

    # Check for existence of video file
    if os.path.exists(video_file_path):
        logging.info(f"Video file already exists: {video_file_path}")
        return video_file_path

    # Setup path handling for ffmpeg on different OSs
    if sys.platform.startswith('win'):
        ffmpeg_path = os.path.join(os.getcwd(), 'Bin', 'ffmpeg.exe')
    elif sys.platform.startswith('linux'):
        ffmpeg_path = 'ffmpeg'
    elif sys.platform.startswith('darwin'):
        ffmpeg_path = 'ffmpeg'

    if download_video_flag:
        video_file_path = os.path.join(download_path, f"{normalized_video_title}.mp4")

        # Set options for video and audio
        ydl_opts_video = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]',
            'outtmpl': video_file_path,
            'ffmpeg_location': ffmpeg_path
        }

        with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
            logging.debug("yt_dlp: About to download video with youtube-dl")
            ydl.download([video_url])
            logging.debug("yt_dlp: Video successfully downloaded with youtube-dl")
        return video_file_path

    else:
        return None


def save_to_file(video_urls, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(video_urls))
    print(f"Video URLs saved to {filename}")

#
#
#######################################################################################################################
