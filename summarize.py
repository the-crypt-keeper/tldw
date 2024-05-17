#!/usr/bin/env python3
import argparse
import configparser
import json
import logging
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import List, Tuple, Optional
import zipfile
from datetime import datetime
from typing import List, Tuple
from typing import Optional

import gradio as gr
import requests
from SQLite_DB import *
import tiktoken
import unicodedata
import yt_dlp
# OpenAI Tokenizer support
from openai import OpenAI
from tqdm import tqdm
import tiktoken

#######################

log_level = "INFO"
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

#######
# Function Sections
#
# Database Setup
# Config Loading
# System Checks
# DataBase Functions
# Processing Paths and local file handling
# Video Download/Handling
# Audio Transcription
# Diarization
# Chunking-related Techniques & Functions
# Tokenization-related Techniques & Functions
# Summarizers
# Gradio UI
# Main
#
#######

# To Do
# Offline diarization - https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/community/offline_usage_speaker_diarization.ipynb


####
#
#       TL/DW: Too Long Didn't Watch
#
#  Project originally created by https://github.com/the-crypt-keeper
#  Modifications made by https://github.com/rmusser01
#  All credit to the original authors, I've just glued shit together.
#
#
# Usage:
#
#   Download Audio only from URL -> Transcribe audio:
#       python summarize.py https://www.youtube.com/watch?v=4nd1CDZP21s`
#
#   Download Audio+Video from URL -> Transcribe audio from Video:**
#       python summarize.py -v https://www.youtube.com/watch?v=4nd1CDZP21s`
#
#   Download Audio only from URL -> Transcribe audio -> Summarize using (`anthropic`/`cohere`/`openai`/`llama` (llama.cpp)/`ooba` (oobabooga/text-gen-webui)/`kobold` (kobold.cpp)/`tabby` (Tabbyapi)) API:**
#       python summarize.py -v https://www.youtube.com/watch?v=4nd1CDZP21s -api <your choice of API>` - Make sure to put your API key into `config.txt` under the appropriate API variable
#
#   Download Audio+Video from a list of videos in a text file (can be file paths or URLs) and have them all summarized:**
#       python summarize.py ./local/file_on_your/system --api_name <API_name>`
#
#   Run it as a WebApp**
#       python summarize.py -gui` - This requires you to either stuff your API keys into the `config.txt` file, or pass them into the app every time you want to use it.
#           Can be helpful for setting up a shared instance, but not wanting people to perform inference on your server.
#
###

#######################
#       Random issues I've encountered and how I solved them:
#   1. Something about cuda nn library missing, even though cuda is installed...
#       https://github.com/tensorflow/tensorflow/issues/54784 - Basically, installing zlib made it go away. idk.
#
#
#
#
#######################


#######################
# DB Setup

# Handled by SQLite_DB.py

#######################


#######################
# Config loading
#

# Read configuration from file
config = configparser.ConfigParser()
config.read('config.txt')

# API Keys
anthropic_api_key = config.get('API', 'anthropic_api_key', fallback=None)
logging.debug(f"Loaded Anthropic API Key: {anthropic_api_key}")

cohere_api_key = config.get('API', 'cohere_api_key', fallback=None)
logging.debug(f"Loaded cohere API Key: {cohere_api_key}")

groq_api_key = config.get('API', 'groq_api_key', fallback=None)
logging.debug(f"Loaded groq API Key: {groq_api_key}")

openai_api_key = config.get('API', 'openai_api_key', fallback=None)
logging.debug(f"Loaded openAI Face API Key: {openai_api_key}")

huggingface_api_key = config.get('API', 'huggingface_api_key', fallback=None)
logging.debug(f"Loaded HuggingFace Face API Key: {huggingface_api_key}")

# Models
anthropic_model = config.get('API', 'anthropic_model', fallback='claude-3-sonnet-20240229')
cohere_model = config.get('API', 'cohere_model', fallback='command-r-plus')
groq_model = config.get('API', 'groq_model', fallback='FIXME')
openai_model = config.get('API', 'openai_model', fallback='gpt-4-turbo')
huggingface_model = config.get('API', 'huggingface_model', fallback='CohereForAI/c4ai-command-r-plus')

# Local-Models
kobold_api_IP = config.get('Local-API', 'kobold_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
kobold_api_key = config.get('Local-API', 'kobold_api_key', fallback='')
llama_api_IP = config.get('Local-API', 'llama_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
llama_api_key = config.get('Local-API', 'llama_api_key', fallback='')
ooba_api_IP = config.get('Local-API', 'ooba_api_IP', fallback='http://127.0.0.1:5000/v1/chat/completions')
ooba_api_key = config.get('Local-API', 'ooba_api_key', fallback='')

# Retrieve output paths from the configuration file
output_path = config.get('Paths', 'output_path', fallback='results')

# Retrieve processing choice from the configuration file
processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')

# Log file
# logging.basicConfig(filename='debug-runtime.log', encoding='utf-8', level=logging.DEBUG)

#
#
#######################

# Dirty hack - sue me.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

whisper_models = ["small", "medium", "small.en", "medium.en"]
source_languages = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French"
}
source_language_list = [key[0] for key in source_languages.items()]

print(r"""_____  _          ________  _    _                                 
|_   _|| |        / /|  _  \| |  | | _                              
  | |  | |       / / | | | || |  | |(_)                             
  | |  | |      / /  | | | || |/\| |                                
  | |  | |____ / /   | |/ / \  /\  / _                              
  \_/  \_____//_/    |___/   \/  \/ (_)                             


 _                   _                                              
| |                 | |                                             
| |_   ___    ___   | |  ___   _ __    __ _                         
| __| / _ \  / _ \  | | / _ \ | '_ \  / _` |                        
| |_ | (_) || (_) | | || (_) || | | || (_| | _                      
 \__| \___/  \___/  |_| \___/ |_| |_| \__, |( )                     
                                       __/ ||/                      
                                      |___/                         
     _  _      _         _  _                      _          _     
    | |(_)    | |       ( )| |                    | |        | |    
  __| | _   __| | _ __  |/ | |_  __      __  __ _ | |_   ___ | |__  
 / _` || | / _` || '_ \    | __| \ \ /\ / / / _` || __| / __|| '_ \ 
| (_| || || (_| || | | |   | |_   \ V  V / | (_| || |_ | (__ | | | |
 \__,_||_| \__,_||_| |_|    \__|   \_/\_/   \__,_| \__| \___||_| |_|
""")

#######################################################################################################################
# System Checks
#
#

# Perform Platform Check
userOS = ""


def platform_check():
    global userOS
    if platform.system() == "Linux":
        print("Linux OS detected \n Running Linux appropriate commands")
        userOS = "Linux"
    elif platform.system() == "Windows":
        print("Windows OS detected \n Running Windows appropriate commands")
        userOS = "Windows"
    else:
        print("Other OS detected \n Maybe try running things manually?")
        exit()


# Check for NVIDIA GPU and CUDA availability
def cuda_check():
    global processing_choice
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
        if "NVIDIA-SMI" in nvidia_smi:
            print("NVIDIA GPU with CUDA is available.")
            processing_choice = "cuda"  # Set processing_choice to gpu if NVIDIA GPU with CUDA is available
        else:
            print("NVIDIA GPU with CUDA is not available.\nYou either have an AMD GPU, or you're stuck with CPU only.")
            processing_choice = "cpu"  # Set processing_choice to cpu if NVIDIA GPU with CUDA is not available
    except subprocess.CalledProcessError:
        print("NVIDIA GPU with CUDA is not available.\nYou either have an AMD GPU, or you're stuck with CPU only.")
        processing_choice = "cpu"  # Set processing_choice to cpu if nvidia-smi command fails


# Ask user if they would like to use either their GPU or their CPU for transcription
def decide_cpugpu():
    global processing_choice
    processing_input = input("Would you like to use your GPU or CPU for transcription? (1/cuda)GPU/(2/cpu)CPU): ")
    if processing_choice == "cuda" and (processing_input.lower() == "cuda" or processing_input == "1"):
        print("You've chosen to use the GPU.")
        logging.debug("GPU is being used for processing")
        processing_choice = "cuda"
    elif processing_input.lower() == "cpu" or processing_input == "2":
        print("You've chosen to use the CPU.")
        logging.debug("CPU is being used for processing")
        processing_choice = "cpu"
    else:
        print("Invalid choice. Please select either GPU or CPU.")


# check for existence of ffmpeg
def check_ffmpeg():
    if shutil.which("ffmpeg") or (os.path.exists("Bin") and os.path.isfile(".\\Bin\\ffmpeg.exe")):
        logging.debug("ffmpeg found installed on the local system, in the local PATH, or in the './Bin' folder")
        pass
    else:
        logging.debug("ffmpeg not installed on the local system/in local PATH")
        print(
            "ffmpeg is not installed.\n\n You can either install it manually, or through your package manager of "
            "choice.\n Windows users, builds are here: https://www.gyan.dev/ffmpeg/builds/")
        if userOS == "Windows":
            download_ffmpeg()
        elif userOS == "Linux":
            print(
                "You should install ffmpeg using your platform's appropriate package manager, 'apt install ffmpeg',"
                "'dnf install ffmpeg' or 'pacman', etc.")
        else:
            logging.debug("running an unsupported OS")
            print("You're running an unspported/Un-tested OS")
            exit_script = input("Let's exit the script, unless you're feeling lucky? (y/n)")
            if exit_script == "y" or "yes" or "1":
                exit()


# Download ffmpeg
def download_ffmpeg():
    user_choice = input("Do you want to download ffmpeg? (y)Yes/(n)No: ")
    if user_choice.lower() == 'yes' or 'y' or '1':
        print("Downloading ffmpeg")
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        response = requests.get(url)

        if response.status_code == 200:
            print("Saving ffmpeg zip file")
            logging.debug("Saving ffmpeg zip file")
            zip_path = "ffmpeg-release-essentials.zip"
            with open(zip_path, 'wb') as file:
                file.write(response.content)

            logging.debug("Extracting the 'ffmpeg.exe' file from the zip")
            print("Extracting ffmpeg.exe from zip file to '/Bin' folder")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                ffmpeg_path = "ffmpeg-7.0-essentials_build/bin/ffmpeg.exe"

                logging.debug("checking if the './Bin' folder exists, creating if not")
                bin_folder = "Bin"
                if not os.path.exists(bin_folder):
                    logging.debug("Creating a folder for './Bin', it didn't previously exist")
                    os.makedirs(bin_folder)

                logging.debug("Extracting 'ffmpeg.exe' to the './Bin' folder")
                zip_ref.extract(ffmpeg_path, path=bin_folder)

                logging.debug("Moving 'ffmpeg.exe' to the './Bin' folder")
                src_path = os.path.join(bin_folder, ffmpeg_path)
                dst_path = os.path.join(bin_folder, "ffmpeg.exe")
                shutil.move(src_path, dst_path)

            logging.debug("Removing ffmpeg zip file")
            print("Deleting zip file (we've already extracted ffmpeg.exe, no worries)")
            os.remove(zip_path)

            logging.debug("ffmpeg.exe has been downloaded and extracted to the './Bin' folder.")
            print("ffmpeg.exe has been successfully downloaded and extracted to the './Bin' folder.")
        else:
            logging.error("Failed to download the zip file.")
            print("Failed to download the zip file.")
    else:
        logging.debug("User chose to not download ffmpeg")
        print("ffmpeg will not be downloaded.")


#
#
#######################################################################################################################


########################################################################################################################
# DB Setup
#
#

# DB Functions
#     create_tables()
#     add_keyword()
#     delete_keyword()
#     add_keyword()
#     add_media_with_keywords()
#     search_db()
#     format_results()
#     search_and_display()
#     export_to_csv()
#     is_valid_url()
#     is_valid_date()

# Currently supported items are documents, 'video' and articles, need to change 'video' to 'audio transcripts'
# FIXME - need to integrate above functions into script so that DB is used for keyword storage and retrieval and search
#        functionality for tags and media items is available in the UI
# FIXME - also need to integrate the DB functions into the main processing functions so that the DB is updated with the results
#        of the processing
# FIXME - need to integrate the DB functions into the summarization functions so that the results are stored in the DB
#        and can be retrieved later
# FIXME - Need to integrate the DB functions into the Gradio UI so that the user can search the DB for previous results
# FIXME - Need to integrate the DB functions into the Gradio UI so that the user can export the results of a search to a CSV file
# FIXME - Need to integrate the DB functions into the Gradio UI so that the user can add keywords to the DB
# FIXME - Need to integrate the DB functions into the Gradio UI so that the user can delete keywords from the DB
# FIXME - Need to integrate the DB functions into the Gradio UI so that the user can add media items to the DB

#
#
########################################################################################################################


########################################################################################################################
# Processing Paths and local file handling
#
#

def read_paths_from_file(file_path):
    """ Reads a file containing URLs or local file paths and returns them as a list. """
    paths = []  # Initialize paths as an empty list
    with open(file_path, 'r') as file:
        paths = [line.strip() for line in file]
    return paths


def process_path(path):
    """ Decides whether the path is a URL or a local file and processes accordingly. """
    if path.startswith('http'):
        logging.debug("file is a URL")
        # For YouTube URLs, modify to download and extract info
        return get_youtube(path)
    elif os.path.exists(path):
        logging.debug("File is a path")
        # For local files, define a function to handle them
        return process_local_file(path)
    else:
        logging.error(f"Path does not exist: {path}")
        return None


# FIXME
def process_local_file(file_path):
    logging.info(f"Processing local file: {file_path}")
    title = normalize_title(os.path.splitext(os.path.basename(file_path))[0])
    info_dict = {'title': title}
    logging.debug(f"Creating {title} directory...")
    download_path = create_download_directory(title)
    logging.debug(f"Converting '{title}' to an audio file (wav).")
    audio_file = convert_to_wav(file_path)  # Assumes input files are videos needing audio extraction
    logging.debug(f"'{title}' successfully converted to an audio file (wav).")
    return download_path, info_dict, audio_file


#
#
#######################################################################################################################


#######################################################################################################################
# Video Download/Handling
#

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def process_url(url, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video, download_audio, detail_level, question_box, keywords):
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

        results = main(url, api_name=api_name, api_key=api_key, num_speakers=num_speakers, whisper_model=whisper_model,
                       offset=offset, vad_filter=vad_filter, download_video_flag=download_video,
                       custom_prompt=custom_prompt, keywords=keywords)
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

        # FIXME - Dead code?
        #summary_text = transcription_result.get('summary', 'Summary not available')

        # Add media to the database
        try:
            db = Database()
            media_url = url
            media_type = "video"
            media_content = transcription_text
            keyword_list = keywords.split(',') if keywords else ["default"]
            media_keywords = ', '.join(keyword_list)
            logging.info(f"Adding media keywords to the database: {media_keywords}")
            media_title = transcription_result['title'] if 'title' in transcription_result else 'Untitled'
            logging.info(f"Adding media to the database: {media_title}")
            media_author = "auto_generated"
            media_ingestion_date = datetime.now().strftime('%Y-%m-%d')
            add_media_with_keywords(media_url, media_title, media_type, media_content, media_keywords, media_author,
                                    media_ingestion_date)
        except Exception as e:
            logging.error(f"Failed to add media to the database: {e}")

        if summary_file_path and os.path.exists(summary_file_path):
            return transcription_text, summary_text, json_file_path, summary_file_path, video_file_path, None#audio_file_path
        else:
            return transcription_text, summary_text, json_file_path, None, video_file_path, None#audio_file_path
    except Exception as e:
        logging.error(f"Error processing URL: {e}")
        return str(e), 'Error processing the request.', None, None, None, None


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


def save_to_file(video_urls, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(video_urls))
    print(f"Video URLs saved to {filename}")


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


#
#
#######################################################################################################################


#######################################################################################################################
# Audio Transcription
#
# Convert video .m4a into .wav using ffmpeg
#   ffmpeg -i "example.mp4" -ar 16000 -ac 1 -c:a pcm_s16le "output.wav"
#       https://www.gyan.dev/ffmpeg/builds/
#


# os.system(r'.\Bin\ffmpeg.exe -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
def convert_to_wav(video_file_path, offset=0, overwrite=False):
    out_path = os.path.splitext(video_file_path)[0] + ".wav"

    if os.path.exists(out_path) and not overwrite:
        print(f"File '{out_path}' already exists. Skipping conversion.")
        logging.info(f"Skipping conversion as file already exists: {out_path}")
        return out_path
    print("Starting conversion process of .m4a to .WAV")
    out_path = os.path.splitext(video_file_path)[0] + ".wav"

    try:
        if os.name == "nt":
            logging.debug("ffmpeg being ran on windows")

            if sys.platform.startswith('win'):
                ffmpeg_cmd = ".\\Bin\\ffmpeg.exe"
                logging.debug(f"ffmpeg_cmd: {ffmpeg_cmd}")
            else:
                ffmpeg_cmd = 'ffmpeg'  # Assume 'ffmpeg' is in PATH for non-Windows systems

            command = [
                ffmpeg_cmd,  # Assuming the working directory is correctly set where .\Bin exists
                "-ss", "00:00:00",  # Start at the beginning of the video
                "-i", video_file_path,
                "-ar", "16000",  # Audio sample rate
                "-ac", "1",  # Number of audio channels
                "-c:a", "pcm_s16le",  # Audio codec
                out_path
            ]
            try:
                # Redirect stdin from null device to prevent ffmpeg from waiting for input
                with open(os.devnull, 'rb') as null_file:
                    result = subprocess.run(command, stdin=null_file, text=True, capture_output=True)
                if result.returncode == 0:
                    logging.info("FFmpeg executed successfully")
                    logging.debug("FFmpeg output: %s", result.stdout)
                else:
                    logging.error("Error in running FFmpeg")
                    logging.error("FFmpeg stderr: %s", result.stderr)
                    raise RuntimeError(f"FFmpeg error: {result.stderr}")
            except Exception as e:
                logging.error("Error occurred - ffmpeg doesn't like windows")
                raise RuntimeError("ffmpeg failed")
        elif os.name == "posix":
            os.system(f'ffmpeg -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
        else:
            raise RuntimeError("Unsupported operating system")
        logging.info("Conversion to WAV completed: %s", out_path)
    except subprocess.CalledProcessError as e:
        logging.error("Error executing FFmpeg command: %s", str(e))
        raise RuntimeError("Error converting video file to WAV")
    except Exception as e:
        logging.error("Unexpected error occurred: %s", str(e))
        raise RuntimeError("Error converting video file to WAV")
    return out_path


# Transcribe .wav into .segments.json
def speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False):
    logging.info('speech-to-text: Loading faster_whisper model: %s', whisper_model)
    from faster_whisper import WhisperModel
    model = WhisperModel(whisper_model, device=f"{processing_choice}")
    time_start = time.time()
    if audio_file_path is None:
        raise ValueError("speech-to-text: No audio file provided")
    logging.info("speech-to-text: Audio file path: %s", audio_file_path)

    try:
        _, file_ending = os.path.splitext(audio_file_path)
        out_file = audio_file_path.replace(file_ending, ".segments.json")
        prettified_out_file = audio_file_path.replace(file_ending, ".segments_pretty.json")
        if os.path.exists(out_file):
            logging.info("speech-to-text: Segments file already exists: %s", out_file)
            with open(out_file) as f:
                segments = json.load(f)
            return segments

        logging.info('speech-to-text: Starting transcription...')
        options = dict(language=selected_source_lang, beam_size=5, best_of=5, vad_filter=vad_filter)
        transcribe_options = dict(task="transcribe", **options)
        segments_raw, info = model.transcribe(audio_file_path, **transcribe_options)

        segments = []
        for segment_chunk in segments_raw:
            chunk = {
                "start": segment_chunk.start,
                "end": segment_chunk.end,
                "text": segment_chunk.text
            }
            logging.debug("Segment: %s", chunk)
            segments.append(chunk)
        logging.info("speech-to-text: Transcription completed with faster_whisper")

        # Save prettified JSON
        with open(prettified_out_file, 'w') as f:
            json.dump(segments, f, indent=2)

        # Save non-prettified JSON
        with open(out_file, 'w') as f:
            json.dump(segments, f)

    except Exception as e:
        logging.error("speech-to-text: Error transcribing audio: %s", str(e))
        raise RuntimeError("speech-to-text: Error transcribing audio")
    return segments


#
#
#######################################################################################################################


#######################################################################################################################
# Diarization
#
# TODO: https://huggingface.co/pyannote/speaker-diarization-3.1
# embedding_model = "pyannote/embedding", embedding_size=512
# embedding_model = "speechbrain/spkrec-ecapa-voxceleb", embedding_size=192
#     def speaker_diarize(video_file_path, segments, embedding_model = "pyannote/embedding", embedding_size=512, num_speakers=0):
#         """
#         1. Generating speaker embeddings for each segments.
#         2. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
#         """
#         try:
#             from pyannote.audio import Audio
#             from pyannote.core import Segment
#             from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
#             import numpy as np
#             import pandas as pd
#             from sklearn.cluster import AgglomerativeClustering
#             from sklearn.metrics import silhouette_score
#             import tqdm
#             import wave
#
#             embedding_model = PretrainedSpeakerEmbedding( embedding_model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#
#
#             _,file_ending = os.path.splitext(f'{video_file_path}')
#             audio_file = video_file_path.replace(file_ending, ".wav")
#             out_file = video_file_path.replace(file_ending, ".diarize.json")
#
#             logging.debug("getting duration of audio file")
#             with contextlib.closing(wave.open(audio_file,'r')) as f:
#                 frames = f.getnframes()
#                 rate = f.getframerate()
#                 duration = frames / float(rate)
#             logging.debug("duration of audio file obtained")
#             print(f"duration of audio file: {duration}")
#
#             def segment_embedding(segment):
#                 logging.debug("Creating embedding")
#                 audio = Audio()
#                 start = segment["start"]
#                 end = segment["end"]
#
#                 # Enforcing a minimum segment length
#                 if end-start < 0.3:
#                     padding = 0.3-(end-start)
#                     start -= padding/2
#                     end += padding/2
#                     print('Padded segment because it was too short:',segment)
#
#                 # Whisper overshoots the end timestamp in the last segment
#                 end = min(duration, end)
#                 # clip audio and embed
#                 clip = Segment(start, end)
#                 waveform, sample_rate = audio.crop(audio_file, clip)
#                 return embedding_model(waveform[None])
#
#             embeddings = np.zeros(shape=(len(segments), embedding_size))
#             for i, segment in enumerate(tqdm.tqdm(segments)):
#                 embeddings[i] = segment_embedding(segment)
#             embeddings = np.nan_to_num(embeddings)
#             print(f'Embedding shape: {embeddings.shape}')
#
#             if num_speakers == 0:
#             # Find the best number of speakers
#                 score_num_speakers = {}
#
#                 for num_speakers in range(2, 10+1):
#                     clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
#                     score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
#                     score_num_speakers[num_speakers] = score
#                 best_num_speaker = max(score_num_speakers, key=lambda x:score_num_speakers[x])
#                 print(f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score")
#             else:
#                 best_num_speaker = num_speakers
#
#             # Assign speaker label
#             clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
#             labels = clustering.labels_
#             for i in range(len(segments)):
#                 segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
#
#             with open(out_file,'w') as f:
#                 f.write(json.dumps(segments, indent=2))
#
#             # Make CSV output
#             def convert_time(secs):
#                 return datetime.timedelta(seconds=round(secs))
#
#             objects = {
#                 'Start' : [],
#                 'End': [],
#                 'Speaker': [],
#                 'Text': []
#             }
#             text = ''
#             for (i, segment) in enumerate(segments):
#                 if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
#                     objects['Start'].append(str(convert_time(segment["start"])))
#                     objects['Speaker'].append(segment["speaker"])
#                     if i != 0:
#                         objects['End'].append(str(convert_time(segments[i - 1]["end"])))
#                         objects['Text'].append(text)
#                         text = ''
#                 text += segment["text"] + ' '
#             objects['End'].append(str(convert_time(segments[i - 1]["end"])))
#             objects['Text'].append(text)
#
#             save_path = video_file_path.replace(file_ending, ".csv")
#             df_results = pd.DataFrame(objects)
#             df_results.to_csv(save_path)
#             return df_results, save_path
#
#         except Exception as e:
#             raise RuntimeError("Error Running inference with local model", e)
#
#
#######################################################################################################################


#######################################################################################################################
# Chunking-related Techniques & Functions
#
#


# This is dirty and shameful and terrible. It should be replaced with a proper implementation.
# anyways lets get to it....
client = OpenAI(api_key=openai_api_key)


def get_chat_completion(messages, model='gpt-4-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


# This function chunks a text into smaller pieces based on a maximum token count and a delimiter
def chunk_on_delimiter(input_string: str,
                       max_tokens: int,
                       delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True)
    if dropped_chunk_count > 0:
        print(f"Warning: {dropped_chunk_count} chunks were dropped due to exceeding the token limit.")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks


# This function combines text chunks into larger blocks without exceeding a specified token count.
#   It returns the combined chunks, their original indices, and the number of dropped chunks due to overflow.
def combine_chunks_with_no_minimum(
        chunks: List[str],
        max_tokens: int,
        chunk_delimiter="\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int]]:
    dropped_chunk_count = 0
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = (
        [] if header is None else [header]
    )  # list to hold the current combined chunk candidate
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]
        #FIXME MAKE NOT OPENAI SPECIFIC
        if len(openai_tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            print(f"warning: chunk overflow")
            if (
                    add_ellipsis_for_overflow
                    # FIXME MAKE NOT OPENAI SPECIFIC
                    and len(openai_tokenize(chunk_delimiter.join(candidate + ["..."]))) <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # this case would break downstream assumptions
        # estimate token count with the current chunk added
        # FIXME MAKE NOT OPENAI SPECIFIC
        extended_candidate_token_count = len(openai_tokenize(chunk_delimiter.join(candidate + [chunk])))
        # If the token count exceeds max_tokens, add the current candidate to output and start a new candidate
        if extended_candidate_token_count > max_tokens:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header  # re-initialize candidate
            candidate_indices = [chunk_i]
        # otherwise keep extending the candidate
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)
    # add the remaining candidate to output if it's not empty
    if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count


def rolling_summarize(text: str,
                      detail: float = 0,
                      model: str = 'gpt-4-turbo',
                      additional_instructions: Optional[str] = None,
                      minimum_chunk_size: Optional[int] = 500,
                      chunk_delimiter: str = ".",
                      summarize_recursively=False,
                      verbose=False):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually.
    The level of detail in the summary can be adjusted, and the process can optionally be made recursive.

    Parameters: - text (str): The text to be summarized. - detail (float, optional): A value between 0 and 1
    indicating the desired level of detail in the summary. 0 leads to a higher level summary, and 1 results in a more
    detailed summary. Defaults to 0. - model (str, optional): The model to use for generating summaries. Defaults to
    'gpt-3.5-turbo'. - additional_instructions (Optional[str], optional): Additional instructions to provide to the
    model for customizing summaries. - minimum_chunk_size (Optional[int], optional): The minimum size for text
    chunks. Defaults to 500. - chunk_delimiter (str, optional): The delimiter used to split the text into chunks.
    Defaults to ".". - summarize_recursively (bool, optional): If True, summaries are generated recursively,
    using previous summaries for context. - verbose (bool, optional): If True, prints detailed information about the
    chunking process.

    Returns:
    - str: The final compiled summary of the text.

    The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count
    based on the `detail` parameter. It then splits the text into chunks and summarizes each chunk. If
    `summarize_recursively` is True, each summary is based on the previous summaries, adding more context to the
    summarization process. The function returns a compiled summary of all chunks.
    """

    # check detail is set correctly
    assert 0 <= detail <= 1

    # interpolate the number of chunks based to get specified level of detail
    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # adjust chunk_size based on interpolated number of chunks
    # FIXME MAKE NOT OPENAI SPECIFIC
    document_length = len(openai_tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        # FIXME MAKE NOT OPENAI SPECIFIC
        print(f"Chunk lengths are {[len(openai_tokenize(x)) for x in text_chunks]}")

    # set system message
    system_message_content = "Rewrite this text in summarized form."
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in tqdm(text_chunks):
        if summarize_recursively and accumulated_summaries:
            # Creating a structured prompt for recursive summarization
            accumulated_summaries_string = '\n\n'.join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            # Directly passing the chunk for summarization without recursive context
            user_message_content = chunk

        # Constructing messages based on whether recursive summarization is applied
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]

        # Assuming this function gets the completion and works as expected
        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    # Compile final summary from partial summaries
    global final_summary
    final_summary = '\n\n'.join(accumulated_summaries)

    return final_summary


#
#
#######################################################################################################################


#######################################################################################################################
# Tokenization-related Techniques & Functions
#
#

def openai_tokenize(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model('gpt-4-turbo')
    return encoding.encode(text)


# openai summarize chunks

#
#
#######################################################################################################################


#######################################################################################################################
# Summarizers
#
#


def extract_text_from_segments(segments):
    logging.debug(f"Main: extracting text from {segments}")
    text = ' '.join([segment['text'] for segment in segments])
    logging.debug(f"Main: Successfully extracted text from {segments}")
    return text


def summarize_with_openai(api_key, file_path, model, custom_prompt):
    try:
        logging.debug("openai: Loading json data for summarization")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("openai: Extracting text from the segments")
        text = extract_text_from_segments(segments)

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(f"openai: API Key is: {api_key}")
        logging.debug("openai: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt}"
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional summarizer."
                },
                {
                    "role": "user",
                    "content": openai_prompt
                }
            ],
            "max_tokens": 4096,  # Adjust tokens as needed
            "temperature": 0.7
        }
        logging.debug("openai: Posting request")
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            summary = response.json()['choices'][0]['message']['content'].strip()
            logging.debug("openai: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.debug("openai: Summarization failed")
            print("Failed to process summary:", response.text)
            return None
    except Exception as e:
        logging.debug("openai: Error in processing: %s", str(e))
        print("Error occurred while processing summary with openai:", str(e))
        return None


def summarize_with_claude(api_key, file_path, model, custom_prompt):
    try:
        logging.debug("anthropic: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("anthropic: Extracting text from the segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }

        anthropic_prompt = custom_prompt
        logging.debug("anthropic: Prompt is {anthropic_prompt}")
        user_message = {
            "role": "user",
            "content": f"{text} \n\n\n\n{anthropic_prompt}"
        }

        data = {
            "model": model,
            "max_tokens": 4096,  # max _possible_ tokens to return
            "messages": [user_message],
            "stop_sequences": ["\n\nHuman:"],
            "temperature": 0.7,
            "top_k": 0,
            "top_p": 1.0,
            "metadata": {
                "user_id": "example_user_id",
            },
            "stream": False,
            "system": "You are a professional summarizer."
        }

        logging.debug("anthropic: Posting request to API")
        response = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=data)

        # Check if the status code indicates success
        if response.status_code == 200:
            logging.debug("anthropic: Post submittal successful")
            response_data = response.json()
            try:
                summary = response_data['content'][0]['text'].strip()
                logging.debug("anthropic: Summarization successful")
                print("Summary processed successfully.")
                return summary
            except (IndexError, KeyError) as e:
                logging.debug("anthropic: Unexpected data in response")
                print("Unexpected response format from Claude API:", response.text)
                return None
        elif response.status_code == 500:  # Handle internal server error specifically
            logging.debug("anthropic: Internal server error")
            print("Internal server error from API. Retrying may be necessary.")
            return None
        else:
            logging.debug(f"anthropic: Failed to summarize, status code {response.status_code}: {response.text}")
            print(f"Failed to process summary, status code {response.status_code}: {response.text}")
            return None

    except Exception as e:
        logging.debug("anthropic: Error in processing: %s", str(e))
        print("Error occurred while processing summary with anthropic:", str(e))
        return None


# Summarize with Cohere
def summarize_with_cohere(api_key, file_path, model, custom_prompt):
    try:
        logging.debug("cohere: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"cohere: Extracting text from segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        cohere_prompt = f"{text} \n\n\n\n{custom_prompt}"
        logging.debug("cohere: Prompt being sent is {cohere_prompt}")

        data = {
            "chat_history": [
                {"role": "USER", "message": cohere_prompt}
            ],
            "message": "Please provide a summary.",
            "model": model,
            "connectors": [{"id": "web-search"}]
        }

        logging.debug("cohere: Submitting request to API endpoint")
        print("cohere: Submitting request to API endpoint")
        response = requests.post('https://api.cohere.ai/v1/chat', headers=headers, json=data)
        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'text' in response_data:
                summary = response_data['text'].strip()
                logging.debug("cohere: Summarization successful")
                print("Summary processed successfully.")
                return summary
            else:
                logging.error("Expected data not found in API response.")
                return "Expected data not found in API response."
        else:
            logging.error(f"cohere: API request failed with status code {response.status_code}: {response.text}")
            print(f"Failed to process summary, status code {response.status_code}: {response.text}")
            return f"cohere: API request failed: {response.text}"

    except Exception as e:
        logging.error("cohere: Error in processing: %s", str(e))
        return f"cohere: Error occurred while processing summary with Cohere: {str(e)}"


# https://console.groq.com/docs/quickstart
def summarize_with_groq(api_key, file_path, model, custom_prompt):
    try:
        logging.debug("groq: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"groq: Extracting text from segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        groq_prompt = f"{text} \n\n\n\n{custom_prompt}"
        logging.debug("groq: Prompt being sent is {groq_prompt}")

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": groq_prompt
                }
            ],
            "model": model
        }

        logging.debug("groq: Submitting request to API endpoint")
        print("groq: Submitting request to API endpoint")
        response = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=data)

        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("groq: Summarization successful")
                print("Summarization successful.")
                return summary
            else:
                logging.error("Expected data not found in API response.")
                return "Expected data not found in API response."
        else:
            logging.error(f"groq: API request failed with status code {response.status_code}: {response.text}")
            return f"groq: API request failed: {response.text}"

    except Exception as e:
        logging.error("groq: Error in processing: %s", str(e))
        return f"groq: Error occurred while processing summary with groq: {str(e)}"


#################################
#
# Local Summarization

def summarize_with_llama(api_url, file_path, token, custom_prompt):
    try:
        logging.debug("llama: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"llama: Extracting text from segments file")
        text = extract_text_from_segments(segments)  # Define this function to extract text properly

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if len(token) > 5:
            headers['Authorization'] = f'Bearer {token}'

        llama_prompt = f"{text} \n\n\n\n{custom_prompt}"
        logging.debug("llama: Prompt being sent is {llama_prompt}")

        data = {
            "prompt": llama_prompt
        }

        logging.debug("llama: Submitting request to API endpoint")
        print("llama: Submitting request to API endpoint")
        response = requests.post(api_url, headers=headers, json=data)
        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            # if 'X' in response_data:
            logging.debug(response_data)
            summary = response_data['content'].strip()
            logging.debug("llama: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.error(f"llama: API request failed with status code {response.status_code}: {response.text}")
            return f"llama: API request failed: {response.text}"

    except Exception as e:
        logging.error("llama: Error in processing: %s", str(e))
        return f"llama: Error occurred while processing summary with llama: {str(e)}"


# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def summarize_with_kobold(api_url, file_path, kobold_api_token, custom_prompt):
    try:
        logging.debug("kobold: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"kobold: Extracting text from segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        kobold_prompt = f"{text} \n\n\n\n{custom_prompt}"
        logging.debug("kobold: Prompt being sent is {kobold_prompt}")

        # FIXME
        # Values literally c/p from the api docs....
        data = {
            "max_context_length": 8096,
            "max_length": 4096,
            "prompt": kobold_prompt,
        }

        logging.debug("kobold: Submitting request to API endpoint")
        print("kobold: Submitting request to API endpoint")
        response = requests.post(api_url, headers=headers, json=data)
        response_data = response.json()
        logging.debug("kobold: API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'results' in response_data and len(response_data['results']) > 0:
                summary = response_data['results'][0]['text'].strip()
                logging.debug("kobold: Summarization successful")
                print("Summarization successful.")
                return summary
            else:
                logging.error("Expected data not found in API response.")
                return "Expected data not found in API response."
        else:
            logging.error(f"kobold: API request failed with status code {response.status_code}: {response.text}")
            return f"kobold: API request failed: {response.text}"

    except Exception as e:
        logging.error("kobold: Error in processing: %s", str(e))
        return f"kobold: Error occurred while processing summary with kobold: {str(e)}"


# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
def summarize_with_oobabooga(api_url, file_path, ooba_api_token, custom_prompt):
    try:
        logging.debug("ooba: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"ooba: Extracting text from segments file\n\n\n")
        text = extract_text_from_segments(segments)
        logging.debug(f"ooba: Finished extracting text from segments file")

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        # prompt_text = "I like to eat cake and bake cakes. I am a baker. I work in a French bakery baking cakes. It
        # is a fun job. I have been baking cakes for ten years. I also bake lots of other baked goods, but cakes are
        # my favorite." prompt_text += f"\n\n{text}"  # Uncomment this line if you want to include the text variable
        ooba_prompt = "{text}\n\n\n\n{custom_prompt}"
        logging.debug("ooba: Prompt being sent is {ooba_prompt}")

        data = {
            "mode": "chat",
            "character": "Example",
            "messages": [{"role": "user", "content": ooba_prompt}]
        }

        logging.debug("ooba: Submitting request to API endpoint")
        print("ooba: Submitting request to API endpoint")
        response = requests.post(api_url, headers=headers, json=data, verify=False)
        logging.debug("ooba: API Response Data: %s", response)

        if response.status_code == 200:
            response_data = response.json()
            summary = response.json()['choices'][0]['message']['content']
            logging.debug("ooba: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.error(f"oobabooga: API request failed with status code {response.status_code}: {response.text}")
            return f"ooba: API request failed with status code {response.status_code}: {response.text}"

    except Exception as e:
        logging.error("ooba: Error in processing: %s", str(e))
        return f"ooba: Error occurred while processing summary with oobabooga: {str(e)}"


def save_summary_to_file(summary, file_path):
    logging.debug("Now saving summary to file...")
    summary_file_path = file_path.replace('.segments.json', '_summary.txt')
    logging.debug("Opening summary file for writing, *segments.json with *_summary.txt")
    with open(summary_file_path, 'w') as file:
        file.write(summary)
    logging.info(f"Summary saved to file: {summary_file_path}")


#
#
#######################################################################################################################


#######################################################################################################################
# Summarization with Detail
#

def summarize_with_detail_openai(text, detail, verbose=False):
    # FIXME MAKE function not specific to the artifiical intelligence example
    summary_with_detail_variable = rolling_summarize(text, detail=detail, verbose=True)
    print(len(openai_tokenize(summary_with_detail_variable)))
    return summary_with_detail_variable


def summarize_with_detail_recursive_openai(text, detail, verbose=False):
    summary_with_recursive_summarization = rolling_summarize(text, detail=detail, summarize_recursively=True)
    print(summary_with_recursive_summarization)


#
#
#######################################################################################################################


#######################################################################################################################
# Gradio UI
#

# Only to be used when configured with Gradio for HF Space
def summarize_with_huggingface(api_key, file_path, custom_prompt):
    logging.debug(f"huggingface: Summarization process starting...")
    try:
        logging.debug("huggingface: Loading json data for summarization")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("huggingface: Extracting text from the segments")
        logging.debug(f"huggingface: Segments: {segments}")
        text = ' '.join([segment['text'] for segment in segments])

        print(f"huggingface: lets make sure the HF api key exists...\n\t {api_key}")
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        model = "microsoft/Phi-3-mini-128k-instruct"
        API_URL = f"https://api-inference.huggingface.co/models/{model}"

        huggingface_prompt = f"{text}\n\n\n\n{custom_prompt}"
        logging.debug("huggingface: Prompt being sent is {huggingface_prompt}")
        data = {
            "inputs": text,
            "parameters": {"max_length": 512, "min_length": 100}  # You can adjust max_length and min_length as needed
        }

        print(f"huggingface: lets make sure the HF api key is the same..\n\t {huggingface_api_key}")

        logging.debug("huggingface: Submitting request...")

        response = requests.post(API_URL, headers=headers, json=data)

        if response.status_code == 200:
            summary = response.json()[0]['summary_text']
            logging.debug("huggingface: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.error(f"huggingface: Summarization failed with status code {response.status_code}: {response.text}")
            return f"Failed to process summary, status code {response.status_code}: {response.text}"
    except Exception as e:
        logging.error("huggingface: Error in processing: %s", str(e))
        print(f"Error occurred while processing summary with huggingface: {str(e)}")
        return None

    # FIXME
    # This is here for gradio authentication
    # Its just not setup.
    #def same_auth(username, password):
    #    return username == password


def format_transcription(transcription_result):
    if transcription_result:
        json_data = transcription_result['transcription']
        return json.dumps(json_data, indent=2)
    else:
        return ""


def format_file_path(file_path, fallback_path=None):
    if file_path and os.path.exists(file_path):
        logging.debug(f"File exists: {file_path}")
        return file_path
    elif fallback_path and os.path.exists(fallback_path):
        logging.debug(f"File does not exist: {file_path}. Returning fallback path: {fallback_path}")
        return fallback_path
    else:
        logging.debug(f"File does not exist: {file_path}. No fallback path available.")
        return None

def search_media(query, fields, keyword, page):
    try:
        results = search_and_display(query, fields, keyword, page)
        return results
    except Exception as e:
        logger.error(f"Error searching media: {e}")
        return str(e)

# FIXME - Change to use 'check_api()' function - also, create 'check_api()' function
def ask_question(transcription, question, api_name, api_key):
    if not question.strip():
        return "Please enter a question."

        prompt = f"Transcription:\n{transcription}\n\nGiven the above transcription, please answer the following:\n\n{question}"

        # FIXME - Refactor main API checks so they're their own function - api_check()
        # Call api_check() function here

        if api_name.lower() == "openai":
            openai_api_key = api_key if api_key else config.get('API', 'openai_api_key', fallback=None)
            headers = {
                'Authorization': f'Bearer {openai_api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                "model": openai_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the given transcription and summary."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content'].strip()
            return answer
        else:
            return "Failed to process the question."
    else:
        return "Question answering is currently only supported with the OpenAI API."


def launch_ui(demo_mode=False):
    whisper_models = ["small.en", "medium.en", "large"]

    with gr.Blocks() as iface:
        with gr.Tab("Audio Transcription + Summarization"):

            with gr.Row():
                # Light/Dark mode toggle switch
                theme_toggle = gr.Radio(choices=["Light", "Dark"], value="Light",
                                        label="Light/Dark Mode Toggle (Toggle to change UI color scheme)")

                # UI Mode toggle switch
                ui_mode_toggle = gr.Radio(choices=["Simple", "Advanced"], value="Simple",
                                          label="UI Mode (Toggle to show all options)")

            # URL input is always visible
            url_input = gr.Textbox(label="URL (Mandatory)", placeholder="Enter the video URL here")

            # Inputs to be shown or hidden
            num_speakers_input = gr.Number(value=2, label="Number of Speakers(Optional - Currently has no effect)",
                                           visible=False)
            whisper_model_input = gr.Dropdown(choices=whisper_models, value="small.en",
                                              label="Whisper Model(This is the ML model used for transcription.)",
                                              visible=False)
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt (Customize your summarization, or ask a question about the video and have it "
                      "answered)",
                placeholder="Above is the transcript of a video. Please read "
                            "through the transcript carefully. Identify the main topics that are discussed over the "
                            "course of the transcript. Then, summarize the key points about each main topic in a "
                            "concise bullet point. The bullet points should cover the key information conveyed about "
                            "each topic in the video, but should be much shorter than the full transcript. Please "
                            "output your bullet point summary inside <bulletpoints> tags.",
                lines=3, visible=True)
            offset_input = gr.Number(value=0, label="Offset (Seconds into the video to start transcribing at)",
                                     visible=False)
            api_name_input = gr.Dropdown(
                choices=[None, "huggingface", "openai", "anthropic", "cohere", "groq", "llama", "kobold", "ooba"],
                value=None,
                label="API Name (Mandatory Unless you just want a Transcription)", visible=True)
            api_key_input = gr.Textbox(label="API Key (Mandatory if API Name is specified)",
                                       placeholder="Enter your API key here; Ignore if using Local API or Built-in API",
                                       visible=True)
            vad_filter_input = gr.Checkbox(label="VAD Filter (WIP)", value=False, visible=False)
            download_video_input = gr.Checkbox(
                label="Download Video(Select to allow for file download of selected video)", value=False, visible=False)
            download_audio_input = gr.Checkbox(
                label="Download Audio(Select to allow for file download of selected Video's Audio)", value=False,
                visible=False)
            # FIXME - Hide unless advance menu shown
            detail_level_input = gr.Slider(minimum=0.01, maximum=1.0, value=0.01, step=0.01, interactive=True,
                                           label="Summary Detail Level (Slide me) (WIP)", visible=False)
            keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated Example: "
                                                                      "tag_one,tag_two,tag_three)", value="default,no_keyword_set",visible=True)
            question_box_input = gr.Textbox(label="Question",
                                            placeholder="Enter a question to ask about the transcription",
                                            visible=False)
            #question_button = gr.Button("Submit Question")
            #question_answer = gr.Textbox(label="Answer")


            inputs = [num_speakers_input, whisper_model_input, custom_prompt_input, offset_input, api_name_input,
                      api_key_input, vad_filter_input, download_video_input, download_audio_input, detail_level_input,
                      question_box_input, keywords_input]

            outputs = [
                gr.Textbox(label="Transcription (Resulting Transcription from your input URL)"),
                gr.Textbox(label="Summary or Status Message (Current status of Summary or Summary itself)"),
                gr.File(label="Download Transcription as JSON (Download the Transcription as a file)"),
                gr.File(label="Download Summary as Text (Download the Summary as a file)"),
                gr.File(label="Download Video (Download the Video as a file)", visible=False),
                gr.File(label="Download Audio (Download the Audio as a file)", visible=False),
            ]

            def toggle_light(mode):
                if mode == "Dark":
                    return """
                    <style>
                        body {
                            background-color: #1c1c1c;
                            color: #ffffff;
                        }
                        .gradio-container {
                            background-color: #1c1c1c;
                            color: #ffffff;
                        }
                        .gradio-button {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-input {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-dropdown {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-slider {
                            background-color: #4c4c4c;
                        }
                        .gradio-checkbox {
                            background-color: #4c4c4c;
                        }
                        .gradio-radio {
                            background-color: #4c4c4c;
                        }
                        .gradio-textbox {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-label {
                            color: #ffffff;
                        }
                    </style>
                    """
                else:
                    return """
                    <style>
                        body {
                            background-color: #ffffff;
                            color: #000000;
                        }
                        .gradio-container {
                            background-color: #ffffff;
                            color: #000000;
                        }
                        .gradio-button {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-input {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-dropdown {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-slider {
                            background-color: #f0f0f0;
                        }
                        .gradio-checkbox {
                            background-color: #f0f0f0;
                        }
                        .gradio-radio {
                            background-color: #f0f0f0;
                        }
                        .gradio-textbox {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-label {
                            color: #000000;
                        }
                    </style>
                    """

            # Set the event listener for the Light/Dark mode toggle switch
            theme_toggle.change(fn=toggle_light, inputs=theme_toggle, outputs=gr.HTML())

            # Function to toggle visibility of advanced inputs
            def toggle_ui(mode):
                visible = (mode == "Advanced")
                return [gr.update(visible=visible) if i not in [2, 4, 5, 11] else gr.update(visible=True) for i in
                        range(len(inputs))]

            # Set the event listener for the UI Mode toggle switch
            ui_mode_toggle.change(fn=toggle_ui, inputs=ui_mode_toggle, outputs=inputs)

            # Combine URL input and inputs lists
            all_inputs = [url_input] + inputs

            gr.Interface(
                fn=process_url,
                inputs=all_inputs,
                outputs=outputs,
                title="Video Transcription and Summarization",
                description="Submit a video URL for transcription and summarization. Ensure you input all necessary "
                            "information including API keys."
            )

        with gr.Tab("Scrape & Summarize Articles/Websites"):
            gr.Markdown("Plan to put for for ingesting articles/websites here")
            gr.Markdown("Will scrape page and store into SQLite DB")
            gr.Markdown("RAG here we come....:/")

        with gr.Tab("Ingest & Summarize Documents"):
            gr.Markdown("Plan to put ingestion form for documents here")
            gr.Markdown("Will ingest documents and store into SQLite DB")
            gr.Markdown("RAG here we come....:/")

        with gr.Tab("Sample Prompts/Questions"):
            gr.Markdown("Plan to put Sample prompts/questions here")
            gr.Markdown("Fabric prompts/live UI?")

    # Gradio interface setup with tabs
    search_tab = gr.Interface(
        fn=search_and_display,
        inputs=[
            gr.Textbox(label="Search Query", placeholder="Enter your search query here..."),
            gr.CheckboxGroup(label="Search Fields", choices=["Title", "Content"], value=["Title"]),
            gr.Textbox(label="Keyword", placeholder="Enter keywords here..."),
            gr.Number(label="Page", value=1, precision=0)
        ],
        outputs=gr.Dataframe(label="Search Results"),
        title="Search Media Summaries",
        description="Search for media (documents, videos, articles) and their summaries in the database. Use keywords for better filtering.",
        live=True
    )

    export_tab = gr.Interface(
        fn=export_to_csv,
        inputs=[
            gr.Textbox(label="Search Query", placeholder="Enter your search query here..."),
            gr.CheckboxGroup(label="Search Fields", choices=["Title", "Content"], value=["Title"]),
            gr.Textbox(label="Keyword", placeholder="Enter keywords here..."),
            gr.Number(label="Page", value=1, precision=0),
            gr.Number(label="Results per File", value=1000, precision=0)
        ],
        outputs="text",
        title="Export Search Results to CSV",
        description="Export the search results to a CSV file."
    )

    keyword_tab = gr.Interface(
        fn=add_keyword,
        inputs=gr.Textbox(label="Add Keywords (comma-separated)", placeholder="Enter keywords here..."),
        outputs="text",
        title="Add Keywords",
        description="Add multiple keywords to the database."
    )

    delete_keyword_tab = gr.Interface(
        fn=delete_keyword,
        inputs=gr.Textbox(label="Delete Keyword", placeholder="Enter keyword to delete here..."),
        outputs="text",
        title="Delete Keyword",
        description="Delete a keyword from the database."
    )

    # Combine interfaces into a tabbed interface
    tabbed_interface = gr.TabbedInterface([iface, search_tab, export_tab, keyword_tab, delete_keyword_tab],
                                          ["Transcription + Summarization", "Search", "Export", "Add Keywords",
                                           "Delete Keywords"])

    # Launch the interface
    tabbed_interface.launch()


#
#
#######################################################################################################################


#######################################################################################################################
# Main()
#
def main(input_path, api_name=None, api_key=None, num_speakers=2, whisper_model="small.en", offset=0, vad_filter=False,
         download_video_flag=False, demo_mode=False, custom_prompt=None, overwrite=False,
         rolling_summarization=None, detail=0.01, keywords=None):
    global summary, audio_file

    if input_path is None and args.user_interface:
        return []
    start_time = time.monotonic()
    paths = []  # Initialize paths as an empty list
    if os.path.isfile(input_path) and input_path.endswith('.txt'):
        logging.debug("MAIN: User passed in a text file, processing text file...")
        paths = read_paths_from_file(input_path)
    elif os.path.exists(input_path):
        logging.debug("MAIN: Local file path detected")
        paths = [input_path]
    elif (info_dict := get_youtube(input_path)) and 'entries' in info_dict:
        logging.debug("MAIN: YouTube playlist detected")
        print(
            "\n\nSorry, but playlists aren't currently supported. You can run the following command to generate a "
            "text file that you can then pass into this script though! (It may not work... playlist support seems "
            "spotty)" + """\n\n\tpython Get_Playlist_URLs.py <Youtube Playlist URL>\n\n\tThen,\n\n\tpython 
            diarizer.py <playlist text file name>\n\n""")
        return
    else:
        paths = [input_path]
    results = []

    for path in paths:
        try:
            if path.startswith('http'):
                logging.debug("MAIN: URL Detected")
                info_dict = get_youtube(path)
                json_file_path = None
                if info_dict:
                    logging.debug("MAIN: Creating path for video file...")
                    download_path = create_download_directory(info_dict['title'])
                    logging.debug("MAIN: Path created successfully\n MAIN: Now Downloading video from yt_dlp...")
                    try:
                        video_path = download_video(path, download_path, info_dict, download_video_flag)
                    except RuntimeError as e:
                        logging.error(f"Error downloading video: {str(e)}")
                        #FIXME - figure something out for handling this situation....
                        continue
                    logging.debug("MAIN: Video downloaded successfully")
                    logging.debug("MAIN: Converting video file to WAV...")
                    audio_file = convert_to_wav(video_path, offset)
                    logging.debug("MAIN: Audio file converted successfully")
            else:
                if os.path.exists(path):
                    logging.debug("MAIN: Local file path detected")
                    download_path, info_dict, audio_file = process_local_file(path)
                else:
                    logging.error(f"File does not exist: {path}")
                    continue

            if info_dict:
                logging.debug("MAIN: Creating transcription file from WAV")
                segments = speech_to_text(audio_file, whisper_model=whisper_model, vad_filter=vad_filter)
                transcription_result = {
                    'video_path': path,
                    'audio_file': audio_file,
                    'transcription': segments
                }
                results.append(transcription_result)
                logging.info(f"MAIN: Transcription complete: {audio_file}")

                # Perform rolling summarization based on API Name, detail level, and if an API key exists
                # Will remove the API key once rolling is added for llama.cpp
                if rolling_summarization:
                    logging.info("MAIN: Rolling Summarization")

                    # Extract the text from the segments
                    text = extract_text_from_segments(segments)

                    # Set the json_file_path
                    json_file_path = audio_file.replace('.wav', '.segments.json')

                    # Perform rolling summarization
                    summary = summarize_with_detail_openai(text, detail=args.detail_level, verbose=False)

                    # Handle the summarized output
                    if summary:
                        transcription_result['summary'] = summary
                        logging.info("MAIN: Rolling Summarization successful.")
                        save_summary_to_file(summary, json_file_path)
                    else:
                        logging.warning("MAIN: Rolling Summarization failed.")

                    # if api_name and api_key:
                    #     logging.debug(f"MAIN: Rolling summarization being performed by {api_name}")
                    #     json_file_path = audio_file.replace('.wav', '.segments.json')
                    #     if api_name.lower() == 'openai':
                    #         openai_api_key = api_key if api_key else config.get('API', 'openai_api_key',
                    #                                                             fallback=None)
                    #         try:
                    #             logging.debug(f"MAIN: trying to summarize with openAI")
                    #             summary = (openai_api_key, json_file_path, openai_model, custom_prompt)
                    #         except requests.exceptions.ConnectionError:
                    #             requests.status_code = "Connection: "
                # Perform summarization based on the specified API
                elif api_name:
                    logging.debug(f"MAIN: Summarization being performed by {api_name}")
                    json_file_path = audio_file.replace('.wav', '.segments.json')
                    if api_name.lower() == 'openai':
                        openai_api_key = api_key if api_key else config.get('API', 'openai_api_key',
                                                                            fallback=None)
                        try:
                            logging.debug(f"MAIN: trying to summarize with openAI")
                            summary = summarize_with_openai(openai_api_key, json_file_path, openai_model, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "anthropic":
                        anthropic_api_key = api_key if api_key else config.get('API', 'anthropic_api_key',
                                                                               fallback=None)
                        try:
                            logging.debug(f"MAIN: Trying to summarize with anthropic")
                            summary = summarize_with_claude(anthropic_api_key, json_file_path, anthropic_model,
                                                            custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "cohere":
                        cohere_api_key = api_key if api_key else config.get('API', 'cohere_api_key', fallback=None)
                        try:
                            logging.debug(f"MAIN: Trying to summarize with cohere")
                            summary = summarize_with_cohere(cohere_api_key, json_file_path, cohere_model, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "groq":
                        groq_api_key = api_key if api_key else config.get('API', 'groq_api_key', fallback=None)
                        try:
                            logging.debug(f"MAIN: Trying to summarize with Groq")
                            summary = summarize_with_groq(groq_api_key, json_file_path, groq_model, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "llama":
                        llama_token = api_key if api_key else config.get('API', 'llama_api_key', fallback=None)
                        llama_ip = llama_api_IP
                        try:
                            logging.debug(f"MAIN: Trying to summarize with Llama.cpp")
                            summary = summarize_with_llama(llama_ip, json_file_path, llama_token, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "kobold":
                        kobold_token = api_key if api_key else config.get('API', 'kobold_api_key', fallback=None)
                        kobold_ip = kobold_api_IP
                        try:
                            logging.debug(f"MAIN: Trying to summarize with kobold.cpp")
                            summary = summarize_with_kobold(kobold_ip, json_file_path, kobold_token, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "ooba":
                        ooba_token = api_key if api_key else config.get('API', 'ooba_api_key', fallback=None)
                        ooba_ip = ooba_api_IP
                        try:
                            logging.debug(f"MAIN: Trying to summarize with oobabooga")
                            summary = summarize_with_oobabooga(ooba_ip, json_file_path, ooba_token, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "huggingface":
                        huggingface_api_key = api_key if api_key else config.get('API', 'huggingface_api_key',
                                                                                 fallback=None)
                        try:
                            logging.debug(f"MAIN: Trying to summarize with huggingface")
                            summarize_with_huggingface(huggingface_api_key, json_file_path, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "

                    else:
                        logging.warning(f"Unsupported API: {api_name}")
                        summary = None

                    if summary:
                        transcription_result['summary'] = summary
                        logging.info(f"Summary generated using {api_name} API")
                        save_summary_to_file(summary, json_file_path)
                    elif final_summary:
                        logging.info(f"Rolling summary generated using {api_name} API")
                        logging.info(f"Final Rolling summary is {final_summary}\n\n")
                        save_summary_to_file(final_summary, json_file_path)
                    else:
                        logging.warning(f"Failed to generate summary using {api_name} API")
                else:
                    logging.info("MAIN: #2 - No API specified. Summarization will not be performed")
        except Exception as e:
            logging.error(f"Error processing path: {path}")
            logging.error(str(e))
            continue
        #end_time = time.monotonic()
        # print("Total program execution time: " + timedelta(seconds=end_time - start_time))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe and summarize videos.')
    parser.add_argument('input_path', type=str, help='Path or URL of the video', nargs='?')
    parser.add_argument('-v', '--video', action='store_true', help='Download the video instead of just the audio')
    parser.add_argument('-api', '--api_name', type=str, help='API name for summarization (optional)')
    parser.add_argument('-key', '--api_key', type=str, help='API key for summarization (optional)')
    parser.add_argument('-ns', '--num_speakers', type=int, default=2, help='Number of speakers (default: 2)')
    parser.add_argument('-wm', '--whisper_model', type=str, default='small.en',
                        help='Whisper model (default: small.en)')
    parser.add_argument('-off', '--offset', type=int, default=0, help='Offset in seconds (default: 0)')
    parser.add_argument('-vad', '--vad_filter', action='store_true', help='Enable VAD filter')
    parser.add_argument('-log', '--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Log level (default: INFO)')
    parser.add_argument('-gui', '--user_interface', action='store_true', help="Launch the Gradio user interface")
    parser.add_argument('-demo', '--demo_mode', action='store_true', help='Enable demo mode')
    parser.add_argument('-prompt', '--custom_prompt', type=str,
                        help='Pass in a custom prompt to be used in place of the existing one.\n (Probably should just '
                             'modify the script itself...)')
    parser.add_argument('-overwrite', '--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('-roll', '--rolling_summarization', action='store_true', help='Enable rolling summarization')
    parser.add_argument('-detail', '--detail_level', type=float, help='Mandatory if rolling summarization is enabled, '
                                                                      'defines the chunk size.\n Default is 0.01(lots '
                                                                      'of chunks) -> 1.00 (few chunks)\n Currently '
                                                                      'only OpenAI works. ',
                        default=0.01, )
    # parser.add_argument('-o', '--output_path', type=str, help='Path to save the output file')
    # parser.add_argument('--log_file', action=str, help='Where to save logfile (non-default)')
    args = parser.parse_args()

    # Logging setup
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

    custom_prompt = args.custom_prompt

    if custom_prompt == "":
        logging.debug(f"Custom prompt defined, will use \n\nf{custom_prompt} \n\nas the prompt")
        print(f"Custom Prompt has been defined. Custom prompt: \n\n {args.custom_prompt}")
    else:
        logging.debug("No custom prompt defined, will use default")
        args.custom_prompt = ("\n\nabove is the transcript of a video "
                              "Please read through the transcript carefully. Identify the main topics that are "
                              "discussed over the course of the transcript. Then, summarize the key points about each "
                              "main topic in a concise bullet point. The bullet points should cover the key "
                              "information conveyed about each topic in the video, but should be much shorter than "
                              "the full transcript. Please output your bullet point summary inside <bulletpoints> "
                              "tags.")
        print("No custom prompt defined, will use default")

    if args.user_interface:
        launch_ui(demo_mode=False)
    else:
        if not args.input_path:
            parser.print_help()
            sys.exit(1)

        logging.info('Starting the transcription and summarization process.')
        logging.info(f'Input path: {args.input_path}')
        logging.info(f'API Name: {args.api_name}')
        logging.info(f'Number of speakers: {args.num_speakers}')
        logging.info(f'Whisper model: {args.whisper_model}')
        logging.info(f'Offset: {args.offset}')
        logging.info(f'VAD filter: {args.vad_filter}')
        logging.info(f'Log Level: {args.log_level}')  # lol
        logging.info(f'Demo Mode: {args.demo_mode}')
        logging.info(f'Custom Prompt: {args.custom_prompt}')
        logging.info(f'Overwrite: {args.overwrite}')
        logging.info(f'Rolling Summarization: {args.rolling_summarization}')
        logging.info(f'User Interface: {args.user_interface}')
        logging.info(f'Video Download: {args.video}')
        # logging.info(f'Save File location: {args.output_path}')
        # logging.info(f'Log File location: {args.log_file}')

        # Get all API keys from the config
        api_keys = {key: value for key, value in config.items('API') if key.endswith('_api_key')}

        api_name = args.api_name

        # Rolling Summarization will only be performed if an API is specified and the API key is available
        # and the rolling summarization flag is set
        #
        summary = None  # Initialize to ensure it's always defined
        if args.api_name and args.rolling_summarization and any(
                key.startswith(args.api_name) and value is not None for key, value in api_keys.items()):
            logging.info(f'MAIN: API used: {args.api_name}')
            logging.info('MAIN: Rolling Summarization will be performed.')

        elif args.api_name:
            logging.info(f'MAIN: API used: {args.api_name}')
            logging.info('MAIN: Summarization (not rolling) will be performed.')

        else:
            logging.info('No API specified. Summarization will not be performed.')

        logging.debug("Platform check being performed...")
        platform_check()
        logging.debug("CUDA check being performed...")
        cuda_check()
        logging.debug("ffmpeg check being performed...")
        check_ffmpeg()

        try:
            results = main(args.input_path, api_name=args.api_name, api_key=args.api_key,
                           num_speakers=args.num_speakers, whisper_model=args.whisper_model, offset=args.offset,
                           vad_filter=args.vad_filter, download_video_flag=args.video, overwrite=args.overwrite,
                           rolling_summarization=args.rolling_summarization, custom_prompt=args.custom_prompt,
                           demo_mode=args.demo_mode, detail=args.detail_level)
            logging.info('Transcription process completed.')
        except Exception as e:
            logging.error('An error occurred during the transcription process.')
            logging.error(str(e))
            sys.exit(1)
