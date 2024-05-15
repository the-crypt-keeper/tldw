#!/usr/bin/env python3
import argparse
import configparser
import json
import logging
import os
import platform
import requests
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile

import gradio as gr
from huggingface_hub import InferenceClient
import torch
import yt_dlp

log_level = "DEBUG"
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
#######
# Function Sections
#
# System Checks
# Processing Paths and local file handling
# Video Download/Handling
# Audio Transcription
# Diarization
# Summarizers
# Main
#
#######

# To Do
# Offline diarization - https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/community/offline_usage_speaker_diarization.ipynb
# Dark mode changes under gradio
#
# Changes made to app.py version:
# 1. Removal of video files after conversion -> check main function
# 2. Usage of/Hardcoding HF_TOKEN as token for API calls
# 3. Usage of HuggingFace for Inference
# 4. Other stuff I can't remember. Will eventually do a diff and document them.
#


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
# Download Audio only from URL -> Transcribe audio -> Summarize using (`anthropic`/`cohere`/`openai`/`llama` (
# llama.cpp)/`ooba` (oobabooga/text-gen-webui)/`kobold` (kobold.cpp)/`tabby` (Tabbyapi)) API:** python summarize.py
# -v https://www.youtube.com/watch?v=4nd1CDZP21s -api <your choice of API>` - Make sure to put your API key into
# `config.txt` under the appropriate API variable
#
#   Download Audio+Video from a list of videos in a text file (can be file paths or URLs) and have them all summarized:**
#       python summarize.py ./local/file_on_your/system --api_name <API_name>`
#
# Run it as a WebApp** python summarize.py -gui` - This requires you to either stuff your API keys into the
# `config.txt` file, or pass them into the app every time you want to use it. Can be helpful for setting up a shared
# instance, but not wanting people to perform inference on your server.
#
###


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

# API Key Shenanigans
api_key = "UNSET"

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

print(r"""
 _____  _          ________  _    _                                 
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

global summary

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
            print("You're running an unsupported/Un-tested OS")
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
########################################################################################################################


#######################################################################################################################
# Processing Paths and local file handling
#
#

def read_paths_from_file(file_path):
    """ Reads a file containing URLs or local file paths and returns them as a list. """
    paths = []  # Initialize paths as an empty list
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not os.path.exists(
                    os.path.join('results', normalize_title(line.split('/')[-1].split('.')[0]) + '.json')):
                logging.debug("line successfully imported from file and added to list to be transcribed")
                paths.append(line)
    return paths


def process_path(path):
    """ Decides whether the path is a URL or a local file and processes accordingly. """
    if path.startswith('http'):
        logging.debug("file is a URL")
        info_dict = get_youtube(path)
        if info_dict:
            return info_dict
        else:
            logging.error("Failed to get Video info")
            return None
    elif os.path.exists(path):
        logging.debug("File is a path")
        return process_local_file(path)  # For local files, define a function to handle them
    else:
        logging.error(f"Path does not exist: {path}")
        return None


#
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
########################################################################################################################


#######################################################################################################################
# Video Download/Handling
#

def process_url(url, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter,
                download_video, download_audio, chunk_size):
    video_file_path = None
    print("API Name received:", api_name)  # Debugging line
    try:
        results = main(url, api_name=api_name, api_key=api_key, num_speakers=num_speakers,
                       whisper_model=whisper_model, offset=offset, vad_filter=vad_filter,
                       download_video_flag=download_video, custom_prompt=custom_prompt)
        if results:
            transcription_result = results[0]
            json_file_path = transcription_result['audio_file'].replace('.wav', '.segments.json')
            prettified_json_file_path = transcription_result['audio_file'].replace('.wav', '.segments_pretty.json')
            summary_file_path = json_file_path.replace('.segments.json', '_summary.txt')

            json_file_path = format_file_path(json_file_path)
            prettified_json_file_path = format_file_path(prettified_json_file_path)
            summary_file_path = format_file_path(summary_file_path)

            if download_video:
                video_file_path = transcription_result['video_path'] if 'video_path' in transcription_result else None

            formatted_transcription = format_transcription(transcription_result)

            summary_text = transcription_result.get('summary', 'Summary not available')

            if summary_file_path and os.path.exists(summary_file_path):
                return formatted_transcription, summary_text, prettified_json_file_path, summary_file_path, video_file_path, None
            else:
                return formatted_transcription, summary_text, prettified_json_file_path, None, video_file_path, None
        else:
            return "No results found.", "Summary not available", None, None, None, None
    except Exception as e:
        return str(e), "Error processing the request.", None, None, None, None


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
        logging.debug(f"Youtube info successfully extracted: {info_dict}")
        if isinstance(info_dict, dict):
            return info_dict
        else:
            logging.error("Invalid info_dict format")
            return None


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

        if userOS == "Windows":
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


######################################################################################################################
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
                ffmpeg_cmd = "..\\Bin\\ffmpeg.exe"
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
    logging.info('Loading faster_whisper model: %s', whisper_model)
    from faster_whisper import WhisperModel
    model = WhisperModel(whisper_model, device=f"{processing_choice}")
    time_start = time.time()
    if audio_file_path is None:
        raise ValueError("No audio file provided")
    logging.info("Audio file path: %s", audio_file_path)

    try:
        _, file_ending = os.path.splitext(audio_file_path)
        out_file = audio_file_path.replace(file_ending, ".segments.json")
        if os.path.exists(out_file):
            logging.info("Segments file already exists: %s", out_file)
            with open(out_file) as f:
                segments = json.load(f)
            return segments

        logging.info('Starting transcription...')
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
        logging.info("Transcription completed with faster_whisper")
        with open(out_file, 'w') as f:
            json.dump(segments, f, indent=2)
    except Exception as e:
        logging.error("Error transcribing audio: %s", str(e))
        raise RuntimeError("Error transcribing audio")
    return segments


#
#
######################################################################################################################


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
# embedding_model = PretrainedSpeakerEmbedding( embedding_model, device=torch.device("cuda" if
# torch.cuda.is_available() else "cpu"))
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
######################################################################################################################


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
        #        headers = {
        #           'Authorization': f'Bearer {api_key}',
        #           'Content-Type': 'application/json'
        #        }

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
            global summary
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
                global summary
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
        logging.basicConfig(level=logging.DEBUG)
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
                global summary
                summary = response_data['text'].strip()
                logging.debug(f"cohere: Summarization successful:\n\n{summary}\n\n")
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
                global summary
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
            global summary
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
def summarize_with_kobold(kobold_ip, json_file_path, kobold_token, custom_prompt):
    try:
        logging.debug("kobold: Loading JSON data")
        with open(json_file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"kobold: Extracting text from segments file")
        text = extract_text_from_segments(segments)

        # FIXME - API Key generated from copilot...kobold.cpp doesn't mention the header for it either...
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'X_API_KEY': kobold_token
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
        response = requests.post(kobold_ip, headers=headers, json=data)
        response_data = response.json()
        logging.debug("kobold: API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'results' in response_data and len(response_data['results']) > 0:
                global summary
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
def summarize_with_oobabooga(ooba_ip, json_file_path, ooba_token, custom_prompt):
    try:
        logging.debug("ooba: Loading JSON data")
        with open(json_file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"ooba: Extracting text from segments file\n\n\n")
        text = extract_text_from_segments(segments)
        logging.debug(f"ooba: Finished extracting text from segments file")

        # FIXME - Add headers for ooba auth
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        # prompt_text = "I like to eat cake and bake cakes. I am a baker. I work in a French bakery baking cakes. It
        # is a fun job. I have been baking cakes for ten years. I also bake lots of other baked goods, but cakes are
        # my favorite." prompt_text += f"\n\n{text}"  # Uncomment this line if you want to include the text variable
        ooba_prompt = f"{text}\n\n\n\n{custom_prompt}"
        logging.debug("ooba: Prompt being sent is {ooba_prompt}")

        data = {
            "mode": "chat",
            "character": "Example",
            "messages": [{"role": "user", "content": ooba_prompt}]
        }

        logging.debug("ooba: Submitting request to API endpoint")
        print("ooba: Submitting request to API endpoint")
        response = requests.post(ooba_ip, headers=headers, json=data, verify=False)
        logging.debug("ooba: API Response Data: %s", response)

        if response.status_code == 200:
            response_data = response.json()
            global summary
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
    summary_file_path = file_path.replace('.segments.json', '_summary.txt')
    logging.debug("Opening summary file for writing, *segments.json with *_summary.txt")
    with open(summary_file_path, 'w') as file:
        file.write(summary)
    logging.info(f"Summary saved to file: {summary_file_path}")


#
#
########################################################################################################################


#######################################################################################################################
# Gradio UI
#

# Only to be used when configured with Gradio for HF Space
def summarize_with_huggingface(huggingface_api_key, json_file_path, custom_prompt):
    logging.debug(f"huggingface: Summarization process starting...")
    client = InferenceClient()

    #model = "microsoft/Phi-3-mini-128k-instruct"
    model = "CohereForAI/c4ai-command-r-plus"
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}

    client = InferenceClient(model=f"{model}", token=f"{huggingface_api_key}")

    response = client.post(json={"inputs": "The goal of life is [MASK]."}, model="bert-base-uncased")

    with open(json_file_path, 'r') as file:
        segments = json.load(file)
    text = ''.join([segment['text'] for segment in segments])

    hf_prompt = text + "\n\n\n\n" + custom_prompt

    if huggingface_api_key == "":
        api_key = os.getenv(HF_TOKEN)
        logging.debug("HUGGINGFACE API KEY CHECK: " + huggingface_api_key)
    try:
        logging.debug("huggingface: Loading json data for summarization")
        with open(json_file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("huggingface: Extracting text from the segments")
        text = ' '.join([segment['text'] for segment in segments])

        #api_key = os.getenv('HF_TOKEN').replace('"', '')
        logging.debug("HUGGINGFACE API KEY CHECK #2: " + huggingface_api_key)

        logging.debug("huggingface: Submitting request...")
        response = client.text_generation(prompt=hf_prompt, max_new_tokens=4096)
        if response is not None:
            return response
        #if response == FIXME:
            #logging.debug("huggingface: Summarization successful")
            #print("Summarization successful.")
            #return response
        #elif Bad Stuff:
            # logging.debug(f"huggingface: Model is currently loading...{response.status_code}: {response.text}")
            # global waiting_summary
            # pretty_json = json.dumps(json.loads(response.text), indent=4)  # Prettify JSON
            # waiting_summary = f" {pretty_json} "  # Use prettified JSON
            # return waiting_summary
        else:
            logging.error(f"huggingface: Summarization failed with status code {response}")
            return f"Failed to process summary, huggingface library error: {response}"
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


def format_file_path(file_path):
    # Helper function to check file existence and return an appropriate path or message
    return file_path if file_path and os.path.exists(file_path) else None


def update_visibility(mode):
    if mode == "Advanced":
        # Show all inputs below URL
        return [gr.update(visible=True)] * 9
    else:
        # Hide all inputs below URL
        return [gr.update(visible=False)] * 9


# https://www.gradio.app/guides/controlling-layout
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
                label="Custom Prompt (Customize your summary, or ask a different question)",
                placeholder="Q: As a professional summarizer, create a concise and comprehensive summary of the "
                            "provided text.\nA: Here is a detailed, bulleted list of the key points made in the "
                            "transcribed video and supporting arguments:",
                lines=3, visible=True)
            offset_input = gr.Number(value=0, label="Offset (Seconds into the video to start transcribing at)",
                                     visible=False)
            api_name_input = gr.Dropdown(
                choices=[None, "huggingface", "openai", "anthropic", "cohere", "groq", "llama", "kobold", "ooba"],
                value=None,
                label="API Name (Mandatory Unless you just want a Transcription)", visible=True)
            api_key_input = gr.Textbox(label="API Key (Mandatory if API Name is specified)",
                                       placeholder="Enter your API key here", visible=True)
            vad_filter_input = gr.Checkbox(label="VAD Filter(Can safely ignore)", value=False, visible=False)
            download_video_input = gr.Checkbox(
                label="Download Video(Select to allow for file download of selected video)", value=False, visible=False)
            download_audio_input = gr.Checkbox(
                label="Download Audio(Select to allow for file download of selected Video's Audio)", value=False,
                visible=False)
            detail_level_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.1, interactive=False,
                                           label="Detail Level (Slide me)", visible=True)

            inputs = [num_speakers_input, whisper_model_input, custom_prompt_input, offset_input, api_name_input,
                      api_key_input, vad_filter_input, download_video_input, download_audio_input, detail_level_input]

            # Function to toggle Light/Dark Mode
            def toggle_light(mode):
                dark = (mode == "Dark")
                return {"__theme": "dark" if dark else "light"}

            # Set the event listener for the Light/Dark mode toggle switch
            theme_toggle.change(fn=toggle_light, inputs=theme_toggle, outputs=None)

            # Function to toggle visibility of advanced inputs
            def toggle_ui(mode):
                visible = (mode == "Advanced")
                return [visible] * len(inputs)

            # Set the event listener for the UI Mode toggle switch
            ui_mode_toggle.change(fn=toggle_ui, inputs=ui_mode_toggle, outputs=inputs)

            # Combine URL input and inputs
            all_inputs = [url_input] + inputs

            outputs = [
                gr.Textbox(label="Transcription (Resulting Transcription from your input URL)"),
                gr.Textbox(label="Summary or Status Message (Current status of Summary or Summary itself)"),
                gr.File(label="Download Transcription as JSON (Download the Transcription as a file)"),
                gr.File(label="Download Summary as Text (Download the Summary as a file)"),
                gr.File(label="Download Video (Download the Video as a file)"),
                gr.File(label="Download Audio (Download the Audio as a file)")
            ]

            gr.Interface(
                fn=process_url,
                inputs=all_inputs,
                outputs=outputs,
                title="TL/DW: Video Transcription and Summarization with Custom Prompt Support",
                description="Submit a video URL for transcription and summarization. Ensure you input all necessary "
                            "information including API keys."
            )

        with gr.Tab("Transcription & Summarization History"):
            image_input = gr.Image(label="Upload Image")
            image_output = gr.Image(label="Processed Image")

        with gr.Accordion("Open for More!", open=False):
            gr.Markdown("Look at me...")
            gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.1, interactive=True, label="Slide me")

        iface.launch(share=False)


#
#
#####################################################################################################################################


####################################################################################################################################
# Main()
#

def main(input_path, api_name=None, api_key=None, num_speakers=2, whisper_model="small.en", offset=0, vad_filter=False,
         download_video_flag=False, demo_mode=False, custom_prompt=None, overwrite=False):
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
                logging.info(f"Transcription complete: {audio_file}")

                # Perform summarization based on the specified API
                logging.debug(f"MAIN: Summarization being performed by {api_name} API")
                json_file_path = audio_file.replace('.wav', '.segments.json')
                prettified_json_file_path = transcription_result['audio_file'].replace('.wav', '.segments_pretty.json')
                if api_name == "huggingface":
                    huggingface_api_key = os.getenv('HF_TOKEN').replace('"', '')
                    if huggingface_api_key is None:
                        huggingface_api_key = api_key if api_key else config.get('API', 'huggingface_api_key',
                                                                                 fallback=None)
                    try:
                        logging.debug(f"MAIN: Trying to summarize with huggingface")
                        summarize_with_huggingface(huggingface_api_key, json_file_path, custom_prompt)
                    except requests.exceptions.ConnectionError:
                        requests.status_code = "Connection: "
                elif api_name == "cohere":
                    cohere_api_key = os.getenv('COHERE_TOKEN').replace('"', '')
                    if cohere_api_key is None:
                        cohere_api_key = api_key if api_key else config.get('API', 'cohere_api_key',
                                                                                 fallback=None)
                    try:
                        global summary
                        logging.debug(f"MAIN: Trying to summarize with Cohere on HuggingFace Spaces")
                        summary = summarize_with_cohere(cohere_api_key, json_file_path, cohere_model, custom_prompt)
                        transcription_result['summary'] = summary
                        logging.info(f"Summary generated using {api_name} API")
                        save_summary_to_file(summary, json_file_path)
                    except requests.exceptions.ConnectionError:
                        requests.status_code = "Connection: "
                elif api_name and api_key:
                    logging.debug(f"MAIN: Summarization being performed by {api_name}")
                    json_file_path = audio_file.replace('.wav', '.segments.json')
                    if api_name.lower() == 'openai':
                        openai_api_key = api_key if api_key else config.get('API', 'openai_api_key', fallback=None)
                        try:
                            logging.debug(f"MAIN: trying to summarize with openAI")
                            summary = summarize_with_openai(openai_api_key, json_file_path, openai_model, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "huggingface":
                        huggingface_api_key = os.getenv(HF_TOKEN)
                        if huggingface_api_key is None:
                            huggingface_api_key = api_key if api_key else config.get('API', 'huggingface_api_key',
                                                                                     fallback=None)
                        try:
                            logging.debug(f"MAIN: Trying to summarize with huggingface")
                            summarize_with_huggingface(huggingface_api_key, json_file_path, custom_prompt)
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
                    else:
                        logging.warning(f"Unsupported API: {api_name}")
                        summary = None

                    print(f"MAIN: #1 - Summary: {summary}")
                    if summary:
                        transcription_result['summary'] = summary
                        logging.info(f"Summary generated using {api_name} API")
                        save_summary_to_file(summary, json_file_path)
                    else:
                        logging.warning(f"Failed to generate summary using {api_name} API")
                else:
                    logging.info("MAIN: #2 - No API specified. Summarization will not be performed")
        except Exception as e:
            logging.error(f"Error processing path: {path}")
            logging.error(str(e))
            continue
        # end_time = time.monotonic()
        # print("Total program execution time: " + timedelta(seconds=end_time - start_time))

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe and summarize videos.')
    parser.add_argument('input_path', type=str, help='Path or URL of the video', nargs='?')
    parser.add_argument('-v', '--video', action='store_true', help='Download the video instead of just the audio')
    parser.add_argument('-api', '--api_name', type=str, help='API name for summarization (optional)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing audio files')
    parser.add_argument('-ns', '--num_speakers', type=int, default=2, help='Number of speakers (default: 2)')
    parser.add_argument('-wm', '--whisper_model', type=str, default='small.en',
                        help='Whisper model (default: small.en)')
    parser.add_argument('-off', '--offset', type=int, default=0, help='Offset in seconds (default: 0)')
    parser.add_argument('-vad', '--vad_filter', action='store_true', help='Enable VAD filter')
    parser.add_argument('-log', '--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Log level (default: INFO)')
    parser.add_argument('-ui', '--user_interface', action='store_true', help='Launch the Gradio user interface')
    parser.add_argument('-demo', '--demo_mode', action='store_true', help='Enable demo mode')
    parser.add_argument('-prompt', '--custom_prompt', type=str,
                        help='Pass in a custom prompt to be used in place of the existing one.(Probably should just modify the script itself...)')
    # parser.add_argument('--log_file', action=str, help='Where to save logfile (non-default)')
    args = parser.parse_args()

    custom_prompt = args.custom_prompt
    if custom_prompt == "":
        logging.debug(f"Custom prompt defined, will use \n\nf{custom_prompt} \n\nas the prompt")
        print(f"Custom Prompt has been defined. Custom prompt: \n\n {args.custom_prompt}")
    else:
        logging.debug("No custom prompt defined, will use default")
        args.custom_prompt = "\n\nQ: As a professional summarizer, create a concise and comprehensive summary of the provided text.\nA: Here is a detailed, bulleted list of the key points made in the transcribed video and supporting arguments:"
        print("No custom prompt defined, will use default")

    # print(f"Is CUDA available: {torch.cuda.is_available()}")
    # True
    # print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    # Tesla T4

    # Since this is running in HF....
    args.user_interface = True
    if args.user_interface:
        launch_ui(demo_mode=args.demo_mode)
    else:
        if not args.input_path:
            parser.print_help()
            sys.exit(1)

        logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info('Starting the transcription and summarization process.')
        logging.info(f'Input path: {args.input_path}')
        logging.info(f'API Name: {args.api_name}')
        logging.debug(f'API Key: {args.api_key}')  # ehhhhh
        logging.info(f'Number of speakers: {args.num_speakers}')
        logging.info(f'Whisper model: {args.whisper_model}')
        logging.info(f'Offset: {args.offset}')
        logging.info(f'VAD filter: {args.vad_filter}')
        logging.info(f'Log Level: {args.log_level}')  # lol

        if args.api_name and args.api_key:
            logging.info(f'API: {args.api_name}')
            logging.info('Summarization will be performed.')
        else:
            logging.info('MAIN: #1 No API specified. Summarization will not be performed.')

        logging.debug("Platform check being performed...")
        platform_check()
        logging.debug("CUDA check being performed...")
        cuda_check()
        logging.debug("ffmpeg check being performed...")
        check_ffmpeg()

        # Hey, we're in HuggingFace
        launch_ui(demo_mode=args.demo_mode)

        try:
            results = main(args.input_path, api_name=args.api_name, api_key=args.api_key,
                           num_speakers=args.num_speakers, whisper_model=args.whisper_model, offset=args.offset,
                           vad_filter=args.vad_filter, download_video_flag=args.video, overwrite=args.overwrite)
            logging.info('Transcription process completed.')
        except Exception as e:
            logging.error('An error occurred during the transcription process.')
            logging.error(str(e))
            sys.exit(1)
