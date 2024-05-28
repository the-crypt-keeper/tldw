#!/usr/bin/env python3
# Std Lib Imports
import argparse
import asyncio
import atexit
import configparser
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import unicodedata
from multiprocessing import process
from typing import  Callable, Dict, List, Optional, Tuple
import webbrowser
import zipfile

# Local Module Imports (Libraries specific to this project)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'App_Function_Libraries')))
from App_Function_Libraries import *
from App_Function_Libraries.Web_UI_Lib import launch_ui
# from App_Function_Libraries import Article-Extractor-Lib
# from App_Function_Libraries import Article-Summarization-Lib
# from App_Function_Libraries import Audio-Transcription-Lib
# from App_Function_Libraries import Chunk-Lib
# from App_Function_Libraries import Diarization-Lib
# from App_Function_Libraries import Local-File-Handling-Lib
# from App_Function_Libraries import Local-LLM-Inference-Engine-Lib
# from App_Function_Libraries import Local-Summarization-Lib
# from App_Function_Libraries import Summarization-General-Lib
# from App_Function_Libraries import System-Checks-Lib
# from App_Function_Libraries import Tokenization-Methods-Lib
# from App_Function_Libraries import Video-DL-Ingestion-Lib
# from App_Function_Libraries import Web-UI-Lib


# 3rd-Party Module Imports
from bs4 import BeautifulSoup
import gradio as gr
import nltk
from playwright.async_api import async_playwright
import requests
from requests.exceptions import RequestException
import trafilatura
import yt_dlp

# OpenAI Tokenizer support
from openai import OpenAI
from tqdm import tqdm
import tiktoken

# Other Tokenizers
from transformers import GPT2Tokenizer


#######################
# Logging Setup
#

log_level = "DEBUG"
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

#
#
#######################

#######################
# Function Sections
#

abc_xyz = """
    Database Setup
    Config Loading
    System Checks
    DataBase Functions
    Processing Paths and local file handling
    Video Download/Handling
    Audio Transcription
    Diarization
    Chunking-related Techniques & Functions
    Tokenization-related Techniques & Functions
    Summarizers
    Gradio UI
    Main
"""

#
#
#######################


#######################
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
#######################


#######################
# Random issues I've encountered and how I solved them:
#   1. Something about cuda nn library missing, even though cuda is installed...
#       https://github.com/tensorflow/tensorflow/issues/54784 - Basically, installing zlib made it go away. idk.
#       Or https://github.com/SYSTRAN/faster-whisper/issues/85
#
#   2. ERROR: Could not install packages due to an OSError: [WinError 2] The system cannot find the file specified: 'C:\\Python312\\Scripts\\dateparser-download.exe' -> 'C:\\Python312\\Scripts\\dateparser-download.exe.deleteme'
#       Resolved through adding --user to the pip install command
#
#   3. ?
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

openrouter_api_key = config.get('Local-API', 'openrouter', fallback=None)
logging.debug(f"Loaded OpenRouter API Key: {openrouter_api_key}")

# Models
anthropic_model = config.get('API', 'anthropic_model', fallback='claude-3-sonnet-20240229')
cohere_model = config.get('API', 'cohere_model', fallback='command-r-plus')
groq_model = config.get('API', 'groq_model', fallback='llama3-70b-8192')
openai_model = config.get('API', 'openai_model', fallback='gpt-4-turbo')
huggingface_model = config.get('API', 'huggingface_model', fallback='CohereForAI/c4ai-command-r-plus')
openrouter_model = config.get('API', 'openrouter_model', fallback='microsoft/wizardlm-2-8x22b')

# Local-Models
kobold_api_IP = config.get('Local-API', 'kobold_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
kobold_api_key = config.get('Local-API', 'kobold_api_key', fallback='')

llama_api_IP = config.get('Local-API', 'llama_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
llama_api_key = config.get('Local-API', 'llama_api_key', fallback='')

ooba_api_IP = config.get('Local-API', 'ooba_api_IP', fallback='http://127.0.0.1:5000/v1/chat/completions')
ooba_api_key = config.get('Local-API', 'ooba_api_key', fallback='')

tabby_api_IP = config.get('Local-API', 'tabby_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
tabby_api_key = config.get('Local-API', 'tabby_api_key', fallback=None)

vllm_api_url = config.get('Local-API', 'vllm_api_IP', fallback='http://127.0.0.1:500/api/v1/chat/completions')
vllm_api_key = config.get('Local-API', 'vllm_api_key', fallback=None)

# Chunk settings for timed chunking summarization
DEFAULT_CHUNK_DURATION = config.getint('Settings', 'chunk_duration', fallback='30')
WORDS_PER_SECOND = config.getint('Settings', 'words_per_second', fallback='3')

# Retrieve output paths from the configuration file
output_path = config.get('Paths', 'output_path', fallback='results')

# Retrieve processing choice from the configuration file
processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')

# Log file
# logging.basicConfig(filename='debug-runtime.log', encoding='utf-8', level=logging.DEBUG)

#
#
#######################


#######################
# System Startup Notice
#

# Dirty hack - sue me. - FIXME - fix this...
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
time.sleep(1)

#
#
#######################


#######################
# System Check Functions
#
# 1. platform_check()
# 2. cuda_check()
# 3. decide_cpugpu()
# 4. check_ffmpeg()
# 5. download_ffmpeg()
#
#######################


#######################
# DB Functions
#
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
#
########################################################################################################################


########################################################################################################################
# Processing Paths and local file handling
#
# Function List
# 1. read_paths_from_file(file_path)
# 2. process_path(path)
# 3. process_local_file(file_path)
# 4. read_paths_from_file(file_path: str) -> List[str]
#
#
########################################################################################################################


#######################################################################################################################
# Online Article Extraction / Handling
#
# Function List
# 1. get_page_title(url)
# 2. get_article_text(url)
# 3. get_article_title(article_url_arg)
#
#
#######################################################################################################################


#######################################################################################################################
# Video Download/Handling
# Video-DL-Ingestion-Lib
#
# Function List
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
#######################################################################################################################


#######################################################################################################################
# Audio Transcription
#
# Function List
# 1. convert_to_wav(video_file_path, offset=0, overwrite=False)
# 2. speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False)
#
#
#######################################################################################################################


#######################################################################################################################
# Diarization
#
# Function List 1. speaker_diarize(video_file_path, segments, embedding_model = "pyannote/embedding",
#                                   embedding_size=512, num_speakers=0)
#
#
#######################################################################################################################


#######################################################################################################################
# Chunking-related Techniques & Functions
#
#
# FIXME
#
#
#######################################################################################################################


#######################################################################################################################
# Tokenization-related Functions
#
#

# FIXME

#
#
#######################################################################################################################


#######################################################################################################################
# Website-related Techniques & Functions
#
#

#
#
#######################################################################################################################


#######################################################################################################################
# Summarizers
#
# Function List
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. summarize_with_openai(api_key, file_path, custom_prompt_arg)
# 3. summarize_with_claude(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5)
# 4. summarize_with_cohere(api_key, file_path, model, custom_prompt_arg)
# 5. summarize_with_groq(api_key, file_path, model, custom_prompt_arg)
#
#################################
# Local Summarization
#
# Function List
#
# 1. summarize_with_local_llm(file_path, custom_prompt_arg)
# 2. summarize_with_llama(api_url, file_path, token, custom_prompt)
# 3. summarize_with_kobold(api_url, file_path, kobold_api_token, custom_prompt)
# 4. summarize_with_oobabooga(api_url, file_path, ooba_api_token, custom_prompt)
# 5. summarize_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg)
# 6. summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt)
# 7. save_summary_to_file(summary, file_path)
#
#######################################################################################################################


#######################################################################################################################
# Summarization with Detail
#

# FIXME - see 'Old_Chunking_Lib.py'

#
#
#######################################################################################################################


#######################################################################################################################
# Gradio UI
#
# Function List
# 1. summarize_with_huggingface(api_key, file_path, custom_prompt_arg)
# 2. format_transcription(transcription_result)
# 3. format_file_path(file_path, fallback_path=None)
# 4. search_media(query, fields, keyword, page)
# 5. ask_question(transcription, question, api_name, api_key)
# 6. launch_ui(demo_mode=False)
#
#
#######################################################################################################################


#######################################################################################################################
# Local LLM Setup / Running
#
# Function List
# 1. download_latest_llamafile(repo, asset_name_prefix, output_filename)
# 2. download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5)
# 3. verify_checksum(file_path, expected_checksum)
# 4. cleanup_process()
# 5. signal_handler(sig, frame)
# 6. local_llm_function()
# 7. launch_in_new_terminal_windows(executable, args)
# 8. launch_in_new_terminal_linux(executable, args)
# 9. launch_in_new_terminal_mac(executable, args)
#
#
#######################################################################################################################


#######################################################################################################################
# Main()
#

def main(input_path, api_name=None, api_key=None,
         num_speakers=2,
         whisper_model="small.en",
         offset=0,
         vad_filter=False,
         download_video_flag=False,
         custom_prompt=None,
         overwrite=False,
         rolling_summarization=False,
         detail=0.01,
         keywords=None,
         chunk_summarization=False,
         chunk_duration=None,
         words_per_second=None,
         llm_model=None,
         time_based=False):
    global detail_level_number, summary, audio_file

    global detail_level, summary, audio_file

    detail_level = detail

    print(f"Keywords: {keywords}")

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
                        # FIXME - figure something out for handling this situation....
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

                # FIXME - Add input for model name for tabby and vllm

                if rolling_summarization:
                    logging.info("MAIN: Rolling Summarization")

                    # Extract the text from the segments
                    text = extract_text_from_segments(segments)

                    # Set the json_file_path
                    json_file_path = audio_file.replace('.wav', '.segments.json')

                    # Perform rolling summarization
                    summary = summarize_with_detail_openai(text, detail=detail_level, verbose=False)

                    # Handle the summarized output
                    if summary:
                        transcription_result['summary'] = summary
                        logging.info("MAIN: Rolling Summarization successful.")
                        save_summary_to_file(summary, json_file_path)
                    else:
                        logging.warning("MAIN: Rolling Summarization failed.")

                # FIXME - fucking mess of a function.
                # # Time-based Summarization
                # elif args.time_based:
                #     logging.info("MAIN: Time-based Summarization")
                #     global time_based_value
                #     time_based_value = args.time_based
                #     # Set the json_file_path
                #     json_file_path = audio_file.replace('.wav', '.segments.json')
                #
                #     # Perform time-based summarization
                #     summary = time_chunk_summarize(api_name, api_key, segments, args.time_based, custom_prompt,
                #                                    llm_model)
                #
                #     # Handle the summarized output
                #     if summary:
                #         transcription_result['summary'] = summary
                #         logging.info("MAIN: Time-based Summarization successful.")
                #         save_summary_to_file(summary, json_file_path)
                #     else:
                #         logging.warning("MAIN: Time-based Summarization failed.")

                # Perform chunk summarization - FIXME
                elif chunk_summarization:
                    logging.info("MAIN: Chunk Summarization")

                    # Set the json_file_path
                    json_file_path = audio_file.replace('.wav', '.segments.json')

                    # Perform chunk summarization
                    summary = summarize_chunks(api_name, api_key, segments, chunk_duration, words_per_second)

                    # Handle the summarized output
                    if summary:
                        transcription_result['summary'] = summary
                        logging.info("MAIN: Chunk Summarization successful.")
                        save_summary_to_file(summary, json_file_path)
                    else:
                        logging.warning("MAIN: Chunk Summarization failed.")
                # Perform summarization based on the specified API
                elif api_name:
                    logging.debug(f"MAIN: Summarization being performed by {api_name}")
                    json_file_path = audio_file.replace('.wav', '.segments.json')
                    if api_name.lower() == 'openai':
                        openai_api_key = api_key if api_key else config.get('API', 'openai_api_key',
                                                                            fallback=None)
                        try:
                            logging.debug(f"MAIN: trying to summarize with openAI")
                            summary = summarize_with_openai(openai_api_key, json_file_path, custom_prompt)
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
                    elif api_name.lower() == "openrouter":
                        openrouter_api_key = api_key if api_key else config.get('API', 'openrouter_api_key', fallback=None)
                        try:
                            logging.debug(f"MAIN: Trying to summarize with OpenRouter")
                            summary = summarize_with_openrouter(openrouter_api_key, json_file_path, custom_prompt)
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
                    elif api_name.lower() == "tabbyapi":
                        tabbyapi_key = api_key if api_key else config.get('API', 'tabby_api_key', fallback=None)
                        tabbyapi_ip = tabby_api_IP
                        try:
                            logging.debug(f"MAIN: Trying to summarize with tabbyapi")
                            tabby_model = llm_model
                            summary = summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, json_file_path, tabby_model,
                                                              custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "vllm":
                        logging.debug(f"MAIN: Trying to summarize with VLLM")
                        summary = summarize_with_vllm(vllm_api_url, vllm_api_key, llm_model, json_file_path,
                                                      custom_prompt)
                    elif api_name.lower() == "local-llm":
                        logging.debug(f"MAIN: Trying to summarize with the local LLM, Mistral Instruct v0.2")
                        local_llm_url = "http://127.0.0.1:8080"
                        summary = summarize_with_local_llm(json_file_path, custom_prompt)
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

                # Add media to the database
                add_media_with_keywords(
                    url=path,
                    title=info_dict.get('title', 'Untitled'),
                    media_type='video',
                    content=' '.join([segment['text'] for segment in segments]),
                    keywords=','.join(keywords),
                    prompt=custom_prompt or 'No prompt provided',
                    summary=summary or 'No summary provided',
                    transcription_model=whisper_model,
                    author=info_dict.get('uploader', 'Unknown'),
                    ingestion_date=datetime.now().strftime('%Y-%m-%d')
                )

        except Exception as e:
            logging.error(f"Error processing {path}: {str(e)}")
            logging.error(str(e))
            continue
        # end_time = time.monotonic()
        # print("Total program execution time: " + timedelta(seconds=end_time - start_time))

    return results

def signal_handler(signal, frame):
    logging.info('Signal received, exiting...')
    sys.exit(0)

############################## MAIN ##############################
#
#

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # Establish logging baseline
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(
        description='Transcribe and summarize videos.',
        epilog='''
Sample commands:
    1. Simple Sample command structure:
        summarize.py <path_to_video> -api openai -k tag_one tag_two tag_three

    2. Rolling Summary Sample command structure:
        summarize.py <path_to_video> -api openai -prompt "custom_prompt_goes_here-is-appended-after-transcription" -roll -detail 0.01 -k tag_one tag_two tag_three

    3. FULL Sample command structure:
        summarize.py <path_to_video> -api openai -ns 2 -wm small.en -off 0 -vad -log INFO -prompt "custom_prompt" -overwrite -roll -detail 0.01 -k tag_one tag_two tag_three

    4. Sample command structure for UI:
        summarize.py -gui -log DEBUG
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )
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
                                                                      'defines the chunk  size.\n Default is 0.01(lots '
                                                                      'of chunks) -> 1.00 (few chunks)\n Currently '
                                                                      'only OpenAI works. ',
                        default=0.01, )
    # FIXME - This or time based...
    parser.add_argument('--chunk_duration', type=int, default=DEFAULT_CHUNK_DURATION,
                        help='Duration of each chunk in seconds')
    # FIXME - This or chunk_duration.... -> Maybe both???
    parser.add_argument('-time', '--time_based', type=int,
                        help='Enable time-based summarization and specify the chunk duration in seconds (minimum 60 seconds, increments of 30 seconds)')
    parser.add_argument('-model', '--llm_model', type=str, default='',
                        help='Model to use for LLM summarization (only used for vLLM/TabbyAPI)')
    parser.add_argument('-k', '--keywords', nargs='+', default=['cli_ingest_no_tag'],
                        help='Keywords for tagging the media, can use multiple separated by spaces (default: cli_ingest_no_tag)')
    parser.add_argument('--log_file', type=str, help='Where to save logfile (non-default)')
    parser.add_argument('--local_llm', action='store_true', help="Use a local LLM from the script(Downloads llamafile from github and 'mistral-7b-instruct-v0.2.Q8' - 8GB model from Huggingface)")
    parser.add_argument('--server_mode', action='store_true', help='Run in server mode (This exposes the GUI/Server to the network)')
    parser.add_argument('--share_public', type=int, default=7860, help="This will use Gradio's built-in ngrok tunneling to share the server publicly on the internet. Specify the port to use (default: 7860)")
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    # parser.add_argument('-o', '--output_path', type=str, help='Path to save the output file')

    args = parser.parse_args()
    share_public = args.share_public
    server_mode = args.server_mode
    server_port = args.port

    ########## Logging setup
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.log_level))

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, args.log_level))
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    if args.log_file:
        # Create file handler
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(getattr(logging, args.log_level))
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file created at: {args.log_file}")

    ########## Custom Prompt setup
    custom_prompt = args.custom_prompt

    if custom_prompt is None or custom_prompt == "":
        logging.debug("No custom prompt defined, will use default")
        args.custom_prompt = ("\n\nabove is the transcript of a video "
                              "Please read through the transcript carefully. Identify the main topics that are "
                              "discussed over the course of the transcript. Then, summarize the key points about each "
                              "main topic in a concise bullet point. The bullet points should cover the key "
                              "information conveyed about each topic in the video, but should be much shorter than "
                              "the full transcript. Please output your bullet point summary inside <bulletpoints> "
                              "tags.")
        custom_prompt = args.custom_prompt
        print("No custom prompt defined, will use default")
    else:
        logging.debug(f"Custom prompt defined, will use \n\nf{custom_prompt} \n\nas the prompt")
        print(f"Custom Prompt has been defined. Custom prompt: \n\n {args.custom_prompt}")

    # Check if the user wants to use the local LLM from the script
    local_llm = args.local_llm
    logging.info(f'Local LLM flag: {local_llm}')

    if args.user_interface:
        if local_llm:
            local_llm_function()
            time.sleep(2)
            webbrowser.open_new_tab('http://127.0.0.1:7860')
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
        logging.info(f'Log Level: {args.log_level}')
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
        if args.detail_level == None:
            args.detail_level = 0.01
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

        llm_model = args.llm_model or None


        try:
            results = main(args.input_path, api_name=args.api_name,
                           api_key=args.api_key,
                           num_speakers=args.num_speakers,
                           whisper_model=args.whisper_model,
                           offset=args.offset,
                           vad_filter=args.vad_filter,
                           download_video_flag=args.video,
                           custom_prompt=args.custom_prompt,
                           overwrite=args.overwrite,
                           rolling_summarization=args.rolling_summarization,
                           detail=args.detail_level,
                           keywords=args.keywords,
                           chunk_summarization=False,
                           chunk_duration=None,
                           words_per_second=None,
                           llm_model=args.llm_model,
                           time_based=args.time_based)

            logging.info('Transcription process completed.')
            atexit.register(cleanup_process)
        except Exception as e:
            logging.error('An error occurred during the transcription process.')
            logging.error(str(e))
            sys.exit(1)

        finally:
            cleanup_process()
