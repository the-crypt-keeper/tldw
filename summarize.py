#!/usr/bin/env python3
# Std Lib Imports
import argparse
import asyncio
import atexit
import configparser
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
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
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import webbrowser
import zipfile

# Local Module Imports (Libraries specific to this project)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'App_Function_Libraries')))
from App_Function_Libraries import *
from App_Function_Libraries.Web_UI_Lib import *
from App_Function_Libraries.Article_Extractor_Lib import *
from App_Function_Libraries.Article_Summarization_Lib import *
from App_Function_Libraries.Audio_Transcription_Lib import *
from App_Function_Libraries.Audio_Transcription_Lib import convert_to_wav
from App_Function_Libraries.Chunk_Lib import *
from App_Function_Libraries.Diarization_Lib import *
from App_Function_Libraries.Local_File_Processing_Lib import *
from App_Function_Libraries.Local_LLM_Inference_Engine_Lib import *
from App_Function_Libraries.Local_Summarization_Lib import *
from App_Function_Libraries.Summarization_General_Lib import *
from App_Function_Libraries.System_Checks_Lib import *
from App_Function_Libraries.Tokenization_Methods_Lib import *
from App_Function_Libraries.Video_DL_Ingestion_Lib import *
from App_Function_Libraries.Video_DL_Ingestion_Lib import normalize_title
# from App_Function_Libraries.Web_UI_Lib import *

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

#############
# Global variables setup

custom_prompt_input = ("Above is the transcript of a video. Please read through the transcript carefully. Identify the "
"main topics that are discussed over the course of the transcript. Then, summarize the key points about each main "
"topic in bullet points. The bullet points should cover the key information conveyed about each topic in the video, "
"but should be much shorter than the full transcript. Please output your bullet point summary inside <bulletpoints> "
"tags.")

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

def load_comprehensive_config():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the config file in the current directory
    config_path = os.path.join(current_dir, 'config.txt')

    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    files_read = config.read(config_path)

    if not files_read:
        raise FileNotFoundError(f"Config file not found at {config_path}")

    return config


def load_and_log_configs():
    try:
        config = load_comprehensive_config()
        if config is None:
            logging.error("Config is None, cannot proceed")
            return None
        # API Keys
        anthropic_api_key = config.get('API', 'anthropic_api_key', fallback=None)
        logging.debug(
            f"Loaded Anthropic API Key: {anthropic_api_key[:5]}...{anthropic_api_key[-5:] if anthropic_api_key else None}")

        cohere_api_key = config.get('API', 'cohere_api_key', fallback=None)
        logging.debug(
            f"Loaded Cohere API Key: {cohere_api_key[:5]}...{cohere_api_key[-5:] if cohere_api_key else None}")

        groq_api_key = config.get('API', 'groq_api_key', fallback=None)
        logging.debug(f"Loaded Groq API Key: {groq_api_key[:5]}...{groq_api_key[-5:] if groq_api_key else None}")

        openai_api_key = config.get('API', 'openai_api_key', fallback=None)
        logging.debug(
            f"Loaded OpenAI API Key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else None}")

        huggingface_api_key = config.get('API', 'huggingface_api_key', fallback=None)
        logging.debug(
            f"Loaded HuggingFace API Key: {huggingface_api_key[:5]}...{huggingface_api_key[-5:] if huggingface_api_key else None}")

        openrouter_api_key = config.get('API', 'openrouter_api_key', fallback=None)
        logging.debug(
            f"Loaded OpenRouter API Key: {openrouter_api_key[:5]}...{openrouter_api_key[-5:] if openrouter_api_key else None}")

        deepseek_api_key = config.get('API', 'deepseek_api_key', fallback=None)
        logging.debug(
            f"Loaded DeepSeek API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:] if deepseek_api_key else None}")

        # Models
        anthropic_model = config.get('API', 'anthropic_model', fallback='claude-3-sonnet-20240229')
        cohere_model = config.get('API', 'cohere_model', fallback='command-r-plus')
        groq_model = config.get('API', 'groq_model', fallback='llama3-70b-8192')
        openai_model = config.get('API', 'openai_model', fallback='gpt-4-turbo')
        huggingface_model = config.get('API', 'huggingface_model', fallback='CohereForAI/c4ai-command-r-plus')
        openrouter_model = config.get('API', 'openrouter_model', fallback='microsoft/wizardlm-2-8x22b')
        deepseek_model = config.get('API', 'deepseek_model', fallback='deepseek-chat')

        logging.debug(f"Loaded Anthropic Model: {anthropic_model}")
        logging.debug(f"Loaded Cohere Model: {cohere_model}")
        logging.debug(f"Loaded Groq Model: {groq_model}")
        logging.debug(f"Loaded OpenAI Model: {openai_model}")
        logging.debug(f"Loaded HuggingFace Model: {huggingface_model}")
        logging.debug(f"Loaded OpenRouter Model: {openrouter_model}")

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

        logging.debug(f"Loaded Kobold API IP: {kobold_api_IP}")
        logging.debug(f"Loaded Llama API IP: {llama_api_IP}")
        logging.debug(f"Loaded Ooba API IP: {ooba_api_IP}")
        logging.debug(f"Loaded Tabby API IP: {tabby_api_IP}")
        logging.debug(f"Loaded VLLM API URL: {vllm_api_url}")

        # Retrieve output paths from the configuration file
        output_path = config.get('Paths', 'output_path', fallback='results')
        logging.debug(f"Output path set to: {output_path}")

        # Retrieve processing choice from the configuration file
        processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')
        logging.debug(f"Processing choice set to: {processing_choice}")

        # Prompts - FIXME
        prompt_path = config.get('Prompts', 'prompt_path', fallback='prompts.db')

        return {
            'api_keys': {
                'anthropic': anthropic_api_key,
                'cohere': cohere_api_key,
                'groq': groq_api_key,
                'openai': openai_api_key,
                'huggingface': huggingface_api_key,
                'openrouter': openrouter_api_key,
                'deepseek': deepseek_api_key
            },
            'models': {
                'anthropic': anthropic_model,
                'cohere': cohere_model,
                'groq': groq_model,
                'openai': openai_model,
                'huggingface': huggingface_model,
                'openrouter': openrouter_model,
                'deepseek': deepseek_model
            },
            'local_apis': {
                'kobold': {'ip': kobold_api_IP, 'key': kobold_api_key},
                'llama': {'ip': llama_api_IP, 'key': llama_api_key},
                'ooba': {'ip': ooba_api_IP, 'key': ooba_api_key},
                'tabby': {'ip': tabby_api_IP, 'key': tabby_api_key},
                'vllm': {'ip': vllm_api_url, 'key': vllm_api_key}
            },
            'output_path': output_path,
            'processing_choice': processing_choice
        }

    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return None



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

whisper_models = ["small", "medium", "small.en", "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3",
                  "distil-large-v2", "distil-medium.en", "distil-small.en", "distil-large-v3"]
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


def print_hello():
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
    return


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
# 10. process_url(url, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video, download_audio, rolling_summarization, detail_level, question_box, keywords, ) # FIXME - UPDATE
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
# 3. summarize_with_anthropic(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5)
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
#######################################################################################################################
# Function Definitions
#

# Only to be used when configured with Gradio for HF Space

# New
def format_transcription(content):
    # Add extra space after periods for better readability
    content = content.replace('.', '. ').replace('.  ', '. ')
    # Split the content into lines for multiline display; assuming simple logic here
    lines = content.split('. ')
    # Join lines with HTML line break for better presentation in HTML
    formatted_content = "<br>".join(lines)
    return formatted_content

# Old
# def format_transcription(transcription_text_arg):
#     if transcription_text_arg:
#         json_data = transcription_text_arg['transcription']
#         return json.dumps(json_data, indent=2)
#     else:
#         return ""


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


#######################################################

def display_details(media_id):
    # Gradio Search Function-related stuff
    if media_id:
        details = display_item_details(media_id)
        details_html = ""
        for detail in details:
            details_html += f"<h4>Prompt:</h4><p>{detail[0]}</p>"
            details_html += f"<h4>Summary:</h4><p>{detail[1]}</p>"
            details_html += f"<h4>Transcription:</h4><pre>{detail[2]}</pre><hr>"
        return details_html
    return "No details available."


def fetch_items_by_title_or_url(search_query: str, search_type: str):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if search_type == 'Title':
                cursor.execute("SELECT id, title, url FROM Media WHERE title LIKE ?", (f'%{search_query}%',))
            elif search_type == 'URL':
                cursor.execute("SELECT id, title, url FROM Media WHERE url LIKE ?", (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by {search_type}: {e}")


def fetch_items_by_keyword(search_query: str):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.id, m.title, m.url
                FROM Media m
                JOIN MediaKeywords mk ON m.id = mk.media_id
                JOIN Keywords k ON mk.keyword_id = k.id
                WHERE k.keyword LIKE ?
            """, (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by keyword: {e}")


def fetch_items_by_content(search_query: str):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, url FROM Media WHERE content LIKE ?", (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by content: {e}")


def fetch_item_details(media_id: int):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT prompt, summary FROM MediaModifications WHERE media_id = ?", (media_id,))
            prompt_summary_results = cursor.fetchall()

            cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
            content_result = cursor.fetchone()
            content = content_result[0] if content_result else ""

            return prompt_summary_results, content
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching item details: {e}")


def browse_items(search_query, search_type):
    if search_type == 'Keyword':
        results = fetch_items_by_keyword(search_query)
    elif search_type == 'Content':
        results = fetch_items_by_content(search_query)
    else:
        results = fetch_items_by_title_or_url(search_query, search_type)
    return results


def display_item_details(media_id):
    # Function to display item details
    prompt_summary_results, content = fetch_item_details(media_id)
    content_section = f"<h4>Transcription:</h4><pre>{content}</pre><hr>"
    prompt_summary_section = ""
    for prompt, summary in prompt_summary_results:
        prompt_summary_section += f"<h4>Prompt:</h4><p>{prompt}</p>"
        prompt_summary_section += f"<h4>Summary:</h4><p>{summary}</p><hr>"
    return prompt_summary_section, content_section


def update_dropdown(search_query, search_type):
    # Function to update the dropdown choices
    results = browse_items(search_query, search_type)
    item_options = [f"{item[1]} ({item[2]})" for item in results]
    item_mapping = {f"{item[1]} ({item[2]})": item[0] for item in results}  # Map item display to media ID
    return gr.update(choices=item_options), item_mapping



def get_media_id(selected_item, item_mapping):
    return item_mapping.get(selected_item)


def update_detailed_view(item, item_mapping):
    # Function to update the detailed view based on selected item
    if item:
        item_id = item_mapping.get(item)
        if item_id:
            prompt_summary_results, content = fetch_item_details(item_id)
            if prompt_summary_results:
                details_html = "<h4>Details:</h4>"
                for prompt, summary in prompt_summary_results:
                    details_html += f"<h4>Prompt:</h4>{prompt}</p>"
                    details_html += f"<h4>Summary:</h4>{summary}</p>"
                # Format the transcription content for better readability
                content_html = f"<h4>Transcription:</h4><div style='white-space: pre-wrap;'>{format_transcription(content)}</div>"
                return details_html, content_html
            else:
                return "No details available.", "No details available."
        else:
            return "No item selected", "No item selected"
    else:
        return "No item selected", "No item selected"


def format_content(content):
    # Format content using markdown
    formatted_content = f"```\n{content}\n```"
    return formatted_content


def update_prompt_dropdown():
    prompt_names = list_prompts()
    return gr.update(choices=prompt_names)


def display_prompt_details(selected_prompt):
    if selected_prompt:
        details = fetch_prompt_details(selected_prompt)
        if details:
            details_str = f"<h4>Details:</h4><p>{details[0]}</p>"
            system_str = f"<h4>System:</h4><p>{details[1]}</p>"
            user_str = f"<h4>User:</h4><p>{details[2]}</p>" if details[2] else ""
            return details_str + system_str + user_str
    return "No details available."


def insert_prompt_to_db(title, description, system_prompt, user_prompt):
    try:
        conn = sqlite3.connect('prompts.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Prompts (name, details, system, user) VALUES (?, ?, ?, ?)",
            (title, description, system_prompt, user_prompt)
        )
        conn.commit()
        conn.close()
        return "Prompt added successfully!"
    except sqlite3.Error as e:
        return f"Error adding prompt: {e}"


def display_search_results(query):
    if not query.strip():
        return "Please enter a search query."

    results = search_prompts(query)

    # Debugging: Print the results to the console to see what is being returned
    print(f"Processed search results for query '{query}': {results}")

    if results:
        result_md = "## Search Results:\n"
        for result in results:
            # Debugging: Print each result to see its format
            print(f"Result item: {result}")

            if len(result) == 2:
                name, details = result
                result_md += f"**Title:** {name}\n\n**Description:** {details}\n\n---\n"
            else:
                result_md += "Error: Unexpected result format.\n\n---\n"
        return result_md
    return "No results found."


#
# End of Gradio Search Function-related stuff
############################################################


# def gradio UI
def launch_ui(demo_mode=False):
    whisper_models = ["small", "medium", "small.en", "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3",
                      "distil-large-v2", "distil-medium.en", "distil-small.en", "distil-large-v3"]
    # Set theme value with https://www.gradio.app/guides/theming-guide - 'theme='
    my_theme = gr.Theme.from_hub("gradio/seafoam")
    global custom_prompt_input
    with gr.Blocks(theme=my_theme) as iface:
        # Tab 1: Video Transcription + Summarization
        with gr.Tab("Video Transcription + Summarization"):

            with gr.Row():
                # Light/Dark mode toggle switch
                theme_toggle = gr.Radio(choices=["Light", "Dark"], value="Light",
                                        label="Light/Dark Mode Toggle (Toggle to change UI color scheme)")

                # UI Mode toggle switch
                ui_frontpage_mode_toggle = gr.Radio(choices=["Simple List", "Advanced List"], value="Simple List",
                                                    label="UI Mode Options Toggle(Toggle to show a few/all options)")

                # Add the new toggle switch
                chunk_summarization_toggle = gr.Radio(choices=["Non-Chunked", "Chunked-Summarization"],
                                                      value="Non-Chunked",
                                                      label="Summarization Mode")

            # URL input is always visible
            url_input = gr.Textbox(label="URL (Mandatory) --> Playlist URLs will be stripped and only the linked video"
                                         " will be downloaded)", placeholder="Enter the video URL here. Multiple at once supported, one per line")

            # Inputs to be shown or hidden
            num_speakers_input = gr.Number(value=2, label="Number of Speakers(Optional - Currently has no effect)",
                                           visible=False)
            whisper_model_input = gr.Dropdown(choices=whisper_models, value="small.en",
                                              label="Whisper Model(This is the ML model used for transcription.)",
                                              visible=False)
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt (Customize your summarization, or ask a question about the video and have it "
                      "answered)\n Does not work against the summary currently.",
                placeholder="Above is the transcript of a video. Please read "
                            "through the transcript carefully. Identify the main topics that are discussed over the "
                            "course of the transcript. Then, summarize the key points about each main topic in"
                            " bullet points. The bullet points should cover the key information conveyed about "
                            "each topic in the video, but should be much shorter than the full transcript. Please "
                            "output your bullet point summary inside <bulletpoints> tags.",
                lines=3, visible=True)
            offset_input = gr.Number(value=0, label="Offset (Seconds into the video to start transcribing at)",
                                     visible=False)
            api_name_input = gr.Dropdown(
                choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "OpenRouter", "Llama.cpp",
                         "Kobold", "Ooba", "Tabbyapi", "VLLM", "HuggingFace",],
                value=None,
                label="API Name (Mandatory) --> Unless you just want a Transcription", visible=True)
            api_key_input = gr.Textbox(
                label="API Key (Mandatory) --> Unless you're running a local model/server OR have no API selected",
                placeholder="Enter your API key here; Ignore if using Local API or Built-in API('Local-LLM')",
                visible=True)
            vad_filter_input = gr.Checkbox(label="VAD Filter (WIP)", value=False,
                                           visible=False)
            rolling_summarization_input = gr.Checkbox(label="Enable Rolling Summarization", value=False,
                                                      visible=False)
            download_video_input = gr.components.Checkbox(label="Download Video(Select to allow for file download of "
                                                                "selected video)", value=False, visible=False)
            download_audio_input = gr.components.Checkbox(label="Download Audio(Select to allow for file download of "
                                                                "selected Video's Audio)", value=False, visible=False)
            detail_level_input = gr.Slider(minimum=0.01, maximum=1.0, value=0.01, step=0.01, interactive=True,
                                           label="Summary Detail Level (Slide me) (Only OpenAI currently supported)",
                                           visible=False)
            keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated Example: "
                                                                      "tag_one,tag_two,tag_three)",
                                        value="default,no_keyword_set",
                                        visible=True)
            question_box_input = gr.Textbox(label="Question",
                                            placeholder="Enter a question to ask about the transcription",
                                            visible=False)
            # Add the additional input components
            chunk_text_by_words_checkbox = gr.Checkbox(label="Chunk Text by Words", value=False, visible=False)
            max_words_input = gr.Number(label="Max Words", value=300, precision=0, visible=False)

            chunk_text_by_sentences_checkbox = gr.Checkbox(label="Chunk Text by Sentences", value=False,
                                                           visible=False)
            max_sentences_input = gr.Number(label="Max Sentences", value=10, precision=0, visible=False)

            chunk_text_by_paragraphs_checkbox = gr.Checkbox(label="Chunk Text by Paragraphs", value=False,
                                                            visible=False)
            max_paragraphs_input = gr.Number(label="Max Paragraphs", value=5, precision=0, visible=False)

            chunk_text_by_tokens_checkbox = gr.Checkbox(label="Chunk Text by Tokens", value=False, visible=False)
            max_tokens_input = gr.Number(label="Max Tokens", value=1000, precision=0, visible=False)

            inputs = [
                num_speakers_input, whisper_model_input, custom_prompt_input, offset_input, api_name_input,
                api_key_input, vad_filter_input, download_video_input, download_audio_input,
                rolling_summarization_input, detail_level_input, question_box_input, keywords_input,
                chunk_text_by_words_checkbox, max_words_input, chunk_text_by_sentences_checkbox,
                max_sentences_input, chunk_text_by_paragraphs_checkbox, max_paragraphs_input,
                chunk_text_by_tokens_checkbox, max_tokens_input
            ]


            # FIgure out how to check for url vs list of urls

            all_inputs = [url_input] + inputs

            outputs = [
                gr.Textbox(label="Transcription (Resulting Transcription from your input URL)"),
                gr.Textbox(label="Summary or Status Message (Current status of Summary or Summary itself)"),
                gr.File(label="Download Transcription as JSON (Download the Transcription as a file)"),
                gr.File(label="Download Summary as Text (Download the Summary as a file)"),
                gr.File(label="Download Video (Download the Video as a file)", visible=False),
                gr.File(label="Download Audio (Download the Audio as a file)", visible=False),
            ]

            # Function to toggle visibility of advanced inputs
            def toggle_frontpage_ui(mode):
                visible_simple = mode == "Simple List"
                visible_advanced = mode == "Advanced List"

                return [
                    gr.update(visible=True),  # URL input should always be visible
                    gr.update(visible=visible_advanced),  # num_speakers_input
                    gr.update(visible=visible_advanced),  # whisper_model_input
                    gr.update(visible=True),  # custom_prompt_input
                    gr.update(visible=visible_advanced),  # offset_input
                    gr.update(visible=True),  # api_name_input
                    gr.update(visible=True),  # api_key_input
                    gr.update(visible=visible_advanced),  # vad_filter_input
                    gr.update(visible=visible_advanced),  # download_video_input
                    gr.update(visible=visible_advanced),  # download_audio_input
                    gr.update(visible=visible_advanced),  # rolling_summarization_input
                    gr.update(visible_advanced),  # detail_level_input
                    gr.update(visible_advanced),  # question_box_input
                    gr.update(visible=True),  # keywords_input
                    gr.update(visible_advanced),  # chunk_text_by_words_checkbox
                    gr.update(visible_advanced),  # max_words_input
                    gr.update(visible_advanced),  # chunk_text_by_sentences_checkbox
                    gr.update(visible_advanced),  # max_sentences_input
                    gr.update(visible_advanced),  # chunk_text_by_paragraphs_checkbox
                    gr.update(visible_advanced),  # max_paragraphs_input
                    gr.update(visible_advanced),  # chunk_text_by_tokens_checkbox
                    gr.update(visible_advanced),  # max_tokens_input
                ]

            def toggle_chunk_summarization(mode):
                visible = (mode == "Chunked-Summarization")
                return [
                    gr.update(visible=visible),  # chunk_text_by_words_checkbox
                    gr.update(visible=visible),  # max_words_input
                    gr.update(visible=visible),  # chunk_text_by_sentences_checkbox
                    gr.update(visible=visible),  # max_sentences_input
                    gr.update(visible=visible),  # chunk_text_by_paragraphs_checkbox
                    gr.update(visible=visible),  # max_paragraphs_input
                    gr.update(visible=visible),  # chunk_text_by_tokens_checkbox
                    gr.update(visible=visible)  # max_tokens_input
                ]

            chunk_summarization_toggle.change(fn=toggle_chunk_summarization, inputs=chunk_summarization_toggle,
                                              outputs=[
                                                  chunk_text_by_words_checkbox, max_words_input,
                                                  chunk_text_by_sentences_checkbox, max_sentences_input,
                                                  chunk_text_by_paragraphs_checkbox, max_paragraphs_input,
                                                  chunk_text_by_tokens_checkbox, max_tokens_input
                                              ])

            def start_llamafile(*args):
                # Unpack arguments
                (am_noob, verbose_checked, threads_checked, threads_value, http_threads_checked, http_threads_value,
                 model_checked, model_value, hf_repo_checked, hf_repo_value, hf_file_checked, hf_file_value,
                 ctx_size_checked, ctx_size_value, ngl_checked, ngl_value, host_checked, host_value, port_checked,
                 port_value) = args

                # Construct command based on checked values
                command = []
                if am_noob:
                    am_noob = True
                if verbose_checked is not None and verbose_checked:
                    command.append('-v')
                if threads_checked and threads_value is not None:
                    command.extend(['-t', str(threads_value)])
                if http_threads_checked and http_threads_value is not None:
                    command.extend(['--threads', str(http_threads_value)])
                if model_checked is not None and model_value is not None:
                    command.extend(['-m', model_value])
                if hf_repo_checked and hf_repo_value is not None:
                    command.extend(['-hfr', hf_repo_value])
                if hf_file_checked and hf_file_value is not None:
                    command.extend(['-hff', hf_file_value])
                if ctx_size_checked and ctx_size_value is not None:
                    command.extend(['-c', str(ctx_size_value)])
                if ngl_checked and ngl_value is not None:
                    command.extend(['-ngl', str(ngl_value)])
                if host_checked and host_value is not None:
                    command.extend(['--host', host_value])
                if port_checked and port_value is not None:
                    command.extend(['--port', str(port_value)])

                # Code to start llamafile with the provided configuration
                local_llm_gui_function(am_noob, verbose_checked, threads_checked, threads_value,
                                       http_threads_checked, http_threads_value, model_checked,
                                       model_value, hf_repo_checked, hf_repo_value, hf_file_checked,
                                       hf_file_value, ctx_size_checked, ctx_size_value, ngl_checked,
                                       ngl_value, host_checked, host_value, port_checked, port_value, )

                # Example command output to verify
                return f"Command built and ran: {' '.join(command)} \n\nLlamafile started successfully."

            def stop_llamafile():
                # Code to stop llamafile
                # ...
                return "Llamafile stopped"

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

            ui_frontpage_mode_toggle.change(fn=toggle_frontpage_ui, inputs=ui_frontpage_mode_toggle, outputs=inputs)

            # Combine URL input and inputs lists
            all_inputs = [url_input] + inputs

            # lets try embedding the theme here - FIXME?
            # Adding a check in process_url to identify if passed multiple URLs or just one
            gr.Interface(
                fn=process_url,
                inputs=all_inputs,
                outputs=outputs,
                title="Video Transcription and Summarization",
                description="Submit a video URL for transcription and summarization. Ensure you input all necessary "
                            "information including API keys.",
                theme='freddyaboulton/dracula_revamped',
                allow_flagging="never"
            )


        # Tab 2: Transcribe & Summarize Audio file
        with gr.Tab("Audio File Processing"):
            audio_url_input = gr.Textbox(
                label="Audio File URL",
                placeholder="Enter the URL of the audio file"
            )
            audio_file_input = gr.File(label="Upload Audio File", file_types=["audio/*"])
            process_audio_button = gr.Button("Process Audio File")
            audio_progress_output = gr.Textbox(label="Progress")
            audio_transcriptions_output = gr.Textbox(label="Transcriptions")

            process_audio_button.click(
                fn=process_audio_file,
                inputs=[audio_url_input, audio_file_input],
                outputs=[audio_progress_output, audio_transcriptions_output]
            )

        # Tab 3: Scrape & Summarize Articles/Websites
        with gr.Tab("Scrape & Summarize Articles/Websites"):
            url_input = gr.Textbox(
                label="Article URLs",
                placeholder="Enter article URLs here, one per line",
                lines=5
            )
            custom_article_title_input = gr.Textbox(
                label="Custom Article Titles (Optional, one per line)",
                placeholder="Enter custom titles for the articles, one per line",
                lines=5
            )
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt (Optional)",
                placeholder="Provide a custom prompt for summarization",
                lines=3
            )
            api_name_input = gr.Dropdown(
                choices=[None, "huggingface", "deepseek", "openrouter", "openai", "anthropic", "cohere", "groq",
                         "llama", "kobold", "ooba"],
                value=None,
                label="API Name (Mandatory for Summarization)"
            )
            api_key_input = gr.Textbox(
                label="API Key (Mandatory if API Name is specified)",
                placeholder="Enter your API key here; Ignore if using Local API or Built-in API"
            )
            keywords_input = gr.Textbox(
                label="Keywords",
                placeholder="Enter keywords here (comma-separated)",
                value="default,no_keyword_set",
                visible=True
            )

            scrape_button = gr.Button("Scrape and Summarize")
            result_output = gr.Textbox(label="Result", lines=20)

            scrape_button.click(
                scrape_and_summarize_multiple,
                inputs=[url_input, custom_prompt_input, api_name_input, api_key_input, keywords_input,
                        custom_article_title_input],
                outputs=result_output
            )
        # with gr.Tab("Scrape & Summarize Articles/Websites"):
        #     url_input = gr.Textbox(label="Article URL", placeholder="Enter the article URL here")
        #     custom_article_title_input = gr.Textbox(label="Custom Article Title (Optional)",
        #                                             placeholder="Enter a custom title for the article")
        #     custom_prompt_input = gr.Textbox(
        #         label="Custom Prompt (Optional)",
        #         placeholder="Provide a custom prompt for summarization",
        #         lines=3
        #     )
        #     api_name_input = gr.Dropdown(
        #         choices=[None, "huggingface", "deepseek", "openrouter", "openai", "anthropic", "cohere", "groq", "llama", "kobold",
        #                  "ooba"],
        #         value=None,
        #         label="API Name (Mandatory for Summarization)"
        #     )
        #     api_key_input = gr.Textbox(label="API Key (Mandatory if API Name is specified)",
        #                                placeholder="Enter your API key here; Ignore if using Local API or Built-in API")
        #     keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated)",
        #                                 value="default,no_keyword_set", visible=True)
        #
        #     scrape_button = gr.Button("Scrape and Summarize")
        #     result_output = gr.Textbox(label="Result")
        #
        #     scrape_button.click(scrape_and_summarize, inputs=[url_input, custom_prompt_input, api_name_input,
        #                                                       api_key_input, keywords_input,
        #                                                       custom_article_title_input], outputs=result_output)
        #
        #     gr.Markdown("### Or Paste Unstructured Text Below (Will use settings from above)")
        #     text_input = gr.Textbox(label="Unstructured Text", placeholder="Paste unstructured text here", lines=10)
        #     text_ingest_button = gr.Button("Ingest Unstructured Text")
        #     text_ingest_result = gr.Textbox(label="Result")
        #
        #     text_ingest_button.click(ingest_unstructured_text,
        #                              inputs=[text_input, custom_prompt_input, api_name_input, api_key_input,
        #                                      keywords_input, custom_article_title_input], outputs=text_ingest_result)

        # Tab 4: Ingest & Summarize Documents
        with gr.Tab("Ingest & Summarize Documents"):
            gr.Markdown("Plan to put ingestion form for documents here")
            gr.Markdown("Will ingest documents and store into SQLite DB")
            gr.Markdown("RAG here we come....:/")

        # Function to update the visibility of the UI elements for Llamafile Settings
        # def toggle_advanced_llamafile_mode(is_advanced):
        #     if is_advanced:
        #         return [gr.update(visible=True)] * 14
        #     else:
        #         return [gr.update(visible=False)] * 11 + [gr.update(visible=True)] * 3
        # FIXME
        def toggle_advanced_mode(advanced_mode):
            # Show all elements if advanced mode is on
            if advanced_mode:
                return {elem: gr.update(visible=True) for elem in all_elements}
            else:
                # Show only specific elements if advanced mode is off
                return {elem: gr.update(visible=elem in simple_mode_elements) for elem in all_elements}

    # Top-Level Gradio Tab #2 - 'Search / Detailed View'
    with gr.Blocks() as search_interface:
        with gr.Tab("Search Ingested Materials / Detailed Entry View / Prompts"):
            search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
            search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                         label="Search By")

            search_button = gr.Button("Search")
            items_output = gr.Dropdown(label="Select Item", choices=[])
            item_mapping = gr.State({})

            search_button.click(fn=update_dropdown,
                                inputs=[search_query_input, search_type_input],
                                outputs=[items_output, item_mapping]
                                )

            prompt_summary_output = gr.HTML(label="Prompt & Summary", visible=True)
            # FIXME - temp change; see if markdown works nicer...
            content_output = gr.Markdown(label="Content", visible=True)
            items_output.change(fn=update_detailed_view,
                                inputs=[items_output, item_mapping],
                                outputs=[prompt_summary_output, content_output]
                                )
        # sub-tab #2 for Search / Detailed view
        with gr.Tab("View Prompts"):
            with gr.Column():
                prompt_dropdown = gr.Dropdown(
                    label="Select Prompt (Thanks to the 'Fabric' project for this initial set: https://github.com/danielmiessler/fabric",
                    choices=[])
                prompt_details_output = gr.HTML()

                prompt_dropdown.change(
                    fn=display_prompt_details,
                    inputs=prompt_dropdown,
                    outputs=prompt_details_output
                )

                prompt_list_button = gr.Button("List Prompts")
                prompt_list_button.click(
                    fn=update_prompt_dropdown,
                    outputs=prompt_dropdown
                )

        # FIXME
        # Sub-tab #3 for Search / Detailed view
        with gr.Tab("Search Prompts"):
            with gr.Column():
                search_query_input = gr.Textbox(label="Search Query (It's broken)",
                                                placeholder="Enter your search query...")
                search_results_output = gr.Markdown()

                search_button = gr.Button("Search Prompts")
                search_button.click(
                    fn=display_search_results,
                    inputs=[search_query_input],
                    outputs=[search_results_output]
                )

                search_query_input.change(
                    fn=display_search_results,
                    inputs=[search_query_input],
                    outputs=[search_results_output]
                )

        # Sub-tab #4 for Search / Detailed view
        with gr.Tab("Add Prompts"):
            gr.Markdown("### Add Prompt")
            title_input = gr.Textbox(label="Title", placeholder="Enter the prompt title")
            description_input = gr.Textbox(label="Description", placeholder="Enter the prompt description", lines=3)
            system_prompt_input = gr.Textbox(label="System Prompt", placeholder="Enter the system prompt", lines=3)
            user_prompt_input = gr.Textbox(label="User Prompt", placeholder="Enter the user prompt", lines=3)
            add_prompt_button = gr.Button("Add Prompt")
            add_prompt_output = gr.HTML()

            add_prompt_button.click(
                fn=add_prompt,
                inputs=[title_input, description_input, system_prompt_input, user_prompt_input],
                outputs=add_prompt_output
            )

    # Top-Level Gradio Tab #3
    with gr.Blocks() as llamafile_interface:
        with gr.Tab("Llamafile Settings"):
            gr.Markdown("Settings for Llamafile")

            # Toggle switch for Advanced/Simple mode
            am_noob = gr.Checkbox(
                label="Check this to enable sane defaults and then download(if not already downloaded) a model, click 'Start Llamafile' and then go to --> 'Llamafile Chat Interface')\n\n",
                value=False, visible=True)
            advanced_mode_toggle = gr.Checkbox(
                label="Advanced Mode - Enable to show all settings\n\n",
                value=False)

            # Simple mode elements
            model_checked = gr.Checkbox(label="Enable Setting Local LLM Model Path", value=False, visible=True)
            model_value = gr.Textbox(label="Path to Local Model File", value="", visible=True)
            ngl_checked = gr.Checkbox(label="Enable Setting GPU Layers", value=False, visible=True)
            ngl_value = gr.Number(label="Number of GPU Layers", value=None, precision=0, visible=True)

            # Advanced mode elements
            verbose_checked = gr.Checkbox(label="Enable Verbose Output", value=False, visible=False)
            threads_checked = gr.Checkbox(label="Set CPU Threads", value=False, visible=False)
            threads_value = gr.Number(label="Number of CPU Threads", value=None, precision=0, visible=False)
            http_threads_checked = gr.Checkbox(label="Set HTTP Server Threads", value=False, visible=False)
            http_threads_value = gr.Number(label="Number of HTTP Server Threads", value=None, precision=0,
                                           visible=False)
            hf_repo_checked = gr.Checkbox(label="Use Huggingface Repo Model", value=False, visible=False)
            hf_repo_value = gr.Textbox(label="Huggingface Repo Name", value="", visible=False)
            hf_file_checked = gr.Checkbox(label="Set Huggingface Model File", value=False, visible=False)
            hf_file_value = gr.Textbox(label="Huggingface Model File", value="", visible=False)
            ctx_size_checked = gr.Checkbox(label="Set Prompt Context Size", value=False, visible=False)
            ctx_size_value = gr.Number(label="Prompt Context Size", value=8124, precision=0, visible=False)
            host_checked = gr.Checkbox(label="Set IP to Listen On", value=False, visible=False)
            host_value = gr.Textbox(label="Host IP Address", value="", visible=False)
            port_checked = gr.Checkbox(label="Set Server Port", value=False, visible=False)
            port_value = gr.Number(label="Port Number", value=None, precision=0, visible=False)

            # Start and Stop buttons
            start_button = gr.Button("Start Llamafile")
            stop_button = gr.Button("Stop Llamafile")
            output_display = gr.Markdown()

            all_elements = [
                verbose_checked, threads_checked, threads_value, http_threads_checked, http_threads_value,
                model_checked, model_value, hf_repo_checked, hf_repo_value, hf_file_checked, hf_file_value,
                ctx_size_checked, ctx_size_value, ngl_checked, ngl_value, host_checked, host_value, port_checked,
                port_value
            ]

            simple_mode_elements = [model_checked, model_value, ngl_checked, ngl_value]

            advanced_mode_toggle.change(
                fn=toggle_advanced_mode,
                inputs=[advanced_mode_toggle],
                outputs=all_elements
            )

            # Function call with the new inputs
            start_button.click(
                fn=start_llamafile,
                inputs=[am_noob, verbose_checked, threads_checked, threads_value, http_threads_checked,
                        http_threads_value,
                        model_checked, model_value, hf_repo_checked, hf_repo_value, hf_file_checked, hf_file_value,
                        ctx_size_checked, ctx_size_value, ngl_checked, ngl_value, host_checked, host_value,
                        port_checked, port_value],
                outputs=output_display
            )

        # Second sub-tab for Llamafile
        with gr.Tab("Llamafile Chat Interface"):
            gr.Markdown("Page to interact with Llamafile Server (iframe to Llamafile server port)")
            # Define the HTML content with the iframe
            html_content = """
            <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Llama.cpp Server Chat Interface - Loaded from  http://127.0.0.1:8080</title>
                    <style>
                        body, html {
                        height: 100%;
                        margin: 0;
                        padding: 0;
                    }
                    iframe {
                        border: none;
                        width: 85%;
                        height: 85vh; /* Full viewport height */
                    }
                </style>
            </head>
            <body>
                <iframe src="http://127.0.0.1:8080" title="Llama.cpp Server Chat Interface - Loaded from  http://127.0.0.1:8080"></iframe>
            </body>
            </html>
            """
            gr.HTML(html_content)

        # Third sub-tab for Llamafile
        # https://github.com/lmg-anon/mikupad/releases
        with gr.Tab("Mikupad Chat Interface"):
            gr.Markdown("Not implemented. Have to wait until I get rid of Gradio")
            gr.HTML(html_content)

    # Top-Level Gradio Tab #4 - Don't ask me how this is tabbed, but it is... #FIXME
    export_keywords_interface = gr.Interface(
        fn=export_keywords_to_csv,
        inputs=[],
        outputs=[gr.File(label="Download Exported Keywords"), gr.Textbox(label="Status")],
        title="Export Keywords",
        description="Export all keywords in the database to a CSV file."
    )

    # Gradio interface for importing data
    def import_data(file):
        # Placeholder for actual import functionality
        return "Data imported successfully"

    # Top-Level Gradio Tab #5 - Export/Import - Same deal as above, not sure why this is auto-tabbed
    import_interface = gr.Interface(
        fn=import_data,
        inputs=gr.File(label="Upload file for import"),
        outputs="text",
        title="Import Data",
        description="Import data into the database from a CSV file."
    )

    # Top-Level Gradio Tab #6 - Export/Import - Same deal as above, not sure why this is auto-tabbed
    import_export_tab = gr.TabbedInterface(
        [gr.TabbedInterface(
            [gr.Interface(
                fn=export_to_csv,
                inputs=[
                    gr.Textbox(label="Search Query", placeholder="Enter your search query here..."),
                    gr.CheckboxGroup(label="Search Fields", choices=["Title", "Content"], value=["Title"]),
                    gr.Textbox(label="Keyword (Match ALL, can use multiple keywords, separated by ',' (comma) )",
                               placeholder="Enter keywords here..."),
                    gr.Number(label="Page", value=1, precision=0),
                    gr.Number(label="Results per File", value=1000, precision=0)
                ],
                outputs="text",
                title="Export Search Results to CSV",
                description="Export the search results to a CSV file."
            ),
                export_keywords_interface],
            ["Export Search Results", "Export Keywords"]
        ),
            import_interface],
        ["Export", "Import"]
    )

    # Second sub-tab for Keywords tab
    keyword_add_interface = gr.Interface(
        fn=add_keyword,
        inputs=gr.Textbox(label="Add Keywords (comma-separated)", placeholder="Enter keywords here..."),
        outputs="text",
        title="Add Keywords",
        description="Add one, or multiple keywords to the database.",
        allow_flagging="never"
    )

    # Third sub-tab for Keywords tab
    keyword_delete_interface = gr.Interface(
        fn=delete_keyword,
        inputs=gr.Textbox(label="Delete Keyword", placeholder="Enter keyword to delete here..."),
        outputs="text",
        title="Delete Keyword",
        description="Delete a keyword from the database.",
        allow_flagging="never"
    )

    # First sub-tab for Keywords tab
    browse_keywords_interface = gr.Interface(
        fn=keywords_browser_interface,
        inputs=[],
        outputs="markdown",
        title="Browse Keywords",
        description="View all keywords currently stored in the database."
    )

    # Combine the keyword interfaces into a tabbed interface
    # So this is how it works... #FIXME
    keyword_tab = gr.TabbedInterface(
        [browse_keywords_interface, keyword_add_interface, keyword_delete_interface],
        ["Browse Keywords", "Add Keywords", "Delete Keywords"]
    )

    def ensure_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def gradio_download_youtube_video(url):
        """Download video using yt-dlp with specified options."""
        # Determine ffmpeg path based on the operating system.
        ffmpeg_path = './Bin/ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'

        # Extract information about the video
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            sanitized_title = sanitize_filename(info_dict['title'])
            original_ext = info_dict['ext']

        # Setup the final directory and filename
        download_dir = Path(f"results/{sanitized_title}")
        download_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = download_dir / f"{sanitized_title}.{original_ext}"

        # Initialize yt-dlp with generic options and the output template
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'ffmpeg_location': ffmpeg_path,
            'outtmpl': str(output_file_path),
            'noplaylist': True, 'quiet': True
        }

        # Execute yt-dlp to download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Final check to ensure file exists
        if not output_file_path.exists():
            raise FileNotFoundError(f"Expected file was not found: {output_file_path}")

        return str(output_file_path)

    # FIXME - example to use for rest of gradio theming, just stuff in HTML/Markdown
    # <-- set description variable with HTML -->
    desc = "<h3>Youtube Video Downloader</h3><p>This Input takes a Youtube URL as input and creates " \
           "a webm file for you to download. </br><em>If you want a full-featured one:</em> " \
           "<strong><em>https://github.com/StefanLobbenmeier/youtube-dl-gui</strong></em> or <strong><em>https://github.com/yt-dlg/yt-dlg</em></strong></p>"

    # Sixth Top Tab - Download Video/Audio Files
    download_videos_interface = gr.Interface(
        fn=gradio_download_youtube_video,
        inputs=gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL here"),
        outputs=gr.File(label="Download Video"),
        title="YouTube Video Downloader",
        description=desc,
        allow_flagging="never"
    )

    # Combine interfaces into a tabbed interface
    tabbed_interface = gr.TabbedInterface(
        [iface, search_interface, llamafile_interface, keyword_tab, import_export_tab, download_videos_interface],
        ["Transcription / Summarization / Ingestion", "Search / Detailed View",
         "Local LLM with Llamafile", "Keywords", "Export/Import", "Download Video/Audio Files"])

    # Launch the interface
    server_port_variable = 7860
    global server_mode, share_public

    if share_public == True:
        tabbed_interface.launch(share=True, )
    elif server_mode == True and share_public is False:
        tabbed_interface.launch(share=False, server_name="0.0.0.0", server_port=server_port_variable)
    else:
        tabbed_interface.launch(share=False, )
        #tabbed_interface.launch(share=True, )


def clean_youtube_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'list' in query_params:
        query_params.pop('list')
    cleaned_query = urlencode(query_params, doseq=True)
    cleaned_url = urlunparse(parsed_url._replace(query=cleaned_query))
    return cleaned_url

def extract_video_info(url):
    info_dict = get_youtube(url)
    title = info_dict.get('title', 'Untitled')
    return info_dict, title


def download_audio_file(url, save_path):
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))
    if file_size > 500 * 1024 * 1024:  # 500 MB limit
        raise ValueError("File size exceeds the 500MB limit.")
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

def process_audio_file(audio_url, audio_file):
    progress = []
    transcriptions = []

    def update_progress(stage, message):
        progress.append(f"{stage}: {message}")
        return "\n".join(progress), "\n".join(transcriptions)

    try:
        if audio_url:
            # Process audio file from URL
            save_path = Path("downloaded_audio_file.wav")
            download_audio_file(audio_url, save_path)
        elif audio_file:
            # Process uploaded audio file
            audio_file_size = os.path.getsize(audio_file.name)
            if audio_file_size > 500 * 1024 * 1024:  # 500 MB limit
                return update_progress("Error", "File size exceeds the 500MB limit.")
            save_path = Path(audio_file.name)
        else:
            return update_progress("Error", "No audio file provided.")

        # Perform transcription and summarization
        transcription, summary, json_file_path, summary_file_path, _, _ = process_url(
            url=None,
            num_speakers=2,
            whisper_model="small.en",
            custom_prompt_input=None,
            offset=0,
            api_name=None,
            api_key=None,
            vad_filter=False,
            download_video_flag=False,
            download_audio=False,
            rolling_summarization=False,
            detail_level=0.01,
            question_box=None,
            keywords="default,no_keyword_set",
            chunk_text_by_words=False,
            max_words=0,
            chunk_text_by_sentences=False,
            max_sentences=0,
            chunk_text_by_paragraphs=False,
            max_paragraphs=0,
            chunk_text_by_tokens=False,
            max_tokens=0,
            local_file_path=str(save_path)
        )
        transcriptions.append(transcription)
        progress.append("Processing complete.")
    except Exception as e:
        progress.append(f"Error: {str(e)}")

    return "\n".join(progress), "\n".join(transcriptions)


def process_url(
        url,
        num_speakers,
        whisper_model,
        custom_prompt_input,
        offset,
        api_name,
        api_key,
        vad_filter,
        download_video_flag,
        download_audio,
        rolling_summarization,
        detail_level,
        # It's for the asking a question about a returned prompt - needs to be removed #FIXME
        question_box,
        keywords,
        chunk_text_by_words,
        max_words,
        chunk_text_by_sentences,
        max_sentences,
        chunk_text_by_paragraphs,
        max_paragraphs,
        chunk_text_by_tokens,
        max_tokens,
        local_file_path=None
):
    # Handle the chunk summarization options
    set_chunk_txt_by_words = chunk_text_by_words
    set_max_txt_chunk_words = max_words
    set_chunk_txt_by_sentences = chunk_text_by_sentences
    set_max_txt_chunk_sentences = max_sentences
    set_chunk_txt_by_paragraphs = chunk_text_by_paragraphs
    set_max_txt_chunk_paragraphs = max_paragraphs
    set_chunk_txt_by_tokens = chunk_text_by_tokens
    set_max_txt_chunk_tokens = max_tokens

    progress = []
    success_message = "All videos processed successfully. Transcriptions and summaries have been ingested into the database."


    # Validate input
    if not url and not local_file_path:
        return "Process_URL: No URL provided.", "No URL provided.", None, None, None, None, None, None

    # FIXME - Chatgpt again?
    if isinstance(url, str):
        urls = url.strip().split('\n')
        if len(urls) > 1:
            return process_video_urls(urls, num_speakers, whisper_model, custom_prompt_input, offset, api_name, api_key, vad_filter,
                                      download_video_flag, download_audio, rolling_summarization, detail_level, question_box,
                                      keywords, chunk_text_by_words, max_words, chunk_text_by_sentences, max_sentences,
                                      chunk_text_by_paragraphs, max_paragraphs, chunk_text_by_tokens, max_tokens)
        else:
            urls = [url]

    if url and not is_valid_url(url):
        return "Process_URL: Invalid URL format.", "Invalid URL format.", None, None, None, None, None, None

    if url:
        # Clean the URL to remove playlist parameters if any
        url = clean_youtube_url(url)
        logging.info(f"Process_URL: Processing URL: {url}")

    if api_name:
        print("Process_URL: API Name received:", api_name)  # Debugging line

    video_file_path = None
    global info_dict

    # FIXME - need to handle local audio file processing
    # If Local audio file is provided
    if local_file_path:
        try:
            pass
            # # insert code to process local audio file
            # # Need to be able to add a title/author/etc for ingestion into the database
            # # Also want to be able to optionally _just_ ingest it, and not ingest.
            # # FIXME
            # #download_path = create_download_directory(title)
            # #audio_path = download_video(url, download_path, info_dict, download_video_flag)
            #
            # audio_file_path = local_file_path
            # global segments
            # audio_file_path, segments = perform_transcription(audio_file_path, offset, whisper_model, vad_filter)
            #
            # if audio_file_path is None or segments is None:
            #     logging.error("Process_URL: Transcription failed or segments not available.")
            #     return "Process_URL: Transcription failed.", "Transcription failed.", None, None, None, None
            #
            # logging.debug(f"Process_URL: Transcription audio_file: {audio_file_path}")
            # logging.debug(f"Process_URL: Transcription segments: {segments}")
            #
            # transcription_text = {'audio_file': audio_file_path, 'transcription': segments}
            # logging.debug(f"Process_URL: Transcription text: {transcription_text}")
            #
            # if rolling_summarization:
            #     text = extract_text_from_segments(segments)
            #     summary_text = rolling_summarize_function(
            #         transcription_text,
            #         detail=detail_level,
            #         api_name=api_name,
            #         api_key=api_key,
            #         custom_prompt=custom_prompt,
            #         chunk_by_words=chunk_text_by_words,
            #         max_words=max_words,
            #         chunk_by_sentences=chunk_text_by_sentences,
            #         max_sentences=max_sentences,
            #         chunk_by_paragraphs=chunk_text_by_paragraphs,
            #         max_paragraphs=max_paragraphs,
            #         chunk_by_tokens=chunk_text_by_tokens,
            #         max_tokens=max_tokens
            #     )
            # if api_name:
            #     summary_text = perform_summarization(api_name, segments_json_path, custom_prompt, api_key, config)
            #     if summary_text is None:
            #         logging.error("Summary text is None. Check summarization function.")
            #         summary_file_path = None  # Set summary_file_path to None if summary is not generated
            # else:
            #     summary_text = 'Summary not available'
            #     summary_file_path = None  # Set summary_file_path to None if summary is not generated
            #
            # json_file_path, summary_file_path = save_transcription_and_summary(transcription_text, summary_text, download_path)
            #
            # add_media_to_database(url, info_dict, segments, summary_text, keywords, custom_prompt, whisper_model)
            #
            # return transcription_text, summary_text, json_file_path, summary_file_path, None, None

        except Exception as e:
            logging.error(f": {e}")
            return str(e), 'process_url: Error processing the request.', None, None, None, None


    # If URL/Local video file is provided
    try:
        info_dict, title = extract_video_info(url)
        download_path = create_download_directory(title)
        video_path = download_video(url, download_path, info_dict, download_video_flag)
        global segments
        audio_file_path, segments = perform_transcription(video_path, offset, whisper_model, vad_filter)

        if audio_file_path is None or segments is None:
            logging.error("Process_URL: Transcription failed or segments not available.")
            return "Process_URL: Transcription failed.", "Transcription failed.", None, None, None, None

        logging.debug(f"Process_URL: Transcription audio_file: {audio_file_path}")
        logging.debug(f"Process_URL: Transcription segments: {segments}")

        transcription_text = {'audio_file': audio_file_path, 'transcription': segments}
        logging.debug(f"Process_URL: Transcription text: {transcription_text}")

        if rolling_summarization:
            text = extract_text_from_segments(segments)
            summary_text = rolling_summarize_function(
                transcription_text,
                detail=detail_level,
                api_name=api_name,
                api_key=api_key,
                custom_prompt_input=custom_prompt_input,
                chunk_by_words=chunk_text_by_words,
                max_words=max_words,
                chunk_by_sentences=chunk_text_by_sentences,
                max_sentences=max_sentences,
                chunk_by_paragraphs=chunk_text_by_paragraphs,
                max_paragraphs=max_paragraphs,
                chunk_by_tokens=chunk_text_by_tokens,
                max_tokens=max_tokens
            )
        if api_name:
            summary_text = perform_summarization(api_name, segments_json_path, custom_prompt_input, api_key)
            if summary_text is None:
                logging.error("Summary text is None. Check summarization function.")
                summary_file_path = None  # Set summary_file_path to None if summary is not generated
        else:
            summary_text = 'Summary not available'
            summary_file_path = None  # Set summary_file_path to None if summary is not generated

        json_file_path, summary_file_path = save_transcription_and_summary(transcription_text, summary_text, download_path)

        add_media_to_database(url, info_dict, segments, summary_text, keywords, custom_prompt_input, whisper_model)

        return transcription_text, summary_text, json_file_path, summary_file_path, None, None

    except Exception as e:
        logging.error(f": {e}")
        return str(e), 'process_url: Error processing the request.', None, None, None, None

# Handle multiple videos as input
# Handle multiple videos as input
def process_video_urls(url_list, num_speakers, whisper_model, custom_prompt_input, offset, api_name, api_key, vad_filter,
                       download_video_flag, download_audio, rolling_summarization, detail_level, question_box,
                       keywords, chunk_text_by_words, max_words, chunk_text_by_sentences, max_sentences,
                       chunk_text_by_paragraphs, max_paragraphs, chunk_text_by_tokens, max_tokens):
    global current_progress
    progress = []  # This must always be a list
    status = []  # This must always be a list

    def update_progress(index, url, message):
        progress.append(f"Processing {index + 1}/{len(url_list)}: {url}")  # Append to list
        status.append(message)  # Append to list
        return "\n".join(progress), "\n".join(status)  # Return strings for display


    for index, url in enumerate(url_list):
        try:
            transcription, summary, json_file_path, summary_file_path, _, _ = process_url(
                url=url,
                num_speakers=num_speakers,
                whisper_model=whisper_model,
                custom_prompt_input=custom_prompt_input,
                offset=offset,
                api_name=api_name,
                api_key=api_key,
                vad_filter=vad_filter,
                download_video_flag=download_video_flag,
                download_audio=download_audio,
                rolling_summarization=rolling_summarization,
                detail_level=detail_level,
                question_box=question_box,
                keywords=keywords,
                chunk_text_by_words=chunk_text_by_words,
                max_words=max_words,
                chunk_text_by_sentences=max_sentences,
                max_sentences=max_sentences,
                chunk_text_by_paragraphs=chunk_text_by_paragraphs,
                max_paragraphs=max_paragraphs,
                chunk_text_by_tokens=chunk_text_by_tokens,
                max_tokens=max_tokens
            )
            # Update progress and transcription properly
            current_progress, current_status = update_progress(index, url, "Video processed and ingested into the database.")
        except Exception as e:
            current_progress, current_status = update_progress(index, url, f"Error: {str(e)}")

    success_message = "All videos have been transcribed, summarized, and ingested into the database successfully."
    return current_progress, success_message, None, None, None, None


# FIXME - Prompt sample box

# Sample data
prompts_category_1 = [
    "What are the key points discussed in the video?",
    "Summarize the main arguments made by the speaker.",
    "Describe the conclusions of the study presented."
]

prompts_category_2 = [
    "How does the proposed solution address the problem?",
    "What are the implications of the findings?",
    "Can you explain the theory behind the observed phenomenon?"
]

all_prompts = prompts_category_1 + prompts_category_2


# Search function
def search_prompts(query):
    filtered_prompts = [prompt for prompt in all_prompts if query.lower() in prompt.lower()]
    return "\n".join(filtered_prompts)


# Handle prompt selection
def handle_prompt_selection(prompt):
    return f"You selected: {prompt}"


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
# Helper Functions for Main() & process_url()
#

def perform_transcription(video_path, offset, whisper_model, vad_filter):
    global segments_json_path
    audio_file_path = convert_to_wav(video_path, offset)
    segments_json_path = audio_file_path.replace('.wav', '.segments.json')

    # Check if segments JSON already exists
    if os.path.exists(segments_json_path):
        logging.info(f"Segments file already exists: {segments_json_path}")
        try:
            with open(segments_json_path, 'r') as file:
                segments = json.load(file)
            if not segments:  # Check if the loaded JSON is empty
                logging.warning(f"Segments JSON file is empty, re-generating: {segments_json_path}")
                raise ValueError("Empty segments JSON file")
            logging.debug(f"Loaded segments from {segments_json_path}")
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to read or parse the segments JSON file: {e}")
            # Remove the corrupted file
            os.remove(segments_json_path)
            # Re-generate the transcription
            logging.info(f"Re-generating transcription for {audio_file_path}")
            audio_file, segments = re_generate_transcription(audio_file_path, whisper_model, vad_filter)
            if segments is None:
                return None, None
    else:
        # Perform speech to text transcription
        audio_file, segments = re_generate_transcription(audio_file_path, whisper_model, vad_filter)

    return audio_file_path, segments


def re_generate_transcription(audio_file_path, whisper_model, vad_filter):
    try:
        segments = speech_to_text(audio_file_path, whisper_model=whisper_model, vad_filter=vad_filter)
        # Save segments to JSON
        segments_json_path = audio_file_path.replace('.wav', '.segments.json')
        with open(segments_json_path, 'w') as file:
            json.dump(segments, file, indent=2)
        logging.debug(f"Transcription segments saved to {segments_json_path}")
        return audio_file_path, segments
    except Exception as e:
        logging.error(f"Error in re-generating transcription: {str(e)}")
        return None, None


def save_transcription_and_summary(transcription_text, summary_text, download_path):
    video_title = sanitize_filename(info_dict.get('title', 'Untitled'))

    json_file_path = os.path.join(download_path, f"{video_title}.segments.json")
    summary_file_path = os.path.join(download_path, f"{video_title}_summary.txt")

    with open(json_file_path, 'w') as json_file:
        json.dump(transcription_text['transcription'], json_file, indent=2)

    if summary_text is not None:
        with open(summary_file_path, 'w') as file:
            file.write(summary_text)
    else:
        logging.warning("Summary text is None. Skipping summary file creation.")
        summary_file_path = None

    return json_file_path, summary_file_path

def add_media_to_database(url, info_dict, segments, summary, keywords, custom_prompt_input, whisper_model):
    content = ' '.join([segment['Text'] for segment in segments if 'Text' in segment])
    add_media_with_keywords(
        url=url,
        title=info_dict.get('title', 'Untitled'),
        media_type='video',
        content=content,
        keywords=','.join(keywords),
        prompt=custom_prompt_input or 'No prompt provided',
        summary=summary or 'No summary provided',
        transcription_model=whisper_model,
        author=info_dict.get('uploader', 'Unknown'),
        ingestion_date=datetime.now().strftime('%Y-%m-%d')
    )


def perform_summarization(api_name, json_file_path, custom_prompt_input, api_key):
    # Load Config
    loaded_config_data = load_and_log_configs()

    if custom_prompt_input is None:
        # FIXME - Setup proper default prompt & extract said prompt from config file or prompts.db file.
        #custom_prompt_input = config.get('Prompts', 'video_summarize_prompt', fallback="Above is the transcript of a video. Please read through the transcript carefully. Identify the main topics that are discussed over the course of the transcript. Then, summarize the key points about each main topic in bullet points. The bullet points should cover the key information conveyed about each topic in the video, but should be much shorter than the full transcript. Please output your bullet point summary inside <bulletpoints> tags. Do not repeat yourself while writing the summary.")
        custom_prompt_input = "Above is the transcript of a video. Please read through the transcript carefully. Identify the main topics that are discussed over the course of the transcript. Then, summarize the key points about each main topic in bullet points. The bullet points should cover the key information conveyed about each topic in the video, but should be much shorter than the full transcript. Please output your bullet point summary inside <bulletpoints> tags. Do not repeat yourself while writing the summary."
    summary = None
    try:
        if not json_file_path or not os.path.exists(json_file_path):
            logging.error(f"JSON file does not exist: {json_file_path}")
            return None

        with open(json_file_path, 'r') as file:
            data = json.load(file)

        segments = data
        if not isinstance(segments, list):
            logging.error(f"Segments is not a list: {type(segments)}")
            return None

        text = extract_text_from_segments(segments)

        if api_name.lower() == 'openai':
            #def summarize_with_openai(api_key, input_data, custom_prompt_arg)
            summary = summarize_with_openai(api_key, text, custom_prompt_input)

        elif api_name.lower() == "anthropic":
            # def summarize_with_anthropic(api_key, input_data, model, custom_prompt_arg, max_retries=3, retry_delay=5):
            summary = summarize_with_anthropic(api_key, text, custom_prompt_input)
        elif api_name.lower() == "cohere":
            # def summarize_with_cohere(api_key, input_data, model, custom_prompt_arg)
            summary = summarize_with_cohere(api_key, text, custom_prompt_input)

        elif api_name.lower() == "groq":
            logging.debug(f"MAIN: Trying to summarize with groq")
            # def summarize_with_groq(api_key, input_data, model, custom_prompt_arg):
            summary = summarize_with_groq(api_key, text, custom_prompt_input)

        elif api_name.lower() == "openrouter":
            logging.debug(f"MAIN: Trying to summarize with OpenRouter")
            # def summarize_with_openrouter(api_key, input_data, custom_prompt_arg):
            summary = summarize_with_openrouter(api_key, text, custom_prompt_input)

        elif api_name.lower() == "deepseek":
            logging.debug(f"MAIN: Trying to summarize with DeepSeek")
            # def summarize_with_deepseek(api_key, input_data, custom_prompt_arg):
            summary = summarize_with_deepseek(api_key, text, custom_prompt_input)

        elif api_name.lower() == "llama.cpp":
            logging.debug(f"MAIN: Trying to summarize with Llama.cpp")
            # def summarize_with_llama(api_url, file_path, token, custom_prompt)
            summary = summarize_with_llama(text, custom_prompt_input)

        elif api_name.lower() == "kobold":
            logging.debug(f"MAIN: Trying to summarize with Kobold.cpp")
            # def summarize_with_kobold(input_data, kobold_api_token, custom_prompt_input, api_url):
            summary = summarize_with_kobold(text, api_key, custom_prompt_input)

        elif api_name.lower() == "ooba":
            # def summarize_with_oobabooga(input_data, api_key, custom_prompt, api_url):
            summary = summarize_with_oobabooga(text, api_key, custom_prompt_input)

        elif api_name.lower() == "tabbyapi":
            # def summarize_with_tabbyapi(input_data, tabby_model, custom_prompt_input, api_key=None, api_IP):
            summary = summarize_with_tabbyapi(text, custom_prompt_input)

        elif api_name.lower() == "vllm":
            logging.debug(f"MAIN: Trying to summarize with VLLM")
            # def summarize_with_vllm(api_key, input_data, custom_prompt_input):
            summary = summarize_with_vllm(text, custom_prompt_input)

        elif api_name.lower() == "local-llm":
            logging.debug(f"MAIN: Trying to summarize with Local LLM")
            summary = summarize_with_local_llm(text, custom_prompt_input)

        elif api_name.lower() == "huggingface":
            logging.debug(f"MAIN: Trying to summarize with huggingface")
            # def summarize_with_huggingface(api_key, input_data, custom_prompt_arg):
            summarize_with_huggingface(api_key, text, custom_prompt_input)
        # Add additional API handlers here...

        else:
            logging.warning(f"Unsupported API: {api_name}")

        if summary is None:
            logging.debug("Summarization did not return valid text.")

        if summary:
            logging.info(f"Summary generated using {api_name} API")
            # Save the summary file in the same directory as the JSON file
            summary_file_path = json_file_path.replace('.json', '_summary.txt')
            with open(summary_file_path, 'w') as file:
                file.write(summary)
        else:
            logging.warning(f"Failed to generate summary using {api_name} API")
        return summary

    except requests.exceptions.ConnectionError:
            logging.error("Connection error while summarizing")
    except Exception as e:
        logging.error(f"Error summarizing with {api_name}: {str(e)}")

    return summary

#
#
#######################################################################################################################


######################################################################################################################
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
         llm_model=None,
         time_based=False,
         set_chunk_txt_by_words=False,
         set_max_txt_chunk_words=0,
         set_chunk_txt_by_sentences=False,
         set_max_txt_chunk_sentences=0,
         set_chunk_txt_by_paragraphs=False,
         set_max_txt_chunk_paragraphs=0,
         set_chunk_txt_by_tokens=False,
         set_max_txt_chunk_tokens=0,
         ):
    global detail_level_number, summary, audio_file, transcription_text, info_dict

    detail_level = detail

    print(f"Keywords: {keywords}")

    if not input_path:
        return []

    start_time = time.monotonic()
    paths = [input_path] if not os.path.isfile(input_path) else read_paths_from_file(input_path)
    results = []

    for path in paths:
        try:
            if path.startswith('http'):
                info_dict, title = extract_video_info(path)
                download_path = create_download_directory(title)
                video_path = download_video(path, download_path, info_dict, download_video_flag)

                if video_path:
                    audio_file, segments = perform_transcription(video_path, offset, whisper_model, vad_filter)
                    transcription_text = {'audio_file': audio_file, 'transcription': segments}
                    # FIXME - V1
                    #transcription_text = {'video_path': path, 'audio_file': audio_file, 'transcription': segments}

                    if rolling_summarization == True:
                        text = extract_text_from_segments(segments)
                        detail = detail_level
                        additional_instructions = custom_prompt_input
                        chunk_text_by_words = set_chunk_txt_by_words
                        max_words = set_max_txt_chunk_words
                        chunk_text_by_sentences = set_chunk_txt_by_sentences
                        max_sentences = set_max_txt_chunk_sentences
                        chunk_text_by_paragraphs = set_chunk_txt_by_paragraphs
                        max_paragraphs = set_max_txt_chunk_paragraphs
                        chunk_text_by_tokens = set_chunk_txt_by_tokens
                        max_tokens = set_max_txt_chunk_tokens
                        # FIXME
                        summarize_recursively = rolling_summarization
                        verbose = False
                        model = None
                        summary = rolling_summarize_function(text, detail, api_name, api_key, model, custom_prompt_input,
                                                             chunk_text_by_words,
                                                             max_words, chunk_text_by_sentences,
                                                             max_sentences, chunk_text_by_paragraphs,
                                                             max_paragraphs, chunk_text_by_tokens,
                                                             max_tokens, summarize_recursively, verbose
                                                             )

                    elif api_name:
                        summary = perform_summarization(api_name, transcription_text, custom_prompt_input, api_key)
                    else:
                        summary = None

                    if summary:
                        # Save the summary file in the download_path directory
                        summary_file_path = os.path.join(download_path, f"{transcription_text}_summary.txt")
                        with open(summary_file_path, 'w') as file:
                            file.write(summary)

                    add_media_to_database(path, info_dict, segments, summary, keywords, custom_prompt_input, whisper_model)
                else:
                    logging.error(f"Failed to download video: {path}")
            else:
                download_path, info_dict, urls_or_media_file = process_local_file(path)
                if isinstance(urls_or_media_file, list):
                    # Text file containing URLs
                    for url in urls_or_media_file:
                        info_dict, title = extract_video_info(url)
                        download_path = create_download_directory(title)
                        video_path = download_video(url, download_path, info_dict, download_video_flag)

                        if video_path:
                            audio_file, segments = perform_transcription(video_path, offset, whisper_model, vad_filter)
                            # FIXME - V1
                            #transcription_text = {'video_path': url, 'audio_file': audio_file, 'transcription': segments}
                            transcription_text = {'audio_file': audio_file, 'transcription': segments}
                            if rolling_summarization:
                                text = extract_text_from_segments(segments)
                                summary = summarize_with_detail_openai(text, detail=detail)
                            elif api_name:
                                summary = perform_summarization(api_name, transcription_text, custom_prompt_input, api_key)
                            else:
                                summary = None

                            if summary:
                                # Save the summary file in the download_path directory
                                summary_file_path = os.path.join(download_path, f"{transcription_text}_summary.txt")
                                with open(summary_file_path, 'w') as file:
                                    file.write(summary)

                            add_media_to_database(url, info_dict, segments, summary, keywords, custom_prompt_input, whisper_model)
                        else:
                            logging.error(f"Failed to download video: {url}")
                else:
                    # Video or audio file
                    media_path = urls_or_media_file

                    if media_path.lower().endswith(('.mp4', '.avi', '.mov')):
                        # Video file
                        audio_file, segments = perform_transcription(media_path, offset, whisper_model, vad_filter)
                    elif media_path.lower().endswith(('.wav', '.mp3', '.m4a')):
                        # Audio file
                        segments = speech_to_text(media_path, whisper_model=whisper_model, vad_filter=vad_filter)
                    else:
                        logging.error(f"Unsupported media file format: {media_path}")
                        continue

                    transcription_text = {'media_path': path, 'audio_file': media_path, 'transcription': segments}

                    if rolling_summarization:
                        text = extract_text_from_segments(segments)
                        summary = summarize_with_detail_openai(text, detail=detail)
                    elif api_name:
                        summary = perform_summarization(api_name, transcription_text, custom_prompt_input, api_key)
                    else:
                        summary = None

                    if summary:
                        # Save the summary file in the download_path directory
                        summary_file_path = os.path.join(download_path, f"{transcription_text}_summary.txt")
                        with open(summary_file_path, 'w') as file:
                            file.write(summary)

                    add_media_to_database(path, info_dict, segments, summary, keywords, custom_prompt_input, whisper_model)

        except Exception as e:
            logging.error(f"Error processing {path}: {str(e)}")
            continue

    return transcription_text


def signal_handler(sig, frame):
    logging.info('Signal handler called with signal: %s', sig)
    cleanup_process()
    sys.exit(0)


############################## MAIN ##############################
#
#

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load Config
    loaded_config_data = load_and_log_configs()

    if loaded_config_data:
        logging.info("Main: Configuration loaded successfully")
        # You can access the configuration data like this:
        # print(f"OpenAI API Key: {config_data['api_keys']['openai']}")
        # print(f"Anthropic Model: {config_data['models']['anthropic']}")
        # print(f"Kobold API IP: {config_data['local_apis']['kobold']['ip']}")
        # print(f"Output Path: {config_data['output_path']}")
        # print(f"Processing Choice: {config_data['processing_choice']}")
    else:
        print("Failed to load configuration")

    # Print ascii_art
    print_hello()

    transcription_text = None

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
    parser.add_argument('-wm', '--whisper_model', type=str, default='small',
                        help='Whisper model (default: small)| Options: tiny.en, tiny, base.en, base, small.en, small, medium.en, '
                             'medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, '
                             'distil-small.en, distil-large-v3 ')
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
    parser.add_argument('-model', '--llm_model', type=str, default='',
                        help='Model to use for LLM summarization (only used for vLLM/TabbyAPI)')
    parser.add_argument('-k', '--keywords', nargs='+', default=['cli_ingest_no_tag'],
                        help='Keywords for tagging the media, can use multiple separated by spaces (default: cli_ingest_no_tag)')
    parser.add_argument('--log_file', type=str, help='Where to save logfile (non-default)')
    parser.add_argument('--local_llm', action='store_true',
                        help="Use a local LLM from the script(Downloads llamafile from github and 'mistral-7b-instruct-v0.2.Q8' - 8GB model from Huggingface)")
    parser.add_argument('--server_mode', action='store_true',
                        help='Run in server mode (This exposes the GUI/Server to the network)')
    parser.add_argument('--share_public', type=int, default=7860,
                        help="This will use Gradio's built-in ngrok tunneling to share the server publicly on the internet. Specify the port to use (default: 7860)")
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    # parser.add_argument('--offload', type=int, default=20, help='Numbers of layers to offload to GPU for Llamafile usage')
    # parser.add_argument('-o', '--output_path', type=str, help='Path to save the output file')

    args = parser.parse_args()

    # Set Chunking values/variables
    set_chunk_txt_by_words = False
    set_max_txt_chunk_words = 0
    set_chunk_txt_by_sentences = False
    set_max_txt_chunk_sentences = 0
    set_chunk_txt_by_paragraphs = False
    set_max_txt_chunk_paragraphs = 0
    set_chunk_txt_by_tokens = False
    set_max_txt_chunk_tokens = 0

    global server_mode

    if args.share_public:
        share_public = args.share_public
    else:
        share_public = None
    if args.server_mode:

        server_mode = args.server_mode
    else:
        server_mode = None
    if args.server_mode is True:
        server_mode = True
    if args.port:
        server_port = args.port
    else:
        server_port = None

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
    custom_prompt_input = args.custom_prompt

    if not args.custom_prompt:
        logging.debug("No custom prompt defined, will use default")
        args.custom_prompt_input = (
            "\n\nabove is the transcript of a video. "
            "Please read through the transcript carefully. Identify the main topics that are "
            "discussed over the course of the transcript. Then, summarize the key points about each "
            "main topic in a concise bullet point. The bullet points should cover the key "
            "information conveyed about each topic in the video, but should be much shorter than "
            "the full transcript. Please output your bullet point summary inside <bulletpoints> "
            "tags."
        )
        print("No custom prompt defined, will use default")

        custom_prompt_input = args.custom_prompt
    else:
        logging.debug(f"Custom prompt defined, will use \n\nf{custom_prompt_input} \n\nas the prompt")
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
    elif not args.input_path:
        parser.print_help()
        sys.exit(1)

    else:
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

        global api_name
        api_name = args.api_name

        summary = None  # Initialize to ensure it's always defined
        if args.detail_level == None:
            args.detail_level = 0.01

        # FIXME
        # if args.api_name and args.rolling_summarization and any(
        #         key.startswith(args.api_name) and value is not None for key, value in api_keys.items()):
        #     logging.info(f'MAIN: API used: {args.api_name}')
        #     logging.info('MAIN: Rolling Summarization will be performed.')

        elif args.api_name:
            logging.info(f'MAIN: API used: {args.api_name}')
            logging.info('MAIN: Summarization (not rolling) will be performed.')

        else:
            logging.info('No API specified. Summarization will not be performed.')

        logging.debug("Platform check being performed...")
        platform_check()
        logging.debug("CUDA check being performed...")
        cuda_check()
        processing_choice = "cpu"
        logging.debug("ffmpeg check being performed...")
        check_ffmpeg()
        # download_ffmpeg()

        llm_model = args.llm_model or None
        # FIXME - dirty hack
        args.time_based = False

        try:
            results = main(args.input_path, api_name=args.api_name, api_key=args.api_key,
                           num_speakers=args.num_speakers, whisper_model=args.whisper_model, offset=args.offset,
                           vad_filter=args.vad_filter, download_video_flag=args.video, custom_prompt=args.custom_prompt_input,
                           overwrite=args.overwrite, rolling_summarization=args.rolling_summarization,
                           detail=args.detail_level, keywords=args.keywords, llm_model=args.llm_model,
                           time_based=args.time_based, set_chunk_txt_by_words=set_chunk_txt_by_words,
                           set_max_txt_chunk_words=set_max_txt_chunk_words,
                           set_chunk_txt_by_sentences=set_chunk_txt_by_sentences,
                           set_max_txt_chunk_sentences=set_max_txt_chunk_sentences,
                           set_chunk_txt_by_paragraphs=set_chunk_txt_by_paragraphs,
                           set_max_txt_chunk_paragraphs=set_max_txt_chunk_paragraphs,
                           set_chunk_txt_by_tokens=set_chunk_txt_by_tokens,
                           set_max_txt_chunk_tokens=set_max_txt_chunk_tokens)

            logging.info('Transcription process completed.')
            atexit.register(cleanup_process)
        except Exception as e:
            logging.error('An error occurred during the transcription process.')
            logging.error(str(e))
            sys.exit(1)

        finally:
            cleanup_process()
