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
from App_Function_Libraries.Chunk_Lib import *
from App_Function_Libraries.Diarization_Lib import *
from App_Function_Libraries.Local_File_Processing_Lib import *
from App_Function_Libraries.Local_LLM_Inference_Engine_Lib import *
from App_Function_Libraries.Local_Summarization_Lib import *
from App_Function_Libraries.Summarization_General_Lib import *
from App_Function_Libraries.System_Checks_Lib import *
from App_Function_Libraries.Tokenization_Methods_Lib import *
from App_Function_Libraries.Video_DL_Ingestion_Lib import *
#from App_Function_Libraries.Web_UI_Lib import *


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

custom_prompt = None

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
#######################################################################################################################
# Function Definitions
#

# Only to be used when configured with Gradio for HF Space


def format_transcription(transcription_result_arg):
    if transcription_result_arg:
        json_data = transcription_result_arg['transcription']
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


# FIXME - code for the 're-prompt' functionality
#- Change to use 'check_api()' function - also, create 'check_api()' function
# def ask_question(transcription, question, api_name, api_key):
#     if not question.strip():
#         return "Please enter a question."
#
#         prompt = f"""Transcription:\n{transcription}
#
#         Given the above transcription, please answer the following:\n\n{question}"""
#
#         # FIXME - Refactor main API checks so they're their own function - api_check()
#         # Call api_check() function here
#
#         if api_name.lower() == "openai":
#             openai_api_key = api_key if api_key else config.get('API', 'openai_api_key', fallback=None)
#             headers = {
#                 'Authorization': f'Bearer {openai_api_key}',
#                 'Content-Type': 'application/json'
#             }
#             if openai_model:
#                 pass
#             else:
#                 openai_model = 'gpt-4-turbo'
#             data = {
#                 "model": openai_model,
#                 "messages": [
#                     {
#                         "role": "system",
#                         "content": "You are a helpful assistant that answers questions based on the given "
#                                    "transcription and summary."
#                     },
#                     {
#                         "role": "user",
#                         "content": prompt
#                     }
#                 ],
#                 "max_tokens": 150000,
#                 "temperature": 0.1
#             }
#             response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
#
#         if response.status_code == 200:
#             answer = response.json()['choices'][0]['message']['content'].strip()
#             return answer
#         else:
#             return "Failed to process the question."
#     else:
#         return "Question answering is currently only supported with the OpenAI API."


# For the above 'ask_question()' function, the following APIs are supported:
# summarizers: Dict[str, Callable[[str, str], str]] = {
#     'tabbyapi': summarize_with_tabbyapi,
#     'openai': summarize_with_openai,
#     'anthropic': summarize_with_claude,
#     'cohere': summarize_with_cohere,
#     'groq': summarize_with_groq,
#     'llama': summarize_with_llama,
#     'kobold': summarize_with_kobold,
#     'oobabooga': summarize_with_oobabooga,
#     'local-llm': summarize_with_local_llm,
#     'huggingface': summarize_with_huggingface,
#     'openrouter': summarize_with_openrouter
#     # Add more APIs here as needed
# }

#########################################################################


# FIXME - Move to 'Web_UI_Lib.py'
# Gradio Search Function-related stuff
def display_details(media_id):
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
    prompt_summary_results, content = fetch_item_details(media_id)
    content_section = f"<h4>Transcription:</h4><pre>{content}</pre><hr>"
    prompt_summary_section = ""
    for prompt, summary in prompt_summary_results:
        prompt_summary_section += f"<h4>Prompt:</h4><p>{prompt}</p>"
        prompt_summary_section += f"<h4>Summary:</h4><p>{summary}</p><hr>"
    return prompt_summary_section, content_section

def update_dropdown(search_query, search_type):
    results = browse_items(search_query, search_type)
    item_options = [f"{item[1]} ({item[2]})" for item in results]
    item_mapping = {f"{item[1]} ({item[2]})": item[0] for item in results}  # Map item display to media ID
    return gr.Dropdown.update(choices=item_options), item_mapping

def get_media_id(selected_item, item_mapping):
    return item_mapping.get(selected_item)

def update_detailed_view(selected_item, item_mapping):
    media_id = get_media_id(selected_item, item_mapping)
    if media_id:
        prompt_summary_html, content_html = display_item_details(media_id)
        return gr.update(value=prompt_summary_html), gr.update(value=content_html)
    return gr.update(value="No details available"), gr.update(value="No details available")

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
    whisper_models = ["small.en", "medium.en", "large"]
    # Set theme value with https://www.gradio.app/guides/theming-guide - 'theme='
    my_theme = gr.Theme.from_hub("gradio/seafoam")
    with gr.Blocks(theme=my_theme) as iface:
        # Tab 1: Audio Transcription + Summarization
        with gr.Tab("Audio Transcription + Summarization"):

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
                          " will be downloaded)", placeholder="Enter the video URL here")
#            url_input = gr.Textbox(label="URL (Mandatory) --> Playlist URLs will be stripped and only the linked video"
#                                         " will be downloaded)", placeholder="Enter the video URL here")

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
                            "course of the transcript. Then, summarize the key points about each main topic in a "
                            "concise bullet point. The bullet points should cover the key information conveyed about "
                            "each topic in the video, but should be much shorter than the full transcript. Please "
                            "output your bullet point summary inside <bulletpoints> tags.",
                lines=3, visible=True)
            offset_input = gr.Number(value=0, label="Offset (Seconds into the video to start transcribing at)",
                                     visible=False)
            api_name_input = gr.Dropdown(
                choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "OpenRouter", "Llama.cpp",
                         "Kobold", "Ooba", "HuggingFace"],
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
            max_words_input = gr.Number(label="Max Words", value=0, precision=0, visible=False)

            chunk_text_by_sentences_checkbox = gr.Checkbox(label="Chunk Text by Sentences", value=False,
                                                           visible=False)
            max_sentences_input = gr.Number(label="Max Sentences", value=0, precision=0, visible=False)

            chunk_text_by_paragraphs_checkbox = gr.Checkbox(label="Chunk Text by Paragraphs", value=False,
                                                            visible=False)
            max_paragraphs_input = gr.Number(label="Max Paragraphs", value=0, precision=0, visible=False)

            chunk_text_by_tokens_checkbox = gr.Checkbox(label="Chunk Text by Tokens", value=False, visible=False)
            max_tokens_input = gr.Number(label="Max Tokens", value=0, precision=0, visible=False)

            inputs = [
                num_speakers_input, whisper_model_input, custom_prompt_input, offset_input, api_name_input,
                api_key_input, vad_filter_input, download_video_input, download_audio_input,
                rolling_summarization_input, detail_level_input, question_box_input, keywords_input,
                chunk_text_by_words_checkbox, max_words_input, chunk_text_by_sentences_checkbox,
                max_sentences_input, chunk_text_by_paragraphs_checkbox, max_paragraphs_input,
                chunk_text_by_tokens_checkbox, max_tokens_input
            ]

            all_inputs = [url_input] + inputs

            outputs = [
                gr.Textbox(label="Transcription (Resulting Transcription from your input URL)"),
                gr.Textbox(label="Summary or Status Message (Current status of Summary or Summary itself)"),
                gr.File(label="Download Transcription as JSON (Download the Transcription as a file)"),
                gr.File(label="Download Summary as Text (Download the Summary as a file)"),
                gr.File(label="Download Video (Download the Video as a file)", visible=True),
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
                                                ngl_value, host_checked, host_value, port_checked, port_value,)

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

        # Tab 2: Scrape & Summarize Articles/Websites
        with gr.Tab("Scrape & Summarize Articles/Websites"):
            url_input = gr.Textbox(label="Article URL", placeholder="Enter the article URL here")
            custom_article_title_input = gr.Textbox(label="Custom Article Title (Optional)",
                                                    placeholder="Enter a custom title for the article")
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt (Optional)",
                placeholder="Provide a custom prompt for summarization",
                lines=3
            )
            api_name_input = gr.Dropdown(
                choices=[None, "huggingface", "openrouter", "openai", "anthropic", "cohere", "groq", "llama", "kobold",
                         "ooba"],
                value=None,
                label="API Name (Mandatory for Summarization)"
            )
            api_key_input = gr.Textbox(label="API Key (Mandatory if API Name is specified)",
                                       placeholder="Enter your API key here; Ignore if using Local API or Built-in API")
            keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated)",
                                        value="default,no_keyword_set", visible=True)

            scrape_button = gr.Button("Scrape and Summarize")
            result_output = gr.Textbox(label="Result")

            scrape_button.click(scrape_and_summarize, inputs=[url_input, custom_prompt_input, api_name_input,
                                                              api_key_input, keywords_input,
                                                              custom_article_title_input], outputs=result_output)

            gr.Markdown("### Or Paste Unstructured Text Below (Will use settings from above)")
            text_input = gr.Textbox(label="Unstructured Text", placeholder="Paste unstructured text here", lines=10)
            text_ingest_button = gr.Button("Ingest Unstructured Text")
            text_ingest_result = gr.Textbox(label="Result")

            text_ingest_button.click(ingest_unstructured_text,
                                     inputs=[text_input, custom_prompt_input, api_name_input, api_key_input,
                                             keywords_input, custom_article_title_input], outputs=text_ingest_result)

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

    with gr.Blocks() as search_interface:
        with gr.Tab("Search Ingested Materials / Detailed Entry View / Prompts"):
            search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
            search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                         label="Search By")

            search_button = gr.Button("Search")
            items_output = gr.Dropdown(label="Select Item", choices=[])
            item_mapping = gr.State({})

            search_button.click(fn=update_dropdown, inputs=[search_query_input, search_type_input],
                                outputs=[items_output, item_mapping])

            prompt_summary_output = gr.HTML(label="Prompt & Summary", visible=True)
            content_output = gr.HTML(label="Content", visible=True)
            items_output.change(fn=update_detailed_view, inputs=[items_output, item_mapping],
                                outputs=[prompt_summary_output, content_output])

        with gr.Tab("View Prompts"):
            with gr.Column():
                prompt_dropdown = gr.Dropdown(label="Select Prompt (Thanks to the 'Fabric' project for this initial set: https://github.com/danielmiessler/fabric", choices=[])
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
        with gr.Tab("Search Prompts"):
            with gr.Column():
                search_query_input = gr.Textbox(label="Search Query (It's broken)", placeholder="Enter your search query...")
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

    with gr.Blocks() as llamafile_interface:
        with gr.Tab("Llamafile Settings"):
            gr.Markdown("Settings for Llamafile")

            # Toggle switch for Advanced/Simple mode
            am_noob = gr.Checkbox(label="Check this to enable sane defaults and then download(if not already downloaded) a model, click 'Start Llamafile' and then go to --> 'Llamafile Chat Interface')\n\n", value=False, visible=True)
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
                inputs=[am_noob, verbose_checked, threads_checked, threads_value, http_threads_checked, http_threads_value,
                        model_checked, model_value, hf_repo_checked, hf_repo_value, hf_file_checked, hf_file_value,
                        ctx_size_checked, ctx_size_value, ngl_checked, ngl_value, host_checked, host_value,
                        port_checked, port_value],
                outputs=output_display
            )
        # FIXME - Possibly dead code?
        #
        #     # Setting inputs with checkboxes
        #     verbose_checked = gr.Checkbox(label="Enable Verbose Output", value=False)
        #     threads_checked = gr.Checkbox(label="Enable Setting CPU Threads", value=False)
        #     threads_value = gr.Number(label="Number of CPU Threads", value="", precision=0)
        #     http_threads_checked = gr.Checkbox(label="Enable Setting HTTP Server Threads", value=False)
        #     http_threads_value = gr.Number(label="Number of HTTP Server Threads", value="", precision=0)
        #     model_checked = gr.Checkbox(label="Enable Setting Local LLM Model Path", value=False)
        #     model_value = gr.Textbox(label="Path to Local Model File", value="")
        #     hf_repo_checked = gr.Checkbox(label="Use Huggingface Repo Model", value=False)
        #     hf_repo_value = gr.Textbox(label="Huggingface Repo Name", value="")
        #     hf_file_checked = gr.Checkbox(label="Enable Setting Huggingface Model File", value=False)
        #     hf_file_value = gr.Textbox(label="Huggingface Model File", value="")
        #     ctx_size_checked = gr.Checkbox(label="Enable Setting Prompt Context Size", value=False)
        #     ctx_size_value = gr.Number(label="Prompt Context Size", value=8124, precision=0)
        #     ngl_checked = gr.Checkbox(label="Enable Setting GPU Layers", value=False)
        #     ngl_value = gr.Number(label="Number of GPU Layers", value="", precision=0)
        #     host_checked = gr.Checkbox(label="Enable Setting IP to Listen On", value=False)
        #     host_value = gr.Textbox(label="Host IP Address", value="")
        #     port_checked = gr.Checkbox(label="Enable Setting Server Port", value=False)
        #     port_value = gr.Number(label="Port Number", value="", precision=0)
        #
        #
        #     # Function call with the new inputs
        #     start_button.click(
        #         fn=start_llamafile,
        #         inputs=[verbose_checked, threads_checked, threads_value, http_threads_checked, http_threads_value,
        #                 model_checked, model_value, hf_repo_checked, hf_repo_value, hf_file_checked, hf_file_value,
        #                 ctx_size_checked, ctx_size_value, ngl_checked, ngl_value, host_checked, host_value,
        #                 port_checked, port_value],
        #         outputs=output_display
        #     )
        #
        #     # This function is not implemented yet...
        #     # FIXME - Implement this function
        #     stop_button.click(stop_llamafile, outputs=output_display)
        #
        # # Toggle event for Advanced/Simple mode
        # advanced_mode_toggle.change(toggle_advanced_llamafile_mode,
        #                             inputs=[advanced_mode_toggle],
        #                             outputs=[])

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

    import_interface = gr.Interface(
        fn=import_data,
        inputs=gr.File(label="Upload file for import"),
        outputs="text",
        title="Import Data",
        description="Import data into the database from a CSV file."
    )

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

    keyword_add_interface = gr.Interface(
        fn=add_keyword,
        inputs=gr.Textbox(label="Add Keywords (comma-separated)", placeholder="Enter keywords here..."),
        outputs="text",
        title="Add Keywords",
        description="Add one, or multiple keywords to the database.",
        allow_flagging="never"
    )

    keyword_delete_interface = gr.Interface(
        fn=delete_keyword,
        inputs=gr.Textbox(label="Delete Keyword", placeholder="Enter keyword to delete here..."),
        outputs="text",
        title="Delete Keyword",
        description="Delete a keyword from the database.",
        allow_flagging="never"
    )

    browse_keywords_interface = gr.Interface(
        fn=keywords_browser_interface,
        inputs=[],
        outputs="markdown",
        title="Browse Keywords",
        description="View all keywords currently stored in the database."
    )

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

    #FIXME - example to use for rest of gradio theming, just stuff in HTML/Markdown
    # <-- set description variable with HTML -->
    desc = "<h3>Youtube Video Downloader</h3><p>This Input takes a Youtube URL as input and creates " \
           "a webm file for you to download. </br><em>If you want a full-featured one:</em> " \
           "<strong><em>https://github.com/StefanLobbenmeier/youtube-dl-gui</strong></em> or <strong><em>https://github.com/yt-dlg/yt-dlg</em></strong></p>"

    download_videos_interface = gr.Interface(
        fn=gradio_download_youtube_video,
        inputs=gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL here"),
        outputs=gr.File(label="Download Video"),
        title="YouTube Video Downloader",
        description=desc,
        allow_flagging="never"
    )

    # Combine interfaces into a tabbed interface
    tabbed_interface = gr.TabbedInterface([iface, search_interface, llamafile_interface, keyword_tab, import_export_tab, download_videos_interface],
                                          ["Transcription / Summarization / Ingestion", "Search / Detailed View",
                                           "Llamafile Interface", "Keywords", "Export/Import",  "Download Video/Audio Files"])
    # Launch the interface
    server_port_variable = 7860
    global server_mode, share_public
    if server_mode is True and share_public is False:
        tabbed_interface.launch(share=True, server_port=server_port_variable, server_name="http://0.0.0.0")
    elif share_public == True:
        tabbed_interface.launch(share=True, )
    else:
        tabbed_interface.launch(share=False, )


def clean_youtube_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'list' in query_params:
        query_params.pop('list')
    cleaned_query = urlencode(query_params, doseq=True)
    cleaned_url = urlunparse(parsed_url._replace(query=cleaned_query))
    return cleaned_url


def process_url(
        url,
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
        chunk_text_by_words,
        max_words,
        chunk_text_by_sentences,
        max_sentences,
        chunk_text_by_paragraphs,
        max_paragraphs,
        chunk_text_by_tokens,
        max_tokens
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

    # Validate input
    if not url:
        return "No URL provided.", "No URL provided.", None, None, None, None, None, None

    if not is_valid_url(url):
        return "Invalid URL format.", "Invalid URL format.", None, None, None, None, None, None

    # Clean the URL to remove playlist parameters if any
    url = clean_youtube_url(url)

    print("API Name received:", api_name)  # Debugging line

    logging.info(f"Processing URL: {url}")
    video_file_path = None
    global info_dict
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
                       )

        if not results:
            return "No URL provided.", "No URL provided.", None, None, None, None, None, None

        transcription_result = results[0]
        transcription_text = json.dumps(transcription_result['transcription'], indent=2)
        summary_text = transcription_result.get('summary', 'Summary not available')

        # Prepare file paths for transcription and summary
        # Sanitize filenames
        audio_file_sanitized = sanitize_filename(transcription_result['audio_file'])
        json_pretty_file_path = os.path.join('Results', audio_file_sanitized.replace('.wav', '.segments_pretty.json'))
        json_file_path = os.path.join('Results', audio_file_sanitized.replace('.wav', '.segments.json'))
        summary_file_path = os.path.join('Results', audio_file_sanitized.replace('.wav', '_summary.txt'))

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

        try:
            if download_video:
                video_file_path = transcription_result.get('video_path', None)
                if video_file_path and os.path.exists(video_file_path):
                    logging.debug(f"Confirmed existence of video file at {video_file_path}")
                else:
                    logging.error(f"Video file not found at expected path: {video_file_path}")
                    video_file_path = None
            else:
                video_file_path = None

            if isinstance(transcription_result['transcription'], list):
                text = ' '.join([segment['Text'] for segment in transcription_result['transcription']])
            else:
                text = ''

        except Exception as e:
            logging.error(f"Error processing video file: {e}")

        # Check if files exist before returning paths
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File not found: {json_file_path}")
        if not os.path.exists(summary_file_path):
            raise FileNotFoundError(f"File not found: {summary_file_path}")

        formatted_transcription = format_transcription(transcription_result)

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
            logging.debug(f"Media Type: {media_type}")
            logging.debug(f"Media Content: {media_content}")
            logging.debug(f"Media Keywords: {media_keywords}")
            logging.debug(f"Media Author: {media_author}")
            logging.debug(f"Ingestion Date: {media_ingestion_date}")
            logging.debug(f"Custom Prompt: {custom_prompt}")
            logging.debug(f"Summary Text: {summary_text}")
            logging.debug(f"Transcription Model: {transcription_model}")

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
            return transcription_text, summary_text, json_file_path, summary_file_path, video_file_path, None
        else:
            return transcription_text, summary_text, json_file_path, None, video_file_path, None
    except KeyError as e:
        logging.error(f"Error processing {url}: {str(e)}")
        return str(e), 'Error processing the request.', None, None, None, None
    except Exception as e:
        logging.error(f"Error processing URL: {e}")
        return str(e), 'Error processing the request.', None, None, None, None


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
    global detail_level_number, summary, audio_file, transcription_result, info_dict

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
                    logging.debug(f"MAIN: info_dict content: {info_dict}")
                    logging.debug("MAIN: Creating path for video file...")
                    download_path = create_download_directory(info_dict['title'])
                    logging.debug("MAIN: Path created successfully\n MAIN: Now Downloading video from yt_dlp...")
                    try:
                        video_path = download_video(path, download_path, info_dict, download_video_flag)
                        if video_path is None:
                            logging.error("MAIN: video_path is None after download_video")
                            continue
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

                if isinstance(segments, dict) and "error" in segments:
                    logging.error(f"Error transcribing audio: {segments['error']}")
                    transcription_result['error'] = segments['error']

                results.append(transcription_result)
                logging.info(f"MAIN: Transcription complete: {audio_file}")

                # Check if segments is a dictionary before proceeding with summarization
                if isinstance(segments, dict):
                    logging.warning("Skipping summarization due to transcription error")
                    continue

                # FIXME
                # Perform rolling summarization based on API Name, detail level, and if an API key exists
                # Will remove the API key once rolling is added for llama.cpp
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
                # Perform summarization based on the specified API
                elif api_name:
                    logging.debug(f"MAIN: Summarization being performed by {api_name}")
                    json_file_path = audio_file.replace('.wav', '.segments.json')
                    if api_name.lower() == 'openai':
                        try:
                            logging.debug(f"MAIN: trying to summarize with openAI")
                            summary = summarize_with_openai(openai_api_key, json_file_path, custom_prompt)
                            if summary != "openai: Error occurred while processing summary":
                                transcription_result['summary'] = summary
                                logging.info(f"Summary generated using {api_name} API")
                                save_summary_to_file(summary, json_file_path)
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
                            else:
                                logging.warning(f"Failed to generate summary using {api_name} API")
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
                        openrouter_api_key = api_key if api_key else config.get('API', 'openrouter_api_key',
                                                                                fallback=None)
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
                    # FIXME
                    # elif final_summary:
                    #     logging.info(f"Rolling summary generated using {api_name} API")
                    #     logging.info(f"Final Rolling summary is {final_summary}\n\n")
                    #     save_summary_to_file(final_summary, json_file_path)
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
    # Establish logging baseline
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print_hello()
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
    #parser.add_argument('--offload', type=int, default=20, help='Numbers of layers to offload to GPU for Llamafile usage')
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
    custom_prompt = args.custom_prompt

    if not args.custom_prompt:
        logging.debug("No custom prompt defined, will use default")
        args.custom_prompt = (
            "\n\nabove is the transcript of a video. "
            "Please read through the transcript carefully. Identify the main topics that are "
            "discussed over the course of the transcript. Then, summarize the key points about each "
            "main topic in a concise bullet point. The bullet points should cover the key "
            "information conveyed about each topic in the video, but should be much shorter than "
            "the full transcript. Please output your bullet point summary inside <bulletpoints> "
            "tags."
        )
        print("No custom prompt defined, will use default")

        custom_prompt = args.custom_prompt
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
        #download_ffmpeg()

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
                           llm_model=args.llm_model,
                           time_based=args.time_based,
                           set_chunk_txt_by_words=set_chunk_txt_by_words,
                           set_max_txt_chunk_words=set_max_txt_chunk_words,
                           set_chunk_txt_by_sentences=set_chunk_txt_by_sentences,
                           set_max_txt_chunk_sentences=set_max_txt_chunk_sentences,
                           set_chunk_txt_by_paragraphs=set_chunk_txt_by_paragraphs,
                           set_max_txt_chunk_paragraphs=set_max_txt_chunk_paragraphs,
                           set_chunk_txt_by_tokens=set_chunk_txt_by_tokens,
                           set_max_txt_chunk_tokens=set_max_txt_chunk_tokens,
                           )

            logging.info('Transcription process completed.')
            atexit.register(cleanup_process)
        except Exception as e:
            logging.error('An error occurred during the transcription process.')
            logging.error(str(e))
            sys.exit(1)

        finally:
            cleanup_process()
