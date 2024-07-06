#!/usr/bin/env python3
# Std Lib Imports
import argparse
import os
import signal
import sys
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import webbrowser
# Local Module Imports (Libraries specific to this project)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'App_Function_Libraries')))
from App_Function_Libraries.Audio_Files import *
from App_Function_Libraries.Book_Ingestion_Lib import ingest_text_file, ingest_folder
from App_Function_Libraries.Chunk_Lib import *
from App_Function_Libraries.Gradio_Related import *
from App_Function_Libraries.Tokenization_Methods_Lib import *
from App_Function_Libraries.Utils import load_and_log_configs
from App_Function_Libraries.Video_DL_Ingestion_Lib import *
# 3rd-Party Module Imports
import requests
# OpenAI Tokenizer support
#
# Other Tokenizers
#
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

# Global variables
whisper_models = ["small", "medium", "small.en", "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3",
                  "distil-large-v2", "distil-medium.en", "distil-small.en"]
server_mode = False
share_public = False

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
#   3. Windows: Could not locate cudnn_ops_infer64_8.dll. Please make sure it is in your library path!
#
#   4.
#
#   5.
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
# 1.
# 2.
#
#
#######################


#######################
# System Startup Notice
#

# Dirty hack - sue me. - FIXME - fix this...
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

whisper_models = ["small", "medium", "small.en", "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3",
                  "distil-large-v2", "distil-medium.en", "distil-small.en"]
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
#
#
#
#
#################################################################################################################
#
#

# Helper functions
def import_data(file):
    # Implement this function to import data from a file
    pass


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


def perform_summarization(api_name, json_file_path, custom_prompt_input, api_key):
    # Load Config
    loaded_config_data = load_and_log_configs()

    if custom_prompt_input is None:
        # FIXME - Setup proper default prompt & extract said prompt from config file or prompts.db file.
        #custom_prompt_input = config.get('Prompts', 'video_summarize_prompt', fallback="Above is the transcript of a video. Please read through the transcript carefully. Identify the main topics that are discussed over the course of the transcript. Then, summarize the key points about each main topic in bullet points. The bullet points should cover the key information conveyed about each topic in the video, but should be much shorter than the full transcript. Please output your bullet point summary inside <bulletpoints> tags. Do not repeat yourself while writing the summary.")
        custom_prompt_input = """
        You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.
**Bulleted Note Creation Guidelines**

**Headings**:
- Based on referenced topics, not categories like quotes or terms
- Surrounded by **bold** formatting 
- Not listed as bullet points
- No space between headings and list items underneath

**Emphasis**:
- **Important terms** set in bold font
- **Text ending in a colon**: also bolded

**Review**:
- Ensure adherence to specified format
- Do not reference these instructions in your response.</s>[INST] {{ .Prompt }} [/INST]"""
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
         ingest_text_file=False,
         chunk=False,
         max_chunk_size=2000,
         chunk_overlap=100,
         chunk_unit='tokens',
         summarize_chunks=None
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

            # FIXME - Need to update so that chunking is fully handled.
            elif chunk and path.lower().endswith('.txt'):
                chunks = chunk_text_file(path, max_chunk_size, chunk_overlap, chunk_unit)
                if chunks:
                    chunks_data = {
                        "file_path": path,
                        "chunk_unit": chunk_unit,
                        "max_chunk_size": max_chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "chunks": []
                    }
                    summaries_data = {
                        "file_path": path,
                        "summarization_method": summarize_chunks,
                        "summaries": []
                    }

                    for i, chunk_text in enumerate(chunks):
                        chunk_info = {
                            "chunk_id": i + 1,
                            "text": chunk_text
                        }
                        chunks_data["chunks"].append(chunk_info)

                        if summarize_chunks:
                            summary = None
                            if summarize_chunks == 'openai':
                                summary = summarize_with_openai(api_key, chunk_text, custom_prompt)
                            elif summarize_chunks == 'anthropic':
                                summary = summarize_with_anthropic(api_key, chunk_text, custom_prompt)
                            elif summarize_chunks == 'cohere':
                                summary = summarize_with_cohere(api_key, chunk_text, custom_prompt)
                            elif summarize_chunks == 'groq':
                                summary = summarize_with_groq(api_key, chunk_text, custom_prompt)
                            elif summarize_chunks == 'local-llm':
                                summary = summarize_with_local_llm(chunk_text, custom_prompt)
                            # Add more summarization methods as needed

                            if summary:
                                summary_info = {
                                    "chunk_id": i + 1,
                                    "summary": summary
                                }
                                summaries_data["summaries"].append(summary_info)
                            else:
                                logging.warning(f"Failed to generate summary for chunk {i + 1}")

                    # Save chunks to a single JSON file
                    chunks_file_path = f"{path}_chunks.json"
                    with open(chunks_file_path, 'w', encoding='utf-8') as f:
                        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
                    logging.info(f"All chunks saved to {chunks_file_path}")

                    # Save summaries to a single JSON file (if summarization was performed)
                    if summarize_chunks:
                        summaries_file_path = f"{path}_summaries.json"
                        with open(summaries_file_path, 'w', encoding='utf-8') as f:
                            json.dump(summaries_data, f, ensure_ascii=False, indent=2)
                        logging.info(f"All summaries saved to {summaries_file_path}")

                    logging.info(f"File {path} chunked into {len(chunks)} parts using {chunk_unit} as the unit.")
                else:
                    logging.error(f"Failed to chunk file {path}")

            # Handle downloading of URLs from a text file or processing local video/audio files
            else:
                download_path, info_dict, urls_or_media_file = process_local_file(path)
                if isinstance(urls_or_media_file, list):
                    # Text file containing URLs
                    for url in urls_or_media_file:
                        for item in urls_or_media_file:
                            if item.startswith(('http://', 'https://')):
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
                    # Video or audio or txt file
                    media_path = urls_or_media_file

                    if media_path.lower().endswith(('.txt', '.md')):
                        if media_path.lower().endswith('.txt'):
                            # Handle text file ingestion
                            result = ingest_text_file(media_path)
                            logging.info(result)
                    elif media_path.lower().endswith(('.mp4', '.avi', '.mov')):
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
                             'distil-small.en')
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
    parser.add_argument('--ingest_text_file', action='store_true',
                        help='Ingest .txt files as content instead of treating them as URL lists')
    parser.add_argument('--text_title', type=str, help='Title for the text file being ingested')
    parser.add_argument('--text_author', type=str, help='Author of the text file being ingested')

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

    # Check if the user wants to ingest a text file (singular or multiple from a folder)
    if args.input_path is not None:
        if os.path.isdir(args.input_path) and args.ingest_text_file:
            results = ingest_folder(args.input_path, keywords=args.keywords)
            for result in results:
                print(result)
        elif args.input_path.lower().endswith('.txt') and args.ingest_text_file:
            result = ingest_text_file(args.input_path, title=args.text_title, author=args.text_author,
                                      keywords=args.keywords)
            print(result)
        sys.exit(0)

    # Launch the GUI
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
