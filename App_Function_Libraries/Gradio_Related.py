# Gradio_Related.py
#########################################
# Gradio UI Functions Library
# This library is used to hold all UI-related functions for Gradio.
# I fucking hate Gradio.
#
#####
# Functions:
#
# download_audio_file(url, save_path)
# process_audio(
# process_audio_file(audio_url, audio_file, whisper_model="small.en", api_name=None, api_key=None)
#
#
#########################################
#
# Built-In Imports
from datetime import datetime
import json
import logging
import os.path
from pathlib import Path
import re
import sqlite3
from typing import Dict, List, Tuple
#
# Import 3rd-Party Libraries
import yt_dlp
import gradio as gr
#
# Local Imports
from App_Function_Libraries.Article_Summarization_Lib import scrape_and_summarize_multiple
from App_Function_Libraries.Audio_Files import process_audio_file
from App_Function_Libraries.PDF_Ingestion_Lib import ingest_pdf_file
from App_Function_Libraries.Local_LLM_Inference_Engine_Lib import local_llm_gui_function
from App_Function_Libraries.Local_Summarization_Lib import summarize_with_llama, summarize_with_kobold, \
    summarize_with_oobabooga, summarize_with_tabbyapi, summarize_with_vllm, summarize_with_local_llm
from App_Function_Libraries.Summarization_General_Lib import summarize_with_openai, summarize_with_cohere, \
    summarize_with_anthropic, summarize_with_groq, summarize_with_openrouter, summarize_with_deepseek, \
    summarize_with_huggingface, process_url
from App_Function_Libraries.SQLite_DB import update_media_content, list_prompts, search_and_display, db, DatabaseError, \
    fetch_prompt_details, keywords_browser_interface, add_keyword, delete_keyword, export_to_csv, export_keywords_to_csv
from App_Function_Libraries.Utils import sanitize_filename
#
#######################################################################################################################
# Function Definitions
#

whisper_models = ["small", "medium", "small.en", "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3",
                  "distil-large-v2", "distil-medium.en", "distil-small.en"]
custom_prompt_input = None
server_mode = False
share_public = False


def load_preset_prompts():
    return list_prompts()


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



def format_transcription(content):
    # Add extra space after periods for better readability
    content = content.replace('.', '. ').replace('.  ', '. ')
    # Split the content into lines for multiline display; assuming simple logic here
    lines = content.split('. ')
    # Join lines with HTML line break for better presentation in HTML
    formatted_content = "<br>".join(lines)
    return formatted_content


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
        logger = logging.getLogger()
        logger.error(f"Error searching media: {e}")
        return str(e)




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


def fetch_item_details_single(media_id: int):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT prompt, summary 
                FROM MediaModifications 
                WHERE media_id = ? 
                ORDER BY modification_date DESC 
                LIMIT 1
            """, (media_id,))
            prompt_summary_result = cursor.fetchone()
            cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
            content_result = cursor.fetchone()

            prompt = prompt_summary_result[0] if prompt_summary_result else ""
            summary = prompt_summary_result[1] if prompt_summary_result else ""
            content = content_result[0] if content_result else ""

            return prompt, summary, content
    except sqlite3.Error as e:
        raise Exception(f"Error fetching item details: {e}")


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

def search_media_database(query: str) -> List[Tuple[int, str, str]]:
    return browse_items(query, 'Title')

def load_media_content(media_id: int) -> dict:
    prompt_summary_results, content = fetch_item_details(media_id)
    return {
        "content": content if content else "No content available",
        "prompt": prompt_summary_results[-1][0] if prompt_summary_results else "No prompt available",
        "summary": prompt_summary_results[-1][1] if prompt_summary_results else "No summary available"
    }

def load_preset_prompts():
    return list_prompts()

def chat(message, history, media_content, selected_parts, api_endpoint, api_key, prompt):
    try:
        # Ensure selected_parts is a list
        if not isinstance(selected_parts, (list, tuple)):
            selected_parts = [selected_parts] if selected_parts else []

        # Combine the selected parts of the media content
        combined_content = " ".join([media_content.get(part, "") for part in selected_parts if part in media_content])

        # Prepare the input for the API
        input_data = f"{combined_content}\n\nUser: {message}\nAI:"

        # Use the existing API request code based on the selected endpoint
        if api_endpoint.lower() == 'openai':
            response = summarize_with_openai(api_key, input_data, prompt)
        elif api_endpoint.lower() == "anthropic":
            response = summarize_with_anthropic(api_key, input_data, prompt)
        elif api_endpoint.lower() == "cohere":
            response = summarize_with_cohere(api_key, input_data, prompt)
        elif api_endpoint.lower() == "groq":
            response = summarize_with_groq(api_key, input_data, prompt)
        elif api_endpoint.lower() == "openrouter":
            response = summarize_with_openrouter(api_key, input_data, prompt)
        elif api_endpoint.lower() == "deepseek":
            response = summarize_with_deepseek(api_key, input_data, prompt)
        elif api_endpoint.lower() == "llama.cpp":
            response = summarize_with_llama(input_data, prompt)
        elif api_endpoint.lower() == "kobold":
            response = summarize_with_kobold(input_data, api_key, prompt)
        elif api_endpoint.lower() == "ooba":
            response = summarize_with_oobabooga(input_data, api_key, prompt)
        elif api_endpoint.lower() == "tabbyapi":
            response = summarize_with_tabbyapi(input_data, prompt)
        elif api_endpoint.lower() == "vllm":
            response = summarize_with_vllm(input_data, prompt)
        elif api_endpoint.lower() == "local-llm":
            response = summarize_with_local_llm(input_data, prompt)
        elif api_endpoint.lower() == "huggingface":
            response = summarize_with_huggingface(api_key, input_data, prompt)
        else:
            raise ValueError(f"Unsupported API endpoint: {api_endpoint}")

        return response

    except Exception as e:
        logging.error(f"Error in chat function: {str(e)}")
        return f"An error occurred: {str(e)}"


def save_chat_history(history: List[List[str]], media_content: Dict[str, str], selected_parts: List[str],
                      api_endpoint: str, prompt: str):
    """
    Save the chat history along with context information to a JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"

    chat_data = {
        "timestamp": timestamp,
        "history": history,
        "context": {
            "selected_media": {
                part: media_content.get(part, "") for part in selected_parts
            },
            "api_endpoint": api_endpoint,
            "prompt": prompt
        }
    }

    json_data = json.dumps(chat_data, indent=2)

    return filename, json_data


def generate_timestamped_url(url, hours, minutes, seconds):
    # Extract video ID from the URL
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if not video_id_match:
        return "Invalid YouTube URL"

    video_id = video_id_match.group(1)

    # Calculate total seconds
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)

    # Generate the new URL
    new_url = f"https://www.youtube.com/watch?v={video_id}&t={total_seconds}s"

    return new_url





def create_chunking_inputs():
    chunk_text_by_words_checkbox = gr.Checkbox(label="Chunk Text by Words", value=False, visible=False)
    max_words_input = gr.Number(label="Max Words", value=300, precision=0, visible=False)
    chunk_text_by_sentences_checkbox = gr.Checkbox(label="Chunk Text by Sentences", value=False, visible=False)
    max_sentences_input = gr.Number(label="Max Sentences", value=10, precision=0, visible=False)
    chunk_text_by_paragraphs_checkbox = gr.Checkbox(label="Chunk Text by Paragraphs", value=False, visible=False)
    max_paragraphs_input = gr.Number(label="Max Paragraphs", value=5, precision=0, visible=False)
    chunk_text_by_tokens_checkbox = gr.Checkbox(label="Chunk Text by Tokens", value=False, visible=False)
    max_tokens_input = gr.Number(label="Max Tokens", value=1000, precision=0, visible=False)
    return [chunk_text_by_words_checkbox, max_words_input, chunk_text_by_sentences_checkbox, max_sentences_input,
            chunk_text_by_paragraphs_checkbox, max_paragraphs_input, chunk_text_by_tokens_checkbox, max_tokens_input]


def create_video_transcription_tab():
    with gr.Group():
        with gr.Row():
            theme_toggle = gr.Radio(choices=["Light", "Dark"], value="Light", label="Light/Dark Mode Toggle")
            ui_frontpage_mode_toggle = gr.Radio(choices=["Simple List", "Advanced List"], value="Simple List",
                                                label="UI Mode Options Toggle")
            chunk_summarization_toggle = gr.Radio(choices=["Non-Chunked", "Chunked-Summarization"], value="Non-Chunked",
                                                  label="Summarization Mode")

        url_input = gr.Textbox(label="URL (Mandatory)",
                               placeholder="Enter the video URL here. Multiple at once supported, one per line")
        diarize_input = gr.Checkbox(label="Enable Speaker Diarization", visible=True)
        num_speakers_input = gr.Number(value=2, label="Number of Speakers(Optional - Currently has no effect)",
                                       visible=False)
        whisper_model_input = gr.Dropdown(choices=whisper_models, value="medium", label="Whisper Model", visible=False)
        custom_prompt_input = gr.Textbox(label="Custom Prompt", placeholder="Enter custom prompt here", lines=3,
                                         visible=True)
        offset_input = gr.Number(value=0, label="Offset (Seconds into the video to start transcribing at)",
                                 visible=False)
        api_name_input = gr.Dropdown(
            choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "OpenRouter", "Llama.cpp",
                     "Kobold", "Ooba", "Tabbyapi", "VLLM", "HuggingFace"], value=None, label="API Name (Mandatory)",
            visible=True)
        api_key_input = gr.Textbox(label="API Key (Mandatory)", placeholder="Enter your API key here", visible=True)
        vad_filter_input = gr.Checkbox(label="VAD Filter (WIP)", value=False, visible=False)
        rolling_summarization_input = gr.Checkbox(label="Enable Rolling Summarization", value=False, visible=False)
        download_video_input = gr.Checkbox(label="Download Video", value=False, visible=False)
        download_audio_input = gr.Checkbox(label="Download Audio", value=False, visible=False)
        detail_level_input = gr.Slider(minimum=0.01, maximum=1.0, value=0.01, step=0.01, interactive=True,
                                       label="Summary Detail Level", visible=False)
        keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated)",
                                    value="default,no_keyword_set", visible=True)
        question_box_input = gr.Textbox(label="Question", placeholder="Enter a question to ask about the transcription",
                                        visible=False)
        local_file_path_input = gr.Textbox(label="Local File Path", placeholder="Enter the path to a local file", visible=False)
        chunking_inputs = create_chunking_inputs()

        outputs = [
            gr.Textbox(label="Transcription"),
            gr.Textbox(label="Summary or Status Message"),
            gr.File(label="Download Transcription as JSON"),
            gr.File(label="Download Summary as Text"),
            gr.File(label="Download Video", visible=False),
            gr.File(label="Download Audio", visible=False),
        ]

        process_button = gr.Button("Process Video")

        process_button.click(
            fn=process_url,
            inputs=[url_input, num_speakers_input, whisper_model_input, custom_prompt_input, offset_input,
                    api_name_input, api_key_input, vad_filter_input, download_video_input, download_audio_input,
                    rolling_summarization_input, detail_level_input, question_box_input,
                    keywords_input, local_file_path_input, diarize_input] + chunking_inputs,
            outputs=outputs
        )


def create_audio_processing_tab():
    with gr.Group():
        gr.Markdown("# Transcribe & Summarize Audio Files from URLs or Local Files!")
        audio_url_input = gr.Textbox(label="Audio File URL", placeholder="Enter the URL of the audio file")
        audio_file_input = gr.File(label="Upload Audio File", file_types=["audio/*"])
        whisper_model_input = gr.Dropdown(choices=whisper_models, value="medium", label="Whisper Model")
        api_name_input = gr.Dropdown(
            choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "OpenRouter", "Llama.cpp",
                     "Kobold", "Ooba", "Tabbyapi", "VLLM", "HuggingFace"], value=None,
            label="API for Summarization (Optional)")
        api_key_input = gr.Textbox(label="API Key (if required)", placeholder="Enter your API key here",
                                   type="password")

        process_audio_button = gr.Button("Process Audio File")
        audio_progress_output = gr.Textbox(label="Progress")
        audio_transcriptions_output = gr.Textbox(label="Transcriptions")

        process_audio_button.click(
            fn=process_audio_file,
            inputs=[audio_url_input, audio_file_input, whisper_model_input, api_name_input, api_key_input],
            outputs=[audio_progress_output, audio_transcriptions_output]
        )


def create_website_scraping_tab():
    with gr.Group():
        gr.Markdown("# Scrape Websites & Summarize Articles using a Headless Chrome Browser!")
        gr.Markdown("In the plans to add support for custom cookies/logins...")
        url_input = gr.Textbox(label="Article URLs", placeholder="Enter article URLs here, one per line", lines=5)
        custom_article_title_input = gr.Textbox(label="Custom Article Titles (Optional, one per line)",
                                                placeholder="Enter custom titles for the articles, one per line",
                                                lines=5)
        custom_prompt_input = gr.Textbox(label="Custom Prompt (Optional)",
                                         placeholder="Provide a custom prompt for summarization", lines=3)
        api_name_input = gr.Dropdown(
            choices=[None, "huggingface", "deepseek", "openrouter", "openai", "anthropic", "cohere", "groq", "llama",
                     "kobold", "ooba"], value=None, label="API Name (Mandatory for Summarization)")
        api_key_input = gr.Textbox(label="API Key (Mandatory if API Name is specified)",
                                   placeholder="Enter your API key here; Ignore if using Local API or Built-in API")
        keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated)",
                                    value="default,no_keyword_set", visible=True)

        scrape_button = gr.Button("Scrape and Summarize")
        result_output = gr.Textbox(label="Result", lines=20)

        scrape_button.click(
            fn=scrape_and_summarize_multiple,
            inputs=[url_input, custom_prompt_input, api_name_input, api_key_input, keywords_input,
                    custom_article_title_input],
            outputs=result_output
        )


def create_pdf_ingestion_tab():
    with gr.Group():
        pdf_file_input = gr.File(label="Upload PDF File", file_types=[".pdf"])
        pdf_title_input = gr.Textbox(label="Title (Optional)")
        pdf_author_input = gr.Textbox(label="Author (Optional)")
        pdf_keywords_input = gr.Textbox(label="Keywords (Optional, comma-separated)")
        pdf_ingest_button = gr.Button("Ingest PDF")
        pdf_result_output = gr.Textbox(label="Result")

        pdf_ingest_button.click(
            fn=ingest_pdf_file,
            inputs=[pdf_file_input, pdf_author_input, pdf_title_input, pdf_keywords_input],
            outputs=pdf_result_output
        )


def create_search_tab():
    with gr.Group():
        gr.Markdown(
            "# Search across all ingested items in the Database by Title / URL / Keyword / or Content via SQLite Full-Text-Search")
        search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
        search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title", label="Search By")
        search_button = gr.Button("Search")
        items_output = gr.Dropdown(label="Select Item", choices=[])
        item_mapping = gr.State({})
        prompt_summary_output = gr.HTML(label="Prompt & Summary", visible=True)
        content_output = gr.Markdown(label="Content", visible=True)

        search_button.click(
            fn=update_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[items_output, item_mapping]
        )

        items_output.change(
            fn=update_detailed_view,
            inputs=[items_output, item_mapping],
            outputs=[prompt_summary_output, content_output]
        )


def create_llamafile_settings_tab():
    with gr.Group():
        gr.Markdown("# Settings for Llamafile")
        am_noob = gr.Checkbox(label="Check this to enable sane defaults", value=False, visible=True)
        advanced_mode_toggle = gr.Checkbox(label="Advanced Mode - Enable to show all settings", value=False)

        model_checked = gr.Checkbox(label="Enable Setting Local LLM Model Path", value=False, visible=True)
        model_value = gr.Textbox(label="Select Local Model File", value="", visible=True)
        ngl_checked = gr.Checkbox(label="Enable Setting GPU Layers", value=False, visible=True)
        ngl_value = gr.Number(label="Number of GPU Layers", value=None, precision=0, visible=True)

        advanced_inputs = create_llamafile_advanced_inputs()

        start_button = gr.Button("Start Llamafile")
        stop_button = gr.Button("Stop Llamafile")
        output_display = gr.Markdown()

        start_button.click(
            fn=start_llamafile,
            inputs=[am_noob, model_checked, model_value, ngl_checked, ngl_value] + advanced_inputs,
            outputs=output_display
        )


def create_llamafile_advanced_inputs():
    verbose_checked = gr.Checkbox(label="Enable Verbose Output", value=False, visible=False)
    threads_checked = gr.Checkbox(label="Set CPU Threads", value=False, visible=False)
    threads_value = gr.Number(label="Number of CPU Threads", value=None, precision=0, visible=False)
    http_threads_checked = gr.Checkbox(label="Set HTTP Server Threads", value=False, visible=False)
    http_threads_value = gr.Number(label="Number of HTTP Server Threads", value=None, precision=0, visible=False)
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

    return [verbose_checked, threads_checked, threads_value, http_threads_checked, http_threads_value,
            hf_repo_checked, hf_repo_value, hf_file_checked, hf_file_value, ctx_size_checked, ctx_size_value,
            host_checked, host_value, port_checked, port_value]


def create_chat_interface():
    with gr.Group():
        gr.Markdown("# Chat with a designated LLM Endpoint, using your selected item as starting context")

        with gr.Row():
            with gr.Column(scale=1):
                search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
                search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                             label="Search By")
                search_button = gr.Button("Search")

            with gr.Column(scale=2):
                items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                item_mapping = gr.State({})

        with gr.Row():
            use_content = gr.Checkbox(label="Use Content")
            use_summary = gr.Checkbox(label="Use Summary")
            use_prompt = gr.Checkbox(label="Use Prompt")

        api_endpoint = gr.Dropdown(label="Select API Endpoint",
                                   choices=["Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek",
                                            "OpenRouter", "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM",
                                            "HuggingFace"])
        api_key = gr.Textbox(label="API Key (if required)", type="password")
        preset_prompt = gr.Dropdown(label="Select Preset Prompt", choices=load_preset_prompts())
        user_prompt = gr.Textbox(label="Modify Prompt", lines=3)

        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Enter your message")
        clear = gr.Button("Clear")
        submit = gr.Button("Submit")

        chat_history = gr.State([])
        media_content = gr.State()
        selected_parts = gr.State([])

        save_button = gr.Button("Save Chat History")
        download_file = gr.File(label="Download Chat History")

        def chat_wrapper(message, history, media_content, selected_parts, api_endpoint, api_key, user_prompt):
            bot_message = chat(message, history, media_content, selected_parts, api_endpoint, api_key, user_prompt)
            history.append((message, bot_message))
            return "", history

        submit.click(
            chat_wrapper,
            inputs=[msg, chat_history, media_content, selected_parts, api_endpoint, api_key, user_prompt],
            outputs=[msg, chatbot]
        )

        clear.click(lambda: ([], []), outputs=[chatbot, chat_history])

        def save_chat_history(history):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(history, f)
            return filename

        save_button.click(save_chat_history, inputs=[chat_history], outputs=[download_file])

        search_button.click(
            fn=update_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[items_output, item_mapping]
        )

        def update_user_prompt(preset_name):
            details = fetch_prompt_details(preset_name)
            if details:
                return details[1]  # Return the system prompt
            return ""

        preset_prompt.change(update_user_prompt, inputs=preset_prompt, outputs=user_prompt)

        def update_chat_content(selected_item, use_content, use_summary, use_prompt):
            if selected_item in item_mapping:
                media_id = item_mapping[selected_item]
                content = load_media_content(media_id)
                selected_parts = []
                if use_content:
                    selected_parts.append("content")
                if use_summary:
                    selected_parts.append("summary")
                if use_prompt:
                    selected_parts.append("prompt")
                return content, selected_parts
            return None, []

        items_output.change(
            update_chat_content,
            inputs=[items_output, use_content, use_summary, use_prompt],
            outputs=[media_content, selected_parts]
        )


def create_media_edit_tab():
    with gr.Group():
        gr.Markdown("# Search and Edit Media Items")

        with gr.Row():
            search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
            search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                         label="Search By")
            search_button = gr.Button("Search")

        with gr.Row():
            items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
            item_mapping = gr.State({})

        content_input = gr.Textbox(label="Edit Content", lines=10)
        prompt_input = gr.Textbox(label="Edit Prompt", lines=3)
        summary_input = gr.Textbox(label="Edit Summary", lines=5)

        update_button = gr.Button("Update Media Content")
        status_message = gr.Textbox(label="Status", interactive=False)

        search_button.click(
            fn=update_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[items_output, item_mapping]
        )

        def load_selected_media_content(selected_item, item_mapping):
            if selected_item and item_mapping and selected_item in item_mapping:
                media_id = item_mapping[selected_item]
                content, prompt, summary = fetch_item_details(media_id)
                return content, prompt, summary
            return "No item selected or invalid selection", "", ""

        items_output.change(
            fn=load_selected_media_content,
            inputs=[items_output, item_mapping],
            outputs=[content_input, prompt_input, summary_input]
        )

        update_button.click(
            fn=update_media_content,
            inputs=[items_output, item_mapping, content_input, prompt_input, summary_input],
            outputs=status_message
        )


def create_keyword_tab():
    with gr.Group():
        gr.Markdown("# Keyword Management")

        with gr.Tab("Browse Keywords"):
            browse_output = gr.Markdown()
            browse_button = gr.Button("Browse Keywords")
            browse_button.click(fn=keywords_browser_interface, outputs=browse_output)

        with gr.Tab("Add Keywords"):
            add_input = gr.Textbox(label="Add Keywords (comma-separated)", placeholder="Enter keywords here...")
            add_button = gr.Button("Add Keywords")
            add_output = gr.Textbox(label="Result")
            add_button.click(fn=add_keyword, inputs=add_input, outputs=add_output)

        with gr.Tab("Delete Keywords"):
            delete_input = gr.Textbox(label="Delete Keyword", placeholder="Enter keyword to delete here...")
            delete_button = gr.Button("Delete Keyword")
            delete_output = gr.Textbox(label="Result")
            delete_button.click(fn=delete_keyword, inputs=delete_input, outputs=delete_output)



def import_data(file):
    # Placeholder for actual import functionality
    return "Data imported successfully"

def create_import_export_tab():
    with gr.Group():
        with gr.Tab("Export"):
            with gr.Tab("Export Search Results"):
                search_query = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
                search_fields = gr.CheckboxGroup(label="Search Fields", choices=["Title", "Content"], value=["Title"])
                keyword_input = gr.Textbox(
                    label="Keyword (Match ALL, can use multiple keywords, separated by ',' (comma) )",
                    placeholder="Enter keywords here...")
                page_input = gr.Number(label="Page", value=1, precision=0)
                results_per_file_input = gr.Number(label="Results per File", value=1000, precision=0)
                export_search_button = gr.Button("Export Search Results")
                export_search_output = gr.Textbox(label="Export Status")

                export_search_button.click(
                    fn=export_to_csv,
                    inputs=[search_query, search_fields, keyword_input, page_input, results_per_file_input],
                    outputs=export_search_output
                )

            with gr.Tab("Export Keywords"):
                export_keywords_button = gr.Button("Export Keywords")
                export_keywords_output = gr.File(label="Download Exported Keywords")
                export_keywords_status = gr.Textbox(label="Export Status")

                export_keywords_button.click(
                    fn=export_keywords_to_csv,
                    outputs=[export_keywords_output, export_keywords_status]
                )

        with gr.Tab("Import"):
            import_file = gr.File(label="Upload file for import")
            import_button = gr.Button("Import Data")
            import_output = gr.Textbox(label="Import Status")

            import_button.click(
                fn=import_data,
                inputs=import_file,
                outputs=import_output
            )


def create_utilities_tab():
    with gr.Group():
        with gr.Tab("YouTube Video Downloader"):
            gr.Markdown(
                "<h3>Youtube Video Downloader</h3><p>This Input takes a Youtube URL as input and creates a webm file for you to download. </br><em>If you want a full-featured one:</em> <strong><em>https://github.com/StefanLobbenmeier/youtube-dl-gui</strong></em> or <strong><em>https://github.com/yt-dlg/yt-dlg</em></strong></p>")
            youtube_url_input = gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL here")
            download_button = gr.Button("Download Video")
            output_file = gr.File(label="Download Video")

            download_button.click(
                fn=gradio_download_youtube_video,
                inputs=youtube_url_input,
                outputs=output_file
            )

        with gr.Tab("YouTube Audio Downloader"):
            gr.Markdown(
                "<h3>Youtube Audio Downloader</h3><p>This Input takes a Youtube URL as input and creates an audio file for you to download. </br><em>If you want a full-featured one:</em> <strong><em>https://github.com/StefanLobbenmeier/youtube-dl-gui</strong></em> or <strong><em>https://github.com/yt-dlg/yt-dlg</em></strong></p>")
            youtube_url_input_audio = gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL here")
            download_button_audio = gr.Button("Download Audio")
            output_file_audio = gr.File(label="Download Audio")

            # Implement the audio download functionality here

        with gr.Tab("Grammar Checker"):
            gr.Markdown("# Grammar Check Utility to be added...")

        with gr.Tab("YouTube Timestamp URL Generator"):
            gr.Markdown("## Generate YouTube URL with Timestamp")
            with gr.Row():
                url_input = gr.Textbox(label="YouTube URL")
                hours_input = gr.Number(label="Hours", value=0, minimum=0, precision=0)
                minutes_input = gr.Number(label="Minutes", value=0, minimum=0, maximum=59, precision=0)
                seconds_input = gr.Number(label="Seconds", value=0, minimum=0, maximum=59, precision=0)

            generate_button = gr.Button("Generate URL")
            output_url = gr.Textbox(label="Timestamped URL")

            generate_button.click(
                fn=generate_timestamped_url,
                inputs=[url_input, hours_input, minutes_input, seconds_input],
                outputs=output_url
            )


def launch_ui(demo_mode=False):
    if demo_mode == False:
        share_public = False
    else:
        share_public = True
    with gr.Blocks() as iface:
        gr.Markdown("# TL/DW: Too Long, Didn't Watch - Your Personal Research Multi-Tool")

        with gr.Tabs():
            with gr.TabItem("Transcription / Summarization / Ingestion"):
                create_video_transcription_tab()

            with gr.TabItem("Audio Processing"):
                create_audio_processing_tab()

            with gr.TabItem("Website Scraping"):
                create_website_scraping_tab()

            with gr.TabItem("PDF Ingestion"):
                create_pdf_ingestion_tab()

            with gr.TabItem("Search / Detailed View"):
                create_search_tab()

            with gr.TabItem("Local LLM with Llamafile"):
                create_llamafile_settings_tab()

            with gr.TabItem("Remote LLM Chat"):
                create_chat_interface()

            with gr.TabItem("Edit Existing Items"):
                create_media_edit_tab()

            with gr.TabItem("Keywords"):
                create_keyword_tab()

            with gr.TabItem("Export/Import"):
                create_import_export_tab()

            with gr.TabItem("Utilities"):
                create_utilities_tab()

    # Launch the interface
    server_port_variable = 7860
    if share_public is not None and share_public:
        iface.launch(share=True)
    elif server_mode and not share_public:
        iface.launch(share=False, server_name="0.0.0.0", server_port=server_port_variable)
    else:
        iface.launch(share=False)

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
    if model_checked and model_value is not None:
        model_path = model_value.name
        command.extend(['-m', model_path])
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




