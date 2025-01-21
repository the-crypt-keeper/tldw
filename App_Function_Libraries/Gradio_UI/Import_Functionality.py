# Import_Functionality.py
# Functionality to import content into the DB
#
# Imports
import json
from datetime import datetime
from time import sleep
import logging
import re
import shutil
import tempfile
import os
from pathlib import Path
import sqlite3
import traceback
from typing import Optional, List, Dict, Tuple
import uuid
import zipfile
#
# External Imports
import gradio as gr
from chardet import detect
from docx2txt import docx2txt
from pypandoc import convert_file
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import insert_prompt_to_db, import_obsidian_note_to_db, \
    add_media_to_database, list_prompts
from App_Function_Libraries.Metrics.metrics_logger import log_counter
from App_Function_Libraries.Prompt_Handling import import_prompt_from_file, import_prompts_from_zip#
from App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization
from App_Function_Libraries.Utils.Utils import FileProcessor, ZipValidator
#
###################################################################################################################
#
# Functions:

logger = logging.getLogger()

def import_data(file, title, author, keywords, custom_prompt, summary, auto_summarize, api_name, api_key):
    logging.debug(f"Starting import_data with file: {file} / Title: {title} / Author: {author} / Keywords: {keywords}")
    if file is None:
        return "No file uploaded. Please upload a file."

    # We'll define this here so we can clean it up at the very end
    temp_file_path = None

    try:
        logging.debug(f"File object type: {type(file)}")
        logging.debug(f"File object attributes: {dir(file)}")

        if hasattr(file, 'name'):
            file_name = file.name
        else:
            file_name = 'unknown_file'

        # Create a temporary file for reading the content
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as temp_file:
            # Keep track of the full path so we can remove it later
            temp_file_path = temp_file.name

            if isinstance(file, str):
                # 'file' is actually a string of content
                temp_file.write(file)
            elif hasattr(file, 'read'):
                # 'file' is a file-like object
                content = file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                temp_file.write(content)
            else:
                # If neither a string nor file-like, force string conversion
                temp_file.write(str(file))

            temp_file.flush()  # Make sure data is written
            temp_file.seek(0)
            file_content = temp_file.read()

        logging.debug(f"File name: {file_name}")
        logging.debug(f"File content (first 100 chars): {file_content[:100]}")

        # Build info_dict for DB storage
        info_dict = {
            'title': title or 'Untitled',
            'uploader': author or 'Unknown',
        }

        # Prepare segments (right now just one segment for everything)
        # If you intend to chunk, you can do that here:
        segments = [{'Text': file_content}]

        # Process keywords into a list
        if keywords:
            keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        else:
            keyword_list = []

        # If auto-summarize is enabled and we have an API, do summarization
        if auto_summarize and api_name and api_key:
            # FIXME - Make sure custom_prompt is system prompt
            summary = perform_summarization(api_name, file_content, custom_prompt, api_key)
        # If there's no user-provided summary, and we haven't auto-summarized:
        elif not summary:
            summary = "No summary provided"

        # --- ALWAYS add to database after we've finalized `summary` ---
        result = add_media_to_database(
            url=file_name,             # or any unique identifier
            info_dict=info_dict,
            segments=segments,
            summary=summary,
            keywords=keyword_list,
            custom_prompt_input=custom_prompt,
            whisper_model="Imported",  # indicates it was an imported file
            media_type="document",
            overwrite=False
        )

        return f"File '{file_name}' import attempt complete. Database result: {result}"

    except Exception as e:
        logging.exception(f"Error importing file: {str(e)}")
        return f"Error importing file: {str(e)}"

    finally:
        # Clean up the temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


###############################################################
#
# Plaintext/Markdown/RTF/Docx Import Functionality

def preview_import_handler(
        files,
        author,
        keywords,
        system_prompt,
        user_prompt,
        auto_summarize,
        api_name,
        api_key
):
    """
    Step 1: Read/convert files (or ZIP of multiple text files) + optionally auto-summarize,
    but DO NOT store in the DB.

    Returns:
      - A user-facing status string
      - A JSON string (preview_data_json) containing a list of file results:
          [
            {
               "filename": "...",
               "title": "...",
               "content": "...",
               "summary": "...",
               "author": "...",
               "keywords": [...],
               "system_prompt": "...",
               "user_prompt": "...",
               ...
            },
            ...
          ]
    """
    if not files:
        return "No files uploaded.", None

    results_for_ui = []
    preview_list = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file_obj in files:
            filename = os.path.basename(file_obj.name)
            try:
                # Make a temp copy
                temp_path = os.path.join(temp_dir, filename)
                with open(temp_path, 'wb') as f:
                    f.write(open(file_obj.name, 'rb').read())

                # If the file is a ZIP, extract and preview each valid item
                if temp_path.lower().endswith('.zip'):
                    with tempfile.TemporaryDirectory() as zip_temp_dir:
                        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                            zip_ref.extractall(zip_temp_dir)

                        for root, _, extracted_files in os.walk(zip_temp_dir):
                            for extracted_filename in extracted_files:
                                if extracted_filename.lower().endswith(('.md', '.txt', '.rtf', '.docx')):
                                    extracted_path = os.path.join(root, extracted_filename)
                                    file_info = _preview_single_file(
                                        extracted_path,
                                        author,
                                        keywords,
                                        system_prompt,
                                        user_prompt,
                                        auto_summarize,
                                        api_name,
                                        api_key
                                    )
                                    preview_list.append(file_info)
                                    results_for_ui.append(f"📄 [ZIP] {extracted_filename} => Success")

                    results_for_ui.append(f"📦 {filename} => Extracted successfully.")

                else:
                    # Single file scenario
                    file_info = _preview_single_file(
                        temp_path,
                        author,
                        keywords,
                        system_prompt,
                        user_prompt,
                        auto_summarize,
                        api_name,
                        api_key
                    )
                    preview_list.append(file_info)
                    results_for_ui.append(f"📄 {filename} => Success")

            except Exception as e:
                logging.exception(f"❌ Error with file: {filename}")
                results_for_ui.append(f"❌ {filename} => {str(e)}")

    # Convert list of file info dicts to JSON so we can store in gr.State or similar
    preview_data_json = json.dumps(preview_list, ensure_ascii=False)
    status_message = "\n".join(results_for_ui)

    return status_message, preview_data_json


def _preview_single_file(
        file_path,
        author,
        keywords,
        system_prompt,
        user_prompt,
        auto_summarize,
        api_name,
        api_key
):
    """
    Internal helper to read/convert a single file into plain text,
    optionally auto-summarize, and return a dictionary describing the
    would-be DB record (but does not ingest).
    """
    log_counter("file_preview_attempt", labels={"file_path": file_path})

    # Derive a filename-based title
    filename = os.path.basename(file_path)
    title = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1].lower()

    # 1) Read/convert content
    try:
        if extension == '.rtf':
            with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_md:
                convert_file(file_path, 'md', outputfile=temp_md.name)
                file_path_md = temp_md.name
            with open(file_path_md, 'r', encoding='utf-8') as f:
                content = f.read()
        elif extension == '.docx':
            content = docx2txt.process(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except Exception as e:
        logging.error(f"Error reading file content: {str(e)}")
        return {
            "filename": filename,
            "title": title,
            "content": f"Error reading file: {e}",
            "summary": None,
            "author": author,
            "keywords": keywords.split(",") if keywords else [],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "auto_summarize": auto_summarize,
            "api_name": api_name,
            "api_key": api_key,
        }

    # 2) Optionally auto-summarize
    summary = None
    if auto_summarize and api_name and api_key:
        combined_prompt = (system_prompt or "") + "\n\n" + (user_prompt or "")
        summary = perform_summarization(
            api_name=api_name,
            text=content,
            prompt=combined_prompt,
            api_key=api_key
        )

    if not summary:
        summary = "No summary provided"

    # 3) Return the file info dict (not ingested yet)
    return {
        "filename": filename,
        "title": title,
        "content": content,
        "summary": summary,
        "author": author,
        "keywords": keywords.split(",") if keywords else [],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "auto_summarize": auto_summarize,
        "api_name": api_name,
        "api_key": api_key
    }


def final_ingest_handler(preview_data_json, updated_metadata_json):
    """
    Step 2: Actually ingest data into the database using add_media_to_database.

    - preview_data_json: The JSON output from preview_import_handler()
    - updated_metadata_json: (Optional) JSON from the user specifying
        overrides for each file.

    Returns a status string (success/fail for each file).
    """
    if not preview_data_json:
        return "No preview data found. Please run the preview step first."

    try:
        preview_list = json.loads(preview_data_json)
    except Exception as e:
        logging.exception("Error loading preview data.")
        return f"Error parsing preview data: {e}"

    # Parse user-supplied overrides (if any)
    if updated_metadata_json:
        try:
            overrides_dict = json.loads(updated_metadata_json)
        except Exception as e:
            logging.exception("Error loading user metadata overrides.")
            overrides_dict = {}
    else:
        overrides_dict = {}

    results = []
    for file_info in preview_list:
        fname = file_info["filename"]
        # Attempt to match user overrides by filename (the base name without extension, or the full fname).
        # Typically the user might key by "Doc1" vs "Doc1.txt". Decide how you want to match.
        # Here we assume the user’s JSON keys match exactly the 'filename' in file_info.
        this_file_overrides = overrides_dict.get(fname, {})

        # Combine final metadata
        final_author = this_file_overrides.get("author", file_info["author"])
        final_keywords = this_file_overrides.get("keywords", file_info["keywords"])
        final_title = this_file_overrides.get("title", file_info["title"])
        final_summary = this_file_overrides.get("summary", file_info["summary"])

        text_content = file_info["content"]  # The converted text

        # Construct combined prompts if needed or just store them
        combined_prompt = (file_info["system_prompt"] or "") + "\n\n" + (file_info["user_prompt"] or "")

        # Now do the actual DB ingestion
        try:
            db_result = add_media_to_database(
                url=fname,  # or some unique identifier
                info_dict={
                    "title": final_title,
                    "uploader": final_author,
                },
                segments=[{"Text": text_content}],
                summary=final_summary,
                keywords=final_keywords,
                custom_prompt_input=combined_prompt,
                whisper_model="Imported",
                media_type="document",
                overwrite=False
            )
            results.append(f"✅ {fname} => {db_result}")
        except Exception as e:
            logging.exception(f"Error ingesting file {fname}")
            results.append(f"❌ {fname} => {str(e)}")

    # Return an overall string
    return "\n".join(results)

#
# End of Plaintext/Markdown/RTF/Docx Import Functionality
###############################################################


###############################################################
#
# Plaintext/Markdown Import Functionality

def create_import_item_tab():
    with gr.TabItem("Import Markdown/Text Files", visible=True):
        gr.Markdown("# Import a markdown file or text file into the database")
        gr.Markdown("...and have it tagged + summarized")
        with gr.Row():
            with gr.Column():
                import_file = gr.File(label="Upload file for import", file_types=["txt", "md"])
                title_input = gr.Textbox(label="Title", placeholder="Enter the title of the content")
                author_input = gr.Textbox(label="Author", placeholder="Enter the author's name")
                keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords, comma-separated")
                custom_prompt_input = gr.Textbox(label="Custom Prompt",
                                             placeholder="Enter a custom prompt for summarization (optional)")
                summary_input = gr.Textbox(label="Summary",
                                       placeholder="Enter a summary or leave blank for auto-summarization", lines=3)
                auto_summarize_checkbox = gr.Checkbox(label="Auto-summarize", value=False)
                api_name_input = gr.Dropdown(
                choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                         "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM","ollama", "HuggingFace", "Custom-OpenAI-API"],
                label="API for Auto-summarization"
                )
                api_key_input = gr.Textbox(label="API Key", type="password")
            with gr.Column():
                import_button = gr.Button("Import Data")
                import_output = gr.Textbox(label="Import Status")

        import_button.click(
            fn=import_data,
            inputs=[import_file, title_input, author_input, keywords_input, custom_prompt_input,
                    summary_input, auto_summarize_checkbox, api_name_input, api_key_input],
            outputs=import_output
        )

#
# Import Plaintext/Markdown Tab
###############################################################


###############################################################
#
# Prompt Import Functionality

def create_import_single_prompt_tab():
    with gr.TabItem("Import a Prompt", visible=True):
        gr.Markdown("# Import a prompt into the database")

        with gr.Row():
            with gr.Column():
                import_file = gr.File(label="Upload file for import", file_types=["txt", "md"])
                title_input = gr.Textbox(label="Title", placeholder="Enter the title of the content")
                author_input = gr.Textbox(label="Author", placeholder="Enter the author's name")
                system_input = gr.Textbox(label="System", placeholder="Enter the system message for the prompt", lines=3)
                user_input = gr.Textbox(label="User", placeholder="Enter the user message for the prompt", lines=3)
                keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords separated by commas")
                import_button = gr.Button("Import Prompt")

            with gr.Column():
                import_output = gr.Textbox(label="Import Status")
                save_button = gr.Button("Save to Database")
                save_output = gr.Textbox(label="Save Status")

        def handle_import(file):
            result = import_prompt_from_file(file)
            if isinstance(result, tuple) and len(result) == 5:
                title, author, system, user, keywords = result
                return gr.update(value="File successfully imported. You can now edit the content before saving."), \
                       gr.update(value=title), gr.update(value=author), gr.update(value=system), \
                       gr.update(value=user), gr.update(value=", ".join(keywords))
            else:
                return gr.update(value=result), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        import_button.click(
            fn=handle_import,
            inputs=[import_file],
            outputs=[import_output, title_input, author_input, system_input, user_input, keywords_input]
        )

        def save_prompt_to_db(title, author, system, user, keywords):
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            return insert_prompt_to_db(title, author, system, user, keyword_list)

        save_button.click(
            fn=save_prompt_to_db,
            inputs=[title_input, author_input, system_input, user_input, keywords_input],
            outputs=save_output
        )


def create_import_multiple_prompts_tab():
    with gr.TabItem("Import Multiple Prompts", visible=True):
        gr.Markdown("# Import multiple prompts into the database")
        gr.Markdown("Upload a zip file containing multiple prompt files (txt or md)")

        # Initialize state variables for pagination
        current_page_state = gr.State(value=1)
        total_pages_state = gr.State(value=1)

        with gr.Row():
            with gr.Column():
                zip_file = gr.File(label="Upload zip file for import", file_types=["zip"])
                import_button = gr.Button("Import Prompts")
                prompts_dropdown = gr.Dropdown(label="Select Prompt to Edit", choices=[])
                prev_page_button = gr.Button("Previous Page", visible=False)
                page_display = gr.Markdown("Page 1 of X", visible=False)
                next_page_button = gr.Button("Next Page", visible=False)
                title_input = gr.Textbox(label="Title", placeholder="Enter the title of the content")
                author_input = gr.Textbox(label="Author", placeholder="Enter the author's name")
                system_input = gr.Textbox(label="System", placeholder="Enter the system message for the prompt",
                                          lines=3)
                user_input = gr.Textbox(label="User", placeholder="Enter the user message for the prompt", lines=3)
                keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords separated by commas")

            with gr.Column():
                import_output = gr.Textbox(label="Import Status")
                save_button = gr.Button("Save to Database")
                save_output = gr.Textbox(label="Save Status")
                prompts_display = gr.Textbox(label="Identified Prompts")

        # State to store imported prompts
        zip_import_state = gr.State([])

        # Function to handle zip import
        def handle_zip_import(zip_file):
            result = import_prompts_from_zip(zip_file)
            if isinstance(result, list):
                prompt_titles = [prompt['title'] for prompt in result]
                return gr.update(
                    value="Zip file successfully imported. Select a prompt to edit from the dropdown."), prompt_titles, gr.update(
                    value="\n".join(prompt_titles)), result
            else:
                return gr.update(value=result), [], gr.update(value=""), []

        import_button.click(
            fn=handle_zip_import,
            inputs=[zip_file],
            outputs=[import_output, prompts_dropdown, prompts_display, zip_import_state]
        )

        # Function to handle prompt selection from imported prompts
        def handle_prompt_selection(selected_title, prompts):
            selected_prompt = next((prompt for prompt in prompts if prompt['title'] == selected_title), None)
            if selected_prompt:
                return (
                    selected_prompt['title'],
                    selected_prompt.get('author', ''),
                    selected_prompt['system'],
                    selected_prompt.get('user', ''),
                    ", ".join(selected_prompt.get('keywords', []))
                )
            else:
                return "", "", "", "", ""

        zip_import_state = gr.State([])

        import_button.click(
            fn=handle_zip_import,
            inputs=[zip_file],
            outputs=[import_output, prompts_dropdown, prompts_display, zip_import_state]
        )

        prompts_dropdown.change(
            fn=handle_prompt_selection,
            inputs=[prompts_dropdown, zip_import_state],
            outputs=[title_input, author_input, system_input, user_input, keywords_input]
        )

        # Function to save prompt to the database
        def save_prompt_to_db(title, author, system, user, keywords):
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            result = insert_prompt_to_db(title, author, system, user, keyword_list)
            return result

        save_button.click(
            fn=save_prompt_to_db,
            inputs=[title_input, author_input, system_input, user_input, keywords_input],
            outputs=[save_output]
        )

        # Adding pagination controls to navigate prompts in the database
        def on_prev_page_click(current_page, total_pages):
            new_page = max(current_page - 1, 1)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
            page_display_text = f"Page {current_page} of {total_pages}"
            return (
                gr.update(choices=prompts),
                gr.update(value=page_display_text),
                current_page
            )

        def on_next_page_click(current_page, total_pages):
            new_page = min(current_page + 1, total_pages)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
            page_display_text = f"Page {current_page} of {total_pages}"
            return (
                gr.update(choices=prompts),
                gr.update(value=page_display_text),
                current_page
            )

        prev_page_button.click(
            fn=on_prev_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[prompts_dropdown, page_display, current_page_state]
        )

        next_page_button.click(
            fn=on_next_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[prompts_dropdown, page_display, current_page_state]
        )

        # Function to update prompts dropdown after saving to the database
        def update_prompt_dropdown():
            prompts, total_pages, current_page = list_prompts(page=1, per_page=10)
            page_display_text = f"Page {current_page} of {total_pages}"
            return (
                gr.update(choices=prompts),
                gr.update(visible=True),
                gr.update(value=page_display_text, visible=True),
                current_page,
                total_pages
            )

        # Update the dropdown after saving
        save_button.click(
            fn=update_prompt_dropdown,
            inputs=[],
            outputs=[prompts_dropdown, prev_page_button, page_display, current_page_state, total_pages_state]
        )

#
# End of Prompt Import Functionality
###############################################################


###############################################################
#
# Obsidian Vault Import Functionality

def process_obsidian_zip(zip_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            imported_files, total_files, errors = import_obsidian_vault(temp_dir)

            return imported_files, total_files, errors
        except zipfile.BadZipFile:
            error_msg = "The uploaded file is not a valid zip file."
            logger.error(error_msg)
            return 0, 0, [error_msg]
        except Exception as e:
            error_msg = f"Error processing zip file: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return 0, 0, [error_msg]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)



def scan_obsidian_vault(vault_path):
    markdown_files = []
    for root, dirs, files in os.walk(vault_path):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    return markdown_files


def parse_obsidian_note(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    frontmatter = {}
    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        import yaml
        frontmatter = yaml.safe_load(frontmatter_text)
        content = content[frontmatter_match.end():]

    tags = re.findall(r'#(\w+)', content)
    links = re.findall(r'\[\[(.*?)\]\]', content)

    return {
        'title': os.path.basename(file_path).replace('.md', ''),
        'content': content,
        'frontmatter': frontmatter,
        'tags': tags,
        'links': links,
        'file_path': file_path  # Add this line
    }

def create_import_obsidian_vault_tab():
    with gr.TabItem("Import Obsidian Vault", visible=True):
        gr.Markdown("## Import Obsidian Vault")
        with gr.Row():
            with gr.Column():
                vault_path_input = gr.Textbox(label="Obsidian Vault Path (Local)")
                vault_zip_input = gr.File(label="Upload Obsidian Vault (Zip)")
            with gr.Column():
                import_vault_button = gr.Button("Import Obsidian Vault")
                import_status = gr.Textbox(label="Import Status", interactive=False)


    def import_vault(vault_path, vault_zip):
        if vault_zip:
            imported, total, errors = process_obsidian_zip(vault_zip.name)
        elif vault_path:
            imported, total, errors = import_obsidian_vault(vault_path)
        else:
            return "Please provide either a local vault path or upload a zip file."

        status = f"Imported {imported} out of {total} files.\n"
        if errors:
            status += f"Encountered {len(errors)} errors:\n" + "\n".join(errors)
        return status


    import_vault_button.click(
        fn=import_vault,
        inputs=[vault_path_input, vault_zip_input],
        outputs=[import_status],
    )


def import_obsidian_vault(vault_path, progress=gr.Progress()):
    try:
        markdown_files = scan_obsidian_vault(vault_path)
        total_files = len(markdown_files)
        imported_files = 0
        errors = []

        for i, file_path in enumerate(markdown_files):
            try:
                note_data = parse_obsidian_note(file_path)
                success, error_msg = import_obsidian_note_to_db(note_data)
                if success:
                    imported_files += 1
                else:
                    errors.append(error_msg)
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

            progress((i + 1) / total_files, f"Imported {imported_files} of {total_files} files")
            sleep(0.1)  # Small delay to prevent UI freezing

        return imported_files, total_files, errors
    except Exception as e:
        error_msg = f"Error scanning vault: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return 0, 0, [error_msg]

#
# End of Obsidian Vault Import Functionality
###############################################################


###############################################################
#
# RAG Chat Conversation Import Functionality

class RAGQABatchImporter:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.setup_logging()
        self.file_processor = FileProcessor()
        self.zip_validator = ZipValidator()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_qa_import.log'),
                logging.StreamHandler()
            ]
        )

    def process_markdown_content(self, content: str) -> List[Dict[str, str]]:
        """Process markdown content into a conversation format."""
        messages = []
        sections = content.split('\n\n')

        for section in sections:
            if section.strip():
                messages.append({
                    'role': 'user',
                    'content': section.strip()
                })

        return messages

    def process_keywords(self, db: sqlite3.Connection, conversation_id: str, keywords: str):
        """Process and link keywords to a conversation."""
        if not keywords:
            return

        keyword_list = [k.strip() for k in keywords.split(',')]
        for keyword in keyword_list:
            # Insert keyword if it doesn't exist
            db.execute("""
                INSERT OR IGNORE INTO rag_qa_keywords (keyword)
                VALUES (?)
            """, (keyword,))

            # Get keyword ID
            keyword_id = db.execute("""
                SELECT id FROM rag_qa_keywords WHERE keyword = ?
            """, (keyword,)).fetchone()[0]

            # Link keyword to conversation
            db.execute("""
                INSERT INTO rag_qa_conversation_keywords 
                (conversation_id, keyword_id)
                VALUES (?, ?)
            """, (conversation_id, keyword_id))

    def import_single_file(
            self,
            db: sqlite3.Connection,
            content: str,
            filename: str,
            keywords: str,
            custom_prompt: Optional[str] = None,
            rating: Optional[int] = None
    ) -> str:
        """Import a single file's content into the database"""
        conversation_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        # Process filename into title
        title = FileProcessor.process_filename_to_title(filename)
        if title.lower().endswith(('.md', '.txt')):
            title = title[:-3] if title.lower().endswith('.md') else title[:-4]

        # Insert conversation metadata
        db.execute("""
            INSERT INTO conversation_metadata 
            (conversation_id, created_at, last_updated, title, rating)
            VALUES (?, ?, ?, ?, ?)
        """, (conversation_id, current_time, current_time, title, rating))

        # Process content and insert messages
        messages = self.process_markdown_content(content)
        for msg in messages:
            db.execute("""
                INSERT INTO rag_qa_chats 
                (conversation_id, timestamp, role, content)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, current_time, msg['role'], msg['content']))

        # Process keywords
        self.process_keywords(db, conversation_id, keywords)

        return conversation_id

    def extract_zip(self, zip_path: str) -> List[Tuple[str, str]]:
        """Extract and validate files from zip"""
        is_valid, error_msg, valid_files = self.zip_validator.validate_zip_file(zip_path)
        if not is_valid:
            raise ValueError(error_msg)

        files = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for filename in valid_files:
                with zip_ref.open(filename) as f:
                    content = f.read()
                    # Try to decode with detected encoding
                    try:
                        detected_encoding = detect(content)['encoding'] or 'utf-8'
                        content = content.decode(detected_encoding)
                    except UnicodeDecodeError:
                        content = content.decode('utf-8', errors='replace')

                    filename = os.path.basename(filename)
                    files.append((filename, content))
        return files

    def import_files(
            self,
            files: List[str],
            keywords: str = "",
            custom_prompt: Optional[str] = None,
            rating: Optional[int] = None,
            progress=gr.Progress()
    ) -> Tuple[bool, str]:
        """Import multiple files or zip files into the RAG QA database."""
        try:
            imported_files = []

            with sqlite3.connect(self.db_path) as db:
                # Process each file
                for file_path in progress.tqdm(files, desc="Processing files"):
                    filename = os.path.basename(file_path)

                    # Handle zip files
                    if filename.lower().endswith('.zip'):
                        zip_files = self.extract_zip(file_path)
                        for zip_filename, content in progress.tqdm(zip_files, desc=f"Processing files from {filename}"):
                            conv_id = self.import_single_file(
                                db=db,
                                content=content,
                                filename=zip_filename,
                                keywords=keywords,
                                custom_prompt=custom_prompt,
                                rating=rating
                            )
                            imported_files.append(zip_filename)

                    # Handle individual markdown/text files
                    elif filename.lower().endswith(('.md', '.txt')):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        conv_id = self.import_single_file(
                            db=db,
                            content=content,
                            filename=filename,
                            keywords=keywords,
                            custom_prompt=custom_prompt,
                            rating=rating
                        )
                        imported_files.append(filename)

                db.commit()

            return True, f"Successfully imported {len(imported_files)} files:\n" + "\n".join(imported_files)

        except Exception as e:
            logging.error(f"Import failed: {str(e)}")
            return False, f"Import failed: {str(e)}"


def create_conversation_import_tab() -> gr.Tab:
    """Create the import tab for the Gradio interface"""
    with gr.Tab("Import RAG Chats") as tab:
        gr.Markdown("# Import RAG Chats into the Database")
        gr.Markdown("""
        Import your RAG Chat markdown/text files individually or as a zip archive

        Supported file types:
        - Markdown (.md)
        - Text (.txt)
        - Zip archives containing .md or .txt files

        Maximum zip file size: 100MB
        Maximum files per zip: 100
        """)
        with gr.Row():
            with gr.Column():
                import_files = gr.File(
                    label="Upload Files",
                    file_types=["txt", "md", "zip"],
                    file_count="multiple"
                )

                keywords_input = gr.Textbox(
                    label="Keywords",
                    placeholder="Enter keywords to apply to all imported files (comma-separated)"
                )

                custom_prompt_input = gr.Textbox(
                    label="Custom Prompt",
                    placeholder="Enter a custom prompt for processing (optional)"
                )

                rating_input = gr.Slider(
                    minimum=1,
                    maximum=3,
                    step=1,
                    label="Rating (1-3)",
                    value=None
                )

            with gr.Column():
                import_button = gr.Button("Import Files")
                import_output = gr.Textbox(
                    label="Import Status",
                    lines=10
                )

        def handle_import(files, keywords, custom_prompt, rating):
            importer = RAGQABatchImporter("rag_qa.db")  # Update with your DB path
            success, message = importer.import_files(
                files=[f.name for f in files],
                keywords=keywords,
                custom_prompt=custom_prompt,
                rating=rating
            )
            return message

        import_button.click(
            fn=handle_import,
            inputs=[
                import_files,
                keywords_input,
                custom_prompt_input,
                rating_input
            ],
            outputs=import_output
        )

    return tab

#
# End of Conversation Import Functionality
###############################################################


#
# End of Import_Functionality.py
########################################################################################################################
