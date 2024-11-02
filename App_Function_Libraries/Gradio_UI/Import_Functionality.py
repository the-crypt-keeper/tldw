# Import_Functionality.py
# Functionality to import content into the DB
#
# Imports
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

#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import insert_prompt_to_db, load_preset_prompts, import_obsidian_note_to_db, \
    add_media_to_database
from App_Function_Libraries.Prompt_Handling import import_prompt_from_file, import_prompts_from_zip#
from App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization
#
###################################################################################################################
#
# Functions:

logger = logging.getLogger()


def import_data(file, title, author, keywords, custom_prompt, summary, auto_summarize, api_name, api_key):
    logging.debug(f"Starting import_data with file: {file} / Title: {title} / Author: {author} / Keywords: {keywords}")
    if file is None:
        return "No file uploaded. Please upload a file."

    try:
        logging.debug(f"File object type: {type(file)}")
        logging.debug(f"File object attributes: {dir(file)}")

        if hasattr(file, 'name'):
            file_name = file.name
        else:
            file_name = 'unknown_file'

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as temp_file:
            if isinstance(file, str):
                # If file is a string, it's likely file content
                temp_file.write(file)
            elif hasattr(file, 'read'):
                # If file has a 'read' method, it's likely a file-like object
                content = file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                temp_file.write(content)
            else:
                # If it's neither a string nor a file-like object, try converting it to a string
                temp_file.write(str(file))

            temp_file.seek(0)
            file_content = temp_file.read()

        logging.debug(f"File name: {file_name}")
        logging.debug(f"File content (first 100 chars): {file_content[:100]}")

        # Create info_dict
        info_dict = {
            'title': title or 'Untitled',
            'uploader': author or 'Unknown',
        }

        # FIXME - Add chunking support... I added chapter chunking specifically for this...
        # Create segments (assuming one segment for the entire content)
        segments = [{'Text': file_content}]

        # Process keywords
        keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()] if keywords else []

        # Handle summarization
        if auto_summarize and api_name and api_key:
            summary = perform_summarization(api_name, file_content, custom_prompt, api_key)
        elif not summary:
            summary = "No summary provided"

            # Add to database
            result = add_media_to_database(
                url=file_name,  # Using filename as URL
                info_dict=info_dict,
                segments=segments,
                summary=summary,
                keywords=keyword_list,
                custom_prompt_input=custom_prompt,
                whisper_model="Imported",  # Indicating this was an imported file
                media_type="document",
                overwrite=False  # Set this to True if you want to overwrite existing entries
            )

            # Clean up the temporary file
            os.unlink(temp_file.name)

            return f"File '{file_name}' import attempt complete. Database result: {result}"
    except Exception as e:
        logging.exception(f"Error importing file: {str(e)}")
        return f"Error importing file: {str(e)}"


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

        def update_prompt_dropdown():
            return gr.update(choices=load_preset_prompts())

        save_button.click(
            fn=update_prompt_dropdown,
            inputs=[],
            outputs=[gr.Dropdown(label="Select Preset Prompt")]
        )

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


def create_import_multiple_prompts_tab():
    with gr.TabItem("Import Multiple Prompts", visible=True):
        gr.Markdown("# Import multiple prompts into the database")
        gr.Markdown("Upload a zip file containing multiple prompt files (txt or md)")

        with gr.Row():
            with gr.Column():
                zip_file = gr.File(label="Upload zip file for import", file_types=["zip"])
                import_button = gr.Button("Import Prompts")
                prompts_dropdown = gr.Dropdown(label="Select Prompt to Edit", choices=[])
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

        def handle_zip_import(zip_file):
            result = import_prompts_from_zip(zip_file)
            if isinstance(result, list):
                prompt_titles = [prompt['title'] for prompt in result]
                return gr.update(
                    value="Zip file successfully imported. Select a prompt to edit from the dropdown."), prompt_titles, gr.update(
                    value="\n".join(prompt_titles)), result
            else:
                return gr.update(value=result), [], gr.update(value=""), []

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

        def save_prompt_to_db(title, author, system, user, keywords):
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            return insert_prompt_to_db(title, author, system, user, keyword_list)

        save_button.click(
            fn=save_prompt_to_db,
            inputs=[title_input, author_input, system_input, user_input, keywords_input],
            outputs=save_output
        )

        def update_prompt_dropdown():
            return gr.update(choices=load_preset_prompts())

        save_button.click(
            fn=update_prompt_dropdown,
            inputs=[],
            outputs=[gr.Dropdown(label="Select Preset Prompt")]
        )


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
                        progress.tqdm(desc=f"Processing files from {filename}")
                        for zip_filename, content in zip_files:
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


class FileProcessor:
    """Handles file reading and name processing"""

    VALID_EXTENSIONS = {'.md', '.txt', '.zip'}
    ENCODINGS_TO_TRY = [
        'utf-8',
        'utf-16',
        'windows-1252',
        'iso-8859-1',
        'ascii'
    ]

    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect the file encoding using chardet"""
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = detect(raw_data)
            return result['encoding'] or 'utf-8'

    @staticmethod
    def read_file_content(file_path: str) -> str:
        """Read file content with automatic encoding detection"""
        detected_encoding = FileProcessor.detect_encoding(file_path)

        # Try detected encoding first
        try:
            with open(file_path, 'r', encoding=detected_encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # If detected encoding fails, try others
            for encoding in FileProcessor.ENCODINGS_TO_TRY:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, use utf-8 with error handling
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()

    @staticmethod
    def process_filename_to_title(filename: str) -> str:
        """Convert filename to a readable title"""
        # Remove extension
        name = os.path.splitext(filename)[0]

        # Look for date patterns
        date_pattern = r'(\d{4}[-_]?\d{2}[-_]?\d{2})'
        date_match = re.search(date_pattern, name)
        date_str = ""
        if date_match:
            try:
                date = datetime.strptime(date_match.group(1).replace('_', '-'), '%Y-%m-%d')
                date_str = date.strftime("%b %d, %Y")
                name = name.replace(date_match.group(1), '').strip('-_')
            except ValueError:
                pass

        # Replace separators with spaces
        name = re.sub(r'[-_]+', ' ', name)

        # Remove redundant spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # Capitalize words, excluding certain words
        exclude_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        words = name.split()
        capitalized = []
        for i, word in enumerate(words):
            if i == 0 or word not in exclude_words:
                capitalized.append(word.capitalize())
            else:
                capitalized.append(word.lower())
        name = ' '.join(capitalized)

        # Add date if found
        if date_str:
            name = f"{name} - {date_str}"

        return name


class ZipValidator:
    """Validates zip file contents and structure"""

    MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_FILES = 100
    VALID_EXTENSIONS = {'.md', '.txt'}

    @staticmethod
    def validate_zip_file(zip_path: str) -> Tuple[bool, str, List[str]]:
        """
        Validate zip file and its contents
        Returns: (is_valid, error_message, valid_files)
        """
        try:
            # Check zip file size
            if os.path.getsize(zip_path) > ZipValidator.MAX_ZIP_SIZE:
                return False, "Zip file too large (max 100MB)", []

            valid_files = []
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check number of files
                if len(zip_ref.filelist) > ZipValidator.MAX_FILES:
                    return False, f"Too many files in zip (max {ZipValidator.MAX_FILES})", []

                # Check for directory traversal attempts
                for file_info in zip_ref.filelist:
                    if '..' in file_info.filename or file_info.filename.startswith('/'):
                        return False, "Invalid file paths detected", []

                # Validate each file
                total_size = 0
                for file_info in zip_ref.filelist:
                    # Skip directories
                    if file_info.filename.endswith('/'):
                        continue

                    # Check file size
                    if file_info.file_size > ZipValidator.MAX_ZIP_SIZE:
                        return False, f"File {file_info.filename} too large", []

                    total_size += file_info.file_size
                    if total_size > ZipValidator.MAX_ZIP_SIZE:
                        return False, "Total uncompressed size too large", []

                    # Check file extension
                    ext = os.path.splitext(file_info.filename)[1].lower()
                    if ext in ZipValidator.VALID_EXTENSIONS:
                        valid_files.append(file_info.filename)

            if not valid_files:
                return False, "No valid markdown or text files found in zip", []

            return True, "", valid_files

        except zipfile.BadZipFile:
            return False, "Invalid or corrupted zip file", []
        except Exception as e:
            return False, f"Error processing zip file: {str(e)}", []


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
