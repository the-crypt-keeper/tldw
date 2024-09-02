# Import_Functionality.py
# Functionality to import content into the DB
#
# Imports
from time import sleep
import logging
import re
import shutil
import tempfile
import os
import traceback
import zipfile
#
# External Imports
import gradio as gr
import pypandoc
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import insert_prompt_to_db, load_preset_prompts, import_obsidian_note_to_db, \
    add_media_to_database
from App_Function_Libraries.Prompt_Handling import import_prompt_from_file, import_prompts_from_zip#
from App_Function_Libraries.Summarization_General_Lib import perform_summarization

###################################################################################################################
#
# Functions:

logger = logging.getLogger()


def import_data(file, title, author, keywords, custom_prompt, summary, auto_summarize, api_name, api_key):
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
        keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]

        # Handle summarization
        if auto_summarize and api_name and api_key:
            summary = perform_summarization(api_name, file_content, custom_prompt, api_key)
        elif not summary:
            summary = "No summary provided"

        # Add to database
        add_media_to_database(
            url=file_name,  # Using filename as URL
            info_dict=info_dict,
            segments=segments,
            summary=summary,
            keywords=keyword_list,
            custom_prompt_input=custom_prompt,
            whisper_model="Imported",  # Indicating this was an imported file
            media_type="document"
        )

        # Clean up the temporary file
        os.unlink(temp_file.name)

        return f"File '{file_name}' successfully imported with title '{title}' and author '{author}'."
    except Exception as e:
        logging.error(f"Error importing file: {str(e)}")
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
    with gr.TabItem("Import a Prompt"):
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
    with gr.TabItem("Import Markdown/Text Files"):
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
                         "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM","ollama", "HuggingFace"],
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
    with gr.TabItem("Import Multiple Prompts"):
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
    with gr.TabItem("Import Obsidian Vault"):
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
        show_progress=True
    )



# Using pypandoc to convert EPUB to Markdown
def create_import_book_tab():
    with gr.TabItem("Import .epub/ebook Files"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Ingest an .epub file using pypandoc")
                gr.Markdown("...and have it tagged + summarized")
                gr.Markdown(
                    "How to remove DRM from your ebooks: https://www.reddit.com/r/Calibre/comments/1ck4w8e/2024_guide_on_removing_drm_from_kobo_kindle_ebooks/")
                import_file = gr.File(label="Upload file for import", file_types=[".epub"])
                title_input = gr.Textbox(label="Title", placeholder="Enter the title of the content")
                author_input = gr.Textbox(label="Author", placeholder="Enter the author's name")
                keywords_input = gr.Textbox(label="Keywords(like genre or publish year)",
                                            placeholder="Enter keywords, comma-separated")
                system_prompt_input = gr.Textbox(label="System Prompt",
                                                 lines=3,
                                                 value=""""
                                                    <s>You are a bulleted notes specialist. [INST]```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.[/INST]
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
                                                    - Do not reference these instructions in your response.</s>[INST] {{ .Prompt }} [/INST]
                                                """, )
                custom_prompt_input = gr.Textbox(label="Custom User Prompt",
                                                 placeholder="Enter a custom user prompt for summarization (optional)")
                auto_summarize_checkbox = gr.Checkbox(label="Auto-summarize", value=False)
                api_name_input = gr.Dropdown(
                    choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral",
                             "OpenRouter",
                             "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace"],
                    label="API for Auto-summarization"
                )
                api_key_input = gr.Textbox(label="API Key", type="password")
                import_button = gr.Button("Import eBook")
            with gr.Column():
                with gr.Row():
                    import_output = gr.Textbox(label="Import Status")

        def import_epub(epub_file, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name,
                        api_key):
            try:
                # Create a temporary directory to store the converted file
                with tempfile.TemporaryDirectory() as temp_dir:
                    epub_path = epub_file.name
                    md_path = os.path.join(temp_dir, "converted.md")

                    # Use pypandoc to convert EPUB to Markdown
                    output = pypandoc.convert_file(epub_path, 'md', outputfile=md_path)

                    if output != "":
                        return f"Error converting EPUB: {output}"

                    # Read the converted markdown content
                    with open(md_path, "r", encoding="utf-8") as md_file:
                        content = md_file.read()

                    # Now process the content as you would with a text file
                    return import_data(content, title, author, keywords, system_prompt,
                                       user_prompt, auto_summarize, api_name, api_key)
            except Exception as e:
                return f"Error processing EPUB: {str(e)}"

        import_button.click(
            fn=import_epub,
            inputs=[import_file, title_input, author_input, keywords_input, system_prompt_input,
                    custom_prompt_input, auto_summarize_checkbox, api_name_input, api_key_input],
            outputs=import_output
        )

def import_obsidian_vault(vault_path, progress=gr.Progress()):
    try:
        from App_Function_Libraries.Gradio_UI.Export_Functionality import scan_obsidian_vault
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