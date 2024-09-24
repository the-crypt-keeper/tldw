# Book_Ingestion_tab.py
# Functionality to import epubs/ebooks into the system.
####################
# Function List
#
# 1. create_import_book_tab()
# 2. import_epub(epub_file, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key)
#
####################
# Imports
import tempfile
import os
import zipfile
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.Gradio_UI.Import_Functionality import import_data
from App_Function_Libraries.Books.Book_Ingestion_Lib import epub_to_markdown
#
########################################################################################################################
#
# Functions:

def import_epub(epub_file, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
    try:
        # Create a temporary directory to store the converted file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Handle different types of file objects
            if isinstance(epub_file, (str, os.PathLike)):
                epub_path = epub_file
            elif hasattr(epub_file, 'name'):
                epub_path = epub_file.name
            elif hasattr(epub_file, 'path'):
                epub_path = epub_file.path
            else:
                raise ValueError("Unsupported file object type")

            md_path = os.path.join(temp_dir, "converted.md")

            # Convert EPUB to Markdown
            markdown_content = epub_to_markdown(epub_path)

            # Write the markdown content to a file
            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(markdown_content)

            # Read the converted markdown content
            with open(md_path, "r", encoding="utf-8") as md_file:
                content = md_file.read()

            # Now process the content as you would with a text file
            return import_data(content, title, author, keywords, system_prompt,
                               user_prompt, auto_summarize, api_name, api_key)
    except Exception as e:
        return f"Error processing EPUB: {str(e)}"


def process_zip_file(zip_file, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        if hasattr(zip_file, 'name'):
            zip_path = zip_file.name
        elif hasattr(zip_file, 'path'):
            zip_path = zip_file.path
        else:
            raise ValueError("Unsupported zip file object type")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        for filename in os.listdir(temp_dir):
            if filename.lower().endswith('.epub'):
                file_path = os.path.join(temp_dir, filename)
                result = import_epub(file_path, title, author, keywords, system_prompt,
                                     user_prompt, auto_summarize, api_name, api_key)
                results.append(f"File: {filename} - {result}")

    return "\n".join(results)


def create_import_book_tab():
    with gr.TabItem("Ebook(epub) Files"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Import .epub files")
                gr.Markdown("Upload a single .epub file or a .zip file containing multiple .epub files")
                gr.Markdown(
                    "How to remove DRM from your ebooks: https://www.reddit.com/r/Calibre/comments/1ck4w8e/2024_guide_on_removing_drm_from_kobo_kindle_ebooks/")
                import_file = gr.File(label="Upload file for import", file_types=[".epub", ".zip"])
                title_input = gr.Textbox(label="Title", placeholder="Enter the title of the content (for single files)")
                author_input = gr.Textbox(label="Author", placeholder="Enter the author's name (for single files)")
                keywords_input = gr.Textbox(label="Keywords (like genre or publish year)",
                                            placeholder="Enter keywords, comma-separated")
                system_prompt_input = gr.Textbox(label="System Prompt", lines=3,
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
                             "OpenRouter", "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace"],
                    label="API for Auto-summarization"
                )
                api_key_input = gr.Textbox(label="API Key", type="password")
                import_button = gr.Button("Import eBook(s)")
            with gr.Column():
                with gr.Row():
                    import_output = gr.Textbox(label="Import Status")

        def import_file_handler(file, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
            if file.name.lower().endswith('.epub'):
                return import_epub(file, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key)
            elif file.name.lower().endswith('.zip'):
                return process_zip_file(file, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key)
            else:
                return "Unsupported file type. Please upload an .epub file or a .zip file containing .epub files."

        import_button.click(
            fn=import_file_handler,
            inputs=[import_file, title_input, author_input, keywords_input, system_prompt_input,
                    custom_prompt_input, auto_summarize_checkbox, api_name_input, api_key_input],
            outputs=import_output
        )

    return import_file, title_input, author_input, keywords_input, system_prompt_input, custom_prompt_input, auto_summarize_checkbox, api_name_input, api_key_input, import_button, import_output

#
# End of File
########################################################################################################################