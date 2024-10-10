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
#
# External Imports
import logging
import os

import gradio as gr
#
# Local Imports
from App_Function_Libraries.Books.Book_Ingestion_Lib import process_zip_file, import_epub


#
########################################################################################################################
#
# Functions:

def create_import_book_tab():
    with gr.TabItem("Ebook(epub) Files"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Import .epub files")
                gr.Markdown("Upload a single .epub file or a .zip file containing multiple .epub files")
                gr.Markdown(
                    "🔗 **How to remove DRM from your ebooks:** [Reddit Guide](https://www.reddit.com/r/Calibre/comments/1ck4w8e/2024_guide_on_removing_drm_from_kobo_kindle_ebooks/)")
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

                # Chunking options
                max_chunk_size = gr.Slider(minimum=100, maximum=2000, value=500, step=50, label="Max Chunk Size")
                chunk_overlap = gr.Slider(minimum=0, maximum=500, value=200, step=10, label="Chunk Overlap")
                custom_chapter_pattern = gr.Textbox(label="Custom Chapter Pattern (optional)",
                                                    placeholder="Enter a custom regex pattern for chapter detection")


                import_button = gr.Button("Import eBook(s)")
            with gr.Column():
                with gr.Row():
                    import_output = gr.Textbox(label="Import Status", lines=10, interactive=False)

        def import_file_handler(file, title, author, keywords, system_prompt, custom_prompt, auto_summarize, api_name,
                                api_key, max_chunk_size, chunk_overlap, custom_chapter_pattern):
            try:
                # Handle max_chunk_size
                if isinstance(max_chunk_size, str):
                    max_chunk_size = int(max_chunk_size) if max_chunk_size.strip() else 4000
                elif not isinstance(max_chunk_size, int):
                    max_chunk_size = 4000  # Default value if not a string or int

                # Handle chunk_overlap
                if isinstance(chunk_overlap, str):
                    chunk_overlap = int(chunk_overlap) if chunk_overlap.strip() else 0
                elif not isinstance(chunk_overlap, int):
                    chunk_overlap = 0  # Default value if not a string or int

                chunk_options = {
                    'method': 'chapter',
                    'max_size': max_chunk_size,
                    'overlap': chunk_overlap,
                    'custom_chapter_pattern': custom_chapter_pattern if custom_chapter_pattern else None
                }

                if file is None:
                    return "No file uploaded."

                file_path = file.name
                if not os.path.exists(file_path):
                    return "Uploaded file not found."

                if file_path.lower().endswith('.epub'):
                    status = import_epub(
                        file_path,
                        title,
                        author,
                        keywords,
                        custom_prompt=custom_prompt,
                        system_prompt=system_prompt,
                        summary=None,
                        auto_summarize=auto_summarize,
                        api_name=api_name,
                        api_key=api_key,
                        chunk_options=chunk_options,
                        custom_chapter_pattern=custom_chapter_pattern
                    )
                    return f"📚 EPUB Imported Successfully:\n{status}"
                elif file.name.lower().endswith('.zip'):
                    status = process_zip_file(
                        zip_file=file,
                        title=title,
                        author=author,
                        keywords=keywords,
                        custom_prompt=custom_prompt,
                        system_prompt=system_prompt,
                        summary=None,  # Let the library handle summarization
                        auto_summarize=auto_summarize,
                        api_name=api_name,
                        api_key=api_key,
                        chunk_options=chunk_options
                    )
                    return f"📦 ZIP Processed Successfully:\n{status}"
                elif file.name.lower().endswith(('.chm', '.html', '.pdf', '.xml', '.opml')):
                    file_type = file.name.split('.')[-1].upper()
                    return f"{file_type} file import is not yet supported."
                else:
                    return "❌ Unsupported file type. Please upload an `.epub` file or a `.zip` file containing `.epub` files."

            except ValueError as ve:
                logging.exception(f"Error parsing input values: {str(ve)}")
                return f"❌ Error: Invalid input for chunk size or overlap. Please enter valid numbers."
            except Exception as e:
                logging.exception(f"Error during file import: {str(e)}")
                return f"❌ Error during import: {str(e)}"

        import_button.click(
            fn=import_file_handler,
            inputs=[
                import_file,
                title_input,
                author_input,
                keywords_input,
                custom_prompt_input,
                auto_summarize_checkbox,
                api_name_input,
                api_key_input,
                max_chunk_size,
                chunk_overlap,
                custom_chapter_pattern
            ],
            outputs=import_output
        )

    return import_file, title_input, author_input, keywords_input, system_prompt_input, custom_prompt_input, auto_summarize_checkbox, api_name_input, api_key_input, import_button, import_output

#
# End of File
########################################################################################################################