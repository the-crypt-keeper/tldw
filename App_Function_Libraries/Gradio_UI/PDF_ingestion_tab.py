# PDF_ingestion_tab.py
# Gradio UI for ingesting PDFs into the database
import os
import shutil
import tempfile

# Imports
#
# External Imports
import gradio as gr
import pymupdf4llm
from docling.document_converter import DocumentConverter

#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import list_prompts
from App_Function_Libraries.Gradio_UI.Chat_ui import update_user_prompt
from App_Function_Libraries.PDF.PDF_Ingestion_Lib import extract_metadata_from_pdf, extract_text_and_format_from_pdf, \
    process_and_cleanup_pdf
#
#
########################################################################################################################
#
# Functions:

def create_pdf_ingestion_tab():
    with gr.TabItem("PDF Ingestion", visible=True):
        gr.Markdown("# Ingest PDF Files and Extract Metadata")
        with gr.Row():
            with gr.Column():
                # Changed to support multiple files
                pdf_file_input = gr.File(
                    label="Uploaded PDF Files",
                    file_types=[".pdf"],
                    visible=True,
                    file_count="multiple"
                )
                pdf_upload_button = gr.UploadButton(
                    "Click to Upload PDFs",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                parser_selection = gr.Radio(
                    choices=["pymupdf", "pymupdf4llm", "docling"],
                    label="Select Parser",
                    value="pymupdf"  # default value
                )
                # Common metadata for all files
                pdf_keywords_input = gr.Textbox(label="Keywords (Optional, comma-separated)")
#                 with gr.Row():
#                     custom_prompt_checkbox = gr.Checkbox(
#                         label="Use a Custom Prompt",
#                         value=False,
#                         visible=True
#                     )
#                     preset_prompt_checkbox = gr.Checkbox(
#                         label="Use a pre-set Prompt",
#                         value=False,
#                         visible=True
#                     )
#                 # Initialize state variables for pagination
#                 current_page_state = gr.State(value=1)
#                 total_pages_state = gr.State(value=1)
#                 with gr.Row():
#                     # Add pagination controls
#                     preset_prompt = gr.Dropdown(
#                         label="Select Preset Prompt",
#                         choices=[],
#                         visible=False
#                     )
#                     prev_page_button = gr.Button("Previous Page", visible=False)
#                     page_display = gr.Markdown("Page 1 of X", visible=False)
#                     next_page_button = gr.Button("Next Page", visible=False)
#                 with gr.Row():
#                     custom_prompt_input = gr.Textbox(
#                         label="Custom Prompt",
#                         placeholder="Enter custom prompt here",
#                         lines=3,
#                         visible=False
#                     )
#                 with gr.Row():
#                     system_prompt_input = gr.Textbox(
#                         label="System Prompt",
#                         value="""
# <s>You are a bulleted notes specialist.
# [INST]```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.[/INST]
# **Bulleted Note Creation Guidelines**
#
# **Headings**:
# - Based on referenced topics, not categories like quotes or terms
# - Surrounded by **bold** formatting
# - Not listed as bullet points
# - No space between headings and list items underneath
#
# **Emphasis**:
# - **Important terms** set in bold font
# - **Text ending in a colon**: also bolded
#
# **Review**:
# - Ensure adherence to specified format
# - Do not reference these instructions in your response.</s>[INST] {{ .Prompt }} [/INST]""",
#                         lines=3,
#                         visible=False
#                     )
#
#                 custom_prompt_checkbox.change(
#                     fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
#                     inputs=[custom_prompt_checkbox],
#                     outputs=[custom_prompt_input, system_prompt_input]
#                 )
#
#                 def on_preset_prompt_checkbox_change(is_checked):
#                     if is_checked:
#                         prompts, total_pages, current_page = list_prompts(page=1, per_page=10)
#                         page_display_text = f"Page {current_page} of {total_pages}"
#                         return (
#                             gr.update(visible=True, interactive=True, choices=prompts),  # preset_prompt
#                             gr.update(visible=True),  # prev_page_button
#                             gr.update(visible=True),  # next_page_button
#                             gr.update(value=page_display_text, visible=True),  # page_display
#                             current_page,  # current_page_state
#                             total_pages  # total_pages_state
#                         )
#                     else:
#                         return (
#                             gr.update(visible=False, interactive=False),  # preset_prompt
#                             gr.update(visible=False),  # prev_page_button
#                             gr.update(visible=False),  # next_page_button
#                             gr.update(visible=False),  # page_display
#                             1,  # current_page_state
#                             1   # total_pages_state
#                         )
#
#                 preset_prompt_checkbox.change(
#                     fn=on_preset_prompt_checkbox_change,
#                     inputs=[preset_prompt_checkbox],
#                     outputs=[preset_prompt, prev_page_button, next_page_button, page_display, current_page_state, total_pages_state]
#                 )
#
#                 def on_prev_page_click(current_page, total_pages):
#                     new_page = max(current_page - 1, 1)
#                     prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
#                     page_display_text = f"Page {current_page} of {total_pages}"
#                     return gr.update(choices=prompts), gr.update(value=page_display_text), current_page
#
#                 prev_page_button.click(
#                     fn=on_prev_page_click,
#                     inputs=[current_page_state, total_pages_state],
#                     outputs=[preset_prompt, page_display, current_page_state]
#                 )
#
#                 def on_next_page_click(current_page, total_pages):
#                     new_page = min(current_page + 1, total_pages)
#                     prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
#                     page_display_text = f"Page {current_page} of {total_pages}"
#                     return gr.update(choices=prompts), gr.update(value=page_display_text), current_page
#
#                 next_page_button.click(
#                     fn=on_next_page_click,
#                     inputs=[current_page_state, total_pages_state],
#                     outputs=[preset_prompt, page_display, current_page_state]
#                 )
#
#                 def update_prompts(preset_name):
#                     prompts = update_user_prompt(preset_name)
#                     return (
#                         gr.update(value=prompts["user_prompt"], visible=True),
#                         gr.update(value=prompts["system_prompt"], visible=True)
#                     )
#
#                 preset_prompt.change(
#                     update_prompts,
#                     inputs=preset_prompt,
#                     outputs=[custom_prompt_input, system_prompt_input]
#                 )

                pdf_ingest_button = gr.Button("Ingest PDFs")

                # Update the upload button handler for multiple files
                pdf_upload_button.upload(
                    fn=lambda files: files,
                    inputs=pdf_upload_button,
                    outputs=pdf_file_input
                )

            with gr.Column():
                pdf_result_output = gr.DataFrame(
                    headers=["Filename", "Status", "Message"],
                    label="Processing Results"
                )

            # Define a new function to handle multiple PDFs
            def process_multiple_pdfs(pdf_files, keywords, custom_prompt_checkbox_value, custom_prompt_text, system_prompt_text):
                results = []
                if pdf_files is None:
                    return [["No files", "Error", "No files uploaded"]]

                for pdf_file in pdf_files:
                    try:
                        # Extract metadata from PDF
                        metadata = extract_metadata_from_pdf(pdf_file.name)

                        # Use custom or system prompt if checkbox is checked
                        if custom_prompt_checkbox_value:
                            prompt = custom_prompt_text
                            system_prompt = system_prompt_text
                        else:
                            prompt = None
                            system_prompt = None

                        # Process the PDF with prompts
                        result = process_and_cleanup_pdf(
                            pdf_file,
                            metadata.get('title', os.path.splitext(os.path.basename(pdf_file.name))[0]),
                            metadata.get('author', 'Unknown'),
                            keywords,
                            #prompt=prompt,
                            #system_prompt=system_prompt
                        )

                        results.append([
                            pdf_file.name,
                            "Success" if "successfully" in result else "Error",
                            result
                        ])
                    except Exception as e:
                        results.append([
                            pdf_file.name,
                            "Error",
                            str(e)
                        ])

                return results

            # Update the ingest button click handler
            pdf_ingest_button.click(
                fn=process_multiple_pdfs,
                inputs=[
                    pdf_file_input,
                    pdf_keywords_input,
                    parser_selection,
                    #custom_prompt_checkbox,
                    #custom_prompt_input,
                    #system_prompt_input
                ],
                outputs=pdf_result_output
            )


def test_pymupdf4llm_pdf_ingestion(pdf_file):
    if pdf_file is None:
        return "No file uploaded", ""

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a path for the temporary PDF file
            temp_path = os.path.join(temp_dir, "temp.pdf")

            # Copy the contents of the uploaded file to the temporary file
            shutil.copy(pdf_file.name, temp_path)

            # Extract text and convert to Markdown
            markdown_text = pymupdf4llm.to_markdown(temp_path)

            # Extract metadata from PDF
            metadata = extract_metadata_from_pdf(temp_path)

            # Use metadata for title and author if not provided
            title = metadata.get('title', os.path.splitext(os.path.basename(pdf_file.name))[0])
            author = metadata.get('author', 'Unknown')

        result = f"PDF '{title}' by {author} processed successfully by pymupdf4llm."
        return result, markdown_text
    except Exception as e:
        return f"Error ingesting PDF: {str(e)}", ""


def test_pymupdf_pdf_ingestion(pdf_file):
    if pdf_file is None:
        return "No file uploaded", ""

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a path for the temporary PDF file
            temp_path = os.path.join(temp_dir, "temp.pdf")

            # Copy the contents of the uploaded file to the temporary file
            shutil.copy(pdf_file.name, temp_path)

            # Extract text and convert to Markdown
            markdown_text = extract_text_and_format_from_pdf(temp_path)

            # Extract metadata from PDF
            metadata = extract_metadata_from_pdf(temp_path)

            # Use metadata for title and author if not provided
            title = metadata.get('title', os.path.splitext(os.path.basename(pdf_file.name))[0])
            author = metadata.get('author', 'Unknown')

        result = f"PDF '{title}' by {author} processed successfully by pymupdf."
        return result, markdown_text
    except Exception as e:
        return f"Error ingesting PDF: {str(e)}", ""


def test_docling_pdf_ingestion(pdf_file):
    if pdf_file is None:
        return "No file uploaded", ""

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a path for the temporary PDF file
            temp_path = os.path.join(temp_dir, "temp.pdf")

            # Copy the contents of the uploaded file to the temporary file
            shutil.copy(pdf_file.name, temp_path)

            # Extract text and convert to Markdown
            converter = DocumentConverter()
            parsed_pdf = converter.convert(temp_path)
            markdown_text = parsed_pdf.document.export_to_markdown()
            # Extract metadata from PDF
            metadata = extract_metadata_from_pdf(temp_path)

            # Use metadata for title and author if not provided
            title = metadata.get('title', os.path.splitext(os.path.basename(pdf_file.name))[0])
            author = metadata.get('author', 'Unknown')

        result = f"PDF '{title}' by {author} processed successfully by Docling."
        return result, markdown_text
    except Exception as e:
        return f"Error ingesting PDF: {str(e)}", ""

def create_pdf_ingestion_test_tab():
    with gr.TabItem("Test PDF Ingestion", visible=True):
        with gr.Row():
            with gr.Column():
                pdf_file_input = gr.File(label="Upload PDF for testing")
                test_button = gr.Button("Test pymupdf PDF Ingestion")
                test_button_2 = gr.Button("Test pymupdf4llm PDF Ingestion")
                test_button_3 = gr.Button("Test Docling PDF Ingestion")
            with gr.Column():
                test_output = gr.Textbox(label="Test Result")
                pdf_content_output = gr.Textbox(label="PDF Content", lines=200)
        test_button.click(
            fn=test_pymupdf_pdf_ingestion,
            inputs=[pdf_file_input],
            outputs=[test_output, pdf_content_output]
        )
        test_button_2.click(
            fn=test_pymupdf4llm_pdf_ingestion,
            inputs=[pdf_file_input],
            outputs=[test_output, pdf_content_output]
        )
        test_button_3.click(
            fn=test_docling_pdf_ingestion,
            inputs=[pdf_file_input],
            outputs=[test_output, pdf_content_output]
        )

