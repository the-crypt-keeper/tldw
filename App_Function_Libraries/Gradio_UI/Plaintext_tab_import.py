# Plaintext_tab_import.py
# Contains the code for the "Import Plain Text Files" tab in the Gradio UI.
# This tab allows users to upload plain text files (Markdown, Text, RTF) or a zip file containing multiple files.
# The user can provide a title, author, keywords, system prompt, custom user prompt, and select an API for
#   auto-summarization.
#
#######################################################################################################################
#
# Import necessary libraries
import logging
#
# Import Non-Local
import gradio as gr
#
# Import Local libraries
from App_Function_Libraries.Plaintext.Plaintext_Files import final_ingest_handler, preview_import_handler
from App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name
#
#######################################################################################################################
#
# Functions:

def create_plain_text_import_tab():
    try:
        default_value = None
        if default_api_endpoint and default_api_endpoint in global_api_endpoints:
            default_value = format_api_name(default_api_endpoint)
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None

    with gr.TabItem("Import Plain Text & .docx Files", visible=True):
        gr.Markdown("# Import Plaintext files")
        gr.Markdown("Upload txt/docx/md/rtf/zip collections of them for import")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 1: Upload / Preview")
                import_files = gr.File(
                    label="Upload files or .zip",
                    file_count="multiple",
                    file_types=["text", ".md", ".txt", ".rtf", ".docx", ".zip"]
                )
                author_input = gr.Textbox(label="Author")
                keywords_input = gr.Textbox(label="Keywords (comma-separated)")
                system_prompt_input = gr.Textbox(label="System Prompt", lines=3)
                custom_prompt_input = gr.Textbox(label="User Prompt", lines=2)
                auto_summarize_checkbox = gr.Checkbox(label="Auto Summarize?")
                api_name_input = gr.Dropdown(
                    label="API for Summaries",
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value
                )
                api_key_input = gr.Textbox(label="API Key", type="password")

                preview_button = gr.Button("Preview")

                preview_status = gr.Textbox(label="Preview Status", lines=5)

            with gr.Column():
                # We'll store the preview data in a hidden state
                preview_data_state = gr.State()
                preview_data_display = gr.Textbox(label="Preview Data JSON", lines=15)

        # Step 2: Ingest with possible metadata overrides
        gr.Markdown("### Step 2: Final Ingest with Overrides")
        metadata_overrides_input = gr.Textbox(
            label="Metadata Overrides (JSON)",
            lines=6,
            placeholder='{"myfile.md": {"title": "New Title", "author": "New Author"}}'
        )
        ingest_button = gr.Button("Confirm Ingest")
        ingest_status = gr.Textbox(label="Ingest Status", lines=5)

        # Wire up events
        preview_button.click(
            fn=preview_import_handler,
            inputs=[
                import_files, author_input, keywords_input,
                system_prompt_input, custom_prompt_input,
                auto_summarize_checkbox, api_name_input, api_key_input
            ],
            outputs=[preview_status, preview_data_state]
        )

        # Show the JSON in preview_data_display
        preview_button.click(
            fn=preview_import_handler,
            inputs=[
                import_files, author_input, keywords_input,
                system_prompt_input, custom_prompt_input,
                auto_summarize_checkbox, api_name_input, api_key_input
            ],
            outputs=[preview_status, preview_data_state]
        ).then(
            fn=lambda x: x,  # identity function
            inputs=preview_data_state,
            outputs=preview_data_display
        )

        ingest_button.click(
            fn=final_ingest_handler,
            inputs=[preview_data_state, metadata_overrides_input],
            outputs=ingest_status
        )

    return (
        import_files,
        author_input,
        keywords_input,
        system_prompt_input,
        custom_prompt_input,
        auto_summarize_checkbox,
        api_name_input,
        api_key_input,
        preview_button,
        preview_status,
        preview_data_state,
        preview_data_display,
        metadata_overrides_input,
        ingest_button,
        ingest_status
    )
# v1
# def create_plain_text_import_tab():
#     """Create the Gradio UI tab for importing .md/.txt/.rtf/.docx and .zip files."""
#
#     try:
#         default_value = None
#         if default_api_endpoint:
#             if default_api_endpoint in global_api_endpoints:
#                 default_value = format_api_name(default_api_endpoint)
#             else:
#                 logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
#     except Exception as e:
#         logging.error(f"Error setting default API endpoint: {str(e)}")
#         default_value = None
#
#     with gr.TabItem("Import Plain text & .docx Files", visible=True):
#         with gr.Row():
#             with gr.Column():
#                 gr.Markdown("# Import `.md`/`.txt`/`.rtf`/`.docx`  Files & `.zip` collections of them.")
#                 gr.Markdown("Upload multiple files or a zip file containing multiple files")
#
#                 # Updated to support multiple files
#                 import_files = gr.File(
#                     label="Upload files for import",
#                     file_count="multiple",
#                     file_types=['text', ".md", ".txt", ".rtf", ".docx", ".zip", "zip"]
#                 )
#
#                 # Optional metadata override fields
#                 author_input = gr.Textbox(
#                     label="Author Override (optional)",
#                     placeholder="Enter author name to apply to all files"
#                 )
#                 keywords_input = gr.Textbox(
#                     label="Keywords",
#                     placeholder="Enter keywords, comma-separated - will be applied to all files"
#                 )
#                 system_prompt_input = gr.Textbox(
#                     label="System Prompt (for Summarization)",
#                     lines=3,
#                     value="""
#                     <s>You are a bulleted notes specialist. [INST]```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.[/INST]
#                     **Bulleted Note Creation Guidelines**
#
#                     **Headings**:
#                     - Based on referenced topics, not categories like quotes or terms
#                     - Surrounded by **bold** formatting
#                     - Not listed as bullet points
#                     - No space between headings and list items underneath
#
#                     **Emphasis**:
#                     - **Important terms** set in bold font
#                     - **Text ending in a colon**: also bolded
#
#                     **Review**:
#                     - Ensure adherence to specified format
#                     - Do not reference these instructions in your response.</s>[INST]
#                     """
#                 )
#                 custom_prompt_input = gr.Textbox(
#                     label="Custom User Prompt",
#                     placeholder="Enter a custom user prompt for summarization (optional)"
#                 )
#                 auto_summarize_checkbox = gr.Checkbox(label="Auto-summarize", value=False)
#
#                 # API configuration
#                 api_name_input = gr.Dropdown(
#                     choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
#                     value=default_value,
#                     label="API for Analysis/Summarization (Optional)"
#                 )
#                 api_key_input = gr.Textbox(label="API Key", type="password")
#                 import_button = gr.Button("Import File(s)")
#
#             with gr.Column():
#                 import_output = gr.Textbox(
#                     label="Import Status",
#                     lines=10,
#                     interactive=False
#                 )
#
#         import_button.click(
#             fn=import_file_handler,
#             inputs=[
#                 import_files,
#                 author_input,
#                 keywords_input,
#                 system_prompt_input,
#                 custom_prompt_input,
#                 auto_summarize_checkbox,
#                 api_name_input,
#                 api_key_input
#             ],
#             outputs=import_output
#         )
#
#     return (
#         import_files,
#         author_input,
#         keywords_input,
#         system_prompt_input,
#         custom_prompt_input,
#         auto_summarize_checkbox,
#         api_name_input,
#         api_key_input,
#         import_button,
#         import_output
# )

#
# End of Plain_text_import.py
#######################################################################################################################
