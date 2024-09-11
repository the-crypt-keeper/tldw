# Media_wiki_tab.py
# Description: Gradio UI snippet that allows users to import a MediaWiki XML dump file into the application.
#
# Imports
import os
from threading import Thread
#
# 3rd-party Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.MediaWiki.Media_Wiki import import_mediawiki_dump
#
#######################################################################################################################
#
# Create MediaWiki Import Tab

def create_mediawiki_import_tab():
    with gr.Tab("MediaWiki Import"):
        gr.Markdown("# Import MediaWiki Dump")
        with gr.Row():
            with gr.Column():
                file_path = gr.File(label="MediaWiki XML Dump File")
                wiki_name = gr.Textbox(label="Wiki Name", placeholder="Enter a unique name for this wiki")
                namespaces = gr.Textbox(label="Namespaces (comma-separated integers, leave empty for all)")
                skip_redirects = gr.Checkbox(label="Skip Redirects", value=True)
                single_item = gr.Checkbox(label="Import as Single Item", value=False)
                chunk_method = gr.Dropdown(
                    choices=["sentences", "words", "paragraphs", "tokens"],
                    value="sentences",
                    label="Chunking Method"
                )
                chunk_size = gr.Slider(minimum=100, maximum=2000, value=1000, step=100, label="Chunk Size")
                chunk_overlap = gr.Slider(minimum=0, maximum=500, value=100, step=10, label="Chunk Overlap")
                import_button = gr.Button("Import MediaWiki Dump")
                cancel_button = gr.Button("Cancel Import", visible=False)
            with gr.Column():
                output = gr.Markdown(label="Import Status")
                progress_bar = gr.Progress()

        def validate_inputs(file_path, wiki_name, namespaces):
            if not file_path:
                return "Please select a MediaWiki XML dump file."
            if not wiki_name:
                return "Please enter a name for the wiki."
            if namespaces:
                try:
                    [int(ns.strip()) for ns in namespaces.split(',')]
                except ValueError:
                    return "Invalid namespaces. Please enter comma-separated integers."
            return None

        def check_file_size(file_path):
            max_size_mb = 1000  # 1 GB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return f"Warning: The selected file is {file_size_mb:.2f} MB. Importing large files may take a long time."
            return None

        import_thread = None
        cancel_flag = False

        def run_import(file_path, wiki_name, namespaces, skip_redirects, single_item, chunk_method, chunk_size,
                       chunk_overlap, progress=gr.Progress()):
            validation_error = validate_inputs(file_path, wiki_name, namespaces)
            if validation_error:
                return gr.update(), gr.update(), validation_error

            file_size_warning = check_file_size(file_path.name)
            status_text = "# MediaWiki Import Process\n\n## Initializing\n- Starting import process...\n"
            if file_size_warning:
                status_text += f"- {file_size_warning}\n"

            chunk_options = {
                'method': chunk_method,
                'max_size': chunk_size,
                'overlap': chunk_overlap,
                'adaptive': True,
                'language': 'en'
            }
            namespaces_list = [int(ns.strip()) for ns in namespaces.split(',')] if namespaces else None

            pages_processed = 0

            try:
                for progress_info in import_mediawiki_dump(
                        file_path=file_path.name,
                        wiki_name=wiki_name,
                        namespaces=namespaces_list,
                        skip_redirects=skip_redirects,
                        chunk_options=chunk_options,
                        single_item=single_item,
                        progress_callback=progress
                ):
                    if progress_info.startswith("Found"):
                        status_text += f"\n## Parsing\n- {progress_info}\n"
                    elif progress_info.startswith("Processed page"):
                        pages_processed += 1
                        if pages_processed % 10 == 0:  # Update every 10 pages to avoid too frequent updates
                            status_text += f"- {progress_info}\n"
                    elif progress_info.startswith("Successfully imported"):
                        status_text += f"\n## Completed\n- {progress_info}\n- Total pages processed: {pages_processed}"
                    else:
                        status_text += f"- {progress_info}\n"

                    yield gr.update(), gr.update(), status_text

                status_text += "\n## Import Process Completed Successfully"
            except Exception as e:
                status_text += f"\n## Error\n- An error occurred during the import process: {str(e)}"

            yield gr.update(visible=False), gr.update(visible=True), status_text

        def start_import(*args):
            nonlocal import_thread
            import_thread = Thread(target=run_import, args=args)
            import_thread.start()
            return gr.update(visible=True), gr.update(visible=False), gr.update(
                value="Import process started. Please wait...")

        def cancel_import():
            nonlocal cancel_flag
            cancel_flag = True
            return gr.update(visible=False), gr.update(visible=True)

        import_button.click(
            run_import,
            inputs=[file_path, wiki_name, namespaces, skip_redirects, single_item, chunk_method, chunk_size,
                    chunk_overlap],
            outputs=[cancel_button, import_button, output]
        )

        cancel_button.click(
            cancel_import,
            outputs=[cancel_button, import_button]
        )

    return file_path, wiki_name, namespaces, skip_redirects, single_item, chunk_method, chunk_size, chunk_overlap, import_button, output

#
# End of MediaWiki Import Tab
#######################################################################################################################
