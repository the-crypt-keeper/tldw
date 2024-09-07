# Media_wiki_tab.py
# Description: Gradio UI snippet that allows users to import a MediaWiki XML dump file into the application.
#
# Imports
import asyncio
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
                wiki_name = gr.Textbox(label="Wiki Name")
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
            with gr.Column():
                output = gr.Textbox(label="Import Status", lines=10)

        def run_import(file_path, wiki_name, namespaces, skip_redirects, single_item, chunk_method, chunk_size,
                       chunk_overlap):
            chunk_options = {
                'method': chunk_method,
                'max_size': chunk_size,
                'overlap': chunk_overlap,
                'adaptive': True,
                'language': 'en'
            }
            namespaces_list = [int(ns.strip()) for ns in namespaces.split(',')] if namespaces else None

            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(import_mediawiki_dump(
                file_path=file_path.name,
                wiki_name=wiki_name,
                namespaces=namespaces_list,
                skip_redirects=skip_redirects,
                chunk_options=chunk_options,
                single_item=single_item
            ))
            return result

        import_button.click(
            run_import,
            inputs=[file_path, wiki_name, namespaces, skip_redirects, single_item, chunk_method, chunk_size,
                    chunk_overlap],
            outputs=output
        )

    return file_path, wiki_name, namespaces, skip_redirects, single_item, chunk_method, chunk_size, chunk_overlap, import_button, output

#
# End of MediaWiki Import Tab
#######################################################################################################################
