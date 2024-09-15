# Embeddings_tabc.py
# Description: This file contains the code for the RAG Chat tab in the Gradio UI
#
# Imports
import logging
#
# External Imports
import gradio as gr

from App_Function_Libraries.Chunk_Lib import improved_chunking_process
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import get_all_content_from_database
from App_Function_Libraries.RAG.ChromaDB_Library import chroma_client, \
    check_embedding_status, store_in_chroma
from App_Function_Libraries.RAG.Embeddings_Create import create_embedding
#
########################################################################################################################
#
# Functions:




# FIXME - under construction
def create_embeddings_tab():
    with gr.TabItem("Create Embeddings"):
        gr.Markdown("# Create Embeddings for All Content")

        with gr.Row():
            with gr.Column():
                embedding_provider = gr.Radio(
                    choices=["openai", "local", "huggingface"],
                    label="Select Embedding Provider",
                    value="openai"
                )
                embedding_model = gr.Textbox(
                    label="Embedding Model",
                    value="text-embedding-3-small"
                )
                embedding_api_url = gr.Textbox(
                    label="API URL (for local provider)",
                    value="http://localhost:8080/embedding",
                    visible=False
                )

                # Add chunking options
                chunking_method = gr.Dropdown(
                    choices=["words", "sentences", "paragraphs", "tokens", "semantic"],
                    label="Chunking Method",
                    value="words"
                )
                max_chunk_size = gr.Slider(
                    minimum=100, maximum=2000, step=100, value=500,
                    label="Max Chunk Size"
                )
                chunk_overlap = gr.Slider(
                    minimum=0, maximum=200, step=10, value=50,
                    label="Chunk Overlap"
                )
                adaptive_chunking = gr.Checkbox(
                    label="Use Adaptive Chunking",
                    value=False
                )

                create_button = gr.Button("Create Embeddings")

            with gr.Column():
                status_output = gr.Textbox(label="Status", lines=10)

        def update_provider_options(provider):
            return gr.update(visible=provider == "local")

        embedding_provider.change(
            fn=update_provider_options,
            inputs=[embedding_provider],
            outputs=[embedding_api_url]
        )

        def create_all_embeddings(provider, model, api_url, method, max_size, overlap, adaptive):
            try:
                all_content = get_all_content_from_database()
                if not all_content:
                    return "No content found in the database."

                chunk_options = {
                    'method': method,
                    'max_size': max_size,
                    'overlap': overlap,
                    'adaptive': adaptive
                }

                collection_name = "all_content_embeddings"
                collection = chroma_client.get_or_create_collection(name=collection_name)

                for item in all_content:
                    media_id = item['id']
                    text = item['content']

                    chunks = improved_chunking_process(text, chunk_options)
                    for i, chunk in enumerate(chunks):
                        chunk_text = chunk['text']
                        chunk_id = f"doc_{media_id}_chunk_{i}"

                        existing = collection.get(ids=[chunk_id])
                        if existing['ids']:
                            continue

                        embedding = create_embedding(chunk_text, provider, model, api_url)
                        metadata = {
                            "media_id": str(media_id),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "chunking_method": method,
                            "max_chunk_size": max_size,
                            "chunk_overlap": overlap,
                            "adaptive_chunking": adaptive,
                            "embedding_model": model,
                            "embedding_provider": provider,
                            **chunk['metadata']
                        }
                        store_in_chroma(collection_name, [chunk_text], [embedding], [chunk_id], [metadata])

                return "Embeddings created and stored successfully for all content."
            except Exception as e:
                logging.error(f"Error during embedding creation: {str(e)}")
                return f"Error: {str(e)}"

        create_button.click(
            fn=create_all_embeddings,
            inputs=[embedding_provider, embedding_model, embedding_api_url,
                    chunking_method, max_chunk_size, chunk_overlap, adaptive_chunking],
            outputs=status_output
        )


def create_view_embeddings_tab():
    with gr.TabItem("View/Update Embeddings"):
        gr.Markdown("# View and Update Embeddings")
        item_mapping = gr.State({})
        with gr.Row():
            with gr.Column():
                item_dropdown = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                refresh_button = gr.Button("Refresh Item List")
                embedding_status = gr.Textbox(label="Embedding Status", interactive=False)
                embedding_preview = gr.Textbox(label="Embedding Preview", interactive=False, lines=5)

            with gr.Column():
                create_new_embedding_button = gr.Button("Create New Embedding")
                embedding_provider = gr.Radio(
                    choices=["openai", "local", "huggingface"],
                    label="Embedding Provider",
                    value="openai"
                )
                embedding_model = gr.Textbox(
                    label="Embedding Model",
                    value="text-embedding-3-small",
                    visible=True
                )
                embedding_api_url = gr.Textbox(
                    label="API URL (for local provider)",
                    value="http://localhost:8080/embedding",
                    visible=False
                )

        def get_items_with_embedding_status():
            try:
                items = get_all_content_from_database()
                collection = chroma_client.get_or_create_collection(name="all_content_embeddings")
                choices = []
                new_item_mapping = {}
                for item in items:
                    try:
                        result = collection.get(ids=[f"doc_{item['id']}"])
                        embedding_exists = result is not None and result.get('ids') and len(result['ids']) > 0
                        status = "Embedding exists" if embedding_exists else "No embedding"
                    except Exception as e:
                        print(f"Error checking embedding for item {item['id']}: {str(e)}")
                        status = "Error checking"
                    choice = f"{item['title']} ({status})"
                    choices.append(choice)
                    new_item_mapping[choice] = item['id']
                return gr.update(choices=choices), new_item_mapping
            except Exception as e:
                print(f"Error in get_items_with_embedding_status: {str(e)}")
                return gr.update(choices=["Error: Unable to fetch items"]), {}

        def update_provider_options(provider):
            return gr.update(visible=provider == "local")

        def create_new_embedding_for_item(selected_item, provider, model, api_url, item_mapping):
            if not selected_item:
                return "Please select an item", ""

            try:
                item_id = item_mapping.get(selected_item)
                if item_id is None:
                    return f"Invalid item selected: {selected_item}", ""

                items = get_all_content_from_database()
                item = next((item for item in items if item['id'] == item_id), None)
                if not item:
                    return f"Item not found: {item_id}", ""

                embedding = create_embedding(item['content'], provider, model, api_url)

                collection_name = "all_content_embeddings"
                metadata = {"media_id": item_id, "title": item['title']}
                store_in_chroma(collection_name, [item['content']], [embedding], [f"doc_{item_id}"],
                                [{"media_id": item_id, "title": item['title']}])

                embedding_preview = str(embedding[:50])
                status = f"New embedding created and stored for item: {item['title']} (ID: {item_id})"
                return status, f"First 50 elements of new embedding:\n{embedding_preview}\n\nMetadata: {metadata}"
            except Exception as e:
                logging.error(f"Error in create_new_embedding_for_item: {str(e)}")
                return f"Error creating embedding: {str(e)}", ""

        refresh_button.click(
            get_items_with_embedding_status,
            outputs=[item_dropdown, item_mapping]
        )
        item_dropdown.change(
            check_embedding_status,
            inputs=[item_dropdown, item_mapping],
            outputs=[embedding_status, embedding_preview]
        )
        create_new_embedding_button.click(
            create_new_embedding_for_item,
            inputs=[item_dropdown, embedding_provider, embedding_model, embedding_api_url, item_mapping],
            outputs=[embedding_status, embedding_preview]
        )
        embedding_provider.change(
            update_provider_options,
            inputs=[embedding_provider],
            outputs=[embedding_api_url]
        )

    return item_dropdown, refresh_button, embedding_status, embedding_preview, create_new_embedding_button, embedding_provider, embedding_model, embedding_api_url


def create_purge_embeddings_tab():
    with gr.TabItem("Purge Embeddings"):
        gr.Markdown("# Purge Embeddings")

        with gr.Row():
            with gr.Column():
                purge_button = gr.Button("Purge All Embeddings")
            with gr.Column():
                status_output = gr.Textbox(label="Status", lines=10)

    def purge_all_embeddings():
        try:
            collection_name = "all_content_embeddings"
            chroma_client.delete_collection(collection_name)
            chroma_client.create_collection(collection_name)
            return "All embeddings have been purged successfully."
        except Exception as e:
            logging.error(f"Error during embedding purge: {str(e)}")
            return f"Error: {str(e)}"

    purge_button.click(
        fn=purge_all_embeddings,
        outputs=status_output
    )



#
# End of file
########################################################################################################################
