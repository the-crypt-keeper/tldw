# RAG_QA_Chat_tab.py
# Description: Gradio UI for RAG QA Chat
#
# Imports
#
# External Imports
import logging

import gradio as gr

from App_Function_Libraries.DB.DB_Manager import DatabaseError, get_paginated_files
from App_Function_Libraries.RAG.RAG_QA_Chat import search_database, load_chat_history, \
    save_chat_history, rag_qa_chat


#
# Local Imports
#
########################################################################################################################
#
# Functions:

def create_rag_qa_chat_tab():
    with gr.TabItem("RAG QA Chat"):
        gr.Markdown("# RAG QA Chat")

        with gr.Row():
            with gr.Column(scale=1):
                context_source = gr.Radio(
                    ["Existing File", "Search Database", "Upload File"],
                    label="Context Source",
                    value="Existing File"
                )
                existing_file = gr.Dropdown(label="Select Existing File", choices=[], interactive=True)
                file_page = gr.State(value=1)
                with gr.Row():
                    prev_page_btn = gr.Button("Previous Page")
                    next_page_btn = gr.Button("Next Page")
                    page_info = gr.HTML("Page 1")

                search_query = gr.Textbox(label="Search Query", visible=False)
                search_button = gr.Button("Search", visible=False)
                search_results = gr.Dropdown(label="Search Results", choices=[], visible=False)
                file_upload = gr.File(label="Upload File", visible=False)

                api_choice = gr.Dropdown(
                    choices=["OpenAI", "Anthropic", "Cohere", "Local-LLM"],
                    label="Select API for RAG",
                    value="OpenAI"
                )
                chat_file = gr.File(label="Chat File")
                load_chat = gr.Button("Load Chat")
                clear = gr.Button("Clear Current Chat")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(label="Enter your message")
                submit = gr.Button("Submit")

                save_chat = gr.Button("Save Chat")


        loading_indicator = gr.HTML(visible=False)

        def update_file_list(page):
            files, total_pages, current_page = get_paginated_files(page)
            choices = [f"{title} (ID: {id})" for id, title in files]
            return gr.update(choices=choices), gr.update(value=f"Page {current_page} of {total_pages}"), current_page

        def next_page_fn(current_page):
            return update_file_list(current_page + 1)

        def prev_page_fn(current_page):
            return update_file_list(max(1, current_page - 1))

        def update_context_source(choice):
            return {
                existing_file: gr.update(visible=choice == "Existing File"),
                prev_page_btn: gr.update(visible=choice == "Existing File"),
                next_page_btn: gr.update(visible=choice == "Existing File"),
                page_info: gr.update(visible=choice == "Existing File"),
                search_query: gr.update(visible=choice == "Search Database"),
                search_button: gr.update(visible=choice == "Search Database"),
                search_results: gr.update(visible=choice == "Search Database"),
                file_upload: gr.update(visible=choice == "Upload File")
            }

        context_source.change(update_context_source, context_source,
                              [existing_file, prev_page_btn, next_page_btn, page_info, search_query, search_button,
                               search_results, file_upload])

        next_page_btn.click(next_page_fn, inputs=[file_page], outputs=[existing_file, page_info, file_page])
        prev_page_btn.click(prev_page_fn, inputs=[file_page], outputs=[existing_file, page_info, file_page])

        # Initialize the file list
        context_source.change(lambda: update_file_list(1), outputs=[existing_file, page_info, file_page])

        loading_indicator = gr.HTML(visible=False)

        def rag_qa_chat_wrapper(message, history, context_source, existing_file, search_results, file_upload,
                                api_choice):
            try:
                # Show loading indicator
                yield history, "", gr.update(visible=True)

                if context_source == "Existing File":
                    context = f"media_id:{existing_file.split('(ID: ')[1][:-1]}"
                elif context_source == "Search Database":
                    context = f"media_id:{search_results.split('(ID: ')[1][:-1]}"
                else:  # Upload File
                    if file_upload is None:
                        raise ValueError("No file uploaded")
                    context = file_upload

                new_history, response = rag_qa_chat(message, history, context, api_choice)
                gr.Info("Response generated successfully")
                yield new_history, "", gr.update(visible=False)
            except ValueError as e:
                gr.Error(f"Input error: {str(e)}")
                yield history, "", gr.update(visible=False)
            except DatabaseError as e:
                gr.Error(f"Database error: {str(e)}")
                yield history, "", gr.update(visible=False)
            except Exception as e:
                logging.error(f"Unexpected error in rag_qa_chat_wrapper: {e}")
                gr.Error("An unexpected error occurred. Please try again later.")
                yield history, "", gr.update(visible=False)

        def save_chat_history_wrapper(history):
            try:
                file_path = save_chat_history(history)
                gr.Info("Chat history saved successfully")
                return gr.update(value=file_path)
            except Exception as e:
                gr.Error(f"Error saving chat history: {str(e)}")
                return gr.update(value=None)

        def load_chat_history_wrapper(file):
            try:
                if file is not None:
                    history = load_chat_history(file)
                    gr.Info("Chat history loaded successfully")
                    return history
                return []
            except Exception as e:
                gr.Error(f"Error loading chat history: {str(e)}")
                return []

        def perform_search(query):
            try:
                results = search_database(query)
                return gr.update(choices=results)
            except Exception as e:
                gr.Error(f"Error performing search: {str(e)}")
                return gr.update(choices=[])

        save_chat.click(save_chat_history_wrapper, inputs=[chatbot], outputs=[chat_file])
        load_chat.click(load_chat_history_wrapper, inputs=[chat_file], outputs=[chatbot])

        search_button.click(perform_search, inputs=[search_query], outputs=[search_results])

        submit.click(
            rag_qa_chat_wrapper,
            inputs=[msg, chatbot, context_source, existing_file, search_results, file_upload, api_choice],
            outputs=[chatbot, msg, loading_indicator]
        )

        clear.click(lambda: ([], None), outputs=[chatbot, chat_file])

    return context_source, existing_file, search_query, search_button, search_results, file_upload, api_choice, chatbot, msg, submit, clear, save_chat, load_chat, chat_file

#
# End of RAG_QA_Chat_tab.py
########################################################################################################################
#