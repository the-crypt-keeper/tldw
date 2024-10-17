# RAG_QA_Chat_Notes.py
# Description: This file contains the code for the RAG QA Chat Notes tab in the RAG QA Chat application.
#
# Imports
import logging
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.DB.RAG_QA_Chat_DB import save_message, add_keywords_to_conversation, \
    search_conversations_by_keywords, load_chat_history
#
####################################################################################################
#
# Functions
def create_rag_qa_chat_notes_tab():
    with gr.TabItem("RAG QA Chat"):
        gr.Markdown("# RAG QA Chat")

        state = gr.State({
            "conversation_id": None,
            "page": 1,
            "context_source": "All Files in the Database"
        })

        with gr.Row():
            with gr.Column(scale=1):
                context_source = gr.Radio(
                    ["All Files in the Database", "Search Database", "Upload File"],
                    label="Context Source",
                    value="All Files in the Database"
                )
                existing_file = gr.Dropdown(label="Select Existing File", choices=[], interactive=True)
                file_page = gr.State(value=1)
                with gr.Row():
                    page_number = gr.Number(value=1, label="Page", precision=0)
                    page_size = gr.Number(value=20, label="Items per page", precision=0)
                    total_pages = gr.Number(label="Total Pages", interactive=False)
                with gr.Row():
                    prev_page_btn = gr.Button("Previous Page")
                    next_page_btn = gr.Button("Next Page")
                    page_info = gr.HTML("Page 1")

                search_query = gr.Textbox(label="Search Query", visible=False)
                search_button = gr.Button("Search", visible=False)
                search_results = gr.Dropdown(label="Search Results", choices=[], visible=False)
                file_upload = gr.File(
                    label="Upload File",
                    visible=False,
                    file_types=["txt", "pdf", "epub", "md", "rtf", "json", "csv"]
                )
                convert_to_text = gr.Checkbox(label="Convert to plain text", visible=False)
                keywords = gr.Textbox(label="Keywords (comma-separated)", visible=False)
            with gr.Column(scale=1):
                api_choice = gr.Dropdown(
                    choices=["Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                             "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace"],
                    label="Select API for RAG",
                    value="OpenAI"
                )
                use_query_rewriting = gr.Checkbox(label="Use Query Rewriting", value=True)

                load_conversation = gr.Dropdown(label="Load Conversation", choices=[])
                new_conversation = gr.Button("New Conversation")
                conversation_title = gr.Textbox(label="Conversation Title",
                                                placeholder="Enter a title for the new conversation")

        with gr.Row():
            page_number = gr.Number(value=1, label="Page", precision=0)
            page_size = gr.Number(value=20, label="Items per page", precision=0)
            total_pages = gr.Number(label="Total Pages", interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(label="Enter your message")
                submit = gr.Button("Submit")
                clear_chat = gr.Button("Clear Chat History")

            with gr.Column(scale=1):
                notes = gr.TextArea(label="Notes", placeholder="Enter your notes here...", lines=20)
                save_notes = gr.Button("Save Notes")
                clear_notes = gr.Button("Clear Notes")



        loading_indicator = gr.HTML(visible=False)

        def rag_qa_chat_wrapper(message, history, state, context_source, existing_file, search_results, file_upload,
                                convert_to_text, keywords, api_choice, use_query_rewriting):
            try:
                conversation_id = state["conversation_id"]
                if not conversation_id:
                    conversation_id = start_new_conversation()
                    state = update_state(state, conversation_id=conversation_id)

                save_message(conversation_id, 'human', message)

                if keywords:
                    add_keywords_to_conversation(conversation_id, keywords.split(','))

                # Here you would implement your actual RAG logic
                # For this example, we'll just echo the message
                response = f"RAG QA Chat: You said '{message}'"

                save_message(conversation_id, 'ai', response)

                new_history = history + [(message, response)]

                logging.info(f"Successfully processed message for conversation '{conversation_id}'")
                return new_history, "", gr.update(visible=False), state

            except Exception as e:
                logging.error(f"Error in rag_qa_chat_wrapper: {e}")
                gr.Error("An unexpected error occurred. Please try again later.")
                return history, "", gr.update(visible=False), state

        def load_conversation_history(selected_conversation_id, page, page_size, state):
            if selected_conversation_id:
                history, total_pages, _ = load_chat_history(selected_conversation_id, page, page_size)
                return history, total_pages, update_state(state, conversation_id=selected_conversation_id, page=page)
            return [], 1, state

        def start_new_conversation(title, state):
            new_conversation_id = start_new_conversation(title if title else "Untitled Conversation")
            return [], update_state(state, conversation_id=new_conversation_id, page=1)

        def update_state(state, **kwargs):
            new_state = state.copy()
            new_state.update(kwargs)
            return new_state

        def update_page(direction, current_page, total_pages):
            new_page = max(1, min(current_page + direction, total_pages))
            return new_page

        def update_context_source(choice):
            return {
                existing_file: gr.update(visible=choice == "Existing File"),
                prev_page_btn: gr.update(visible=choice == "Existing File"),
                next_page_btn: gr.update(visible=choice == "Existing File"),
                page_info: gr.update(visible=choice == "Existing File"),
                search_query: gr.update(visible=choice == "Search Database"),
                search_button: gr.update(visible=choice == "Search Database"),
                search_results: gr.update(visible=choice == "Search Database"),
                file_upload: gr.update(visible=choice == "Upload File"),
                convert_to_text: gr.update(visible=choice == "Upload File"),
                keywords: gr.update(visible=choice == "Upload File")
            }

        def perform_search(query):
            try:
                results = search_conversations_by_keywords(query.split())
                return gr.update(choices=[f"{title} (ID: {id})" for id, title in results[0]])
            except Exception as e:
                logging.error(f"Error performing search: {e}")
                gr.Error(f"Error performing search: {str(e)}")
                return gr.update(choices=[])

        def clear_chat_history():
            return [], ""

        def save_notes_function(notes):
            # Implement saving notes functionality here
            logging.info("Notes saved successfully!")
            return notes

        def clear_notes_function():
            return ""

        # Event handlers
        submit.click(
            rag_qa_chat_wrapper,
            inputs=[msg, chatbot, state, context_source, existing_file, search_results, file_upload,
                    convert_to_text, keywords, api_choice, use_query_rewriting],
            outputs=[chatbot, msg, loading_indicator, state]
        )

        load_conversation.change(
            load_conversation_history,
            inputs=[load_conversation, page_number, page_size, state],
            outputs=[chatbot, total_pages, state]
        )

        new_conversation.click(
            start_new_conversation,
            inputs=[conversation_title, state],
            outputs=[chatbot, state]
        )

        # Event handlers
        prev_page_btn.click(
            update_page,
            inputs=[gr.Number(-1, visible=False), page_number, total_pages],
            outputs=[page_number]
        )
        next_page_btn.click(
            update_page,
            inputs=[gr.Number(1, visible=False), page_number, total_pages],
            outputs=[page_number]
        )

        context_source.change(update_context_source, context_source,
                              [existing_file, prev_page_btn, next_page_btn, page_info, search_query, search_button,
                               search_results, file_upload, convert_to_text, keywords])

        search_button.click(perform_search, inputs=[search_query], outputs=[search_results])

        clear_chat.click(clear_chat_history, outputs=[chatbot, msg])

        save_notes.click(save_notes_function, inputs=[notes], outputs=[notes])
        clear_notes.click(clear_notes_function, outputs=[notes])

    return (context_source, existing_file, search_query, search_button, search_results, file_upload,
            convert_to_text, keywords, api_choice, use_query_rewriting, chatbot, msg, submit, clear_chat,
            notes, save_notes, clear_notes, load_conversation, new_conversation, conversation_title,
            prev_page_btn, next_page_btn, page_number, page_size, total_pages)

#
# End of Rag_QA_Chat_Notes.py
####################################################################################################