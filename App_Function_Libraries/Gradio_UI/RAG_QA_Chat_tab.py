# RAG_QA_Chat_tab.py
# Description: Gradio UI for RAG QA Chat
#
# Imports
import csv
import logging
import json
import os
from datetime import datetime
#
# External Imports
import docx2txt
import gradio as gr

# Local Imports
from App_Function_Libraries.Books.Book_Ingestion_Lib import read_epub
from App_Function_Libraries.DB.DB_Manager import DatabaseError, get_paginated_files, add_media_with_keywords
from App_Function_Libraries.DB.RAG_QA_Chat_DB import (
    save_notes,
    add_keywords_to_note,
    start_new_conversation,
    save_message,
    search_conversations_by_keywords,
    load_chat_history,
    get_all_conversations,
    get_note_by_id,
    get_notes_by_keywords,
    get_notes_by_keyword_collection,
    update_note,
    clear_keywords_from_note, get_notes, get_keywords_for_note, delete_conversation, delete_note, execute_query,
    add_keywords_to_conversation,
)
from App_Function_Libraries.PDF.PDF_Ingestion_Lib import extract_text_and_format_from_pdf
from App_Function_Libraries.RAG.RAG_Library_2 import generate_answer, enhanced_rag_pipeline
from App_Function_Libraries.RAG.RAG_QA_Chat import search_database, rag_qa_chat
#
########################################################################################################################
#
# Functions:

def create_rag_qa_chat_tab():
    with gr.TabItem("RAG QA Chat", visible=True):
        gr.Markdown("# RAG QA Chat")

        state = gr.State({
            "conversation_id": None,  # No conversation ID initially
            "page": 1,
            "context_source": "Entire Media Database",
            "conversation_messages": [],  # Store messages before saving
        })

        note_state = gr.State({"note_id": None})

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
                # FIXME - Add pages for search results handling
                file_upload = gr.File(
                    label="Upload File",
                    visible=False,
                    file_types=["txt", "pdf", "epub", "md", "rtf", "json", "csv", "docx"]
                )
                convert_to_text = gr.Checkbox(label="Convert to plain text", visible=False)

            with gr.Column(scale=1):
                load_conversation = gr.Dropdown(label="Load Conversation", choices=[])
                new_conversation = gr.Button("New Conversation")
                save_conversation_button = gr.Button("Save Conversation")
                conversation_title = gr.Textbox(
                    label="Conversation Title", placeholder="Enter a title for the new conversation"
                )
                keywords = gr.Textbox(label="Keywords (comma-separated)", visible=True)

                api_choice = gr.Dropdown(
                    choices=[
                        "Local-LLM",
                        "OpenAI",
                        "Anthropic",
                        "Cohere",
                        "Groq",
                        "DeepSeek",
                        "Mistral",
                        "OpenRouter",
                        "Llama.cpp",
                        "Kobold",
                        "Ooba",
                        "Tabbyapi",
                        "VLLM",
                        "ollama",
                        "HuggingFace",
                    ],
                    label="Select API for RAG",
                    value="OpenAI",
                )
                use_query_rewriting = gr.Checkbox(label="Use Query Rewriting", value=True)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(label="Enter your message")
                submit = gr.Button("Submit")
                clear_chat = gr.Button("Clear Chat History")

            with gr.Column(scale=1):
                # Adding UI elements for notes
                note_title = gr.Textbox(label="Note Title", placeholder="Enter a title for the note")
                notes = gr.TextArea(label="Notes", placeholder="Enter your notes here...", lines=20)
                keywords_for_notes = gr.Textbox(
                    label="Keywords for Notes (comma-separated)",
                    placeholder="Enter keywords for the note",
                    visible=True,
                )
                save_notes_btn = gr.Button("Save Note")
                clear_notes_btn = gr.Button("Clear Current Note text")

                new_note_btn = gr.Button("New Note")
                search_notes_by_keyword = gr.Textbox(label="Search Notes by Keyword")
                search_notes_button = gr.Button("Search Notes")
                note_results = gr.Dropdown(label="Notes", choices=[])
                load_note = gr.Dropdown(label="Load Note", choices=[])

        loading_indicator = gr.HTML("Loading...", visible=False)
        status_message = gr.HTML()
        # Function Definitions

        def update_state(state, **kwargs):
            new_state = state.copy()
            new_state.update(kwargs)
            return new_state

        def create_new_note():
            return gr.update(value='un-named note'), gr.update(value=''), {"note_id": None}

        new_note_btn.click(
            create_new_note,
            outputs=[note_title, notes, note_state]
        )

        def search_notes(keywords):
            if keywords:
                keywords_list = [kw.strip() for kw in keywords.split(',')]
                notes_data, total_pages, total_count = get_notes_by_keywords(keywords_list)
                choices = [f"Note {note_id} ({timestamp})" for note_id, title, content, timestamp in notes_data]
                return gr.update(choices=choices)
            else:
                return gr.update(choices=[])

        search_notes_button.click(
            search_notes,
            inputs=[search_notes_by_keyword],
            outputs=[note_results]
        )

        def load_selected_note(note_selection):
            if note_selection:
                note_id = int(note_selection.split(' ')[1])
                note_data = get_note_by_id(note_id)
                if note_data:
                    note_id, title, content = note_data[0]
                    updated_note_state = {"note_id": note_id}
                    return gr.update(value=title), gr.update(value=content), updated_note_state
            return gr.update(value=''), gr.update(value=''), {"note_id": None}

        note_results.change(
            load_selected_note,
            inputs=[note_results],
            outputs=[note_title, notes, note_state]
        )

        def save_notes_function(note_title_text, notes_content, keywords_content, note_state_value, state_value):
            """Save the notes and associated keywords to the database."""
            conversation_id = state_value["conversation_id"]
            note_id = note_state_value["note_id"]
            if conversation_id and notes_content:
                if note_id:
                    # Update existing note
                    update_note(note_id, note_title_text, notes_content)
                else:
                    # Save new note
                    note_id = save_notes(conversation_id, note_title_text, notes_content)
                    note_state_value["note_id"] = note_id
                if keywords_content:
                    # Clear existing keywords and add new ones
                    clear_keywords_from_note(note_id)
                    add_keywords_to_note(note_id, [kw.strip() for kw in keywords_content.split(',')])

                logging.info("Notes and keywords saved successfully!")
                return notes_content, note_state_value
            else:
                logging.warning("No conversation ID or notes to save.")
                return "", note_state_value

        save_notes_btn.click(
            save_notes_function,
            inputs=[note_title, notes, keywords_for_notes, note_state, state],
            outputs=[notes, note_state]
        )

        def clear_notes_function():
            """Clear notes for the current conversation."""
            return gr.update(value=''), {"note_id": None}

        clear_notes_btn.click(
            clear_notes_function,
            outputs=[notes, note_state]
        )

        def update_conversation_list():
            conversations, total_pages, total_count = get_all_conversations()
            choices = [f"{title} (ID: {conversation_id})" for conversation_id, title in conversations]
            return choices

        # Initialize the conversation list
        load_conversation.choices = update_conversation_list()

        def load_conversation_history(selected_conversation, state_value):
            if selected_conversation:
                conversation_id = selected_conversation.split('(ID: ')[1][:-1]
                chat_data, total_pages_val, _ = load_chat_history(conversation_id, 1, 50)
                # Convert chat data to list of tuples (user_message, assistant_response)
                history = []
                for role, content in chat_data:
                    if role == 'user':
                        history.append((content, ''))
                    else:
                        if history:
                            history[-1] = (history[-1][0], content)
                        else:
                            history.append(('', content))
                # Retrieve notes
                notes_content = get_notes(conversation_id)
                updated_state = update_state(state_value, conversation_id=conversation_id, page=1,
                                             conversation_messages=[])
                return history, updated_state, "\n".join(notes_content)
            return [], state_value, ""

        load_conversation.change(
            load_conversation_history,
            inputs=[load_conversation, state],
            outputs=[chatbot, state, notes]
        )

        def save_conversation_function(conversation_title_text, keywords_text, state_value):
            conversation_messages = state_value.get("conversation_messages", [])
            if not conversation_messages:
                return gr.update(
                    value="<p style='color:red;'>No conversation to save.</p>"), state_value, update_conversation_list()
            # Start a new conversation in the database
            new_conversation_id = start_new_conversation(
                conversation_title_text if conversation_title_text else "Untitled Conversation")
            # Save the messages
            for role, content in conversation_messages:
                save_message(new_conversation_id, role, content)
            # Save keywords if provided
            if keywords_text:
                add_keywords_to_conversation(new_conversation_id, [kw.strip() for kw in keywords_text.split(',')])
            # Update state
            updated_state = update_state(state_value, conversation_id=new_conversation_id)
            # Update the conversation list
            conversation_choices = update_conversation_list()
            return gr.update(
                value="<p style='color:green;'>Conversation saved successfully.</p>"), updated_state, conversation_choices

        save_conversation_button.click(
            save_conversation_function,
            inputs=[conversation_title, keywords, state],
            outputs=[status_message, state, load_conversation]
        )

        def start_new_conversation_wrapper(title, state_value):
            # Reset the state without saving to the database
            updated_state = update_state(state_value, conversation_id=None, page=1, conversation_messages=[])
            # Clear the chat history
            return [], updated_state

        new_conversation.click(
            start_new_conversation_wrapper,
            inputs=[conversation_title, state],
            outputs=[chatbot, state]
        )

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
                file_upload: gr.update(visible=choice == "Upload File"),
                convert_to_text: gr.update(visible=choice == "Upload File"),
                keywords: gr.update(visible=choice == "Upload File")
            }

        context_source.change(update_context_source, context_source,
                              [existing_file, prev_page_btn, next_page_btn, page_info, search_query, search_button,
                               search_results, file_upload, convert_to_text, keywords])

        next_page_btn.click(next_page_fn, inputs=[file_page], outputs=[existing_file, page_info, file_page])
        prev_page_btn.click(prev_page_fn, inputs=[file_page], outputs=[existing_file, page_info, file_page])

        # Initialize the file list
        context_source.change(lambda: update_file_list(1), outputs=[existing_file, page_info, file_page])

        def perform_search(query):
            try:
                results = search_database(query)
                return gr.update(choices=results)
            except Exception as e:
                gr.Error(f"Error performing search: {str(e)}")
                return gr.update(choices=[])

        search_button.click(
            perform_search,
            inputs=[search_query],
            outputs=[search_results]
        )

        def rephrase_question(history, latest_question, api_choice):
            logging.info("RAG QnA: Rephrasing question")
            conversation_history = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history[:-1]])
            prompt = f"""You are a helpful assistant. Given the conversation history and the latest question, resolve any ambiguous references in the latest question.

Conversation History:
{conversation_history}

Latest Question:
{latest_question}

Rewritten Question:"""

            # Use the selected API to generate the rephrased question
            rephrased_question = generate_answer(api_choice, prompt, "")
            logging.info(f"Rephrased question: {rephrased_question}")
            return rephrased_question.strip()

        def rag_qa_chat_wrapper(message, history, context_source, existing_file, search_results, file_upload,
                                convert_to_text, keywords, api_choice, use_query_rewriting, state_value):
            try:
                logging.info(f"Starting rag_qa_chat_wrapper with message: {message}")
                logging.info(f"Context source: {context_source}")
                logging.info(f"API choice: {api_choice}")
                logging.info(f"Query rewriting: {'enabled' if use_query_rewriting else 'disabled'}")

                # Show loading indicator
                yield history, "", gr.update(visible=True), state_value

                conversation_id = state_value.get("conversation_id")
                conversation_messages = state_value.get("conversation_messages", [])

                # Save the user's message
                if conversation_id:
                    save_message(conversation_id, "user", message)
                else:
                    # Append to in-memory messages
                    conversation_messages.append(("user", message))
                    state_value["conversation_messages"] = conversation_messages

                # Ensure api_choice is a string
                api_choice = api_choice.value if isinstance(api_choice, gr.components.Dropdown) else api_choice
                logging.info(f"Resolved API choice: {api_choice}")

                # Only rephrase the question if it's not the first query and query rewriting is enabled
                if len(history) > 0 and use_query_rewriting:
                    rephrased_question = rephrase_question(history, message, api_choice)
                    logging.info(f"Original question: {message}")
                    logging.info(f"Rephrased question: {rephrased_question}")
                else:
                    rephrased_question = message
                    logging.info(f"Using original question: {message}")

                if context_source == "All Files in the Database":
                    # Use the enhanced_rag_pipeline to search the entire database
                    context = enhanced_rag_pipeline(rephrased_question, api_choice)
                    logging.info(f"Using enhanced_rag_pipeline for database search")
                elif context_source == "Search Database":
                    context = f"media_id:{search_results.split('(ID: ')[1][:-1]}"
                    logging.info(f"Using search result with context: {context}")
                else:  # Upload File
                    logging.info("Processing uploaded file")
                    if file_upload is None:
                        raise ValueError("No file uploaded")

                    # Process the uploaded file
                    file_path = file_upload.name
                    file_name = os.path.basename(file_path)
                    logging.info(f"Uploaded file: {file_name}")

                    if convert_to_text:
                        logging.info("Converting file to plain text")
                        content = convert_file_to_text(file_path)
                    else:
                        logging.info("Reading file content")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                    logging.info(f"File content length: {len(content)} characters")

                    # Process keywords
                    if not keywords:
                        keywords = "default,rag-file-upload"
                    logging.info(f"Keywords: {keywords}")

                    # Add the content to the database and get the media_id
                    logging.info("Adding content to database")
                    result = add_media_with_keywords(
                        url=file_name,
                        title=file_name,
                        media_type='document',
                        content=content,
                        keywords=keywords,
                        prompt='No prompt for uploaded files',
                        summary='No summary for uploaded files',
                        transcription_model='None',
                        author='Unknown',
                        ingestion_date=datetime.now().strftime('%Y-%m-%d')
                    )

                    logging.info(f"Result from add_media_with_keywords: {result}")
                    if isinstance(result, tuple):
                        media_id, _ = result
                    else:
                        media_id = result

                    context = f"media_id:{media_id}"
                    logging.info(f"Context for uploaded file: {context}")

                logging.info("Calling rag_qa_chat function")
                new_history, response = rag_qa_chat(rephrased_question, history, context, api_choice)
                # Log first 100 chars of response
                logging.info(f"Response received from rag_qa_chat: {response[:100]}...")

                # Save assistant's response
                if conversation_id:
                    save_message(conversation_id, "assistant", response)
                else:
                    conversation_messages.append(("assistant", response))
                    state_value["conversation_messages"] = conversation_messages

                # Update the state
                state_value["conversation_messages"] = conversation_messages

                # Safely update history
                if new_history:
                    new_history[-1] = (message, response)
                else:
                    new_history = [(message, response)]

                gr.Info("Response generated successfully")
                logging.info("rag_qa_chat_wrapper completed successfully")
                yield new_history, "", gr.update(visible=False), state_value  # Include state_value in outputs
            except ValueError as e:
                logging.error(f"Input error in rag_qa_chat_wrapper: {str(e)}")
                gr.Error(f"Input error: {str(e)}")
                yield history, "", gr.update(visible=False), state_value
            except DatabaseError as e:
                logging.error(f"Database error in rag_qa_chat_wrapper: {str(e)}")
                gr.Error(f"Database error: {str(e)}")
                yield history, "", gr.update(visible=False), state_value
            except Exception as e:
                logging.error(f"Unexpected error in rag_qa_chat_wrapper: {e}", exc_info=True)
                gr.Error("An unexpected error occurred. Please try again later.")
                yield new_history, "", gr.update(visible=False), state_value

        def clear_chat_history():
            return [], ""

        submit.click(
            rag_qa_chat_wrapper,
            inputs=[
                msg,
                chatbot,
                context_source,
                existing_file,
                search_results,
                file_upload,
                convert_to_text,
                keywords,
                api_choice,
                use_query_rewriting,
                state,
            ],
            outputs=[chatbot, msg, loading_indicator, state],
        )

        clear_chat.click(
            clear_chat_history,
            outputs=[chatbot, msg]
        )

    return (
        context_source,
        existing_file,
        search_query,
        search_button,
        search_results,
        file_upload,
        convert_to_text,
        keywords,
        api_choice,
        use_query_rewriting,
        chatbot,
        msg,
        submit,
        clear_chat,
    )


def create_rag_qa_notes_management_tab():
    # New Management Tab
    with gr.TabItem("Notes Management", visible=True):
        gr.Markdown("# RAG QA Notes Management")

        management_state = gr.State({
            "selected_conversation_id": None,
            "selected_note_id": None,
        })

        with gr.Row():
            with gr.Column(scale=1):
                # Search Notes
                search_notes_input = gr.Textbox(label="Search Notes by Keywords")
                search_notes_button = gr.Button("Search Notes")
                notes_list = gr.Dropdown(label="Notes", choices=[])

                # Manage Notes
                load_note_button = gr.Button("Load Note")
                delete_note_button = gr.Button("Delete Note")
                note_title_input = gr.Textbox(label="Note Title")
                note_content_input = gr.TextArea(label="Note Content", lines=15)
                note_keywords_input = gr.Textbox(label="Note Keywords (comma-separated)")
                save_note_button = gr.Button("Save Note")
                create_new_note_button = gr.Button("Create New Note")
                status_message = gr.HTML()

        # Function Definitions
        def search_notes(keywords):
            if keywords:
                keywords_list = [kw.strip() for kw in keywords.split(',')]
                notes_data, total_pages, total_count = get_notes_by_keywords(keywords_list)
                choices = [f"{title} (ID: {note_id})" for note_id, title, content, timestamp in notes_data]
                return gr.update(choices=choices)
            else:
                return gr.update(choices=[])

        search_notes_button.click(
            search_notes,
            inputs=[search_notes_input],
            outputs=[notes_list]
        )

        def load_selected_note(selected_note, state_value):
            if selected_note:
                note_id = int(selected_note.split('(ID: ')[1][:-1])
                note_data = get_note_by_id(note_id)
                if note_data:
                    note_id, title, content = note_data[0]
                    state_value["selected_note_id"] = note_id
                    # Get keywords for the note
                    keywords = get_keywords_for_note(note_id)
                    keywords_str = ', '.join(keywords)
                    return (
                        gr.update(value=title),
                        gr.update(value=content),
                        gr.update(value=keywords_str),
                        state_value
                    )
            return gr.update(value=''), gr.update(value=''), gr.update(value=''), state_value

        load_note_button.click(
            load_selected_note,
            inputs=[notes_list, management_state],
            outputs=[note_title_input, note_content_input, note_keywords_input, management_state]
        )

        def save_note_function(title, content, keywords_str, state_value):
            note_id = state_value["selected_note_id"]
            if note_id:
                update_note(note_id, title, content)
                if keywords_str:
                    # Clear existing keywords and add new ones
                    clear_keywords_from_note(note_id)
                    keywords_list = [kw.strip() for kw in keywords_str.split(',')]
                    add_keywords_to_note(note_id, keywords_list)
                return gr.Info("Note updated successfully.")
            else:
                # Create new note
                conversation_id = state_value.get("selected_conversation_id")
                if conversation_id:
                    note_id = save_notes(conversation_id, title, content)
                    state_value["selected_note_id"] = note_id
                    if keywords_str:
                        keywords_list = [kw.strip() for kw in keywords_str.split(',')]
                        add_keywords_to_note(note_id, keywords_list)
                    return gr.Info("New note created successfully.")
                else:
                    return gr.Error("No conversation selected. Cannot create a new note.")

        save_note_button.click(
            save_note_function,
            inputs=[note_title_input, note_content_input, note_keywords_input, management_state],
            outputs=[]
        )

        def delete_selected_note(state_value):
            note_id = state_value["selected_note_id"]
            if note_id:
                delete_note(note_id)
                # Reset state
                state_value["selected_note_id"] = None
                # Update notes list
                updated_notes = search_notes("")
                return updated_notes, gr.update(value="Note deleted successfully."), state_value
            else:
                return gr.update(), gr.update(value="No note selected."), state_value

        delete_note_button.click(
            delete_selected_note,
            inputs=[management_state],
            outputs=[notes_list, status_message, management_state]
        )

        def create_new_note_function(state_value):
            state_value["selected_note_id"] = None
            return gr.update(value=''), gr.update(value=''), gr.update(value=''), state_value

        create_new_note_button.click(
            create_new_note_function,
            inputs=[management_state],
            outputs=[note_title_input, note_content_input, note_keywords_input, management_state]
        )

    # Return components if needed
    # ...


def create_rag_qa_chat_management_tab():
    # New Management Tab
    with gr.TabItem("Chat Management", visible=True):
        gr.Markdown("# RAG QA Chat Conversation Management")

        management_state = gr.State({
            "selected_conversation_id": None,
            "selected_note_id": None,
        })

        with gr.Row():
            with gr.Column(scale=1):
                # Search Conversations
                search_conversations_input = gr.Textbox(label="Search Conversations by Keywords")
                search_conversations_button = gr.Button("Search Conversations")
                conversations_list = gr.Dropdown(label="Conversations", choices=[])

                # Manage Conversations
                load_conversation_button = gr.Button("Load Conversation")
                delete_conversation_button = gr.Button("Delete Conversation")
                conversation_title_input = gr.Textbox(label="Conversation Title")
                save_conversation_title_button = gr.Button("Save Conversation Title")
                status_message = gr.HTML()

        # Function Definitions
        def search_conversations(keywords):
            if keywords:
                keywords_list = [kw.strip() for kw in keywords.split(',')]
                conversations, total_pages, total_count = search_conversations_by_keywords(keywords_list)
                choices = [f"{title} (ID: {conversation_id})" for conversation_id, title in conversations]
                return gr.update(choices=choices)
            else:
                conversations, total_pages, total_count = get_all_conversations()
                choices = [f"{title} (ID: {conversation_id})" for conversation_id, title in conversations]
                return gr.update(choices=choices)

        search_conversations_button.click(
            search_conversations,
            inputs=[search_conversations_input],
            outputs=[conversations_list]
        )

        def load_selected_conversation(selected_conversation, state_value):
            if selected_conversation:
                conversation_id = selected_conversation.split('(ID: ')[1][:-1]
                # Load conversation title
                conversation_title = selected_conversation.split(' (ID: ')[0]
                # Update state
                state_value["selected_conversation_id"] = conversation_id
                return gr.update(value=conversation_title), state_value
            return gr.update(value=''), state_value

        load_conversation_button.click(
            load_selected_conversation,
            inputs=[conversations_list, management_state],
            outputs=[conversation_title_input, management_state]
        )

        def save_conversation_title(title, state_value):
            conversation_id = state_value["selected_conversation_id"]
            if conversation_id:
                update_conversation_title(conversation_id, title)
                return gr.Info("Conversation title updated successfully.")
            else:
                return gr.Error("No conversation selected.")

        save_conversation_title_button.click(
            save_conversation_title,
            inputs=[conversation_title_input, management_state],
            outputs=[]
        )

        def delete_selected_conversation(state_value):
            conversation_id = state_value["selected_conversation_id"]
            if conversation_id:
                delete_conversation(conversation_id)
                # Reset state
                state_value["selected_conversation_id"] = None
                # Update conversations list
                updated_conversations = search_conversations("")
                return updated_conversations, gr.update(value="Conversation deleted successfully."), state_value
            else:
                return gr.update(), gr.update(value="No conversation selected."), state_value

        delete_conversation_button.click(
            delete_selected_conversation,
            inputs=[management_state],
            outputs=[conversations_list, status_message, management_state]
        )

        def load_selected_note(selected_note, state_value):
            if selected_note:
                note_id = int(selected_note.split('(ID: ')[1][:-1])
                note_data = get_note_by_id(note_id)
                if note_data:
                    note_id, title, content = note_data[0]
                    state_value["selected_note_id"] = note_id
                    # Get keywords for the note
                    keywords = get_keywords_for_note(note_id)
                    keywords_str = ', '.join(keywords)
                    return (
                        gr.update(value=title),
                        gr.update(value=content),
                        gr.update(value=keywords_str),
                        state_value
                    )
            return gr.update(value=''), gr.update(value=''), gr.update(value=''), state_value

    # Return components if needed
    # ...

def update_conversation_title(conversation_id, new_title):
    """Update the title of a conversation."""
    try:
        query = "UPDATE conversation_metadata SET title = ? WHERE conversation_id = ?"
        execute_query(query, (new_title, conversation_id))
        logging.info(f"Conversation '{conversation_id}' title updated to '{new_title}'")
    except Exception as e:
        logging.error(f"Error updating conversation title: {e}")
        raise


def convert_file_to_text(file_path):
    """Convert various file types to plain text."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        return extract_text_and_format_from_pdf(file_path)
    elif file_extension == '.epub':
        return read_epub(file_path)
    elif file_extension in ['.json', '.csv']:
        return read_structured_file(file_path)
    elif file_extension == '.docx':
        return docx2txt.process(file_path)
    elif file_extension in ['.txt', '.md', '.rtf']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def read_structured_file(file_path):
    """Read and convert JSON or CSV files to text."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.json':
        with open(file_path, 'r') as file:
            data = json.load(file)
        return json.dumps(data, indent=2)

    elif file_extension == '.csv':
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            return '\n'.join([','.join(row) for row in csv_reader])

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

#
# End of RAG_QA_Chat_tab.py
########################################################################################################################
