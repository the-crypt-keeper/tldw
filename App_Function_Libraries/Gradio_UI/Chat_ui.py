# Chat_ui.py
# Description: Chat interface functions for Gradio
#
# Imports
import logging
import os
import sqlite3
import time
from datetime import datetime
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.Chat.Chat_Functions import approximate_token_count, chat, save_chat_history, \
    update_chat_content, save_chat_history_to_db_wrapper
from App_Function_Libraries.DB.DB_Manager import search_chat_conversations, update_chat_message, delete_chat_message, \
    load_preset_prompts, db, load_chat_history, start_new_conversation, save_message, search_conversations_by_keywords, \
    get_all_conversations, delete_messages_in_conversation
from App_Function_Libraries.DB.RAG_QA_Chat_DB import get_db_connection
from App_Function_Libraries.Gradio_UI.Gradio_Shared import update_dropdown, update_user_prompt
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from App_Function_Libraries.Utils.Utils import default_api_endpoint, format_api_name, global_api_endpoints
#
#
########################################################################################################################
#
# Functions:


def show_edit_message(selected):
    if selected:
        return gr.update(value=selected[0], visible=True), gr.update(value=selected[1], visible=True), gr.update(
            visible=True)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def show_delete_message(selected):
    if selected:
        return gr.update(value=selected[1], visible=True), gr.update(visible=True)
    return gr.update(visible=False), gr.update(visible=False)


def debug_output(media_content, selected_parts):
    print(f"Debug - Media Content: {media_content}")
    print(f"Debug - Selected Parts: {selected_parts}")
    return ""


def update_selected_parts(use_content, use_summary, use_prompt):
    selected_parts = []
    if use_content:
        selected_parts.append("content")
    if use_summary:
        selected_parts.append("summary")
    if use_prompt:
        selected_parts.append("prompt")
    print(f"Debug - Update Selected Parts: {selected_parts}")
    return selected_parts


# Old update_user_prompt shim for backwards compatibility
def get_system_prompt(preset_name):
    # For backwards compatibility
    prompts = update_user_prompt(preset_name)
    return prompts["system_prompt"]

def clear_chat():
    """
    Return empty list for chatbot and None for conversation_id
    @return:
    """
    return gr.update(value=[]), None


def clear_chat_single():
    """
    Clears the chatbot and chat history.

    Returns:
        list: Empty list for chatbot messages.
        list: Empty list for chat history.
    """
    return [], []

# FIXME - add additional features....
def chat_wrapper(message, history, media_content, selected_parts, api_endpoint, api_key, custom_prompt, conversation_id,
                 save_conversation, temperature, system_prompt, max_tokens=None, top_p=None, frequency_penalty=None,
                 presence_penalty=None, stop_sequence=None):
    try:
        if save_conversation:
            if conversation_id is None:
                # Create a new conversation
                media_id = media_content.get('id', None)
                conversation_name = f"Chat about {media_content.get('title', 'Unknown Media')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                conversation_id = start_new_conversation(title=conversation_name, media_id=media_id)
            # Add user message to the database
            user_message_id = save_message(conversation_id, role="user", content=message)

        # Include the selected parts and custom_prompt only for the first message
        if not history and selected_parts:
            message_body = "\n".join(selected_parts)
            full_message = f"{custom_prompt}\n\n{message}\n\n{message_body}"
        elif custom_prompt:
            full_message = f"{custom_prompt}\n\n{message}"
        else:
            full_message = message

        # Generate bot response
        bot_message = chat(full_message, history, media_content, selected_parts, api_endpoint, api_key, custom_prompt,
                           temperature, system_prompt)

        logging.debug(f"Bot message being returned: {bot_message}")

        if save_conversation:
            # Add assistant message to the database
            save_message(conversation_id, role="assistant", content=bot_message)

        # Update history
        new_history = history + [(message, bot_message)]

        return bot_message, new_history, conversation_id
    except Exception as e:
        logging.error(f"Error in chat wrapper: {str(e)}")
        return "An error occurred.", history, conversation_id


def search_conversations(query):
    """Convert existing chat search to use RAG chat functions"""
    try:
        # Use the RAG search function - search by title if given a query
        if query and query.strip():
            results, _, _ = search_conversations_by_keywords(
                title_query=query.strip()
            )
        else:
            # Get all conversations if no query
            results, _, _ = get_all_conversations()

        if not results:
            return gr.update(choices=[])

        # Format choices to match existing UI format
        conversation_options = [
            (f"{conv['title']} (ID: {conv['conversation_id'][:8]})", conv['conversation_id'])
            for conv in results
        ]

        return gr.update(choices=conversation_options)
    except Exception as e:
        logging.error(f"Error searching conversations: {str(e)}")
        return gr.update(choices=[])


def load_conversation(conversation_id):
    """Convert existing load to use RAG chat functions"""
    if not conversation_id:
        return [], None

    try:
        # Use RAG load function
        messages, _, _ = load_chat_history(conversation_id)

        # Convert to chatbot history format
        history = [
            (content, None) if role == 'user' else (None, content)
            for role, content in messages
        ]

        return history, conversation_id
    except Exception as e:
        logging.error(f"Error loading conversation: {str(e)}")
        return [], None


def update_message_in_chat(message_id, new_text, history):
    update_chat_message(message_id, new_text)
    updated_history = [(msg1, msg2) if msg1[1] != message_id and msg2[1] != message_id
                       else ((new_text, msg1[1]) if msg1[1] == message_id else (new_text, msg2[1]))
                       for msg1, msg2 in history]
    return updated_history


def delete_message_from_chat(message_id, history):
    delete_chat_message(message_id)
    updated_history = [(msg1, msg2) for msg1, msg2 in history if msg1[1] != message_id and msg2[1] != message_id]
    return updated_history


def regenerate_last_message(history, media_content, selected_parts, api_endpoint, api_key, custom_prompt, temperature,
                            system_prompt):
    if not history:
        return history, "No messages to regenerate."

    last_entry = history[-1]
    last_user_message, last_bot_message = last_entry

    if last_bot_message is None:
        return history, "The last message is not from the bot."

    new_history = history[:-1]

    if not last_user_message:
        return new_history, "No user message to regenerate the bot response."

    full_message = last_user_message

    bot_message = chat(
        full_message,
        new_history,
        media_content,
        selected_parts,
        api_endpoint,
        api_key,
        custom_prompt,
        temperature,
        system_prompt
    )

    new_history.append((last_user_message, bot_message))

    return new_history, "Last message regenerated successfully."


def create_chat_interface():
    try:
        default_value = None
        if default_api_endpoint:
            if default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None
    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    """
    with gr.TabItem("Remote LLM Chat (Horizontal)", visible=True):
        gr.Markdown("# Chat with a designated LLM Endpoint, using your selected item as starting context")
        chat_history = gr.State([])
        media_content = gr.State({})
        selected_parts = gr.State([])
        conversation_id = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
                search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                             label="Search By")
                search_button = gr.Button("Search")
                items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                item_mapping = gr.State({})
                with gr.Row():
                    use_content = gr.Checkbox(label="Use Content")
                    use_summary = gr.Checkbox(label="Use Summary")
                    use_prompt = gr.Checkbox(label="Use Prompt")
                    save_conversation = gr.Checkbox(label="Save Conversation", value=False, visible=True)
                with gr.Row():
                    temperature = gr.Slider(label="Temperature", minimum=0.00, maximum=1.0, step=0.05, value=0.7)
                with gr.Row():
                    conversation_search = gr.Textbox(label="Search Conversations")
                with gr.Row():
                    search_conversations_btn = gr.Button("Search Conversations")
                with gr.Row():
                    previous_conversations = gr.Dropdown(label="Select Conversation", choices=[], interactive=True)
                with gr.Row():
                    load_conversations_btn = gr.Button("Load Selected Conversation")

                # Refactored API selection dropdown
                api_endpoint = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Chat Interaction (Optional)"
                )
                api_key = gr.Textbox(label="API Key (if required)", type="password")
                custom_prompt_checkbox = gr.Checkbox(label="Use a Custom Prompt",
                                                     value=False,
                                                     visible=True)
                preset_prompt_checkbox = gr.Checkbox(label="Use a pre-set Prompt",
                                                     value=False,
                                                     visible=True)
                preset_prompt = gr.Dropdown(label="Select Preset Prompt",
                                            choices=load_preset_prompts(),
                                            visible=False)
                user_prompt = gr.Textbox(label="Custom Prompt",
                                         placeholder="Enter custom prompt here",
                                         lines=3,
                                         visible=False)
                system_prompt_input = gr.Textbox(label="System Prompt",
                                                 value="You are a helpful AI assitant",
                                                 lines=3,
                                                 visible=False)
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=800, elem_classes="chatbot-container")
                msg = gr.Textbox(label="Enter your message")
                submit = gr.Button("Submit")
                regenerate_button = gr.Button("Regenerate Last Message")
                token_count_display = gr.Number(label="Approximate Token Count", value=0, interactive=False)
                clear_chat_button = gr.Button("Clear Chat")

                edit_message_id = gr.Number(label="Message ID to Edit", visible=False)
                edit_message_text = gr.Textbox(label="Edit Message", visible=False)
                update_message_button = gr.Button("Update Message", visible=False)

                delete_message_id = gr.Number(label="Message ID to Delete", visible=False)
                delete_message_button = gr.Button("Delete Message", visible=False)

                chat_media_name = gr.Textbox(label="Custom Chat Name(optional)")
                save_chat_history_to_db = gr.Button("Save Chat History to DataBase")
                save_status = gr.Textbox(label="Save Status", interactive=False)
                save_chat_history_as_file = gr.Button("Save Chat History as File")
                download_file = gr.File(label="Download Chat History")

        # Restore original functionality
        search_button.click(
            fn=update_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[items_output, item_mapping]
        )

        def save_chat_wrapper(history, conversation_id, media_content):
            file_path = save_chat_history(history, conversation_id, media_content)
            if file_path:
                return file_path, f"Chat history saved successfully as {os.path.basename(file_path)}!"
            else:
                return None, "Error saving chat history. Please check the logs and try again."

        save_chat_history_as_file.click(
            save_chat_wrapper,
            inputs=[chatbot, conversation_id, media_content],
            outputs=[download_file, save_status]
        )

        def update_prompts(preset_name):
            prompts = update_user_prompt(preset_name)
            return (
                gr.update(value=prompts["user_prompt"], visible=True),
                gr.update(value=prompts["system_prompt"], visible=True)
            )

        def clear_chat():
            return [], None  # Return empty list for chatbot and None for conversation_id

        clear_chat_button.click(
            clear_chat,
            outputs=[chatbot, conversation_id]
        )

        preset_prompt.change(
            update_prompts,
            inputs=preset_prompt,
            outputs=[user_prompt, system_prompt_input]
        )

        custom_prompt_checkbox.change(
            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[custom_prompt_checkbox],
            outputs=[user_prompt, system_prompt_input]
        )

        preset_prompt_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[preset_prompt_checkbox],
            outputs=[preset_prompt]
        )

        submit.click(
            chat_wrapper,
            inputs=[msg, chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt, conversation_id,
                    save_conversation, temperature, system_prompt_input],
            outputs=[msg, chatbot, conversation_id]
        ).then(  # Clear the message box after submission
            lambda x: gr.update(value=""),
            inputs=[chatbot],
            outputs=[msg]
        ).then(  # Clear the user prompt after the first message
            lambda: (gr.update(value=""), gr.update(value="")),
            outputs=[user_prompt, system_prompt_input]
        ).then(
        lambda history: approximate_token_count(history),
        inputs=[chatbot],
        outputs=[token_count_display]
        )

        items_output.change(
            update_chat_content,
            inputs=[items_output, use_content, use_summary, use_prompt, item_mapping],
            outputs=[media_content, selected_parts]
        )

        use_content.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                           outputs=[selected_parts])
        use_summary.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                           outputs=[selected_parts])
        use_prompt.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                          outputs=[selected_parts])
        items_output.change(debug_output, inputs=[media_content, selected_parts], outputs=[])

        search_conversations_btn.click(
            search_conversations,
            inputs=[conversation_search],
            outputs=[previous_conversations]
        )

        load_conversations_btn.click(
            clear_chat,
            outputs=[chatbot, chat_history]
        ).then(
            load_conversation,
            inputs=[previous_conversations],
            outputs=[chatbot, conversation_id]
        )

        previous_conversations.change(
            load_conversation,
            inputs=[previous_conversations],
            outputs=[chat_history]
        )

        update_message_button.click(
            update_message_in_chat,
            inputs=[edit_message_id, edit_message_text, chat_history],
            outputs=[chatbot]
        )

        delete_message_button.click(
            delete_message_from_chat,
            inputs=[delete_message_id, chat_history],
            outputs=[chatbot]
        )

        save_chat_history_as_file.click(
            save_chat_history,
            inputs=[chatbot, conversation_id],
            outputs=[download_file]
        )

        save_chat_history_to_db.click(
            save_chat_history_to_db_wrapper,
            inputs=[chatbot, conversation_id, media_content, chat_media_name],
            outputs=[conversation_id, gr.Textbox(label="Save Status")]
        )

        regenerate_button.click(
            regenerate_last_message,
            inputs=[chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt, temperature,
                    system_prompt_input],
            outputs=[chatbot, save_status]
        ).then(
            lambda history: approximate_token_count(history),
            inputs=[chatbot],
            outputs=[token_count_display]
        )

        chatbot.select(show_edit_message, None, [edit_message_text, edit_message_id, update_message_button])
        chatbot.select(show_delete_message, None, [delete_message_id, delete_message_button])


def create_chat_interface_stacked():
    try:
        default_value = None
        if default_api_endpoint:
            if default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None

    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    """
    with gr.TabItem("Remote LLM Chat - Stacked", visible=True):
        gr.Markdown("# Stacked Chat")
        chat_history = gr.State([])
        media_content = gr.State({})
        selected_parts = gr.State([])
        conversation_id = gr.State(None)

        with gr.Row():
            with gr.Column():
                search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
                search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                             label="Search By")
                search_button = gr.Button("Search")
                items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                item_mapping = gr.State({})
                with gr.Row():
                    use_content = gr.Checkbox(label="Use Content")
                    use_summary = gr.Checkbox(label="Use Summary")
                    use_prompt = gr.Checkbox(label="Use Prompt")
                    save_conversation = gr.Checkbox(label="Save Conversation", value=False, visible=True)
                    temp = gr.Slider(label="Temperature", minimum=0.00, maximum=1.0, step=0.05, value=0.7)
                with gr.Row():
                    conversation_search = gr.Textbox(label="Search Conversations")
                with gr.Row():
                    previous_conversations = gr.Dropdown(label="Select Conversation", choices=[], interactive=True)
                with gr.Row():
                    search_conversations_btn = gr.Button("Search Conversations")
                    load_conversations_btn = gr.Button("Load Selected Conversation")
            with gr.Column():
                # Refactored API selection dropdown
                api_endpoint = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Chat Interaction (Optional)"
                )
                api_key = gr.Textbox(label="API Key (if required)", type="password")
                preset_prompt = gr.Dropdown(label="Select Preset Prompt",
                                            choices=load_preset_prompts(),
                                            visible=True)
                system_prompt = gr.Textbox(label="System Prompt",
                                           value="You are a helpful AI assistant.",
                                           lines=4,
                                           visible=True)
                user_prompt = gr.Textbox(label="Custom User Prompt",
                                         placeholder="Enter custom prompt here",
                                         lines=4,
                                         visible=True)
                gr.Markdown("Scroll down for the chat window...")
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(height=800, elem_classes="chatbot-container")
                msg = gr.Textbox(label="Enter your message")
        with gr.Row():
            with gr.Column():
                submit = gr.Button("Submit")
                regenerate_button = gr.Button("Regenerate Last Message")
                token_count_display = gr.Number(label="Approximate Token Count", value=0, interactive=False)
                clear_chat_button = gr.Button("Clear Chat")
                chat_media_name = gr.Textbox(label="Custom Chat Name(optional)", visible=True)
                save_chat_history_to_db = gr.Button("Save Chat History to DataBase")
                save_status = gr.Textbox(label="Save Status", interactive=False)
                save_chat_history_as_file = gr.Button("Save Chat History as File")
            with gr.Column():
                download_file = gr.File(label="Download Chat History")

        # Restore original functionality
        search_button.click(
            fn=update_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[items_output, item_mapping]
        )

        def search_conversations(query):
            try:
                # Use RAG search with title search
                if query and query.strip():
                    results, _, _ = search_conversations_by_keywords(title_query=query.strip())
                else:
                    results, _, _ = get_all_conversations()

                if not results:
                    return gr.update(choices=[])

                # Format choices to match UI
                conversation_options = [
                    (f"{conv['title']} (ID: {conv['conversation_id'][:8]})", conv['conversation_id'])
                    for conv in results
                ]

                return gr.update(choices=conversation_options)
            except Exception as e:
                logging.error(f"Error searching conversations: {str(e)}")
                return gr.update(choices=[])

        def load_conversation(conversation_id):
            if not conversation_id:
                return [], None

            try:
                # Use RAG load function
                messages, _, _ = load_chat_history(conversation_id)

                # Convert to chatbot history format
                history = [
                    (content, None) if role == 'user' else (None, content)
                    for role, content in messages
                ]

                return history, conversation_id
            except Exception as e:
                logging.error(f"Error loading conversation: {str(e)}")
                return [], None

        def save_chat_history_to_db_wrapper(chatbot, conversation_id, media_content, chat_name=None):
            log_counter("save_chat_history_to_db_attempt")
            start_time = time.time()
            logging.info(f"Attempting to save chat history. Media content type: {type(media_content)}")

            try:
                # First check if we can access the database
                try:
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                except sqlite3.DatabaseError as db_error:
                    logging.error(f"Database is corrupted or inaccessible: {str(db_error)}")
                    return conversation_id, gr.update(
                        value="Database error: The database file appears to be corrupted. Please contact support.")

                # For both new and existing conversations
                try:
                    if not conversation_id:
                        title = chat_name if chat_name else "Untitled Conversation"
                        conversation_id = start_new_conversation(title=title)
                        logging.info(f"Created new conversation with ID: {conversation_id}")

                    # Update existing messages
                    delete_messages_in_conversation(conversation_id)
                    for user_msg, assistant_msg in chatbot:
                        if user_msg:
                            save_message(conversation_id, "user", user_msg)
                        if assistant_msg:
                            save_message(conversation_id, "assistant", assistant_msg)
                except sqlite3.DatabaseError as db_error:
                    logging.error(f"Database error during message save: {str(db_error)}")
                    return conversation_id, gr.update(
                        value="Database error: Unable to save messages. Please try again or contact support.")

                save_duration = time.time() - start_time
                log_histogram("save_chat_history_to_db_duration", save_duration)
                log_counter("save_chat_history_to_db_success")

                return conversation_id, gr.update(value="Chat history saved successfully!")

            except Exception as e:
                log_counter("save_chat_history_to_db_error", labels={"error": str(e)})
                error_message = f"Failed to save chat history: {str(e)}"
                logging.error(error_message, exc_info=True)
                return conversation_id, gr.update(value=error_message)

        def update_prompts(preset_name):
            prompts = update_user_prompt(preset_name)
            return (
                gr.update(value=prompts["user_prompt"], visible=True),
                gr.update(value=prompts["system_prompt"], visible=True)
            )

        def clear_chat():
            return [], None, 0  # Empty history, conversation_id, and token count

        clear_chat_button.click(
            clear_chat,
            outputs=[chatbot, conversation_id, token_count_display]
        )

        preset_prompt.change(
            update_prompts,
            inputs=preset_prompt,
            outputs=[user_prompt, system_prompt]
        )

        submit.click(
            chat_wrapper,
            inputs=[msg, chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt,
                    conversation_id, save_conversation, temp, system_prompt],
            outputs=[msg, chatbot, conversation_id]
        ).then(
            lambda x: gr.update(value=""),
            inputs=[chatbot],
            outputs=[msg]
        ).then(
            lambda history: approximate_token_count(history),
            inputs=[chatbot],
            outputs=[token_count_display]
        )

        items_output.change(
            update_chat_content,
            inputs=[items_output, use_content, use_summary, use_prompt, item_mapping],
            outputs=[media_content, selected_parts]
        )
        use_content.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                           outputs=[selected_parts])
        use_summary.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                           outputs=[selected_parts])
        use_prompt.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                          outputs=[selected_parts])
        items_output.change(debug_output, inputs=[media_content, selected_parts], outputs=[])

        search_conversations_btn.click(
            search_conversations,
            inputs=[conversation_search],
            outputs=[previous_conversations]
        )

        load_conversations_btn.click(
            clear_chat,
            outputs=[chatbot, chat_history]
        ).then(
            load_conversation,
            inputs=[previous_conversations],
            outputs=[chatbot, conversation_id]
        )

        previous_conversations.change(
            load_conversation,
            inputs=[previous_conversations],
            outputs=[chat_history]
        )

        save_chat_history_as_file.click(
            save_chat_history,
            inputs=[chatbot, conversation_id],
            outputs=[download_file]
        )

        save_chat_history_to_db.click(
            save_chat_history_to_db_wrapper,
            inputs=[chatbot, conversation_id, media_content, chat_media_name],
            outputs=[conversation_id, save_status]
        )

        regenerate_button.click(
            regenerate_last_message,
            inputs=[chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt, temp, system_prompt],
            outputs=[chatbot, gr.Textbox(label="Regenerate Status")]
        ).then(
            lambda history: approximate_token_count(history),
            inputs=[chatbot],
            outputs=[token_count_display]
        )


def create_chat_interface_multi_api():
    try:
        default_value = None
        if default_api_endpoint:
            if default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None
    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    .chat-window {
        height: 400px;
        overflow-y: auto;
    }
    """
    with gr.TabItem("One Prompt - Multiple APIs", visible=True):
        gr.Markdown("# One Prompt but Multiple APIs Chat Interface")

        with gr.Row():
            with gr.Column(scale=1):
                search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
                search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                             label="Search By")
                search_button = gr.Button("Search")
                items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                item_mapping = gr.State({})
                with gr.Row():
                    use_content = gr.Checkbox(label="Use Content")
                    use_summary = gr.Checkbox(label="Use Summary")
                    use_prompt = gr.Checkbox(label="Use Prompt")
            with gr.Column():
                preset_prompt = gr.Dropdown(label="Select Preset Prompt", choices=load_preset_prompts(), visible=True)
                system_prompt = gr.Textbox(label="System Prompt", value="You are a helpful AI assistant.", lines=5)
                user_prompt = gr.Textbox(label="Modify Prompt (Prefixed to your message every time)", lines=5,
                                         value="", visible=True)

        with gr.Row():
            chatbots = []
            api_endpoints = []
            api_keys = []
            temperatures = []
            regenerate_buttons = []
            token_count_displays = []
            for i in range(3):
                with gr.Column():
                    gr.Markdown(f"### Chat Window {i + 1}")
                    # Refactored API selection dropdown
                    api_endpoint = gr.Dropdown(
                        choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                        value=default_value,
                        label="API for Chat Interaction (Optional)"
                    )
                    api_key = gr.Textbox(label=f"API Key {i + 1} (if required)", type="password")
                    temperature = gr.Slider(label=f"Temperature {i + 1}", minimum=0.0, maximum=1.0, step=0.05,
                                            value=0.7)
                    chatbot = gr.Chatbot(height=800, elem_classes="chat-window")
                    token_count_display = gr.Number(label=f"Approximate Token Count {i + 1}", value=0,
                                                    interactive=False)
                    token_count_displays.append(token_count_display)
                    regenerate_button = gr.Button(f"Regenerate Last Message {i + 1}")
                    chatbots.append(chatbot)
                    api_endpoints.append(api_endpoint)
                    api_keys.append(api_key)
                    temperatures.append(temperature)
                    regenerate_buttons.append(regenerate_button)

        with gr.Row():
            msg = gr.Textbox(label="Enter your message", scale=4)
            submit = gr.Button("Submit", scale=1)
            clear_chat_button = gr.Button("Clear All Chats")

        # State variables
        chat_history = [gr.State([]) for _ in range(3)]
        media_content = gr.State({})
        selected_parts = gr.State([])
        conversation_id = gr.State(None)

        # Event handlers
        search_button.click(
            fn=update_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[items_output, item_mapping]
        )

        preset_prompt.change(update_user_prompt, inputs=preset_prompt, outputs=user_prompt)

        def clear_all_chats():
            return [[]] * 3 + [[]] * 3 + [0] * 3

        clear_chat_button.click(
            clear_all_chats,
            outputs=chatbots + chat_history + token_count_displays
        )

        def chat_wrapper_multi(message, custom_prompt, system_prompt, *args):
            chat_histories = args[:3]
            chatbots = args[3:6]
            api_endpoints = args[6:9]
            api_keys = args[9:12]
            temperatures = args[12:15]
            media_content = args[15]
            selected_parts = args[16]

            new_chat_histories = []
            new_chatbots = []

            for i in range(3):
                # Call chat_wrapper with dummy values for conversation_id and save_conversation
                bot_message, new_history, _ = chat_wrapper(
                    message, chat_histories[i], media_content, selected_parts,
                    api_endpoints[i], api_keys[i], custom_prompt, None,  # None for conversation_id
                    False,  # False for save_conversation
                    temperature=temperatures[i],
                    system_prompt=system_prompt
                )

                new_chatbot = chatbots[i] + [(message, bot_message)]

                new_chat_histories.append(new_history)
                new_chatbots.append(new_chatbot)

            return [gr.update(value="")] + new_chatbots + new_chat_histories

        def update_token_counts(*histories):
            token_counts = []
            for history in histories:
                token_counts.append(approximate_token_count(history))
            return token_counts

        def regenerate_last_message(chat_history, chatbot, media_content, selected_parts, api_endpoint, api_key, custom_prompt, temperature, system_prompt):
            if not chat_history:
                return chatbot, chat_history, "No messages to regenerate."

            last_entry = chat_history[-1]
            last_user_message, last_bot_message = last_entry

            if last_bot_message is None:
                return chatbot, chat_history, "The last message is not from the bot."

            new_history = chat_history[:-1]

            if not last_user_message:
                return chatbot[:-1], new_history, "No user message to regenerate the bot response."

            bot_message = chat(
                last_user_message,
                new_history,
                media_content,
                selected_parts,
                api_endpoint,
                api_key,
                custom_prompt,
                temperature,
                system_prompt
            )

            new_history.append((last_user_message, bot_message))
            new_chatbot = chatbot[:-1] + [(last_user_message, bot_message)]

            return new_chatbot, new_history, "Last message regenerated successfully."

        for i in range(3):
            regenerate_buttons[i].click(
                regenerate_last_message,
                inputs=[chat_history[i], chatbots[i], media_content, selected_parts, api_endpoints[i], api_keys[i],
                        user_prompt, temperatures[i], system_prompt],
                outputs=[chatbots[i], chat_history[i], gr.Textbox(label=f"Regenerate Status {i + 1}")]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[chat_history[i]],
                outputs=[token_count_displays[i]]
            )

        # In the create_chat_interface_multi_api function:
        submit.click(
            chat_wrapper_multi,
            inputs=[msg, user_prompt,
                    system_prompt] + chat_history + chatbots + api_endpoints + api_keys + temperatures +
                   [media_content, selected_parts],
            outputs=[msg] + chatbots + chat_history
        ).then(
            lambda: (gr.update(value=""), gr.update(value="")),
            outputs=[msg, user_prompt]
        ).then(
            update_token_counts,
            inputs=chat_history,
            outputs=token_count_displays
        )

        items_output.change(
            update_chat_content,
            inputs=[items_output, use_content, use_summary, use_prompt, item_mapping],
            outputs=[media_content, selected_parts]
        )

        for checkbox in [use_content, use_summary, use_prompt]:
            checkbox.change(
                update_selected_parts,
                inputs=[use_content, use_summary, use_prompt],
                outputs=[selected_parts]
            )


def create_chat_interface_four():
    try:
        default_value = None
        if default_api_endpoint:
            if default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None
    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    .chat-window {
        height: 400px;
        overflow-y: auto;
    }
    """

    with gr.TabItem("Four Independent API Chats", visible=True):
        gr.Markdown("# Four Independent API Chat Interfaces")

        with gr.Row():
            with gr.Column():
                preset_prompt = gr.Dropdown(
                    label="Select Preset Prompt",
                    choices=load_preset_prompts(),
                    visible=True
                )
                user_prompt = gr.Textbox(
                    label="Modify Prompt",
                    lines=3
                )
            with gr.Column():
                gr.Markdown("Scroll down for the chat windows...")

        chat_interfaces = []

        def create_single_chat_interface(index, user_prompt_component):
            with gr.Column():
                gr.Markdown(f"### Chat Window {index + 1}")
                # Refactored API selection dropdown
                api_endpoint = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Chat Interaction (Optional)"
                )
                api_key = gr.Textbox(
                    label=f"API Key {index + 1} (if required)",
                    type="password"
                )
                temperature = gr.Slider(
                    label=f"Temperature {index + 1}",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.7
                )
                chatbot = gr.Chatbot(height=400, elem_classes="chat-window")
                msg = gr.Textbox(label=f"Enter your message for Chat {index + 1}")
                submit = gr.Button(f"Submit to Chat {index + 1}")
                regenerate_button = gr.Button(f"Regenerate Last Message {index + 1}")
                token_count_display = gr.Number(label=f"Approximate Token Count {index + 1}", value=0,
                                                interactive=False)
                clear_chat_button = gr.Button(f"Clear Chat {index + 1}")


                # State to maintain chat history
                chat_history = gr.State([])

                # Append to chat_interfaces list
                chat_interfaces.append({
                    'api_endpoint': api_endpoint,
                    'api_key': api_key,
                    'temperature': temperature,
                    'chatbot': chatbot,
                    'msg': msg,
                    'submit': submit,
                    'regenerate_button': regenerate_button,
                    'clear_chat_button': clear_chat_button,
                    'chat_history': chat_history,
                    'token_count_display': token_count_display
                })

        # Create four chat interfaces arranged in a 2x2 grid
        with gr.Row():
            for i in range(2):
                with gr.Column():
                    for j in range(2):
                        create_single_chat_interface(i * 2 + j, user_prompt)

        # Update user_prompt based on preset_prompt selection
        preset_prompt.change(
            fn=update_user_prompt,
            inputs=preset_prompt,
            outputs=user_prompt
        )

        def chat_wrapper_single(message, chat_history, api_endpoint, api_key, temperature, user_prompt):
            logging.debug(f"Chat Wrapper Single - Message: {message}, Chat History: {chat_history}")

            new_msg, new_history, _ = chat_wrapper(
                message,
                chat_history,
                {},  # Empty media_content
                [],  # Empty selected_parts
                api_endpoint,
                api_key,
                user_prompt,  # custom_prompt
                None,  # conversation_id
                False,  # save_conversation
                temperature,  # temperature
                system_prompt="",  # system_prompt
                max_tokens=None,
                top_p=None,
                frequency_penalty=None,
                presence_penalty=None,
                stop_sequence=None
            )
            if "API request failed" not in new_msg:
                chat_history.append((message, new_msg))
            else:
                logging.error(f"API request failed: {new_msg}")

            return "", chat_history, chat_history

        def regenerate_last_message(chat_history, api_endpoint, api_key, temperature, user_prompt):
            if not chat_history:
                return chat_history, chat_history, "No messages to regenerate."

            last_user_message, _ = chat_history[-1]

            new_msg, new_history, _ = chat_wrapper(
                last_user_message,
                chat_history[:-1],
                {},  # Empty media_content
                [],  # Empty selected_parts
                api_endpoint,
                api_key,
                user_prompt,  # custom_prompt
                None,  # conversation_id
                False,  # save_conversation
                temperature,  # temperature
                system_prompt="",  # system_prompt
                max_tokens=None,
                top_p=None,
                frequency_penalty=None,
                presence_penalty=None,
                stop_sequence=None
            )

            if "API request failed" not in new_msg:
                new_history.append((last_user_message, new_msg))
                return new_history, new_history, "Last message regenerated successfully."
            else:
                logging.error(f"API request failed during regeneration: {new_msg}")
                return chat_history, chat_history, f"Failed to regenerate: {new_msg}"

        # Attach click events for each chat interface
        for interface in chat_interfaces:
            interface['submit'].click(
                chat_wrapper_single,
                inputs=[
                    interface['msg'],
                    interface['chat_history'],
                    interface['api_endpoint'],
                    interface['api_key'],
                    interface['temperature'],
                    user_prompt
                ],
                outputs=[
                    interface['msg'],
                    interface['chatbot'],
                    interface['chat_history']
                ]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[interface['chat_history']],
                outputs=[interface['token_count_display']]
            )

            interface['regenerate_button'].click(
                regenerate_last_message,
                inputs=[
                    interface['chat_history'],
                    interface['api_endpoint'],
                    interface['api_key'],
                    interface['temperature'],
                    user_prompt
                ],
                outputs=[
                    interface['chatbot'],
                    interface['chat_history'],
                    gr.Textbox(label="Regenerate Status")
                ]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[interface['chat_history']],
                outputs=[interface['token_count_display']]
            )

            def clear_chat_single():
                return [], [], 0

            for interface in chat_interfaces:
                interface['clear_chat_button'].click(
                    clear_chat_single,
                    outputs=[interface['chatbot'], interface['chat_history'], interface['token_count_display']]
                )


def chat_wrapper_single(message, chat_history, chatbot, api_endpoint, api_key, temperature, media_content,
                        selected_parts, conversation_id, save_conversation, user_prompt):
    new_msg, new_history, new_conv_id = chat_wrapper(
        message, chat_history, media_content, selected_parts,
        api_endpoint, api_key, user_prompt, conversation_id,
        save_conversation, temperature, system_prompt=""
    )

    if new_msg:
        updated_chatbot = chatbot + [(message, new_msg)]
    else:
        updated_chatbot = chatbot

    return new_msg, updated_chatbot, new_history, new_conv_id

# Mock function to simulate LLM processing
def process_with_llm(workflow, context, prompt, api_endpoint, api_key):
    api_key_snippet = api_key[:5] + "..." if api_key else "Not provided"
    return f"LLM output using {api_endpoint} (API Key: {api_key_snippet}) for {workflow} with context: {context[:30]}... and prompt: {prompt[:30]}..."

#
# End of Chat_ui.py
#######################################################################################################################