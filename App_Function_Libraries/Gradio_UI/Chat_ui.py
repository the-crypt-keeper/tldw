# Chat_ui.py
# Description: Chat interface functions for Gradio
#
# Imports
import html
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.Chat import chat, save_chat_history, update_chat_content, save_chat_history_to_db_wrapper
from App_Function_Libraries.DB.DB_Manager import add_chat_message, search_chat_conversations, create_chat_conversation, \
    get_chat_messages, update_chat_message, delete_chat_message, load_preset_prompts, db
from App_Function_Libraries.Gradio_UI.Gradio_Shared import update_dropdown, update_user_prompt


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
    # Return empty list for chatbot and None for conversation_id
    return gr.update(value=[]), None


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
                conversation_id = create_chat_conversation(media_id, conversation_name)

            # Add user message to the database
            user_message_id = add_chat_message(conversation_id, "user", message)

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

        if save_conversation:
            # Add assistant message to the database
            add_chat_message(conversation_id, "assistant", bot_message)

        # Update history
        history.append((message, bot_message))

        return bot_message, history, conversation_id
    except Exception as e:
        logging.error(f"Error in chat wrapper: {str(e)}")
        return "An error occurred.", history, conversation_id


def search_conversations(query):
    try:
        conversations = search_chat_conversations(query)
        if not conversations:
            print(f"Debug - Search Conversations - No results found for query: {query}")
            return gr.update(choices=[])

        conversation_options = [
            (f"{c['conversation_name']} (Media: {c['media_title']}, ID: {c['id']})", c['id'])
            for c in conversations
        ]
        print(f"Debug - Search Conversations - Options: {conversation_options}")
        return gr.update(choices=conversation_options)
    except Exception as e:
        print(f"Debug - Search Conversations - Error: {str(e)}")
        return gr.update(choices=[])


def load_conversation(conversation_id):
    if not conversation_id:
        return [], None

    messages = get_chat_messages(conversation_id)
    history = [
        (msg['message'], None) if msg['sender'] == 'user' else (None, msg['message'])
        for msg in messages
    ]
    return history, conversation_id


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


def create_chat_interface():
    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    """
    with gr.TabItem("Remote LLM Chat (Horizontal)"):
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

                api_endpoint = gr.Dropdown(label="Select API Endpoint",
                                           choices=["Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek",
                                                    "Mistral", "OpenRouter",
                                                    "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama",
                                                    "HuggingFace"])
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
            with gr.Column():
                chatbot = gr.Chatbot(height=600, elem_classes="chatbot-container")
                msg = gr.Textbox(label="Enter your message")
                submit = gr.Button("Submit")
                clear_chat_button = gr.Button("Clear Chat")

                edit_message_id = gr.Number(label="Message ID to Edit", visible=False)
                edit_message_text = gr.Textbox(label="Edit Message", visible=False)
                update_message_button = gr.Button("Update Message", visible=False)

                delete_message_id = gr.Number(label="Message ID to Delete", visible=False)
                delete_message_button = gr.Button("Delete Message", visible=False)

                save_chat_history_to_db = gr.Button("Save Chat History to DataBase")
                save_chat_history_as_file = gr.Button("Save Chat History as File")
                download_file = gr.File(label="Download Chat History")
                save_status = gr.Textbox(label="Save Status", interactive=False)

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
            inputs=[msg, chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt,
                    conversation_id, save_conversation, temperature, system_prompt_input],
            outputs=[msg, chatbot, conversation_id]
        ).then(  # Clear the message box after submission
            lambda x: gr.update(value=""),
            inputs=[chatbot],
            outputs=[msg]
        ).then(  # Clear the user prompt after the first message
            lambda: (gr.update(value=""), gr.update(value="")),
            outputs=[user_prompt, system_prompt_input]
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
            inputs=[chatbot, conversation_id, media_content],
            outputs=[conversation_id, gr.Textbox(label="Save Status")]
        )

        chatbot.select(show_edit_message, None, [edit_message_text, edit_message_id, update_message_button])
        chatbot.select(show_delete_message, None, [delete_message_id, delete_message_button])


def create_chat_interface_stacked():
    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    """
    with gr.TabItem("Remote LLM Chat - Stacked"):
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
                api_endpoint = gr.Dropdown(label="Select API Endpoint",
                                           choices=["Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek",
                                                    "OpenRouter", "Mistral", "Llama.cpp", "Kobold", "Ooba", "Tabbyapi",
                                                    "VLLM", "ollama", "HuggingFace"])
                api_key = gr.Textbox(label="API Key (if required)", type="password")
                preset_prompt = gr.Dropdown(label="Select Preset Prompt",
                                            choices=load_preset_prompts(),
                                            visible=True)
                system_prompt = gr.Textbox(label="System Prompt",
                                           value="You are a helpful AI assistant.",
                                           lines=3,
                                           visible=True)
                user_prompt = gr.Textbox(label="Custom User Prompt",
                                         placeholder="Enter custom prompt here",
                                         lines=3,
                                         visible=True)
                gr.Markdown("Scroll down for the chat window...")
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(height=600, elem_classes="chatbot-container")
                msg = gr.Textbox(label="Enter your message")
        with gr.Row():
            with gr.Column():
                submit = gr.Button("Submit")
                clear_chat_button = gr.Button("Clear Chat")

                edit_message_id = gr.Number(label="Message ID to Edit", visible=False)
                edit_message_text = gr.Textbox(label="Edit Message", visible=False)
                update_message_button = gr.Button("Update Message", visible=False)

                delete_message_id = gr.Number(label="Message ID to Delete", visible=False)
                delete_message_button = gr.Button("Delete Message", visible=False)
                save_chat_history_to_db = gr.Button("Save Chat History to DataBase")
                save_chat_history_as_file = gr.Button("Save Chat History as File")
            with gr.Column():
                download_file = gr.File(label="Download Chat History")

        # Restore original functionality
        search_button.click(
            fn=update_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[items_output, item_mapping]
        )

        def update_prompts(preset_name):
            prompts = update_user_prompt(preset_name)
            return (
                gr.update(value=prompts["user_prompt"], visible=True),
                gr.update(value=prompts["system_prompt"], visible=True)
            )

        clear_chat_button.click(
            clear_chat,
            outputs=[chatbot, conversation_id]
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
        ).then(  # Clear the message box after submission
            lambda x: gr.update(value=""),
            inputs=[chatbot],
            outputs=[msg]
        ).then(  # Clear the user prompt after the first message
            lambda: gr.update(value=""),
            outputs=[user_prompt, system_prompt]
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
            inputs=[chatbot, conversation_id, media_content],
            outputs=[conversation_id, gr.Textbox(label="Save Status")]
        )

        chatbot.select(show_edit_message, None, [edit_message_text, edit_message_id, update_message_button])
        chatbot.select(show_delete_message, None, [delete_message_id, delete_message_button])


# FIXME - System prompts
def create_chat_interface_multi_api():
    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    .chat-window {
        height: 400px;
        overflow-y: auto;
    }
    """
    with gr.TabItem("One Prompt - Multiple APIs"):
        gr.Markdown("# One Prompt but Multiple API Chat Interface")

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
                user_prompt = gr.Textbox(label="Modify Prompt", lines=5, value=".")

        with gr.Row():
            chatbots = []
            api_endpoints = []
            api_keys = []
            temperatures = []
            for i in range(3):
                with gr.Column():
                    gr.Markdown(f"### Chat Window {i + 1}")
                    api_endpoint = gr.Dropdown(label=f"API Endpoint {i + 1}",
                                               choices=["Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq",
                                                        "DeepSeek", "Mistral", "OpenRouter", "Llama.cpp", "Kobold",
                                                        "Ooba",
                                                        "Tabbyapi", "VLLM", "ollama", "HuggingFace"])
                    api_key = gr.Textbox(label=f"API Key {i + 1} (if required)", type="password")
                    temperature = gr.Slider(label=f"Temperature {i + 1}", minimum=0.0, maximum=1.0, step=0.05,
                                            value=0.7)
                    chatbot = gr.Chatbot(height=800, elem_classes="chat-window")
                    chatbots.append(chatbot)
                    api_endpoints.append(api_endpoint)
                    api_keys.append(api_key)
                    temperatures.append(temperature)

        with gr.Row():
            msg = gr.Textbox(label="Enter your message", scale=4)
            submit = gr.Button("Submit", scale=1)
            # FIXME - clear chat
        #     clear_chat_button = gr.Button("Clear Chat")
        #
        # clear_chat_button.click(
        #     clear_chat,
        #     outputs=[chatbot]
        # )

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
    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    .chat-window {
        height: 400px;
        overflow-y: auto;
    }
    """
    with gr.TabItem("Four Independent API Chats"):
        gr.Markdown("# Four Independent API Chat Interfaces")

        with gr.Row():
            with gr.Column():
                preset_prompt = gr.Dropdown(label="Select Preset Prompt", choices=load_preset_prompts(), visible=True)
                user_prompt = gr.Textbox(label="Modify Prompt", lines=3, value=".")
            with gr.Column():
                gr.Markdown("Scroll down for the chat windows...")
        chat_interfaces = []
        for row in range(2):
            with gr.Row():
                for col in range(2):
                    i = row * 2 + col
                    with gr.Column():
                        gr.Markdown(f"### Chat Window {i + 1}")
                        api_endpoint = gr.Dropdown(label=f"API Endpoint {i + 1}",
                                                   choices=["Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq",
                                                            "DeepSeek", "Mistral", "OpenRouter", "Llama.cpp", "Kobold",
                                                            "Ooba",
                                                            "Tabbyapi", "VLLM", "ollama", "HuggingFace"])
                        api_key = gr.Textbox(label=f"API Key {i + 1} (if required)", type="password")
                        temperature = gr.Slider(label=f"Temperature {i + 1}", minimum=0.0, maximum=1.0, step=0.05,
                                                value=0.7)
                        chatbot = gr.Chatbot(height=400, elem_classes="chat-window")
                        msg = gr.Textbox(label=f"Enter your message for Chat {i + 1}")
                        submit = gr.Button(f"Submit to Chat {i + 1}")

                        chat_interfaces.append({
                            'api_endpoint': api_endpoint,
                            'api_key': api_key,
                            'temperature': temperature,
                            'chatbot': chatbot,
                            'msg': msg,
                            'submit': submit,
                            'chat_history': gr.State([])
                        })

        preset_prompt.change(update_user_prompt, inputs=preset_prompt, outputs=user_prompt)

        def chat_wrapper_single(message, chat_history, api_endpoint, api_key, temperature, user_prompt):
            logging.debug(f"Chat Wrapper Single - Message: {message}, Chat History: {chat_history}")
            new_msg, new_history, _ = chat_wrapper(
                message, chat_history, {}, [],  # Empty media_content and selected_parts
                api_endpoint, api_key, user_prompt, None,  # No conversation_id
                False,  # Not saving conversation
                temperature=temperature, system_prompt=""
            )
            chat_history.append((message, new_msg))
            return "", chat_history, chat_history

        for interface in chat_interfaces:
            logging.debug(f"Chat Interface - Clicked Submit for Chat {interface['chatbot']}"),
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


# FIXME - Finish implementing functions + testing/valdidation
def create_chat_management_tab():
    with gr.TabItem("Chat Management"):
        gr.Markdown("# Chat Management")

        with gr.Row():
            search_query = gr.Textbox(label="Search Conversations")
            search_button = gr.Button("Search")

        conversation_list = gr.Dropdown(label="Select Conversation", choices=[])
        conversation_mapping = gr.State({})

        with gr.Tabs():
            with gr.TabItem("Edit"):
                chat_content = gr.TextArea(label="Chat Content (JSON)", lines=20, max_lines=50)
                save_button = gr.Button("Save Changes")
                delete_button = gr.Button("Delete Conversation", variant="stop")

            with gr.TabItem("Preview"):
                chat_preview = gr.HTML(label="Chat Preview")
        result_message = gr.Markdown("")

        def search_conversations(query):
            conversations = search_chat_conversations(query)
            choices = [f"{conv['conversation_name']} (Media: {conv['media_title']}, ID: {conv['id']})" for conv in
                       conversations]
            mapping = {choice: conv['id'] for choice, conv in zip(choices, conversations)}
            return gr.update(choices=choices), mapping

        def load_conversations(selected, conversation_mapping):
            logging.info(f"Selected: {selected}")
            logging.info(f"Conversation mapping: {conversation_mapping}")

            try:
                if selected and selected in conversation_mapping:
                    conversation_id = conversation_mapping[selected]
                    messages = get_chat_messages(conversation_id)
                    conversation_data = {
                        "conversation_id": conversation_id,
                        "messages": messages
                    }
                    json_content = json.dumps(conversation_data, indent=2)

                    # Create HTML preview
                    html_preview = "<div style='max-height: 500px; overflow-y: auto;'>"
                    for msg in messages:
                        sender_style = "background-color: #e6f3ff;" if msg[
                                                                           'sender'] == 'user' else "background-color: #f0f0f0;"
                        html_preview += f"<div style='margin-bottom: 10px; padding: 10px; border-radius: 5px; {sender_style}'>"
                        html_preview += f"<strong>{msg['sender']}:</strong> {html.escape(msg['message'])}<br>"
                        html_preview += f"<small>Timestamp: {msg['timestamp']}</small>"
                        html_preview += "</div>"
                    html_preview += "</div>"

                    logging.info("Returning json_content and html_preview")
                    return json_content, html_preview
                else:
                    logging.warning("No conversation selected or not in mapping")
                    return "", "<p>No conversation selected</p>"
            except Exception as e:
                logging.error(f"Error in load_conversations: {str(e)}")
                return f"Error: {str(e)}", "<p>Error loading conversation</p>"

        def validate_conversation_json(content):
            try:
                data = json.loads(content)
                if not isinstance(data, dict):
                    return False, "Invalid JSON structure: root should be an object"
                if "conversation_id" not in data or not isinstance(data["conversation_id"], int):
                    return False, "Missing or invalid conversation_id"
                if "messages" not in data or not isinstance(data["messages"], list):
                    return False, "Missing or invalid messages array"
                for msg in data["messages"]:
                    if not all(key in msg for key in ["sender", "message"]):
                        return False, "Invalid message structure: missing required fields"
                return True, data
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}"

        def save_conversation(selected, conversation_mapping, content):
            if not selected or selected not in conversation_mapping:
                return "Please select a conversation before saving.", "<p>No changes made</p>"

            conversation_id = conversation_mapping[selected]
            is_valid, result = validate_conversation_json(content)

            if not is_valid:
                return f"Error: {result}", "<p>No changes made due to error</p>"

            conversation_data = result
            if conversation_data["conversation_id"] != conversation_id:
                return "Error: Conversation ID mismatch.", "<p>No changes made due to ID mismatch</p>"

            try:
                with db.get_connection() as conn:
                    conn.execute("BEGIN TRANSACTION")
                    cursor = conn.cursor()

                    # Backup original conversation
                    cursor.execute("SELECT * FROM ChatMessages WHERE conversation_id = ?", (conversation_id,))
                    original_messages = cursor.fetchall()
                    backup_data = json.dumps({"conversation_id": conversation_id, "messages": original_messages})

                    # You might want to save this backup_data somewhere

                    # Delete existing messages
                    cursor.execute("DELETE FROM ChatMessages WHERE conversation_id = ?", (conversation_id,))

                    # Insert updated messages
                    for message in conversation_data["messages"]:
                        cursor.execute('''
                            INSERT INTO ChatMessages (conversation_id, sender, message, timestamp)
                            VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                        ''', (conversation_id, message["sender"], message["message"], message.get("timestamp")))

                    conn.commit()

                    # Create updated HTML preview
                    html_preview = "<div style='max-height: 500px; overflow-y: auto;'>"
                    for msg in conversation_data["messages"]:
                        sender_style = "background-color: #e6f3ff;" if msg[
                                                                           'sender'] == 'user' else "background-color: #f0f0f0;"
                        html_preview += f"<div style='margin-bottom: 10px; padding: 10px; border-radius: 5px; {sender_style}'>"
                        html_preview += f"<strong>{msg['sender']}:</strong> {html.escape(msg['message'])}<br>"
                        html_preview += f"<small>Timestamp: {msg.get('timestamp', 'N/A')}</small>"
                        html_preview += "</div>"
                    html_preview += "</div>"

                    return "Conversation updated successfully.", html_preview
            except sqlite3.Error as e:
                conn.rollback()
                logging.error(f"Database error in save_conversation: {e}")
                return f"Error updating conversation: {str(e)}", "<p>Error occurred while saving</p>"
            except Exception as e:
                conn.rollback()
                logging.error(f"Unexpected error in save_conversation: {e}")
                return f"Unexpected error: {str(e)}", "<p>Unexpected error occurred</p>"

        def delete_conversation(selected, conversation_mapping):
            if not selected or selected not in conversation_mapping:
                return "Please select a conversation before deleting.", "<p>No changes made</p>", gr.update(choices=[])

            conversation_id = conversation_mapping[selected]

            try:
                with db.get_connection() as conn:
                    cursor = conn.cursor()

                    # Delete messages associated with the conversation
                    cursor.execute("DELETE FROM ChatMessages WHERE conversation_id = ?", (conversation_id,))

                    # Delete the conversation itself
                    cursor.execute("DELETE FROM ChatConversations WHERE id = ?", (conversation_id,))

                    conn.commit()

                # Update the conversation list
                remaining_conversations = [choice for choice in conversation_mapping.keys() if choice != selected]
                updated_mapping = {choice: conversation_mapping[choice] for choice in remaining_conversations}

                return "Conversation deleted successfully.", "<p>Conversation deleted</p>", gr.update(choices=remaining_conversations)
            except sqlite3.Error as e:
                conn.rollback()
                logging.error(f"Database error in delete_conversation: {e}")
                return f"Error deleting conversation: {str(e)}", "<p>Error occurred while deleting</p>", gr.update()
            except Exception as e:
                conn.rollback()
                logging.error(f"Unexpected error in delete_conversation: {e}")
                return f"Unexpected error: {str(e)}", "<p>Unexpected error occurred</p>", gr.update()

        def parse_formatted_content(formatted_content):
            lines = formatted_content.split('\n')
            conversation_id = int(lines[0].split(': ')[1])
            timestamp = lines[1].split(': ')[1]
            history = []
            current_role = None
            current_content = None
            for line in lines[3:]:
                if line.startswith("Role: "):
                    if current_role is not None:
                        history.append({"role": current_role, "content": ["", current_content]})
                    current_role = line.split(': ')[1]
                elif line.startswith("Content: "):
                    current_content = line.split(': ', 1)[1]
            if current_role is not None:
                history.append({"role": current_role, "content": ["", current_content]})
            return json.dumps({
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "history": history
            }, indent=2)

        search_button.click(
            search_conversations,
            inputs=[search_query],
            outputs=[conversation_list, conversation_mapping]
        )

        conversation_list.change(
            load_conversations,
            inputs=[conversation_list, conversation_mapping],
            outputs=[chat_content, chat_preview]
        )

        save_button.click(
            save_conversation,
            inputs=[conversation_list, conversation_mapping, chat_content],
            outputs=[result_message, chat_preview]
        )

        delete_button.click(
            delete_conversation,
            inputs=[conversation_list, conversation_mapping],
            outputs=[result_message, chat_preview, conversation_list]
        )

    return search_query, search_button, conversation_list, conversation_mapping, chat_content, save_button, delete_button, result_message, chat_preview


# Load workflows from a JSON file
json_path = Path('./Helper_Scripts/Workflows/Workflows.json')
with json_path.open('r') as f:
    workflows = json.load(f)


# FIXME - broken Completely. Doesn't work.
def chat_workflows_tab():
    with gr.TabItem("Chat Workflows"):
        gr.Markdown("# Workflows using LLMs")

        with gr.Row():
            workflow_selector = gr.Dropdown(label="Select Workflow", choices=[wf['name'] for wf in workflows])
            api_selector = gr.Dropdown(
                label="Select API Endpoint",
                choices=["OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                         "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace"],
                value="OpenAI"
            )
            api_key_input = gr.Textbox(label="API Key (optional)", type="password")

        context_input = gr.Textbox(label="Initial Context (optional)", lines=5)

        # Create a container for dynamic components
        with gr.Column() as dynamic_components:
            prompt_displays = []
            user_inputs = []
            output_boxes = []
            process_buttons = []
            regenerate_buttons = []

            # Create the maximum number of components needed
            max_steps = max(len(wf['prompts']) for wf in workflows)
            for i in range(max_steps):
                prompt_displays.append(gr.Markdown(visible=False))
                user_inputs.append(gr.Textbox(label=f"Your Response", lines=2, visible=False))
                output_boxes.append(gr.Textbox(label=f"AI Output", lines=5, visible=False))
                with gr.Row():
                    process_buttons.append(gr.Button(f"Process Step {i + 1}", visible=False))
                    regenerate_buttons.append(gr.Button(f"🔄 Regenerate", visible=False))

        def update_workflow_ui(workflow_name):
            selected_workflow = next(wf for wf in workflows if wf['name'] == workflow_name)
            num_prompts = len(selected_workflow['prompts'])

            prompt_updates = []
            input_updates = []
            output_updates = []
            button_updates = []
            regenerate_updates = []

            for i in range(max_steps):
                if i < num_prompts:
                    prompt_updates.append(
                        gr.update(value=f"**Step {i + 1}:** {selected_workflow['prompts'][i]}", visible=True))
                    input_updates.append(gr.update(value="", visible=True, interactive=(i == 0)))
                    output_updates.append(gr.update(value="", visible=True))
                    button_updates.append(gr.update(visible=(i == 0)))
                    regenerate_updates.append(gr.update(visible=False))
                else:
                    prompt_updates.append(gr.update(visible=False))
                    input_updates.append(gr.update(visible=False))
                    output_updates.append(gr.update(visible=False))
                    button_updates.append(gr.update(visible=False))
                    regenerate_updates.append(gr.update(visible=False))

            return prompt_updates + input_updates + output_updates + button_updates + regenerate_updates

        def process(context, user_inputs, workflow_name, api_endpoint, api_key, step):
            selected_workflow = next(wf for wf in workflows if wf['name'] == workflow_name)

            # Build up the context from previous steps
            full_context = context + "\n\n"
            for i in range(step + 1):
                full_context += f"Question: {selected_workflow['prompts'][i]}\n"
                full_context += f"Answer: {user_inputs[i]}\n"
                if i < step:
                    full_context += f"AI Output: {output_boxes[i].value}\n\n"

            result = process_with_llm(workflow_name, full_context, selected_workflow['prompts'][step], api_endpoint,
                                      api_key)

            prompt_updates = [gr.update() for _ in range(max_steps)]
            input_updates = []
            output_updates = [gr.update() for _ in range(max_steps)]
            button_updates = []
            regenerate_updates = []

            for i in range(len(selected_workflow['prompts'])):
                if i == step:
                    regenerate_updates.append(gr.update(visible=True))
                elif i == step + 1:
                    input_updates.append(gr.update(interactive=True))
                    button_updates.append(gr.update(visible=True))
                    regenerate_updates.append(gr.update(visible=False))
                elif i > step + 1:
                    input_updates.append(gr.update(interactive=False))
                    button_updates.append(gr.update(visible=False))
                    regenerate_updates.append(gr.update(visible=False))
                else:
                    input_updates.append(gr.update(interactive=False))
                    button_updates.append(gr.update(visible=False))
                    regenerate_updates.append(gr.update(visible=True))

            return [result] + prompt_updates + input_updates + output_updates + button_updates + regenerate_updates

        # Set up event handlers
        workflow_selector.change(
            update_workflow_ui,
            inputs=[workflow_selector],
            outputs=prompt_displays + user_inputs + output_boxes + process_buttons + regenerate_buttons
        )

        # Set up process button click events
        for i, button in enumerate(process_buttons):
            button.click(
                fn=lambda context, *user_inputs, wf_name, api_endpoint, api_key, step=i: process(context, user_inputs,
                                                                                                 wf_name, api_endpoint,
                                                                                                 api_key, step),
                inputs=[context_input] + user_inputs + [workflow_selector, api_selector, api_key_input],
                outputs=[output_boxes[
                             i]] + prompt_displays + user_inputs + output_boxes + process_buttons + regenerate_buttons
            )

        # Set up regenerate button click events
        for i, button in enumerate(regenerate_buttons):
            button.click(
                fn=lambda context, *user_inputs, wf_name, api_endpoint, api_key, step=i: process(context, user_inputs,
                                                                                                 wf_name, api_endpoint,
                                                                                                 api_key, step),
                inputs=[context_input] + user_inputs + [workflow_selector, api_selector, api_key_input],
                outputs=[output_boxes[
                             i]] + prompt_displays + user_inputs + output_boxes + process_buttons + regenerate_buttons
            )

    return workflow_selector, api_selector, api_key_input, context_input, dynamic_components


# Mock function to simulate LLM processing
def process_with_llm(workflow, context, prompt, api_endpoint, api_key):
    api_key_snippet = api_key[:5] + "..." if api_key else "Not provided"
    return f"LLM output using {api_endpoint} (API Key: {api_key_snippet}) for {workflow} with context: {context[:30]}... and prompt: {prompt[:30]}..."

#
# End of Chat_ui.py
#######################################################################################################################