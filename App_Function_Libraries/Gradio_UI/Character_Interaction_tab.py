# Character_Interaction_tab.py
# Description: This file contains the functions that are used for Character Interactions in the Gradio UI.
#
# Imports
import base64
import io
from datetime import datetime as datetime
import logging
import json
import os
from typing import List, Dict, Tuple, Union

#
# External Imports
import gradio as gr
from PIL import Image
#
# Local Imports
from App_Function_Libraries.Chat import chat, load_characters
from App_Function_Libraries.Gradio_UI.Chat_ui import chat_wrapper
from App_Function_Libraries.Gradio_UI.Writing_tab import generate_writing_feedback
#
########################################################################################################################
#
# Single-Character chat Functions:


def chat_with_character(user_message, history, char_data, api_name_input, api_key):
    if char_data is None:
        return history, "Please import a character card first."

    bot_message = generate_writing_feedback(user_message, char_data['name'], "Overall", api_name_input,
                                            api_key)
    history.append((user_message, bot_message))
    return history, ""


def import_character_card(file):
    if file is None:
        logging.warning("No file provided for character card import")
        return None
    try:
        if file.name.lower().endswith(('.png', '.webp')):
            logging.info(f"Attempting to import character card from image: {file.name}")
            json_data = extract_json_from_image(file)
            if json_data:
                logging.info("JSON data extracted from image, attempting to parse")
                card_data = import_character_card_json(json_data)
                if card_data:
                    # Save the image data
                    with Image.open(file) as img:
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        card_data['image'] = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                return card_data
            else:
                logging.warning("No JSON data found in the image")
        else:
            logging.info(f"Attempting to import character card from JSON file: {file.name}")
            content = file.read().decode('utf-8')
            return import_character_card_json(content)
    except Exception as e:
        logging.error(f"Error importing character card: {e}")
    return None


def import_character_card_json(json_content):
    try:
        # Remove any leading/trailing whitespace
        json_content = json_content.strip()

        # Log the first 100 characters of the content
        logging.debug(f"JSON content (first 100 chars): {json_content[:100]}...")

        card_data = json.loads(json_content)
        logging.debug(f"Parsed JSON data keys: {list(card_data.keys())}")
        if 'spec' in card_data and card_data['spec'] == 'chara_card_v2':
            logging.info("Detected V2 character card")
            return card_data['data']
        else:
            logging.info("Assuming V1 character card")
            return card_data
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        logging.error(f"Problematic JSON content: {json_content[:500]}...")
    except Exception as e:
        logging.error(f"Unexpected error parsing JSON: {e}")
    return None


def extract_json_from_image(image_file):
    logging.debug(f"Attempting to extract JSON from image: {image_file.name}")
    try:
        with Image.open(image_file) as img:
            logging.debug("Image opened successfully")
            metadata = img.info
            if 'chara' in metadata:
                logging.debug("Found 'chara' in image metadata")
                chara_content = metadata['chara']
                logging.debug(f"Content of 'chara' metadata (first 100 chars): {chara_content[:100]}...")
                try:
                    decoded_content = base64.b64decode(chara_content).decode('utf-8')
                    logging.debug(f"Decoded content (first 100 chars): {decoded_content[:100]}...")
                    return decoded_content
                except Exception as e:
                    logging.error(f"Error decoding base64 content: {e}")

            logging.debug("'chara' not found in metadata, checking for base64 encoded data")
            raw_data = img.tobytes()
            possible_json = raw_data.split(b'{', 1)[-1].rsplit(b'}', 1)[0]
            if possible_json:
                try:
                    decoded = base64.b64decode(possible_json).decode('utf-8')
                    if decoded.startswith('{') and decoded.endswith('}'):
                        logging.debug("Found and decoded base64 JSON data")
                        return '{' + decoded + '}'
                except Exception as e:
                    logging.error(f"Error decoding base64 data: {e}")

            logging.warning("No JSON data found in the image")
    except Exception as e:
        logging.error(f"Error extracting JSON from image: {e}")
    return None


def load_chat_history(file):
    try:
        content = file.read().decode('utf-8')
        chat_data = json.loads(content)
        return chat_data['history'], chat_data['character']
    except Exception as e:
        logging.error(f"Error loading chat history: {e}")
        return None, None


# FIXME This should be in the chat tab....
def create_character_card_interaction_tab():
    with gr.TabItem("Chat with a Character Card"):
        gr.Markdown("# Chat with a Character Card")
        with gr.Row():
            with gr.Column(scale=1):
                character_image = gr.Image(label="Character Image", type="filepath")
                character_card_upload = gr.File(label="Upload Character Card")
                import_card_button = gr.Button("Import Character Card")
                load_characters_button = gr.Button("Load Existing Characters")
                from App_Function_Libraries.Chat import get_character_names
                character_dropdown = gr.Dropdown(label="Select Character", choices=get_character_names())
                user_name_input = gr.Textbox(label="Your Name", placeholder="Enter your name here")
                api_name_input = gr.Dropdown(
                    choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral",
                             "OpenRouter", "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace",
                             "Custom-OpenAI-API"],
                    value=None,
                    # FIXME - make it so the user cant' click `Send Message` without first setting an API + Chatbot
                    label="API for Interaction(Mandatory)"
                )
                api_key_input = gr.Textbox(label="API Key (if not set in Config_Files/config.txt)",
                                           placeholder="Enter your API key here", type="password")
                temperature_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.05, label="Temperature")
                import_chat_button = gr.Button("Import Chat History")
                chat_file_upload = gr.File(label="Upload Chat History JSON", visible=False)

            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Conversation", height=800)
                user_input = gr.Textbox(label="Your message")
                send_message_button = gr.Button("Send Message")
                regenerate_button = gr.Button("Regenerate Last Message")
                clear_chat_button = gr.Button("Clear Chat")
                save_chat_button = gr.Button("Save This Chat")
                save_status = gr.Textbox(label="Save Status", interactive=False)

    character_data = gr.State(None)
    user_name = gr.State("")

    def import_chat_history(file, current_history, char_data):
        loaded_history, char_name = load_chat_history(file)
        if loaded_history is None:
            return current_history, char_data, "Failed to load chat history."

        # Check if the loaded chat is for the current character
        if char_data and char_data.get('name') != char_name:
            return current_history, char_data, f"Warning: Loaded chat is for character '{char_name}', but current character is '{char_data.get('name')}'. Chat not imported."

        # If no character is selected, try to load the character from the chat
        if not char_data:
            new_char_data = load_character(char_name)[0]
            if new_char_data:
                char_data = new_char_data
            else:
                return current_history, char_data, f"Warning: Character '{char_name}' not found. Please select the character manually."

        return loaded_history, char_data, f"Chat history for '{char_name}' imported successfully."

    def import_character(file):
        card_data = import_character_card(file)
        if card_data:
            from App_Function_Libraries.Chat import save_character
            save_character(card_data)
            return card_data, gr.update(choices=get_character_names())
        else:
            return None, gr.update()

    def load_character(name):
        from App_Function_Libraries.Chat import load_characters
        characters = load_characters()
        char_data = characters.get(name)
        if char_data:
            first_message = char_data.get('first_mes', "Hello! I'm ready to chat.")
            return char_data, [(None, first_message)] if first_message else [], None
        return None, [], None

    def load_character_image(name):
        from App_Function_Libraries.Chat import load_characters
        characters = load_characters()
        char_data = characters.get(name)
        if char_data and 'image_path' in char_data:
            image_path = char_data['image_path']
            if os.path.exists(image_path):
                return image_path
            else:
                logging.warning(f"Image file not found: {image_path}")
        return None

    def load_character_and_image(name):
        char_data, chat_history, _ = load_character(name)
        image_path = load_character_image(name)
        logging.debug(f"Character: {name}")
        logging.debug(f"Character data: {char_data}")
        logging.debug(f"Image path: {image_path}")
        return char_data, chat_history, image_path

    def character_chat_wrapper(message, history, char_data, api_endpoint, api_key, temperature, user_name):
        logging.debug("Entered character_chat_wrapper")
        if char_data is None:
            return "Please select a character first.", history

        if not user_name:
            user_name = "User"

        char_name = char_data.get('name', 'AI Assistant')

        # Prepare the character's background information
        char_background = f"""
        Name: {char_name}
        Description: {char_data.get('description', 'N/A')}
        Personality: {char_data.get('personality', 'N/A')}
        Scenario: {char_data.get('scenario', 'N/A')}
        """

        # Prepare the system prompt for character impersonation
        system_message = f"""You are roleplaying as {char_name}, the character described below. Respond to the user's messages in character, maintaining the personality and background provided. Do not break character or refer to yourself as an AI. Always refer to yourself as "{char_name}" and refer to the user as "{user_name}".

        {char_background}

        Additional instructions: {char_data.get('post_history_instructions', '')}
        """

        # Prepare media_content and selected_parts
        media_content = {
            'id': char_name,
            'title': char_name,
            'content': char_background,
            'description': char_data.get('description', ''),
            'personality': char_data.get('personality', ''),
            'scenario': char_data.get('scenario', '')
        }
        selected_parts = ['description', 'personality', 'scenario']

        prompt = char_data.get('post_history_instructions', '')

        # Prepare the input for the chat function
        if not history:
            full_message = f"{prompt}\n\n{user_name}: {message}" if prompt else f"{user_name}: {message}"
        else:
            full_message = f"{user_name}: {message}"

        # Call the chat function
        bot_message = chat(
            full_message,
            history,
            media_content,
            selected_parts,
            api_endpoint,
            api_key,
            prompt,
            temperature,
            system_message
        )

        # Update history
        history.append((message, bot_message))
        return history

    def save_chat_history(history, character_name):
        # Create the Saved_Chats folder if it doesn't exist
        save_directory = "Saved_Chats"
        os.makedirs(save_directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{character_name}_{timestamp}.json"
        filepath = os.path.join(save_directory, filename)

        chat_data = {
            "character": character_name,
            "timestamp": timestamp,
            "history": history
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
            return filepath
        except Exception as e:
            return f"Error saving chat: {str(e)}"

    def save_current_chat(history, char_data):
        if not char_data or not history:
            return "No chat to save or character not selected."

        character_name = char_data.get('name', 'Unknown')
        result = save_chat_history(history, character_name)
        if result.startswith("Error"):
            return result
        return f"Chat saved successfully as {result}"

    def regenerate_last_message(history, char_data, api_name, api_key, temperature, user_name):
        if not history:
            return history

        last_user_message = history[-1][0]
        new_history = history[:-1]

        return character_chat_wrapper(last_user_message, new_history, char_data, api_name, api_key, temperature,
                                      user_name)

    import_chat_button.click(
        fn=lambda: gr.update(visible=True),
        outputs=chat_file_upload
    )

    chat_file_upload.change(
        fn=import_chat_history,
        inputs=[chat_file_upload, chat_history, character_data],
        outputs=[chat_history, character_data, save_status]
    )

    def update_character_info(name):
        from App_Function_Libraries.Chat import load_characters
        characters = load_characters()
        char_data = characters.get(name)

        image_path = char_data.get('image_path') if char_data else None

        logging.debug(f"Character: {name}")
        logging.debug(f"Character data: {char_data}")
        logging.debug(f"Image path: {image_path}")

        if image_path:
            if os.path.exists(image_path):
                logging.debug(f"Image file exists at {image_path}")
                if os.access(image_path, os.R_OK):
                    logging.debug(f"Image file is readable")
                else:
                    logging.warning(f"Image file is not readable: {image_path}")
                    image_path = None
            else:
                logging.warning(f"Image file does not exist: {image_path}")
                image_path = None
        else:
            logging.warning("No image path provided for the character")

        return char_data, None, image_path  # Return None for chat_history

    def on_character_select(name):
        logging.debug(f"Character selected: {name}")
        return update_character_info_with_error_handling(name)

    def clear_chat_history():
        return [], None  # Return empty list for chat_history and None for character_data

    def update_character_info_with_error_handling(name):
        logging.debug(f"Entering update_character_info_with_error_handling for character: {name}")
        try:
            char_data, _, image_path = update_character_info(name)
            logging.debug(f"Retrieved data: char_data={bool(char_data)}, image_path={image_path}")

            if char_data:
                first_message = char_data.get('first_mes', "Hello! I'm ready to chat.")
                chat_history = [(None, first_message)] if first_message else []
            else:
                chat_history = []

            logging.debug(f"Created chat_history with length: {len(chat_history)}")

            if image_path and os.path.exists(image_path):
                logging.debug(f"Image file exists at {image_path}")
                return char_data, chat_history, image_path
            else:
                logging.warning(f"Image not found or invalid path: {image_path}")
                return char_data, chat_history, None
        except Exception as e:
            logging.error(f"Error updating character info: {str(e)}", exc_info=True)
            return None, [], None
        finally:
            logging.debug("Exiting update_character_info_with_error_handling")

    import_card_button.click(
        fn=import_character,
        inputs=[character_card_upload],
        outputs=[character_data, character_dropdown]
    )

    load_characters_button.click(
        fn=lambda: gr.update(choices=get_character_names()),
        outputs=character_dropdown
    )

    clear_chat_button.click(
        fn=clear_chat_history,
        inputs=[],
        outputs=[chat_history, character_data]
    )

    character_dropdown.change(
        fn=on_character_select,
        inputs=[character_dropdown],
        outputs=[character_data, chat_history, character_image]
    )

    send_message_button.click(
        fn=character_chat_wrapper,
        inputs=[user_input, chat_history, character_data, api_name_input, api_key_input, temperature_slider,
                user_name_input],
        outputs=[chat_history]
    ).then(lambda: "", outputs=user_input)

    regenerate_button.click(
        fn=regenerate_last_message,
        inputs=[chat_history, character_data, api_name_input, api_key_input, temperature_slider, user_name_input],
        outputs=[chat_history]
    )

    user_name_input.change(
        fn=lambda name: name,
        inputs=[user_name_input],
        outputs=[user_name]
    )

    save_chat_button.click(
        fn=save_current_chat,
        inputs=[chat_history, character_data],
        outputs=[save_status]
    )

    return character_data, chat_history, user_input, user_name, character_image


#
# End of Character chat tab
######################################################################################################################
#
# Multi-Character Chat Interface

def character_interaction_setup():
    characters = load_characters()
    return characters, [], None, None


def extract_character_response(response: Union[str, Tuple]) -> str:
    if isinstance(response, tuple):
        # If it's a tuple, try to extract the first string element
        for item in response:
            if isinstance(item, str):
                return item.strip()
        # If no string found, return a default message
        return "I'm not sure how to respond."
    elif isinstance(response, str):
        # If it's already a string, just return it
        return response.strip()
    else:
        # For any other type, return a default message
        return "I'm having trouble forming a response."

# def process_character_response(response: str) -> str:
#     # Remove any leading explanatory text before the first '---'
#     parts = response.split('---')
#     if len(parts) > 1:
#         return '---' + '---'.join(parts[1:])
#     return response.strip()
def process_character_response(response: Union[str, Tuple]) -> str:
    if isinstance(response, tuple):
        response = ' '.join(str(item) for item in response if isinstance(item, str))

    if isinstance(response, str):
        # Remove any leading explanatory text before the first '---'
        parts = response.split('---')
        if len(parts) > 1:
            return '---' + '---'.join(parts[1:])
        return response.strip()
    else:
        return "I'm having trouble forming a response."

def character_turn(characters: Dict, conversation: List[Tuple[str, str]],
                   current_character: str, other_characters: List[str],
                   api_endpoint: str, api_key: str, temperature: float,
                   scenario: str = "") -> Tuple[List[Tuple[str, str]], str]:
    if not current_character:
        return conversation, current_character

    if not conversation and scenario:
        conversation.append(("Scenario", scenario))

    current_char = characters[current_character]
    other_chars = [characters[char] for char in other_characters if char != current_character]

    prompt = f"{current_char['name']}'s personality: {current_char['personality']}\n"
    for char in other_chars:
        prompt += f"{char['name']}'s personality: {char['personality']}\n"
    prompt += "Conversation so far:\n" + "\n".join([f"{sender}: {message}" for sender, message in conversation])
    prompt += f"\n\nHow would {current_char['name']} respond?"

    try:
        response = chat_wrapper(prompt, conversation, {}, [], api_endpoint, api_key, "", None, False, temperature, "")
        processed_response = process_character_response(response)
        conversation.append((current_char['name'], processed_response))
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        conversation.append((current_char['name'], error_message))

    return conversation, current_character


def character_interaction(character1: str, character2: str, api_endpoint: str, api_key: str,
                          num_turns: int, scenario: str, temperature: float,
                          user_interjection: str = "") -> List[str]:
    characters = load_characters()
    char1 = characters[character1]
    char2 = characters[character2]
    conversation = []
    current_speaker = char1
    other_speaker = char2

    # Add scenario to the conversation start
    if scenario:
        conversation.append(f"Scenario: {scenario}")

    for turn in range(num_turns):
        # Construct the prompt for the current speaker
        prompt = f"{current_speaker['name']}'s personality: {current_speaker['personality']}\n"
        prompt += f"{other_speaker['name']}'s personality: {other_speaker['personality']}\n"
        prompt += f"Conversation so far:\n" + "\n".join(
            [msg if isinstance(msg, str) else f"{msg[0]}: {msg[1]}" for msg in conversation])

        # Add user interjection if provided
        if user_interjection and turn == num_turns // 2:
            prompt += f"\n\nUser interjection: {user_interjection}\n"
            conversation.append(f"User: {user_interjection}")

        prompt += f"\n\nHow would {current_speaker['name']} respond?"

        # Get response from the LLM
        response = chat_wrapper(prompt, conversation, {}, [], api_endpoint, api_key, "", None, False, temperature, "")

        # Add the response to the conversation
        conversation.append((current_speaker['name'], response))

        # Switch speakers
        current_speaker, other_speaker = other_speaker, current_speaker

    # Convert the conversation to a list of strings for output
    return [f"{msg[0]}: {msg[1]}" if isinstance(msg, tuple) else msg for msg in conversation]


def create_multiple_character_chat_tab():
    with gr.TabItem("Multi-Character Chat"):
        characters, conversation, current_character, other_character = character_interaction_setup()

        with gr.Blocks() as character_interaction:
            gr.Markdown("# Multi-Character Chat")

            with gr.Row():
                num_characters = gr.Dropdown(label="Number of Characters", choices=["2", "3", "4"], value="2")
                character_selectors = [gr.Dropdown(label=f"Character {i + 1}", choices=list(characters.keys())) for i in
                                       range(4)]

            api_endpoint = gr.Dropdown(label="API Endpoint",
                                       choices=["OpenAI", "Anthropic", "Local-LLM", "Cohere", "Groq", "DeepSeek",
                                                "Mistral", "OpenRouter"])
            api_key = gr.Textbox(label="API Key (if required)", type="password")
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, step=0.1, value=0.7)
            scenario = gr.Textbox(label="Scenario (optional)", lines=3)

            chat_display = gr.Chatbot(label="Character Interaction")

            next_turn_btn = gr.Button("Next Turn")
            narrator_input = gr.Textbox(label="Narrator Input", placeholder="Add a narration or description...")
            add_narration_btn = gr.Button("Add Narration")
            reset_btn = gr.Button("Reset Conversation")

            def update_character_selectors(num):
                return [gr.update(visible=True) if i < int(num) else gr.update(visible=False) for i in range(4)]

            num_characters.change(
                update_character_selectors,
                inputs=[num_characters],
                outputs=character_selectors
            )

            def reset_conversation():
                # Clear chat, reset current_index, other_character, and scenario
                return [], None, None, gr.update(value="")

            def take_turn(characters, conversation, current_index, char_selectors, api_endpoint,
                          api_key, temperature, scenario):
                num_chars = sum(1 for selector in char_selectors if selector)
                if not conversation:
                    conversation = []
                    if scenario:
                        conversation.append(("Scenario", scenario))

                if current_index is None:
                    current_index = 0

                current_character = char_selectors[current_index]
                next_index = (current_index + 1) % num_chars

                new_conversation, _ = character_turn(
                    characters, conversation, current_character, char_selectors,
                    api_endpoint, api_key, temperature, scenario
                )

                return new_conversation, next_index

            def add_narration(narration, conversation):
                if narration:
                    conversation.append(("Narrator", narration))
                return conversation, ""

            next_turn_btn.click(
                take_turn,
                inputs=[gr.State(characters), gr.State(conversation), gr.State(None),
                        character_selectors[:4], api_endpoint, api_key, temperature, scenario],
                outputs=[chat_display, gr.State(None)]
            )

            add_narration_btn.click(
                add_narration,
                inputs=[narrator_input, chat_display],
                outputs=[chat_display, narrator_input]
            )

            reset_btn.click(
                reset_conversation,
                outputs=[chat_display, gr.State(None), gr.State(None), scenario]
            )

        return character_interaction
