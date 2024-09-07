# Writing_tab.py
# Description: This file contains the functions that are used for writing in the Gradio UI.
#
# Imports
import base64
from datetime import datetime as datetime
import logging
import json
import os
#
# External Imports
import gradio as gr
from PIL import Image
import textstat
#
# Local Imports
from App_Function_Libraries.Summarization_General_Lib import perform_summarization
from App_Function_Libraries.Chat import chat
#
########################################################################################################################
#
# Functions:

def adjust_tone(text, concise, casual, api_name, api_key):
    tones = [
        {"tone": "concise", "weight": concise},
        {"tone": "casual", "weight": casual},
        {"tone": "professional", "weight": 1 - casual},
        {"tone": "expanded", "weight": 1 - concise}
    ]
    tones = sorted(tones, key=lambda x: x['weight'], reverse=True)[:2]

    tone_prompt = " and ".join([f"{t['tone']} (weight: {t['weight']:.2f})" for t in tones])

    prompt = f"Rewrite the following text to match these tones: {tone_prompt}. Text: {text}"
    # Performing tone adjustment request...
    adjusted_text = perform_summarization(api_name, text, prompt, api_key)

    return adjusted_text


def grammar_style_check(input_text, custom_prompt, api_name, api_key, system_prompt):
    default_prompt = "Please analyze the following text for grammar and style. Offer suggestions for improvement and point out any misused words or incorrect spellings:\n\n"
    full_prompt = custom_prompt if custom_prompt else default_prompt
    full_text = full_prompt + input_text

    return perform_summarization(api_name, full_text, custom_prompt, api_key, system_prompt)


def create_grammar_style_check_tab():
    with gr.TabItem("Grammar and Style Check"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Grammar and Style Check")
                gr.Markdown("This utility checks the grammar and style of the provided text by feeding it to an LLM and returning suggestions for improvement.")
                input_text = gr.Textbox(label="Input Text", lines=10)
                custom_prompt_checkbox = gr.Checkbox(label="Use Custom Prompt", value=False, visible=True)
                system_prompt_input = gr.Textbox(label="System Prompt", placeholder="Please analyze the provided text for grammar and style. Offer any suggestions or points to improve you can identify. Additionally please point out any misuses of any words or incorrect spellings.", lines=5, visible=False)
                custom_prompt_input = gr.Textbox(label="user Prompt",
                                                     value="""<s>You are a bulleted notes specialist. [INST]```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.[/INST]
**Bulleted Note Creation Guidelines**

**Headings**:
- Based on referenced topics, not categories like quotes or terms
- Surrounded by **bold** formatting 
- Not listed as bullet points
- No space between headings and list items underneath

**Emphasis**:
- **Important terms** set in bold font
- **Text ending in a colon**: also bolded

**Review**:
- Ensure adherence to specified format
- Do not reference these instructions in your response.</s>[INST] {{ .Prompt }} [/INST]
""",
                                                     lines=3,
                                                     visible=False)
                custom_prompt_checkbox.change(
                    fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
                    inputs=[custom_prompt_checkbox],
                    outputs=[custom_prompt_input, system_prompt_input]
                )
                api_name_input = gr.Dropdown(
                    choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                             "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM","ollama", "HuggingFace", "Custom-OpenAI-API"],
                    value=None,
                    label="API for Grammar Check"
                )
                api_key_input = gr.Textbox(label="API Key (if not set in config.txt)", placeholder="Enter your API key here",
                                               type="password")
                check_grammar_button = gr.Button("Check Grammar and Style")

            with gr.Column():
                gr.Markdown("# Resulting Suggestions")
                gr.Markdown("(Keep in mind the API used can affect the quality of the suggestions)")

                output_text = gr.Textbox(label="Grammar and Style Suggestions", lines=15)

            check_grammar_button.click(
                fn=grammar_style_check,
                inputs=[input_text, custom_prompt_input, api_name_input, api_key_input, system_prompt_input],
                outputs=output_text
            )


def create_tone_adjustment_tab():
    with gr.TabItem("Tone Analyzer & Editor"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Input Text", lines=10)
                concise_slider = gr.Slider(minimum=0, maximum=1, value=0.5, label="Concise vs Expanded")
                casual_slider = gr.Slider(minimum=0, maximum=1, value=0.5, label="Casual vs Professional")
                api_name_input = gr.Dropdown(
                    choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                             "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM","ollama", "HuggingFace", "Custom-OpenAI-API"],
                    value=None,
                    label="API for Grammar Check"
                )
                api_key_input = gr.Textbox(label="API Key (if not set in config.txt)", placeholder="Enter your API key here",
                                               type="password")
                adjust_btn = gr.Button("Adjust Tone")

            with gr.Column():
                output_text = gr.Textbox(label="Adjusted Text", lines=15)

                adjust_btn.click(
                    adjust_tone,
                    inputs=[input_text, concise_slider, casual_slider],
                    outputs=output_text
                )


persona_prompts = {
    "Hemingway": "As Ernest Hemingway, known for concise and straightforward prose, provide feedback on the following text:",
    "Shakespeare": "Channel William Shakespeare's poetic style and provide feedback on the following text:",
    "Jane Austen": "Embodying Jane Austen's wit and social commentary, critique the following text:",
    "Stephen King": "With Stephen King's flair for suspense and horror, analyze the following text:",
    "J.K. Rowling": "As J.K. Rowling, creator of the magical world of Harry Potter, review the following text:"
}

def generate_writing_feedback(text, persona, aspect, api_name, api_key):
    if isinstance(persona, dict):  # If it's a character card
        base_prompt = f"You are {persona['name']}. {persona['personality']}\n\nScenario: {persona['scenario']}\n\nRespond to the following message in character:"
    else:  # If it's a regular persona
        base_prompt = persona_prompts.get(persona, f"As {persona}, provide feedback on the following text:")

    if aspect != "Overall":
        prompt = f"{base_prompt}\n\nFocus specifically on the {aspect.lower()} in the following text:\n\n{text}"
    else:
        prompt = f"{base_prompt}\n\n{text}"

    return perform_summarization(api_name, text, prompt, api_key, system_message="You are a helpful AI assistant. You will respond to the user as if you were the persona declared in the user prompt.")

def generate_writing_prompt(persona, api_name, api_key):
    prompt = f"Generate a writing prompt in the style of {persona}. The prompt should inspire a short story or scene that reflects {persona}'s typical themes and writing style."
    #FIXME
    return perform_summarization(api_name, prompt, "", api_key, system_message="You are a helpful AI assistant. You will respond to the user as if you were the persona declared in the user prompt." )

def calculate_readability(text):
    ease = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)
    return f"Readability: Flesch Reading Ease: {ease:.2f}, Flesch-Kincaid Grade Level: {grade:.2f}"


def generate_feedback_history_html(history):
    html = "<h3>Recent Feedback History</h3>"
    for entry in reversed(history):
        html += f"<details><summary>{entry['persona']} Feedback</summary>"
        html += f"<p><strong>Original Text:</strong> {entry['text'][:100]}...</p>"

        feedback = entry.get('feedback')
        if feedback:
            html += f"<p><strong>Feedback:</strong> {feedback[:200]}...</p>"
        else:
            html += "<p><strong>Feedback:</strong> No feedback provided.</p>"

        html += "</details>"
    return html


# FIXME
def create_document_feedback_tab():
    with gr.TabItem("Writing Feedback"):
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(label="Your Writing", lines=10)
                persona_dropdown = gr.Dropdown(
                    label="Select Persona",
                    choices=[
                        "Agatha Christie",
                        "Arthur Conan Doyle",
                        "Charles Bukowski",
                        "Charles Dickens",
                        "Chinua Achebe",
                        "Cormac McCarthy",
                        "David Foster Wallace",
                        "Edgar Allan Poe",
                        "F. Scott Fitzgerald",
                        "Flannery O'Connor",
                        "Franz Kafka",
                        "Fyodor Dostoevsky",
                        "Gabriel Garcia Marquez",
                        "George R.R. Martin",
                        "George Orwell",
                        "Haruki Murakami",
                        "Hemingway",
                        "Herman Melville",
                        "Isabel Allende",
                        "James Joyce",
                        "Jane Austen",
                        "J.K. Rowling",
                        "J.R.R. Tolkien",
                        "Jorge Luis Borges",
                        "Kurt Vonnegut",
                        "Leo Tolstoy",
                        "Margaret Atwood",
                        "Mark Twain",
                        "Mary Shelley",
                        "Milan Kundera",
                        "Naguib Mahfouz",
                        "Neil Gaiman",
                        "Octavia Butler",
                        "Philip K Dick",
                        "Ray Bradbury",
                        "Salman Rushdie",
                        "Shakespeare",
                        "Stephen King",
                        "Toni Morrison",
                        "T.S. Eliot",
                        "Ursula K. Le Guin",
                        "Virginia Woolf",
                        "Virginia Woolf",
                        "Zadie Smith"],
                    value="Hemingway"
                )
                custom_persona_name = gr.Textbox(label="Custom Persona Name")
                custom_persona_description = gr.Textbox(label="Custom Persona Description", lines=3)
                add_custom_persona_button = gr.Button("Add Custom Persona")
                aspect_dropdown = gr.Dropdown(
                    label="Focus Feedback On",
                    choices=["Overall", "Grammar", "Word choice", "Structure of delivery", "Character Development", "Character Dialogue", "Descriptive Language", "Plot Structure"],
                    value="Overall"
                )
                api_name_input = gr.Dropdown(
                    choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                             "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace", "Custom-OpenAI-API"],
                    value=None,
                    label="API for Feedback"
                )
                api_key_input = gr.Textbox(label="API Key (if not set in config.txt)", type="password")
                get_feedback_button = gr.Button("Get Feedback")
                generate_prompt_button = gr.Button("Generate Writing Prompt")

            with gr.Column(scale=2):
                feedback_output = gr.Textbox(label="Feedback", lines=15)
                readability_output = gr.Textbox(label="Readability Metrics")
                feedback_history_display = gr.HTML(label="Feedback History")

        with gr.Row():
            compare_personas = gr.CheckboxGroup(
                choices=[
                    "Agatha Christie",
                    "Arthur Conan Doyle",
                    "Charles Bukowski",
                    "Charles Dickens",
                    "Chinua Achebe",
                    "Cormac McCarthy",
                    "David Foster Wallace",
                    "Edgar Allan Poe",
                    "F. Scott Fitzgerald",
                    "Flannery O'Connor",
                    "Franz Kafka",
                    "Fyodor Dostoevsky",
                    "Gabriel Garcia Marquez",
                    "George R.R. Martin",
                    "George Orwell",
                    "Haruki Murakami",
                    "Hemingway",
                    "Herman Melville",
                    "Isabel Allende",
                    "James Joyce",
                    "Jane Austen",
                    "J.K. Rowling",
                    "J.R.R. Tolkien",
                    "Jorge Luis Borges",
                    "Kurt Vonnegut",
                    "Leo Tolstoy",
                    "Margaret Atwood",
                    "Mark Twain",
                    "Mary Shelley",
                    "Milan Kundera",
                    "Naguib Mahfouz",
                    "Neil Gaiman",
                    "Octavia Butler",
                    "Philip K Dick",
                    "Ray Bradbury",
                    "Salman Rushdie",
                    "Shakespeare",
                    "Stephen King",
                    "Toni Morrison",
                    "T.S. Eliot",
                    "Ursula K. Le Guin",
                    "Virginia Woolf",
                    "Virginia Woolf",
                    "Zadie Smith"],
                label="Compare Multiple Persona's Feedback at Once(Compares existing feedback, doesn't create new ones)"
            )
        with gr.Row():
            compare_button = gr.Button("Compare Feedback")

    feedback_history = gr.State([])

    def add_custom_persona(name, description):
        updated_choices = persona_dropdown.choices + [name]
        persona_prompts[name] = f"As {name}, {description}, provide feedback on the following text:"
        return gr.update(choices=updated_choices)

    def update_feedback_history(current_text, persona, feedback):
        # Ensure feedback_history.value is initialized and is a list
        if feedback_history.value is None:
            feedback_history.value = []

        history = feedback_history.value

        # Append the new entry to the history
        history.append({"text": current_text, "persona": persona, "feedback": feedback})

        # Keep only the last 5 entries in the history
        feedback_history.value = history[-10:]

        # Generate and return the updated HTML
        return generate_feedback_history_html(feedback_history.value)

    def compare_feedback(text, selected_personas, api_name, api_key):
        results = []
        for persona in selected_personas:
            feedback = generate_writing_feedback(text, persona, "Overall", api_name, api_key)
            results.append(f"### {persona}'s Feedback:\n{feedback}\n\n")
        return "\n".join(results)

    add_custom_persona_button.click(
        fn=add_custom_persona,
        inputs=[custom_persona_name, custom_persona_description],
        outputs=persona_dropdown
    )

    get_feedback_button.click(
        fn=lambda text, persona, aspect, api_name, api_key: (
            generate_writing_feedback(text, persona, aspect, api_name, api_key),
            calculate_readability(text),
            update_feedback_history(text, persona, generate_writing_feedback(text, persona, aspect, api_name, api_key))
        ),
        inputs=[input_text, persona_dropdown, aspect_dropdown, api_name_input, api_key_input],
        outputs=[feedback_output, readability_output, feedback_history_display]
    )

    compare_button.click(
        fn=compare_feedback,
        inputs=[input_text, compare_personas, api_name_input, api_key_input],
        outputs=feedback_output
    )

    generate_prompt_button.click(
        fn=generate_writing_prompt,
        inputs=[persona_dropdown, api_name_input, api_key_input],
        outputs=input_text
    )

    return input_text, feedback_output, readability_output, feedback_history_display


def create_creative_writing_tab():
    with gr.TabItem("Creative Writing Assistant"):
        gr.Markdown("# Utility to be added...")


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
                return import_character_card_json(json_data)
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
                character_card_upload = gr.File(label="Upload Character Card")
                import_card_button = gr.Button("Import Character Card")
                load_characters_button = gr.Button("Load Existing Characters")
                from App_Function_Libraries.Chat import get_character_names
                character_dropdown = gr.Dropdown(label="Select Character", choices=get_character_names())
                api_name_input = gr.Dropdown(
                    choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral",
                             "OpenRouter", "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace", "Custom-OpenAI-API"],
                    value=None,
                    # FIXME - make it so the user cant' click `Send Message` without first setting an API + Chatbot
                    label="API for Interaction(Mandatory)"
                )
                api_key_input = gr.Textbox(label="API Key (if not set in config.txt)",
                                           placeholder="Enter your API key here", type="password")
                temperature_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.05, label="Temperature")
                import_chat_button = gr.Button("Import Chat History")
                chat_file_upload = gr.File(label="Upload Chat History JSON", visible=False)


            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Conversation")
                user_input = gr.Textbox(label="Your message")
                send_message_button = gr.Button("Send Message")
                regenerate_button = gr.Button("Regenerate Last Message")
                save_chat_button = gr.Button("Save This Chat")
                save_status = gr.Textbox(label="Save Status", interactive=False)

    character_data = gr.State(None)

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
            return char_data, [(None, first_message)] if first_message else []
        return None, []

    def character_chat_wrapper(message, history, char_data, api_endpoint, api_key, temperature):
        logging.debug("Entered character_chat_wrapper")
        if char_data is None:
            return "Please select a character first.", history

        # Prepare the character's background information
        char_background = f"""
        Name: {char_data.get('name', 'Unknown')}
        Description: {char_data.get('description', 'N/A')}
        Personality: {char_data.get('personality', 'N/A')}
        Scenario: {char_data.get('scenario', 'N/A')}
        """

        # Prepare the system prompt for character impersonation
        system_message = f"""You are roleplaying as the character described below. Respond to the user's messages in character, maintaining the personality and background provided. Do not break character or refer to yourself as an AI.

        {char_background}

        Additional instructions: {char_data.get('post_history_instructions', '')}
        """

        # Prepare media_content and selected_parts
        media_content = {
            'id': char_data.get('name'),
            'title': char_data.get('name', 'Unknown Character'),
            'content': char_background,
            'description': char_data.get('description', ''),
            'personality': char_data.get('personality', ''),
            'scenario': char_data.get('scenario', '')
        }
        selected_parts = ['description', 'personality', 'scenario']

        prompt = char_data.get('post_history_instructions', '')

        # Prepare the input for the chat function
        if not history:
            full_message = f"{prompt}\n\n{message}" if prompt else message
        else:
            full_message = message

        # Call the chat function
        bot_message = chat(
            message,
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

    def regenerate_last_message(history, char_data, api_name, api_key, temperature):
        if not history:
            return history

        last_user_message = history[-1][0]
        new_history = history[:-1]

        return character_chat_wrapper(last_user_message, new_history, char_data, api_name, api_key, temperature)

    import_chat_button.click(
        fn=lambda: gr.update(visible=True),
        outputs=chat_file_upload
    )

    chat_file_upload.change(
        fn=import_chat_history,
        inputs=[chat_file_upload, chat_history, character_data],
        outputs=[chat_history, character_data, save_status]
    )

    import_card_button.click(
        fn=import_character,
        inputs=[character_card_upload],
        outputs=[character_data, character_dropdown]
    )

    load_characters_button.click(
        fn=lambda: gr.update(choices=get_character_names()),
        outputs=character_dropdown
    )

    character_dropdown.change(
        fn=load_character,
        inputs=[character_dropdown],
        outputs=[character_data, chat_history]
    )

    send_message_button.click(
        fn=character_chat_wrapper,
        inputs=[user_input, chat_history, character_data, api_name_input, api_key_input, temperature_slider],
        outputs=[chat_history]
    ).then(lambda: "", outputs=user_input)

    regenerate_button.click(
        fn=regenerate_last_message,
        inputs=[chat_history, character_data, api_name_input, api_key_input, temperature_slider],
        outputs=[chat_history]
    )

    save_chat_button.click(
        fn=save_current_chat,
        inputs=[chat_history, character_data],
        outputs=[save_status]
    )

    return character_data, chat_history, user_input


def create_mikupad_tab():
    with gr.TabItem("Mikupad"):
        gr.Markdown("I Wish. Gradio won't embed it successfully...")

#
# End of Writing_tab.py
########################################################################################################################
