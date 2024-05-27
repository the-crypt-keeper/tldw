# Web-UI-Lib.py
#########################################
# Web-based UI library
# This library is used to handle creating a GUI using Gradio for a web-based interface.
#
####



####################
# Function List
#
# 1. summarize_with_huggingface(api_key, file_path, custom_prompt_arg)
# 2. format_transcription(transcription_result)
# 3. format_file_path(file_path, fallback_path=None)
# 4. search_media(query, fields, keyword, page)
# 5. ask_question(transcription, question, api_name, api_key)
# 6. launch_ui(demo_mode=False)
#
####################

# Import necessary libraries
import json
import logging
import requests
import sys
import os
# Import 3rd-pary Libraries
import gradio as gr



#######################################################################################################################
# Function Definitions
#

# Only to be used when configured with Gradio for HF Space
def summarize_with_huggingface(api_key, file_path, custom_prompt_arg):
    logging.debug(f"huggingface: Summarization process starting...")
    try:
        logging.debug("huggingface: Loading json data for summarization")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("huggingface: Extracting text from the segments")
        logging.debug(f"huggingface: Segments: {segments}")
        text = ' '.join([segment['text'] for segment in segments])

        print(f"huggingface: lets make sure the HF api key exists...\n\t {api_key}")
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        model = "microsoft/Phi-3-mini-128k-instruct"
        API_URL = f"https://api-inference.huggingface.co/models/{model}"

        huggingface_prompt = f"{text}\n\n\n\n{custom_prompt_arg}"
        logging.debug("huggingface: Prompt being sent is {huggingface_prompt}")
        data = {
            "inputs": text,
            "parameters": {"max_length": 512, "min_length": 100}  # You can adjust max_length and min_length as needed
        }

        print(f"huggingface: lets make sure the HF api key is the same..\n\t {huggingface_api_key}")

        logging.debug("huggingface: Submitting request...")

        response = requests.post(API_URL, headers=headers, json=data)

        if response.status_code == 200:
            summary = response.json()[0]['summary_text']
            logging.debug("huggingface: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.error(f"huggingface: Summarization failed with status code {response.status_code}: {response.text}")
            return f"Failed to process summary, status code {response.status_code}: {response.text}"
    except Exception as e:
        logging.error("huggingface: Error in processing: %s", str(e))
        print(f"Error occurred while processing summary with huggingface: {str(e)}")
        return None

    # FIXME
    # This is here for gradio authentication
    # Its just not setup.
    # def same_auth(username, password):
    #    return username == password


def format_transcription(transcription_result):
    if transcription_result:
        json_data = transcription_result['transcription']
        return json.dumps(json_data, indent=2)
    else:
        return ""


def format_file_path(file_path, fallback_path=None):
    if file_path and os.path.exists(file_path):
        logging.debug(f"File exists: {file_path}")
        return file_path
    elif fallback_path and os.path.exists(fallback_path):
        logging.debug(f"File does not exist: {file_path}. Returning fallback path: {fallback_path}")
        return fallback_path
    else:
        logging.debug(f"File does not exist: {file_path}. No fallback path available.")
        return None


def search_media(query, fields, keyword, page):
    try:
        results = search_and_display(query, fields, keyword, page)
        return results
    except Exception as e:
        logger.error(f"Error searching media: {e}")
        return str(e)


# FIXME - Change to use 'check_api()' function - also, create 'check_api()' function
def ask_question(transcription, question, api_name, api_key):
    if not question.strip():
        return "Please enter a question."

        prompt = f"""Transcription:\n{transcription}

        Given the above transcription, please answer the following:\n\n{question}"""

        # FIXME - Refactor main API checks so they're their own function - api_check()
        # Call api_check() function here

        if api_name.lower() == "openai":
            openai_api_key = api_key if api_key else config.get('API', 'openai_api_key', fallback=None)
            headers = {
                'Authorization': f'Bearer {openai_api_key}',
                'Content-Type': 'application/json'
            }
            if openai_model:
                pass
            else:
                openai_model = 'gpt-4-turbo'
            data = {
                "model": openai_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the given "
                                   "transcription and summary."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 150000,
                "temperature": 0.1
            }
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content'].strip()
            return answer
        else:
            return "Failed to process the question."
    else:
        return "Question answering is currently only supported with the OpenAI API."




# def gradio UI
def launch_ui(demo_mode=False):
    whisper_models = ["small.en", "medium.en", "large"]

    with gr.Blocks() as iface:
        # Tab 1: Audio Transcription + Summarization
        with gr.Tab("Audio Transcription + Summarization"):

            with gr.Row():
                # Light/Dark mode toggle switch
                theme_toggle = gr.Radio(choices=["Light", "Dark"], value="Light",
                                        label="Light/Dark Mode Toggle (Toggle to change UI color scheme)")

                # UI Mode toggle switch
                ui_mode_toggle = gr.Radio(choices=["Simple", "Advanced"], value="Simple",
                                          label="UI Mode (Toggle to show all options)")

            # URL input is always visible
            url_input = gr.Textbox(label="URL (Mandatory)", placeholder="Enter the video URL here")

            # Inputs to be shown or hidden
            num_speakers_input = gr.Number(value=2, label="Number of Speakers(Optional - Currently has no effect)",
                                           visible=False)
            whisper_model_input = gr.Dropdown(choices=whisper_models, value="small.en",
                                              label="Whisper Model(This is the ML model used for transcription.)",
                                              visible=False)
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt (Customize your summarization, or ask a question about the video and have it "
                      "answered)",
                placeholder="Above is the transcript of a video. Please read "
                            "through the transcript carefully. Identify the main topics that are discussed over the "
                            "course of the transcript. Then, summarize the key points about each main topic in a "
                            "concise bullet point. The bullet points should cover the key information conveyed about "
                            "each topic in the video, but should be much shorter than the full transcript. Please "
                            "output your bullet point summary inside <bulletpoints> tags.",
                lines=3, visible=True)
            offset_input = gr.Number(value=0, label="Offset (Seconds into the video to start transcribing at)",
                                     visible=False)
            api_name_input = gr.Dropdown(
                choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "Llama.cpp", "Kobold", "Ooba", "HuggingFace"],
                value=None,
                label="API Name (Mandatory Unless you just want a Transcription)", visible=True)
            api_key_input = gr.Textbox(label="API Key (Mandatory unless you're running a local model/server/no API selected)",
                                       placeholder="Enter your API key here; Ignore if using Local API or Built-in API('Local-LLM')",
                                       visible=True)
            vad_filter_input = gr.Checkbox(label="VAD Filter (WIP)", value=False,
                                           visible=False)
            rolling_summarization_input = gr.Checkbox(label="Enable Rolling Summarization", value=False,
                                                      visible=False)
            download_video_input = gr.components.Checkbox(label="Download Video(Select to allow for file download of "
                                                                "selected video)", value=False, visible=False)
            download_audio_input = gr.components.Checkbox(label="Download Audio(Select to allow for file download of "
                                                                "selected Video's Audio)", value=False, visible=False)
            detail_level_input = gr.Slider(minimum=0.01, maximum=1.0, value=0.01, step=0.01, interactive=True,
                                           label="Summary Detail Level (Slide me) (Only OpenAI currently supported)",
                                           visible=False)
            keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated Example: "
                                                                      "tag_one,tag_two,tag_three)",
                                        value="default,no_keyword_set",
                                        visible=True)
            question_box_input = gr.Textbox(label="Question",
                                            placeholder="Enter a question to ask about the transcription",
                                            visible=False)
            chunk_summarization_input = gr.Checkbox(label="Time-based Chunk Summarization",
                                                    value=False,
                                                    visible=False)
            chunk_duration_input = gr.Number(label="Chunk Duration (seconds)", value=DEFAULT_CHUNK_DURATION,
                                             visible=False)
            words_per_second_input = gr.Number(label="Words per Second", value=WORDS_PER_SECOND,
                                               visible=False)
            # time_based_summarization_input = gr.Checkbox(label="Enable Time-based Summarization", value=False,
            # visible=False) time_chunk_duration_input = gr.Number(label="Time Chunk Duration (seconds)", value=60,
            # visible=False) llm_model_input = gr.Dropdown(label="LLM Model", choices=["gpt-4o", "gpt-4-turbo",
            # "claude-3-sonnet-20240229", "command-r-plus", "CohereForAI/c4ai-command-r-plus", "llama3-70b-8192"],
            # value="gpt-4o", visible=False)

            inputs = [
                num_speakers_input, whisper_model_input, custom_prompt_input, offset_input, api_name_input,
                api_key_input, vad_filter_input, download_video_input, download_audio_input,
                rolling_summarization_input, detail_level_input, question_box_input, keywords_input,
                chunk_summarization_input, chunk_duration_input, words_per_second_input
            ]
            # inputs_1 = [
            #     url_input_1,
            #     num_speakers_input, whisper_model_input, custom_prompt_input_1, offset_input, api_name_input_1,
            #     api_key_input_1, vad_filter_input, download_video_input, download_audio_input,
            #     rolling_summarization_input, detail_level_input, question_box_input, keywords_input_1,
            #     chunk_summarization_input, chunk_duration_input, words_per_second_input,
            #     time_based_summarization_input, time_chunk_duration_input, llm_model_input
            # ]

            outputs = [
                gr.Textbox(label="Transcription (Resulting Transcription from your input URL)"),
                gr.Textbox(label="Summary or Status Message (Current status of Summary or Summary itself)"),
                gr.File(label="Download Transcription as JSON (Download the Transcription as a file)"),
                gr.File(label="Download Summary as Text (Download the Summary as a file)"),
                gr.File(label="Download Video (Download the Video as a file)", visible=False),
                gr.File(label="Download Audio (Download the Audio as a file)", visible=False),
            ]

            def toggle_light(mode):
                if mode == "Dark":
                    return """
                    <style>
                        body {
                            background-color: #1c1c1c;
                            color: #ffffff;
                        }
                        .gradio-container {
                            background-color: #1c1c1c;
                            color: #ffffff;
                        }
                        .gradio-button {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-input {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-dropdown {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-slider {
                            background-color: #4c4c4c;
                        }
                        .gradio-checkbox {
                            background-color: #4c4c4c;
                        }
                        .gradio-radio {
                            background-color: #4c4c4c;
                        }
                        .gradio-textbox {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-label {
                            color: #ffffff;
                        }
                    </style>
                    """
                else:
                    return """
                    <style>
                        body {
                            background-color: #ffffff;
                            color: #000000;
                        }
                        .gradio-container {
                            background-color: #ffffff;
                            color: #000000;
                        }
                        .gradio-button {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-input {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-dropdown {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-slider {
                            background-color: #f0f0f0;
                        }
                        .gradio-checkbox {
                            background-color: #f0f0f0;
                        }
                        .gradio-radio {
                            background-color: #f0f0f0;
                        }
                        .gradio-textbox {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-label {
                            color: #000000;
                        }
                    </style>
                    """

            # Set the event listener for the Light/Dark mode toggle switch
            theme_toggle.change(fn=toggle_light, inputs=theme_toggle, outputs=gr.HTML())

            # Function to toggle visibility of advanced inputs
            def toggle_ui(mode):
                visible = (mode == "Advanced")
                return [
                    gr.update(visible=True) if i in [0, 3, 5, 6, 13] else gr.update(visible=visible)
                    for i in range(len(inputs))
                ]

            # Set the event listener for the UI Mode toggle switch
            ui_mode_toggle.change(fn=toggle_ui, inputs=ui_mode_toggle, outputs=inputs)

            # Combine URL input and inputs lists
            all_inputs = [url_input] + inputs

            gr.Interface(
                fn=process_url,
                inputs=all_inputs,
                outputs=outputs,
                title="Video Transcription and Summarization",
                description="Submit a video URL for transcription and summarization. Ensure you input all necessary "
                            "information including API keys."
            )

        # Tab 2: Scrape & Summarize Articles/Websites
        with gr.Tab("Scrape & Summarize Articles/Websites"):
            url_input = gr.Textbox(label="Article URL", placeholder="Enter the article URL here")
            custom_article_title_input = gr.Textbox(label="Custom Article Title (Optional)",
                                                    placeholder="Enter a custom title for the article")
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt (Optional)",
                placeholder="Provide a custom prompt for summarization",
                lines=3
            )
            api_name_input = gr.Dropdown(
                choices=[None, "huggingface", "openai", "anthropic", "cohere", "groq", "llama", "kobold", "ooba"],
                value=None,
                label="API Name (Mandatory for Summarization)"
            )
            api_key_input = gr.Textbox(label="API Key (Mandatory if API Name is specified)",
                                       placeholder="Enter your API key here; Ignore if using Local API or Built-in API")
            keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated)",
                                        value="default,no_keyword_set", visible=True)

            scrape_button = gr.Button("Scrape and Summarize")
            result_output = gr.Textbox(label="Result")

            scrape_button.click(scrape_and_summarize, inputs=[url_input, custom_prompt_input, api_name_input,
                                                              api_key_input, keywords_input,
                                                              custom_article_title_input], outputs=result_output)

            gr.Markdown("### Or Paste Unstructured Text Below (Will use settings from above)")
            text_input = gr.Textbox(label="Unstructured Text", placeholder="Paste unstructured text here", lines=10)
            text_ingest_button = gr.Button("Ingest Unstructured Text")
            text_ingest_result = gr.Textbox(label="Result")

            text_ingest_button.click(ingest_unstructured_text,
                                     inputs=[text_input, custom_prompt_input, api_name_input, api_key_input,
                                             keywords_input, custom_article_title_input], outputs=text_ingest_result)

        with gr.Tab("Ingest & Summarize Documents"):
            gr.Markdown("Plan to put ingestion form for documents here")
            gr.Markdown("Will ingest documents and store into SQLite DB")
            gr.Markdown("RAG here we come....:/")

        with gr.Tab("Sample Prompts/Questions"):
            gr.Markdown("Plan to put Sample prompts/questions here")
            gr.Markdown("Fabric prompts/live UI?")
            # Searchable list
            with gr.Row():
                search_box = gr.Textbox(label="Search prompts", placeholder="Type to filter prompts")
                search_result = gr.Textbox(label="Matching prompts", interactive=False)
                search_box.change(search_prompts, inputs=search_box, outputs=search_result)

            # Interactive list
            with gr.Row():
                prompt_selector = gr.Radio(choices=all_prompts, label="Select a prompt")
                selected_output = gr.Textbox(label="Selected prompt")
                prompt_selector.change(handle_prompt_selection, inputs=prompt_selector, outputs=selected_output)

            # Categorized display
            with gr.Accordion("Category 1"):
                gr.Markdown("\n".join(prompts_category_1))
            with gr.Accordion("Category 2"):
                gr.Markdown("\n".join(prompts_category_2))

    with gr.Blocks() as search_interface:
        with gr.Tab("Search & Detailed View"):
            search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
            search_fields_input = gr.CheckboxGroup(label="Search Fields",
                                                   choices=["Title", "Content", "URL", "Type", "Author"],
                                                   value=["Title"])
            keywords_input = gr.Textbox(label="Keywords to Match against", placeholder="Enter keywords here (comma-separated)...")
            page_input = gr.Slider(label="Pages of results to display", minimum=1, maximum=10, step=1, value=1)

            search_button = gr.Button("Search")
            results_output = gr.Dataframe()
            index_input = gr.Number(label="Select index of the result", value=None)
            details_button = gr.Button("Show Details")
            details_output = gr.HTML()

            search_button.click(
                fn=search_and_display,
                inputs=[search_query_input, search_fields_input, keywords_input, page_input],
                outputs=results_output
            )

            details_button.click(
                fn=display_details,
                inputs=[index_input, results_output],
                outputs=details_output
            )
    # search_tab = gr.Interface(
    #     fn=search_and_display,
    #     inputs=[
    #         gr.Textbox(label="Search Query", placeholder="Enter your search query here..."),
    #         gr.CheckboxGroup(label="Search Fields", choices=["Title", "Content", "URL", "Type", "Author"],
    #                          value=["Title"]),
    #         gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated)..."),
    #         gr.Slider(label="Page", minimum=1, maximum=10, step=1, value=1)
    #     ],
    #     outputs=gr.Dataframe(label="Search Results", height=300)  # Height in pixels
    #     #outputs=gr.Dataframe(label="Search Results")
    # )



    export_keywords_interface = gr.Interface(
        fn=export_keywords_to_csv,
        inputs=[],
        outputs=[gr.File(label="Download Exported Keywords"), gr.Textbox(label="Status")],
        title="Export Keywords",
        description="Export all keywords in the database to a CSV file."
    )

    # Gradio interface for importing data
    def import_data(file):
        # Placeholder for actual import functionality
        return "Data imported successfully"

    import_interface = gr.Interface(
        fn=import_data,
        inputs=gr.File(label="Upload file for import"),
        outputs="text",
        title="Import Data",
        description="Import data into the database from a CSV file."
    )

    import_export_tab = gr.TabbedInterface(
        [gr.TabbedInterface(
            [gr.Interface(
                fn=export_to_csv,
                inputs=[
                    gr.Textbox(label="Search Query", placeholder="Enter your search query here..."),
                    gr.CheckboxGroup(label="Search Fields", choices=["Title", "Content"], value=["Title"]),
                    gr.Textbox(label="Keyword (Match ALL, can use multiple keywords, separated by ',' (comma) )",
                               placeholder="Enter keywords here..."),
                    gr.Number(label="Page", value=1, precision=0),
                    gr.Number(label="Results per File", value=1000, precision=0)
                ],
                outputs="text",
                title="Export Search Results to CSV",
                description="Export the search results to a CSV file."
            ),
                export_keywords_interface],
            ["Export Search Results", "Export Keywords"]
        ),
            import_interface],
        ["Export", "Import"]
    )

    keyword_add_interface = gr.Interface(
        fn=add_keyword,
        inputs=gr.Textbox(label="Add Keywords (comma-separated)", placeholder="Enter keywords here..."),
        outputs="text",
        title="Add Keywords",
        description="Add one, or multiple keywords to the database.",
        allow_flagging="never"
    )

    keyword_delete_interface = gr.Interface(
        fn=delete_keyword,
        inputs=gr.Textbox(label="Delete Keyword", placeholder="Enter keyword to delete here..."),
        outputs="text",
        title="Delete Keyword",
        description="Delete a keyword from the database.",
        allow_flagging="never"
    )

    browse_keywords_interface = gr.Interface(
        fn=keywords_browser_interface,
        inputs=[],
        outputs="markdown",
        title="Browse Keywords",
        description="View all keywords currently stored in the database."
    )

    keyword_tab = gr.TabbedInterface(
        [browse_keywords_interface, keyword_add_interface, keyword_delete_interface],
        ["Browse Keywords", "Add Keywords", "Delete Keywords"]
    )

    # Combine interfaces into a tabbed interface
    tabbed_interface = gr.TabbedInterface([iface, search_interface, import_export_tab, keyword_tab],
                                          ["Transcription + Summarization", "Search and Detail View", "Export/Import", "Keywords"])
    # Launch the interface
    server_port_variable = 7860
    if server_mode:
        tabbed_interface.launch(share=True, server_port=server_port_variable, server_name="http://0.0.0.0")
    elif share_public == True:
        tabbed_interface.launch(share=True,)
    else:
        tabbed_interface.launch(share=False,)

#
#
#################################################################################




# FIXME - Prompt sample box

# Sample data
prompts_category_1 = [
    "What are the key points discussed in the video?",
    "Summarize the main arguments made by the speaker.",
    "Describe the conclusions of the study presented."
]

prompts_category_2 = [
    "How does the proposed solution address the problem?",
    "What are the implications of the findings?",
    "Can you explain the theory behind the observed phenomenon?"
]

all_prompts = prompts_category_1 + prompts_category_2


# Search function
def search_prompts(query):
    filtered_prompts = [prompt for prompt in all_prompts if query.lower() in prompt.lower()]
    return "\n".join(filtered_prompts)


# Handle prompt selection
def handle_prompt_selection(prompt):
    return f"You selected: {prompt}"

