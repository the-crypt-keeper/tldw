# Pieces

## Table of Contents
- [What is this page?](#what-is-this-page)
- [Introduction](#introduction)
- [How Does it All Start?](#how-does-it-start)
- [Transcription / Summarization / Ingestion Tab](#transcription--summarization--ingestion-tab)
- [](#)
- [](#)


------------------------------------------------------------------------------------------------------------------
### <a name="what-is-this-page"></a>What is this page?
- Goal of this page is to help provide documentation and guidance for the various pieces of the project.
- This page will be updated as the project progresses and more pieces are added.
- Specifically, this page will walk through each of the functions exposed by the project and provide a brief overview of what they do and how they work by mapping the code functions used throughout the process.
- 
------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------
### <a name="introduction"></a>Introduction
- The project is broken down into several pieces, each of which is responsible for a specific task.
- The pieces are designed to be:
    - Modular and can be used independently of each other.
    - Used in conjunction with each other to provide a seamless experience for the user.
    - Easily extendable and can be modified to suit the needs of the user.
- The pieces are:
  - Ingestion
    - video/audio: yt-dlp, ffmpeg, faster_whisper, ctranslate2
    - text/ebook: pypandoc
    - website: requests, BeautifulSoup/trafilatura
    - pdf: marker pipeline
  - LLM Chat
    - plain web requests
  - Database Management
    - sqlite3 + python sqlite3 module
  - GUI
    - Gradio
------------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------------
### <a name="how-does-it-start"></a> How does it all start?
1. Execution starts with the `summarize.py` file.
2. The `summarize.py` file is responsible for starting the GUI and handling the user input.
3. Depending on the user input, the appropriate function is called to handle the user request.
   - This occurs on lines 696-910 of the `summarize.py` file.
4. For this, we are going to assume the user has passed the `-gui` argument to the script with no other arguments presented.
5. The GUI is started by calling the `launch_ui()` function on line 838.
   - The script first checks to see if the `--local_llm` argument was passed, first running it if it was, and then `launch_ui(share_public=False)`
   - The `launch_ui()` function is defined in `Gradio_Related.py` on line 3712.
6. The `launch_ui()` function is responsible for starting the GUI using the Gradio library.
   - As part of this function, the various tabs are aligned and ordered.
   - Each of the tabs are created using the appropriate `create_X_tab()` function.
7. From here, all functions are called as needed by the user from each individual tab within the Gradio UI.
8. The following sections will walk through each of the tabs and the functions they call to handle the user input.
------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------
### <a name="transcription--summarization--ingestion-tab"></a> Transcription / Summarization / Ingestion Tab
#### Video Transcription + Summarization
- The tab for video transcription is the `create_video_transcription_tab()`
- The function is defined in `Gradio_Related.py` on line 561.
- The function is responsible for creating the tab and handling the user input.
- The function first creates the tab using the `gr.Tabitem()` -> `gr.Markdown()` -> `gr.Row():` functions.
- The function then creates the input fields for the user to input the video URL and the desired summary length.
  - `custom_prompt_checkbox`, `preset_prompt_checkbox`, `preset_prompt`, `use_time_input`, `chunking_options_checkbox`, and `use_cookies_input` are used to dynamically show their corresponding items depending on the user selection.
- The function then creates the output fields for the user to view the summary and the transcription.
- Finally, the button `Process Videos` is created to handle the user input with the variable `process_button`
- The function then defines the `def process_videos_with_error_handling(<args>):` function to handle the user input.
    - `def process_videos_with_error_handling(<args>):`:
      - The function first checks to see if there's any input in the URL field.
      - The function then checks/sets the batch size
      - Then separates URLs and local files
      - Validates existence of local files
      - Sets `all_inputs` variable as result of validation
      - Starts a `for` loop iterating through every item in the current batch
      - If the item is a URL, metadata extraction is attempted (using `yt-dlp`)
      - Chunking options are then set using the variables fed into the function
      - Line 765: `result = process_url_with_metadata<args>` - Stuff
        - Args: `input_item, 2, whisper_model, custom_prompt if custom_prompt_checkbox else None, start_seconds, api_name, api_key, False, False, False, False, 0.01, None, keywords, None, diarize, end_time=end_seconds, include_timestamps=(timestamp_option == "Include Timestamps"), metadata=video_metadata, use_chunking=chunking_options_checkbox, chunk_options=chunk_options, keep_original_video=keep_original_video`
      - If/else decision based on returned result, success or failure
        - If success, secondary sanity check, then: `transcription, summary = result`
            * (`batch_results` is an array)
            * `batch_results.append((input_item, transcription, "Success", result_metadata, json_file, summary_file))`
        - If failure, `batch_results.append((input_item, error_message, "Error", result_metadata, None, None))errors.append(error_message)`
      - Next, the length of the results is checked and printed to the screen
      - Then Lines 809-853 create the HTML to display the results nicely
      - Lines 857-858, 860-861 save the transcriptions + summaries to JSON
      - Finally Line 866 holds the following line which returns the final results to the user:
        - `return ( f"Processed {total_inputs} videos. {len(errors)} errors occurred.", error_summary, results_html, 'all_transcriptions.json', 'all_summaries.txt')`
- **Let's now dig a bit deeper into Line 765: `process_url_with_metadata<args>`**
  - `process_url_with_metadata<args>`:
    - Defined on line 934: `def process_url_with_metadata(input_item, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video_flag, download_audio, rolling_summarization, detail_level, question_box, keywords, local_file_path, diarize, end_time=None, include_timestamps=True, metadata=None, use_chunking=False, chunk_options=None, keep_original_video=False):```
    - First creates the folder `Video_Downloads` if it doesn't already exist in the current folder.
    - We then create the `infodict` dict
    - We then check if the input item is a URL or a local file
      - If it's a file we simply set the `video_metadata` to `None` with the `video_description` set to `Local file`
      - If it's a URL, we extract the metadata using `yt-dlp`
    - Next metadata is set,
    - Video is downloaded 1002: 
      - `Line video_file_path = download_video(input_item, download_path, full_info, download_video_flag)`
    - Video is transcribed Line 1011:
      - `audio_file_path, segments = perform_transcription(video_file_path, offset, whisper_model, vad_filter, diarize)`
    - Line 1027 We save the transcription to a JSON file
    - Lines 1032-1039 We delete the wav file 
    - Lines 1043-1054 we delete the video file if the user doesn't want to keep it
    - Line 1063 we create the `transcription_text` variable from compiling all the transcription segments
    - Lines 1076-1084 we summarize the transcription
      - `perform_summarization(api_name, full_text_with_metadata, custom_prompt, api_key)`
    - Lines 1086-1089 we save the summary to a text file
      - `save_transcription_and_summary(full_text_with_metadata, summary_text, download_path, info_dict)`
    - Lines 1095-1102 We prep the keywords
    - Lines 1104-1108 We save the media to the DB:
      - `add_media_to_database(info_dict['webpage_url'], info_dict, full_text_with_metadata, summary_text, keywords_list, custom_prompt, whisper_model)`
    - Finally, we return the transcription and summary on lines 1110-1111:
      - `return info_dict['webpage_url'], full_text_with_metadata, summary_text, json_file_path, summary_file_path, info_dict`
- **Let's now dig deeper into Line 1002: `download_video()`**
  - This function is imported from `Video_DL_Ingestion_Lib.py` and is defined on Line 97 of that file: 
    - `def download_video(video_url, download_path, info_dict, download_video_flag):` 
  - It first checks to see if the video has a valid name set in the info_dict{} dict
  - Does path handling for calling ffmpeg
  - Uses yt-dlp to download the audio of the video, or the full video if the user wants
- **Let's now dig deeper into Line 1011: `perform_transcription()`**
  - This function is imported from `Summarization_General.py` and is defined on Line 831 of that file:
    - `def perform_transcription(video_path, offset, whisper_model, vad_filter, diarize=False):`
  - It itself calls `audio_file_path = convert_to_wav(video_path, offset)`
  - Then checks for diarization and performs it if set
  - Then checks for a segments JSON file and loads it if it exists
  - Returns:
    - `return audio_file_path, diarized_segments`
- **Let's now dig deper into `convert_to_wav()` from the `Audio_Transcription_Lib.py` Library**
  - Sets the out-path for the file as the same path as the input file, but with a .wav extension
    - If the file already exists it will skip the conversion process
    - If the file doesn't exist, it will call ffmpeg to convert the file to a .wav file
  - Returns:
    - `return out_path`
- **Let's now dig deeper into Line 1076: `perform_summarization()`**
  - This function is defined in `Summarization_General.py` on Line 999:
    * `def perform_summarization(api_name, input_data, custom_prompt_input, api_key, recursive_summarization=False):`
  - First evaluates if a custom prompt was passed, if not it uses the default prompt
  - Then extracts the metadata + content from the input_data
    - `extract_metadata_and_content(input_data)`
  - Input is structured for summarization:
    - `structured_input = format_input_with_metadata(metadata, content)`
  - Then its checked if `recursive_summarization` was set (Line 1035), 
    - if so, it first performs chunking, calling `improved_chunking_process()`
      - `chunks = improved_chunking_process(structured_input, chunk_options)`
    - Finally, it performs recursive summarization using `recursive_summarize_chunks()`
      - `summary = recursive_summarize_chunks([chunk['text'] for chunk in chunks], lambda x: summarize_chunk(api_name, x, custom_prompt_input, api_key), custom_prompt_input)`      
  - If `recursive_summarization` was not set, then `summarize_chunk()` is called directly
    - `summary = summarize_chunk(api_name, structured_input, custom_prompt_input, api_key)`
  - Lines 1051-1081: It then evaluates if a summary was generated, and if so, saves it as a `.txt` file.
  - Once complete, it returns the summary
    - `return summary`
- **Let's now dig deeper into `improved_chunking_process()`:**
  - This function is defined in `Chunking_Lib.py` on Line 52:
    - `def improved_chunking_process(text: str, chunk_options: Dict[str, Any]) -> List[Dict[str, Any]]:`
    - First sets the various chunking options:
    - `chunk_method`, `max_chunk_size`, `overlap`, `language`, `adaptive`, `multi_level`
  - It then proceeds to perform chunking on the input text based on the selected chunking method: (All these functions are defined in the `Chunking_Lib.py` file)
    - `adaptive` - `adpativ_chunk_size()`
    - `multi_level` - `multi_level_chunking()`
    - `chunk_method == words` - `chunk_text_by_words()`
    - `chunk_method == sentences` - `chunk_text_by_sentences()`
    - `chunk_method == paragraps` - `chunk_text_by_paragraphs()`
    - `chunk_method == tokens` - `chunk_text_by_tokens()`
    - `chunks = [text]` - If no chunking method is selected / No chunking is performed.
  - Finally, it returns the chunks:
    - `return [{'text': chunk, 'metadata': get_chunk_metadata(chunk, text)} for chunk in chunks]`
- **Let's now dig deeper into `recursive_summarize_chunks()`:**
  - Defined on Line 215 of `Chunking_Lib.py`:
    - `def recursive_summarize_chunks(chunks, summarize_func, custom_prompt):`
  - First initializes the variables: `summarized_chunks = []`, `current_summary = ""`
  - Then iterates through each chunk in the chunks list
  - If it's the first chunk, it summarizes it using the `summarize_func()` function
  - If it's not the first chunk, it combines the current summary with the chunk and summarizes the combined text
  - Next it appends the current summary to the `summarized_chunks` list
  - Finally, it returns the summarized chunks list:
    - `return summarized_chunks`
- **Let's now dig deeper into `summarize_chunk()`:**
  - This function is defined on line 922 of `Summarization_General.py`:
    - `def summarize_chunk(api_name, text, custom_prompt_input, api_key):`
  - This function is simply one long if/else statement matching the `api_name` with the appropriate summarization function to call.
    - i.e. `if api_name == "openai":` -> `return summarize_with_openai(api_key, text, custom_prompt_input)`
    - Each of these functions are defined in the `Summarization_Lib.py` file.
    - The first one can be found at Line 66 of that file: `def summarize_with_openai(api_key, input_data, custom_prompt_arg):`
      - It continues downward from there.
  - If no api_name is matched, and none was passed, it simply returns `None`
  - If no api_name is matched and an argument for it was passed, it returns an error message.
- **Let's now dig deeper into Line 1086-1089 `save_transcription_and_summary()`**
  - The function is defined in `Summarization_General.py` on Line 900:
    - `def save_transcription_and_summary(transcription_text, summary_text, download_path, info_dict):`
  - The function first sanitizes the video title
    - `video_title = sanitize_filename(info_dict.get('title', 'Untitled'))`
    - `sanitize_filename()` is defined on Line 138 in `Utils.py`.
  - Then, Lines 904-907 it saves the transcription to a `.txt` file
    - `transcription_file_path = os.path.join(download_path, f"{video_title}_transcription.txt")`
  - Then, Lines 909-914 it saves the summary to a `.txt` file if it exists
    - `summary_file_path = os.path.join(download_path, f"{video_title}_summary.txt")`
  - Finally, it returns the paths to the transcription and summary files:
    - `return transcription_file_path, summary_file_path`
- **Let's now dig deeper into Line 1104-1108 `add_media_to_database()`**
  - Function is defined in `SQLite_DB.py` on Line 721:
    - `def add_media_to_database(url, info_dict, segments, summary, keywords, custom_prompt_input, whisper_model, media_type='video'):`
  - The function first extracts the media contents from the `segments` list
  - It then sets the custom prompt if not set
    - ```
          result = add_media_with_keywords(
                    url=url,
                    title=info_dict.get('title', 'Untitled'),
                    media_type=media_type,
                    content=content,
                    keywords=','.join(keywords) if isinstance(keywords, list) else keywords,
                    prompt=custom_prompt_input or 'No prompt provided',
                    summary=summary or 'No summary provided',
                    transcription_model=whisper_model,
                    author=info_dict.get('uploader', 'Unknown'),
                    ingestion_date=datetime.now().strftime('%Y-%m-%d')
         )
      ```
  - Finally, it returns `result`
    - `return result`
- **Let's now dig deeper into `add_media_with_keywords()`**
  - It is defined on Lines 303-304 of `SQLite_DB.py`:
    - `def add_media_with_keywords(url, title, media_type, content, keywords, prompt, summary, transcription_model, author, ingestion_date):`
  - The function first sets default values for all the variables
  - Next it ensures the URL is valid and if it isn't, it sets it to `localhost`
  - It then performs a check of the media type, seeing if it matches one of the following, raising an error if it doesn't:
    - `if media_type not in ['article', 'audio', 'document', 'obsidian_note', 'podcast', 'text', 'video', 'unknown']:`
  - Next it checks to ensure the ingestion date/time is correct, raising an error if it isn't:
    - `if ingestion_date and not is_valid_date(ingestion_date):`
  - Next it ensure proper formatting of the keywords
  - Next, Lines 344-399 it attempts to connect to the DB and insert the generated items into the database.
  - Finally, it returns the following string on success (throwing appropriate errors if it fails at any part of the process):
    - `return f"Media '{title}' added/updated successfully with keywords: {', '.join(keyword_list)}"`

------------------------------------------------------------------------------------------------------------------





#### Audio File Transcription + Summarization



#### Podcast Transcription + Summarization



#### Import .epub/ebook Files



#### Website Scraping



#### PDF Ingestion


#### Re-Summarize

