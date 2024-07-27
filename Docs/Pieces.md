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
  - `custom_prompt_checkbox`, `preset_prompt_checkbox`, `preset_prompt`, `use_time_input`, and `use_cookies_input` are used to dynamically show their corresponding items depending on the user selection.
- The function then creates the output fields for the user to view the summary and the transcription.




#### Audio File Transcription + Summarization



#### Podcast Transcription + Summarization



#### Import .epub/ebook Files



#### Website Scraping



#### PDF Ingestion


#### Re-Summarize

