# Web_UI_Lib.py
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

# # Import necessary libraries
# import json
# import logging
# import requests
# import sys
# import os
# # Import 3rd-pary Libraries
# import gradio as gr
# # Import Local
# from summarize import main
# from Article_Summarization_Lib import *
# from Article_Extractor_Lib import *
# from Audio_Transcription_Lib import *
# from Chunk_Lib import *
# from Diarization_Lib import *
# from Local_File_Processing_Lib import *
# from Local_LLM_Inference_Engine_Lib import *
# from Local_Summarization_Lib import *
# from Old_Chunking_Lib import *
# from SQLite_DB import *
# from Summarization_General_Lib import *
# from System_Checks_Lib import *
# from Tokenization_Methods_Lib import *
# from Video_DL_Ingestion_Lib import *
# #from Web_UI_Lib import *



server_mode = False
share_public = False

# Revisit later.
#
# def gradio_download_youtube_audio(url):
#     """Download audio using yt-dlp with specified options."""
#     # Determine ffmpeg path based on the operating system.
#     ffmpeg_path = './Bin/ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
#
#     # Extract information about the video
#     with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
#         info_dict = ydl.extract_info(url, download=False)
#         sanitized_title = sanitize_filename(info_dict['title'])
#
#     # Setup the final directory and filename
#     download_dir = Path(f"results/{sanitized_title}")
#     download_dir.mkdir(parents=True, exist_ok=True)
#     output_file_path = download_dir / f"{sanitized_title}.ogg"
#
#     # Initialize yt-dlp with generic options and the output template
#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'ffmpeg_location': ffmpeg_path,
#         'outtmpl': str(output_file_path),
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'vorbis',
#             'preferredquality': '192',
#         }],
#         'noplaylist': True,
#         'quiet': True
#     }
#
#     # Execute yt-dlp to download the audio
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([url])
#
#     # Final check to ensure file exists
#     if not output_file_path.exists():
#         raise FileNotFoundError(f"Expected file was not found: {output_file_path}")
#
#     return str(output_file_path)
#
# download_audio_interface = gr.Interface(
#     fn=gradio_download_youtube_audio,
#     inputs=gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL here"),
#     outputs=gr.File(label="Download Audio"),
#     title="YouTube Audio Downloader",
#     description="Enter a YouTube URL to download the audio as an .ogg file."
# )
#
#
#     tabbed_interface = gr.TabbedInterface([iface, search_interface, import_export_tab, keyword_tab, download_videos_interface, download_audio_interface],


# An idea - FIXME - Add a feedback mechanism to the summarization interface
# def submit_feedback(summary_text, user_rating, user_comments):
#     """
#     Store or process user feedback on summaries.
#     """
#     print(f"Received rating: {user_rating}/5")
#     print(f"User comments: {user_comments}")
#     # Here you could log this to a database or use it to further train a model.
#
# # Adding to Gradio interface
# feedback_interface = gr.Interface(
#     fn=submit_feedback,
#     inputs=[
#         gr.Textbox(label="Summary Text", default="Generated summary will appear here...", readonly=True),
#         gr.Slider(minimum=1, maximum=5, label="Rate the summary (1-5 stars)"),
#         gr.Textbox(label="Comments", placeholder="Additional comments...")
#     ],
#     outputs=[]
# )
#
# feedback_interface.launch()