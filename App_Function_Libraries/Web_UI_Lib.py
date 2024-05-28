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


