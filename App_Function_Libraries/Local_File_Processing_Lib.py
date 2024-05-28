# Local_File_Processing_Lib.py
#########################################
# Local File Processing and File Path Handling Library
# This library is used to handle processing local filepaths and URLs.
# It checks for the OS, the availability of the GPU, and the availability of the ffmpeg executable.
# If the GPU is available, it asks the user if they would like to use it for processing.
# If ffmpeg is not found, it asks the user if they would like to download it.
# The script will exit if the user chooses not to download ffmpeg.
####

####################
# Function List
#
# 1. read_paths_from_file(file_path)
# 2. process_path(path)
# 3. process_local_file(file_path)
# 4. read_paths_from_file(file_path: str) -> List[str]
#
####################

# Import necessary libraries
import os
import logging
# Import Local
import summarize
from Article_Summarization_Lib import *
from Article_Extractor_Lib import *
from Audio_Transcription_Lib import *
from Chunk_Lib import *
from Diarization_Lib import *
#from Local_File_Processing_Lib import *
from Local_LLM_Inference_Engine_Lib import *
from Local_Summarization_Lib import *
from Old_Chunking_Lib import *
from SQLite_DB import *
from Summarization_General_Lib import *
from System_Checks_Lib import *
from Tokenization_Methods_Lib import *
from Video_DL_Ingestion_Lib import *
from Web_UI_Lib import *

#######################################################################################################################
# Function Definitions
#

def read_paths_from_file(file_path):
    """ Reads a file containing URLs or local file paths and returns them as a list. """
    paths = []  # Initialize paths as an empty list
    with open(file_path, 'r') as file:
        paths = [line.strip() for line in file]
    return paths


def process_path(path):
    """ Decides whether the path is a URL or a local file and processes accordingly. """
    if path.startswith('http'):
        logging.debug("file is a URL")
        # For YouTube URLs, modify to download and extract info
        return get_youtube(path)
    elif os.path.exists(path):
        logging.debug("File is a path")
        # For local files, define a function to handle them
        return process_local_file(path)
    else:
        logging.error(f"Path does not exist: {path}")
        return None


# FIXME
def process_local_file(file_path):
    logging.info(f"Processing local file: {file_path}")
    title = normalize_title(os.path.splitext(os.path.basename(file_path))[0])
    info_dict = {'title': title}
    logging.debug(f"Creating {title} directory...")
    download_path = create_download_directory(title)
    logging.debug(f"Converting '{title}' to an audio file (wav).")
    audio_file = convert_to_wav(file_path)  # Assumes input files are videos needing audio extraction
    logging.debug(f"'{title}' successfully converted to an audio file (wav).")
    return download_path, info_dict, audio_file


def read_paths_from_file(file_path: str) -> List[str]:
    """Read paths from a text file."""
    with open(file_path, 'r') as file:
        paths = file.readlines()
    return [path.strip() for path in paths]


#
#
#######################################################################################################################