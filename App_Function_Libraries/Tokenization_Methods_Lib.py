# Tokenization_Methods_Lib.py
#########################################
# Tokenization Methods Library
# This library is used to handle tokenization of text for summarization.
#
####

# Import Local
import summarize
from Article_Summarization_Lib import *
from Article_Extractor_Lib import *
from Audio_Transcription_Lib import *
from Chunk_Lib import *
from Diarization_Lib import *
from Local_File_Processing_Lib import *
from Local_LLM_Inference_Engine_Lib import *
from Local_Summarization_Lib import *
from Old_Chunking_Lib import *
from SQLite_DB import *
from Summarization_General_Lib import *
from System_Checks_Lib import *
#from Tokenization_Methods_Lib import *
from Video_DL_Ingestion_Lib import *
#from Web_UI_Lib import *

####################
# Function List
#
# 1. openai_tokenize(text: str) -> List[str]
#
####################


#######################################################################################################################
# Function Definitions
#

def openai_tokenize(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model('gpt-4-turbo')
    return encoding.encode(text)

#
#
#######################################################################################################################
