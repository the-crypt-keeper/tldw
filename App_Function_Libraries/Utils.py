# Utils.py
#########################################
# General Utilities Library
# This library is used to hold random utilities used by various other libraries.
#
####
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2.
#
#
####################
# Import necessary libraries
import logging





#######################################################################################################################
# Function Definitions
#

def extract_text_from_segments(segments):
    logging.debug(f"Segments received: {segments}")
    logging.debug(f"Type of segments: {type(segments)}")

    text = ""

    if isinstance(segments, list):
        for segment in segments:
            logging.debug(f"Current segment: {segment}")
            logging.debug(f"Type of segment: {type(segment)}")
            if 'Text' in segment:
                text += segment['Text'] + " "
            else:
                logging.warning(f"Skipping segment due to missing 'Text' key: {segment}")
    else:
        logging.warning(f"Unexpected type of 'segments': {type(segments)}")

    return text.strip()