# Confabulation_check.py
#
# This file contains the functions that are used to check the confabulation of the user's input.
#
#
# Imports
#
# External Imports
#
# Local Imports
#
#
####################################################################################################
#
# Functions:
from App_Function_Libraries.ms_g_eval import validate_inputs, detailed_api_error, get_summarize_function


def simplified_geval(transcript: str, summary: str, api_name: str, api_key: str) -> str:
    """
    Perform a simplified version of G-Eval using a single query to evaluate the summary.

    Args:
        transcript (str): The original transcript
        summary (str): The summary to be evaluated
        api_name (str): The name of the LLM API to use
        api_key (str): The API key for the chosen LLM

    Returns:
        str: The evaluation result
    """
    try:
        validate_inputs(transcript, summary, api_name, api_key)
    except ValueError as e:
        return str(e)

    prompt = f"""You are an AI assistant tasked with evaluating the quality of a summary. You will be given an original transcript and a summary of that transcript. Your task is to evaluate the summary based on the following criteria:

1. Coherence (1-5): How well-structured and organized is the summary?
2. Consistency (1-5): How factually aligned is the summary with the original transcript?
3. Fluency (1-3): How well-written is the summary in terms of grammar, spelling, and readability?
4. Relevance (1-5): How well does the summary capture the important information from the transcript?

Please provide a score for each criterion and a brief explanation for your scoring. Then, give an overall assessment of the summary's quality.

Original Transcript:
{transcript}

Summary to Evaluate:
{summary}

Please provide your evaluation in the following format:
Coherence: [score] - [brief explanation]
Consistency: [score] - [brief explanation]
Fluency: [score] - [brief explanation]
Relevance: [score] - [brief explanation]

Overall Assessment: [Your overall assessment of the summary's quality]
"""

    # FIXME
    summarize_function = get_summarize_function(api_name)

    try:
        response = summarize_function(api_key, prompt, "", temp=0.7, system_prompt="You are a helpful AI assistant tasked with evaluating summaries.")
    except Exception as e:
        return detailed_api_error(api_name, e)

    return response