#######################################################################################################################
#
# geval.py
#
# Description: This file contains the code to evaluate the generated text using G-Eval metric.
#
# Scripts taken from https://github.com/microsoft/promptflow/tree/main/examples/flows/evaluation/eval-summarization and modified.
#
import gradio as gr
import inspect
import logging
import re
from typing import Dict, Callable, List
import openai
from tenacity import (
    RetryError,
    Retrying,
    after_log,
    before_sleep_log,
    stop_after_attempt,
    wait_random_exponential,
)
#
#######################################################################################################################
#
# Start of G-Eval.py

logger = logging.getLogger(__name__)

def aggregate(
    fluency_list: List[float],
    consistency_list: List[float],
    relevance_list: List[float],
    coherence_list: List[float],
) -> Dict[str, float]:
    """
    Takes list of scores for 4 dims and outputs average for them.

    Args:
        fluency_list (List(float)): list of fluency scores
        consistency_list (List(float)): list of consistency scores
        relevance_list (List(float)): list of relevance scores
        coherence_list (List(float)): list of coherence scores

    Returns:
        Dict[str, float]: Returns average scores
    """
    average_fluency = sum(fluency_list) / len(fluency_list)
    average_consistency = sum(consistency_list) / len(consistency_list)
    average_relevance = sum(relevance_list) / len(relevance_list)
    average_coherence = sum(coherence_list) / len(coherence_list)

    log_metric("average_fluency", average_fluency)
    log_metric("average_consistency", average_consistency)
    log_metric("average_relevance", average_relevance)
    log_metric("average_coherence", average_coherence)

    return {
        "average_fluency": average_fluency,
        "average_consistency": average_consistency,
        "average_relevance": average_relevance,
        "average_coherence": average_coherence,
    }

def run_geval(document, summary, api_key):
    prompts = {
        "coherence": "...",  # Add the coherence prompt here
        "consistency": "...",  # Add the consistency prompt here
        "fluency": "...",  # Add the fluency prompt here
        "relevance": "..."  # Add the relevance prompt here
    }

    scores = {}
    for metric, prompt in prompts.items():
        full_prompt = prompt.replace("{{Document}}", document).replace("{{Summary}}", summary)
        score = geval_summarization(full_prompt, 5 if metric != "fluency" else 3, api_key)
        scores[metric] = score

    avg_scores = aggregate([scores['fluency']], [scores['consistency']],
                           [scores['relevance']], [scores['coherence']])

    return (f"Coherence: {scores['coherence']:.2f}\n"
            f"Consistency: {scores['consistency']:.2f}\n"
            f"Fluency: {scores['fluency']:.2f}\n"
            f"Relevance: {scores['relevance']:.2f}\n"
            f"Average Scores:\n"
            f"  Fluency: {avg_scores['average_fluency']:.2f}\n"
            f"  Consistency: {avg_scores['average_consistency']:.2f}\n"
            f"  Relevance: {avg_scores['average_relevance']:.2f}\n"
            f"  Coherence: {avg_scores['average_coherence']:.2f}")


def create_geval_tab():
    with gr.Tab("G-Eval"):
        gr.Markdown("# G-Eval Summarization Evaluation")
        with gr.Row():
            with gr.Column():
                document_input = gr.Textbox(label="Source Document", lines=10)
                summary_input = gr.Textbox(label="Summary", lines=5)
                api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
                evaluate_button = gr.Button("Evaluate Summary")
            with gr.Column():
                output = gr.Textbox(label="Evaluation Results", lines=10)

        evaluate_button.click(
            fn=run_geval,
            inputs=[document_input, summary_input, api_key_input],
            outputs=output
        )

    return document_input, summary_input, api_key_input, evaluate_button, output


def parse_output(output: str, max: float) -> float:
    """
    Function that extracts numerical score from the beginning of string

    Args:
        output (str): String to search
        max (float): Maximum score allowed

    Returns:
        float: The extracted score
    """
    # match with either non-negative float or integer
    # if number has non-whitespace characture before that, it won't match
    matched: List[str] = re.findall(r"(?<!\S)\d+(?:\.\d+)?", output)
    if matched:
        if len(matched) == 1:
            score = float(matched[0])
            if score > max:
                raise ValueError(
                    f"Parsed number: {score} was larger than max score: {max}"
                )
        else:
            raise ValueError(
                f"More than one number detected in input. Input to parser was: {output}"
            )
    else:
        raise ValueError(
            f'No number detected in input. Input to parser was "{output}". '
        )
    return score

def geval_summarization(
    prompt_with_src_and_gen: str,
    max_score: float,
    api_key: str,
    model: str = "gpt-4",
) -> float:
    openai.api_key = api_key

    message = {"role": "system", "content": prompt_with_src_and_gen}
    try:
        for attempt in Retrying(
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.INFO),
            after=after_log(logger, logging.INFO),
            wait=wait_random_exponential(multiplier=1, min=1, max=120),
            stop=stop_after_attempt(10),
        ):
            with attempt:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[message],
                    temperature=2,
                    max_tokens=5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    n=20,
                )
    except RetryError:
        logger.exception(f"geval openai call failed\nInput prompt was: {message}")
        raise

    all_responses = []
    for choice in response.choices:
        try:
            content = choice.message.content
            all_responses.append(content)
        except AttributeError:
            logger.exception(
                f"Data with missing content: {choice}\nInput prompt was: {message}"
            )

    return aggregate_llm_scores(all_responses, max_score=max_score)

def aggregate_llm_scores(llm_responses: List[str], max_score: float) -> float:
    """Parse and average valid scores from the generated responses of
    the G-Eval LLM call.

    Args:
        llm_responses (List[str]): List of scores from multiple LLMs
        max_score (float): The maximum score allowed.

    Returns:
        float: The average of all the valid scores
    """
    all_scores = []
    error_count = 0
    for generated in llm_responses:
        try:
            parsed = parse_output(generated, max_score)
            all_scores.append(parsed)
        except ValueError as e:
            logger.warning(e)
            error_count += 1
    if error_count:
        logger.warning(
            f"{error_count} out of 20 scores were discarded due to corrupt g-eval generation"
        )
    score = sum(all_scores) / len(all_scores)
    return score\


#######################################################################################################################
#
# Taken from: https://github.com/microsoft/promptflow/blob/b5a68f45e4c3818a29e2f79a76f2e73b8ea6be44/src/promptflow-core/promptflow/_core/metric_logger.py

class MetricLoggerManager:
    _instance = None

    def __init__(self):
        self._metric_loggers = []

    @staticmethod
    def get_instance() -> "MetricLoggerManager":
        if MetricLoggerManager._instance is None:
            MetricLoggerManager._instance = MetricLoggerManager()
        return MetricLoggerManager._instance

    def log_metric(self, key, value, variant_id=None):
        for logger in self._metric_loggers:
            if len(inspect.signature(logger).parameters) == 2:
                logger(key, value)  # If the logger only accepts two parameters, we don't pass variant_id
            else:
                logger(key, value, variant_id)

    def add_metric_logger(self, logger_func: Callable):
        existing_logger = next((logger for logger in self._metric_loggers if logger is logger_func), None)
        if existing_logger:
            return
        if not callable(logger_func):
            return
        sign = inspect.signature(logger_func)
        # We accept two kinds of metric loggers:
        # def log_metric(k, v)
        # def log_metric(k, v, variant_id)
        if len(sign.parameters) not in [2, 3]:
            return
        self._metric_loggers.append(logger_func)

    def remove_metric_logger(self, logger_func: Callable):
        self._metric_loggers.remove(logger_func)


def log_metric(key, value, variant_id=None):
    """Log a metric for current promptflow run.

    :param key: Metric name.
    :type key: str
    :param value: Metric value.
    :type value: float
    :param variant_id: Variant id for the metric.
    :type variant_id: str
    """
    MetricLoggerManager.get_instance().log_metric(key, value, variant_id)


def add_metric_logger(logger_func: Callable):
    MetricLoggerManager.get_instance().add_metric_logger(logger_func)


def remove_metric_logger(logger_func: Callable):
    MetricLoggerManager.get_instance().remove_metric_logger(logger_func)
#
# End of G-Eval.py
#######################################################################################################################