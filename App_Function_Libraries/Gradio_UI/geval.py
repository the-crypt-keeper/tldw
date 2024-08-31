###################################################################################################
# geval.py - Gradio code for G-Eval testing
# We will use the G-Eval API to evaluate the quality of the generated summaries.

import gradio as gr
from Tests.Summarization.ms_g_eval import aggregate, geval_summarization


def run_geval(document, summary, api_key):
    prompts = {
        "coherence": """You will be given one summary written for a source document.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Evaluation Steps:

1. Read the source document carefully and identify the main topic and key points.
2. Read the summary and compare it to the source document. Check if the summary covers the main topic and key points of the source document, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


Example:


Source Document:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Coherence:""",
        "consistency": """You will be given a source document. You will then be given one summary written for this source document.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. 

Evaluation Steps:

1. Read the source document carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the source document. Check if the summary contains any factual errors that are not supported by the source document.
3. Assign a score for consistency based on the Evaluation Criteria.


Example:


Source Document: 

{{Document}}

Summary: 

{{Summary}}


Evaluation Form (scores ONLY):

- Consistency:""",
        "fluency": """You will be given one summary written for a source document.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow.


Example:

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Fluency (1-3):""",
        "relevance": """You will be given one summary written for a source document.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the source document.
3. Assess how well the summary covers the main points of the source document, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.


Example:


Source Document:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Relevance:"""
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