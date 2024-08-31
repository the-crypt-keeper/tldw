###################################################################################################
# geval.py - Gradio code for G-Eval testing
# We will use the G-Eval API to evaluate the quality of the generated summaries.

import gradio as gr
from App_Function_Libraries.ms_g_eval import run_geval

def create_geval_tab():
    with gr.Tab("G-Eval"):
        gr.Markdown("# G-Eval Summarization Evaluation")
        with gr.Row():
            with gr.Column():
                document_input = gr.Textbox(label="Source Document", lines=10)
                summary_input = gr.Textbox(label="Summary", lines=5)
                api_name_input = gr.Dropdown(
                    choices=["OpenAI", "Anthropic", "Cohere", "Groq", "OpenRouter", "DeepSeek", "HuggingFace", "Mistral", "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "Local-LLM", "Ollama"],
                    label="Select API"
                )
                api_key_input = gr.Textbox(label="API Key (if required)", type="password")
                evaluate_button = gr.Button("Evaluate Summary")
            with gr.Column():
                output = gr.Textbox(label="Evaluation Results", lines=10)

        evaluate_button.click(
            fn=run_geval,
            inputs=[document_input, summary_input, api_name_input, api_key_input],
            outputs=output
        )

    return document_input, summary_input, api_name_input, api_key_input, evaluate_button, output

# If you want to run this as a standalone Gradio app
if __name__ == "__main__":
    with gr.Blocks() as demo:
        create_geval_tab()
    demo.launch()
