# import gradio as gr
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import os
# import torch
#
# # Assuming models are stored in a 'models' directory
# MODELS_DIR = "models"
#
#
# def get_local_models():
#     if not os.path.exists(MODELS_DIR):
#         os.makedirs(MODELS_DIR)
#     return [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
#
#
# def download_model(model_name):
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#
#         # Save the model and tokenizer
#         save_path = os.path.join(MODELS_DIR, model_name.split('/')[-1])
#         tokenizer.save_pretrained(save_path)
#         model.save_pretrained(save_path)
#
#         return f"Successfully downloaded model: {model_name}"
#     except Exception as e:
#         return f"Failed to download model: {str(e)}"
#
#
# def run_inference(model_name, prompt):
#     try:
#         model_path = os.path.join(MODELS_DIR, model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForCausalLM.from_pretrained(model_path)
#
#         # Use GPU if available
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model.to(device)
#
#         # Create a text-generation pipeline
#         text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
#
#         # Generate text
#         result = text_generator(prompt, max_length=100, num_return_sequences=1)
#
#         return result[0]['generated_text']
#     except Exception as e:
#         return f"Error running inference: {str(e)}"
#
#
# def create_huggingface_tab():
#     with gr.Tab("Hugging Face Transformers"):
#         gr.Markdown("# Hugging Face Transformers Model Management")
#
#         with gr.Row():
#             model_list = gr.Dropdown(label="Available Models", choices=get_local_models())
#             refresh_button = gr.Button("Refresh Model List")
#
#         with gr.Row():
#             new_model_name = gr.Textbox(label="Model to Download (e.g., 'gpt2' or 'EleutherAI/gpt-neo-1.3B')")
#             download_button = gr.Button("Download Model")
#
#         download_output = gr.Textbox(label="Download Status")
#
#         with gr.Row():
#             run_model = gr.Dropdown(label="Model to Run", choices=get_local_models())
#             prompt = gr.Textbox(label="Prompt")
#             run_button = gr.Button("Run Inference")
#
#         run_output = gr.Textbox(label="Model Output")
#
#         def update_model_lists():
#             models = get_local_models()
#             return gr.update(choices=models), gr.update(choices=models)
#
#         refresh_button.click(update_model_lists, outputs=[model_list, run_model])
#         download_button.click(download_model, inputs=[new_model_name], outputs=[download_output])
#         run_button.click(run_inference, inputs=[run_model, prompt], outputs=[run_output])