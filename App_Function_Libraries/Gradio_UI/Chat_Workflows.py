# Chat_Workflows.py
# Description: UI for Chat Workflows
#
# Imports
import json
import logging
from pathlib import Path
#
# External Imports
import gradio as gr

from App_Function_Libraries.DB.DB_Manager import save_chat_history_to_database
#
from App_Function_Libraries.Gradio_UI.Chat_ui import process_with_llm
#
############################################################################################################
#
# Functions:

# Load workflows from a JSON file
json_path = Path('./Helper_Scripts/Workflows/Workflows.json')
with json_path.open('r') as f:
    workflows = json.load(f)


# FIXME - broken Completely. Doesn't work.
def chat_workflows_tab():
    with gr.TabItem("Chat Workflows"):
        gr.Markdown("# Workflows using LLMs")

        with gr.Row():
            workflow_selector = gr.Dropdown(label="Select Workflow", choices=[wf['name'] for wf in workflows])
            api_selector = gr.Dropdown(
                label="Select API Endpoint",
                choices=["OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                         "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace"],
                value="OpenAI"
            )
            api_key_input = gr.Textbox(label="API Key (optional)", type="password")

        context_input = gr.Textbox(label="Initial Context (optional)", lines=5)

        chatbot = gr.Chatbot(label="Workflow Chat")
        msg = gr.Textbox(label="Your Input")
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear Chat")
        save_btn = gr.Button("Save Chat to Database")

        workflow_state = gr.State({"current_step": 0, "max_steps": 0, "conversation_id": None})

        def update_workflow_ui(workflow_name):
            selected_workflow = next(wf for wf in workflows if wf['name'] == workflow_name)
            num_prompts = len(selected_workflow['prompts'])
            return gr.update(value={"current_step": 0, "max_steps": num_prompts, "conversation_id": None})

        def process_step(message, chat_history, context, workflow_name, api_endpoint, api_key, workflow_state):
            selected_workflow = next(wf for wf in workflows if wf['name'] == workflow_name)
            current_step = workflow_state["current_step"]
            max_steps = workflow_state["max_steps"]

            if current_step >= max_steps:
                return chat_history, workflow_state, gr.update(interactive=False)

            prompt = selected_workflow['prompts'][current_step]
            full_context = f"{context}\n\nStep {current_step + 1}: {prompt}\nUser: {message}"

            try:
                result = process_with_llm(workflow_name, full_context, prompt, api_endpoint, api_key)
            except Exception as e:
                result = f"Error processing with LLM: {str(e)}"

            chat_history.append((message, result))

            next_step = current_step + 1
            workflow_state["current_step"] = next_step

            if next_step >= max_steps:
                return chat_history, workflow_state, gr.update(interactive=False)
            else:
                next_prompt = selected_workflow['promts'][next_step]
                chat_history.append((None, f"Step {next_step + 1}: {next_prompt}"))
                return chat_history, workflow_state, gr.update(interactive=True)

        def clear_chat():
            return [], {"current_step": 0, "max_steps": 0, "conversation_id": None}, gr.update(interactive=True)

        def save_chat_to_db(chat_history, workflow_name, workflow_state):
            conversation_id = workflow_state.get("conversation_id")
            new_id, status_message = save_workflow_chat_to_db(db, chat_history, workflow_name, conversation_id)
            if new_id:
                workflow_state["conversation_id"] = new_id
            return status_message, workflow_state

        workflow_selector.change(
            update_workflow_ui,
            inputs=[workflow_selector],
            outputs=[workflow_state]
        )

        submit_btn.click(
            process_step,
            inputs=[msg, chatbot, context_input, workflow_selector, api_selector, api_key_input, workflow_state],
            outputs=[chatbot, workflow_state, msg]
        )

        clear_btn.click(
            clear_chat,
            outputs=[chatbot, workflow_state, msg]
        )

        save_btn.click(
            save_chat_to_db,
            inputs=[chatbot, workflow_selector, workflow_state],
            outputs=[gr.Textbox(label="Save Status"), workflow_state]
        )

    return workflow_selector, api_selector, api_key_input, context_input, chatbot, msg, submit_btn, clear_btn, save_btn

#
# End of script
############################################################################################################
