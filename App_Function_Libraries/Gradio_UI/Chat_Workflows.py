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

        # Create a container for dynamic components
        with gr.Column() as dynamic_components:
            prompt_displays = []
            user_inputs = []
            output_boxes = []
            process_buttons = []
            regenerate_buttons = []

            # Create the maximum number of components needed
            max_steps = max(len(wf['prompts']) for wf in workflows)
            for i in range(max_steps):
                prompt_displays.append(gr.Markdown(visible=False))
                user_inputs.append(gr.Textbox(label=f"Your Input", lines=2, visible=False))
                output_boxes.append(gr.Textbox(label=f"AI Output", lines=5, visible=False))
                with gr.Row():
                    process_buttons.append(gr.Button(f"Process Step {i + 1}", visible=False))
                    regenerate_buttons.append(gr.Button(f"🔄 Regenerate", visible=False))

        def update_workflow_ui(workflow_name):
            selected_workflow = next(wf for wf in workflows if wf['name'] == workflow_name)
            num_prompts = len(selected_workflow['prompts'])

            prompt_updates = []
            input_updates = []
            output_updates = []
            button_updates = []
            regenerate_updates = []

            for i in range(max_steps):
                if i < num_prompts:
                    prompt_updates.append(
                        gr.update(value=f"**Step {i + 1}:** {selected_workflow['prompts'][i]}", visible=True))
                    input_updates.append(gr.update(value="", visible=True, interactive=(i == 0)))
                    output_updates.append(gr.update(value="", visible=True))
                    button_updates.append(gr.update(visible=(i == 0)))
                    regenerate_updates.append(gr.update(visible=False))
                else:
                    prompt_updates.append(gr.update(visible=False))
                    input_updates.append(gr.update(visible=False))
                    output_updates.append(gr.update(visible=False))
                    button_updates.append(gr.update(visible=False))
                    regenerate_updates.append(gr.update(visible=False))

            return prompt_updates + input_updates + output_updates + button_updates + regenerate_updates

        def process(context, workflow_name, api_endpoint, api_key, step, *user_inputs):
            try:
                selected_workflow = next(wf for wf in workflows if wf['name'] == workflow_name)
            except StopIteration:
                # Handle the case where no matching workflow is found
                error_message = f"No workflow found with name: {workflow_name}"
                logging.error(error_message)
                return [gr.update(value=error_message)] * (
                            len(prompt_displays) + len(user_inputs) + len(output_boxes) + len(process_buttons) + len(
                        regenerate_buttons))

            # Ensure we don't go out of bounds
            if step >= len(selected_workflow['prompts']):
                error_message = f"Step {step} is out of range for workflow: {workflow_name}"
                logging.error(error_message)
                return [gr.update(value=error_message)] * (
                            len(prompt_displays) + len(user_inputs) + len(output_boxes) + len(process_buttons) + len(
                        regenerate_buttons))

            # Build up the context from previous steps
            full_context = context + "\n\n"
            for i in range(step + 1):
                full_context += f"Question: {selected_workflow['prompts'][i]}\n"
                full_context += f"Answer: {user_inputs[i]}\n"
                if i < step:
                    full_context += f"AI Output: {output_boxes[i].value}\n\n"

            try:
                result = process_with_llm(workflow_name, full_context, selected_workflow['prompts'][step], api_endpoint,
                                          api_key)
            except Exception as e:
                error_message = f"Error processing with LLM: {str(e)}"
                logging.error(error_message)
                result = error_message

            updates = []
            for i in range(max_steps):
                if i == step:
                    updates.extend([
                        gr.update(),  # Markdown (prompt_displays)
                        gr.update(interactive=False),  # Textbox (user_inputs)
                        gr.update(value=result),  # Textbox (output_boxes)
                        gr.update(visible=False),  # Button (process_buttons)
                        gr.update(visible=True)  # Button (regenerate_buttons)
                    ])
                elif i == step + 1:
                    updates.extend([
                        gr.update(),  # Markdown (prompt_displays)
                        gr.update(interactive=True),  # Textbox (user_inputs)
                        gr.update(),  # Textbox (output_boxes)
                        gr.update(visible=True),  # Button (process_buttons)
                        gr.update(visible=False)  # Button (regenerate_buttons)
                    ])
                elif i > step + 1:
                    updates.extend([
                        gr.update(),  # Markdown (prompt_displays)
                        gr.update(interactive=False),  # Textbox (user_inputs)
                        gr.update(),  # Textbox (output_boxes)
                        gr.update(visible=False),  # Button (process_buttons)
                        gr.update(visible=False)  # Button (regenerate_buttons)
                    ])
                else:
                    updates.extend([
                        gr.update(),  # Markdown (prompt_displays)
                        gr.update(interactive=False),  # Textbox (user_inputs)
                        gr.update(),  # Textbox (output_boxes)
                        gr.update(visible=False),  # Button (process_buttons)
                        gr.update(visible=True)  # Button (regenerate_buttons)
                    ])

            return updates

        # Set up event handlers
        workflow_selector.change(
            update_workflow_ui,
            inputs=[workflow_selector],
            outputs=prompt_displays + user_inputs + output_boxes + process_buttons + regenerate_buttons
        )

        # Set up process button click events
        for i, button in enumerate(process_buttons):
            button.click(
                fn=lambda context, wf_name, api_endpoint, api_key, *inputs, step=i: process(context, wf_name,
                                                                                            api_endpoint, api_key, step,
                                                                                            *inputs),
                inputs=[context_input, workflow_selector, api_selector, api_key_input] + user_inputs,
                outputs=prompt_displays + user_inputs + output_boxes + process_buttons + regenerate_buttons
            ).then(lambda: gr.update(value=""), outputs=[user_inputs[i]])

        # Set up regenerate button click events
        for i, button in enumerate(regenerate_buttons):
            button.click(
                fn=lambda context, wf_name, api_endpoint, api_key, *inputs, step=i: process(context, wf_name,
                                                                                            api_endpoint, api_key, step,
                                                                                            *inputs),
                inputs=[context_input, workflow_selector, api_selector, api_key_input] + user_inputs,
                outputs=prompt_displays + user_inputs + output_boxes + process_buttons + regenerate_buttons
            )

    return workflow_selector, api_selector, api_key_input, context_input, dynamic_components

#
# End of script
############################################################################################################
