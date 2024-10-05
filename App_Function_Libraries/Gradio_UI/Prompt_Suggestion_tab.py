# Description: Gradio UI for Creating and Testing new Prompts
#
# Imports
import gradio as gr

from App_Function_Libraries.Chat import chat
from App_Function_Libraries.Prompt_Engineering.Prompt_Engineering import generate_prompt, test_generated_prompt


#
# Local Imports

#
########################################################################################################################
#
# Functions

# Gradio tab for prompt suggestion and testing
def create_prompt_suggestion_tab():
    with gr.TabItem("Prompt Suggestion/Creation"):
        gr.Markdown("# Generate and Test AI Prompts with the Metaprompt Approach")

        with gr.Row():
            with gr.Column():
                # Task and variable inputs
                task_input = gr.Textbox(label="Task Description",
                                        placeholder="E.g., Draft an email responding to a customer complaint")
                variables_input = gr.Textbox(label="Variables (comma-separated)",
                                             placeholder="E.g., CUSTOMER_COMPLAINT, COMPANY_NAME")

                # API-related inputs
                api_name_input = gr.Dropdown(
                    choices=["OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter", "Llama.cpp",
                             "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace", "Custom-OpenAI-API"],
                    label="API Provider",
                    value="OpenAI"  # Default selection
                )

                api_key_input = gr.Textbox(label="API Key", placeholder="Enter your API key (if required)",
                                           type="password")

                # Temperature slider for controlling randomness of generation
                temperature_input = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="Temperature")

                # Button to generate the prompt
                generate_prompt_button = gr.Button("Generate Prompt")

            with gr.Column():
                # Output for the generated prompt
                generated_prompt_output = gr.Textbox(label="Generated Prompt", interactive=False)

        # Section to test the generated prompt
        with gr.Row():
            with gr.Column():
                # Input to test the prompt with variable values
                variable_values_input = gr.Textbox(label="Variable Values (comma-separated)",
                                                   placeholder="Enter variable values in order, comma-separated")
                test_prompt_button = gr.Button("Test Generated Prompt")
            with gr.Column():
                # Output for the test result
                test_output = gr.Textbox(label="Test Output", interactive=False)

        # Callback function to generate prompt
        def on_generate_prompt(api_name, api_key, task, variables, temperature):
            # Generate the prompt using the metaprompt approach and API
            generated_prompt = generate_prompt(api_name, api_key, task, variables, temperature)
            return generated_prompt

        # Callback function to test the generated prompt
        def on_test_prompt(api_name, api_key, generated_prompt, variable_values, temperature):
            # Test the prompt by filling in variable values
            test_result = test_generated_prompt(api_name, api_key, generated_prompt, variable_values, temperature)
            return test_result

        # Connect the button to the function that generates the prompt
        generate_prompt_button.click(
            fn=on_generate_prompt,
            inputs=[api_name_input, api_key_input, task_input, variables_input, temperature_input],
            outputs=[generated_prompt_output]
        )

        # Connect the button to the function that tests the generated prompt
        test_prompt_button.click(
            fn=on_test_prompt,
            inputs=[api_name_input, api_key_input, generated_prompt_output, variable_values_input, temperature_input],
            outputs=[test_output]
        )


# Example chat function based on your API structure
def chat_api_call(api_endpoint, api_key, input_data, prompt, temp, system_message=None):
    # Here you will call your chat function as defined previously
    response = chat(message=input_data, history=[], media_content={}, selected_parts=[],
                    api_endpoint=api_endpoint, api_key=api_key, prompt=prompt, temperature=temp,
                    system_message=system_message)
    return response
#
# End of Functions
########################################################################################################################
