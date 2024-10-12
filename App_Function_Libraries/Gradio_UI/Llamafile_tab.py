# Llamafile_tab.py
# Description: Functions relating to the Llamafile tab
#
# Imports
import os
import glob
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.Local_LLM.Llamafile import start_llamafile
#
#######################################################################################################################
#
# Functions:


def create_chat_with_llamafile_tab():
    def get_model_files(directory):
        pattern = os.path.join(directory, "*.{gguf,llamafile}")
        return [os.path.basename(f) for f in glob.glob(pattern)]

    def update_dropdowns():
        current_dir_models = get_model_files(".")
        parent_dir_models = get_model_files("..")
        return (
            {"choices": current_dir_models, "value": None},
            {"choices": parent_dir_models, "value": None}
        )

    with gr.TabItem("Local LLM with Llamafile"):
        gr.Markdown("# Settings for Llamafile")
        with gr.Row():
            with gr.Column():
                am_noob = gr.Checkbox(label="Check this to enable sane defaults", value=False, visible=True)
                # FIXME - these get deleted at some point?
                advanced_mode_toggle = gr.Checkbox(label="Advanced Mode - Enable to show all settings", value=False)


            with gr.Column():
                # FIXME - make this actually work
                model_checked = gr.Checkbox(label="Enable Setting Local LLM Model Path", value=False, visible=True)
                current_dir_dropdown = gr.Dropdown(
                    label="Select Model from Current Directory (.)",
                    choices=[],  # Start with an empty list
                    visible=True
                )
                parent_dir_dropdown = gr.Dropdown(
                    label="Select Model from Parent Directory (..)",
                    choices=[],  # Start with an empty list
                    visible=True
                )
                refresh_button = gr.Button("Refresh Model Lists")
                model_value = gr.Textbox(label="Selected Model File", value="", visible=True)
        with gr.Row():
            with gr.Column():
                ngl_checked = gr.Checkbox(label="Enable Setting GPU Layers", value=False, visible=True)
                ngl_value = gr.Number(label="Number of GPU Layers", value=None, precision=0, visible=True)
                advanced_inputs = create_llamafile_advanced_inputs()
            with gr.Column():
                start_button = gr.Button("Start Llamafile")
                stop_button = gr.Button("Stop Llamafile (doesn't work)")
                output_display = gr.Markdown()


        def update_model_value(current_dir_model, parent_dir_model):
            if current_dir_model:
                return current_dir_model
            elif parent_dir_model:
                return os.path.join("..", parent_dir_model)
            else:
                return ""

        current_dir_dropdown.change(
            fn=update_model_value,
            inputs=[current_dir_dropdown, parent_dir_dropdown],
            outputs=model_value
        )
        parent_dir_dropdown.change(
            fn=update_model_value,
            inputs=[current_dir_dropdown, parent_dir_dropdown],
            outputs=model_value
        )

        refresh_button.click(
            fn=update_dropdowns,
            inputs=[],
            outputs=[current_dir_dropdown, parent_dir_dropdown]
        )

        start_button.click(
            fn=start_llamafile,
            inputs=[am_noob, model_checked, model_value, ngl_checked, ngl_value] + advanced_inputs,
            outputs=output_display
        )


def create_llamafile_advanced_inputs():
    verbose_checked = gr.Checkbox(label="Enable Verbose Output", value=False, visible=False)
    threads_checked = gr.Checkbox(label="Set CPU Threads", value=False, visible=False)
    threads_value = gr.Number(label="Number of CPU Threads", value=None, precision=0, visible=False)
    http_threads_checked = gr.Checkbox(label="Set HTTP Server Threads", value=False, visible=False)
    http_threads_value = gr.Number(label="Number of HTTP Server Threads", value=None, precision=0, visible=False)
    hf_repo_checked = gr.Checkbox(label="Use Huggingface Repo Model", value=False, visible=False)
    hf_repo_value = gr.Textbox(label="Huggingface Repo Name", value="", visible=False)
    hf_file_checked = gr.Checkbox(label="Set Huggingface Model File", value=False, visible=False)
    hf_file_value = gr.Textbox(label="Huggingface Model File", value="", visible=False)
    ctx_size_checked = gr.Checkbox(label="Set Prompt Context Size", value=False, visible=False)
    ctx_size_value = gr.Number(label="Prompt Context Size", value=8124, precision=0, visible=False)
    host_checked = gr.Checkbox(label="Set IP to Listen On", value=False, visible=False)
    host_value = gr.Textbox(label="Host IP Address", value="", visible=False)
    port_checked = gr.Checkbox(label="Set Server Port", value=False, visible=False)
    port_value = gr.Number(label="Port Number", value=None, precision=0, visible=False)

    return [verbose_checked, threads_checked, threads_value, http_threads_checked, http_threads_value,
            hf_repo_checked, hf_repo_value, hf_file_checked, hf_file_value, ctx_size_checked, ctx_size_value,
            host_checked, host_value, port_checked, port_value]

#
# End of Llamafile_tab.py
#########################################################################################################################