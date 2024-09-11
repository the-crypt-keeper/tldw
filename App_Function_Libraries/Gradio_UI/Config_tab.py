import gradio as gr
import configparser
import os

CONFIG_PATH = './Config_Files/config.txt'


def load_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config


def save_config(config):
    with open(CONFIG_PATH, 'w') as configfile:
        config.write(configfile)


def update_config(section, key, value):
    config = load_config()
    if section not in config:
        config[section] = {}
    config[section][key] = value
    save_config(config)
    return f"Updated {section}.{key} to {value}"


def get_config_as_text():
    with open(CONFIG_PATH, 'r') as file:
        return file.read()


def save_config_from_text(text):
    with open(CONFIG_PATH, 'w') as file:
        file.write(text)
    return "Config saved successfully"


def create_config_editor_tab():
    with gr.TabItem("Edit Config"):
        gr.Markdown("# Edit Configuration File")

        with gr.Row():
            with gr.Column():
                section_input = gr.Dropdown(choices=[], label="Section")
                key_input = gr.Dropdown(choices=[], label="Key")
                value_input = gr.Textbox(label="Value")
                update_button = gr.Button("Update Config")

            with gr.Column():
                config_text = gr.TextArea(label="Full Config", lines=30)
                save_text_button = gr.Button("Save Config")

        output = gr.Textbox(label="Output")

        def update_sections(dummy):
            config = load_config()
            return gr.Dropdown.update(choices=list(config.sections()))

        def update_keys(section):
            config = load_config()
            if section in config:
                return gr.Dropdown.update(choices=list(config[section].keys()))
            return gr.Dropdown.update(choices=[])

        def update_value(section, key):
            config = load_config()
            if section in config and key in config[section]:
                return config[section][key]
            return ""

        # Event handlers
        section_input.change(update_keys, inputs=[section_input], outputs=[key_input])
        key_input.change(update_value, inputs=[section_input, key_input], outputs=[value_input])
        update_button.click(update_config, inputs=[section_input, key_input, value_input], outputs=[output])

        config_text.change(lambda: None, None, None)  # Dummy handler to enable changes
        save_text_button.click(save_config_from_text, inputs=[config_text], outputs=[output])

        # Initialize the interface
        section_input.change(None, None, None)  # Trigger the section dropdown population
        config_text.value = get_config_as_text()

    return section_input, key_input, value_input, update_button, config_text, save_text_button, output
