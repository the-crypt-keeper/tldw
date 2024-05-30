import gradio as gr

def greet(name):
    return f"Hello {name}!"

theme = gr.Theme.from_hub("gradio/seafoam")  # Ensure this theme is correctly loaded

with gr.Blocks(theme=theme) as demo:
    with gr.Tab("Greeting"):
        name_input = gr.Textbox(label="Enter your name")
        greet_button = gr.Button("Greet")
        output = gr.Textbox()
        greet_button.click(greet, inputs=name_input, outputs=output)

demo.launch()