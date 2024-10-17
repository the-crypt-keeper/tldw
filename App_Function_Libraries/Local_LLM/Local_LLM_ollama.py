import platform

import gradio as gr
import subprocess
import psutil
import os
import signal


def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        models = result.stdout.strip().split('\n')[1:]  # Skip header
        return [model.split()[0] for model in models]
    except subprocess.CalledProcessError:
        return []


def pull_ollama_model(model_name):
    try:
        subprocess.run(['ollama', 'pull', model_name], check=True)
        return f"Successfully pulled model: {model_name}"
    except subprocess.CalledProcessError as e:
        return f"Failed to pull model: {e}"


def serve_ollama_model(model_name, port):
    try:
        # Check if a server is already running on the specified port
        for conn in psutil.net_connections():
            if conn.laddr.port == int(port):
                return f"Port {port} is already in use. Please choose a different port."

        # Start the Ollama server
        port = str(port)
        os.environ["OLLAMA_HOST"] = port
        cmd = f"ollama serve"
        process = subprocess.Popen(cmd, shell=True)
        return f"Started Ollama server for model {model_name} on port {port}. Process ID: {process.pid}"
    except Exception as e:
        return f"Error starting Ollama server: {e}"


def stop_ollama_server(pid):
    try:
        if platform.system() == "Windows":
            os.system(f"taskkill /F /PID {pid}")
            return f"Stopped Ollama server with PID {pid}"
        elif platform.system() == "Linux":
            os.system(f"kill {pid}")
            return f"Stopped Ollama server with PID {pid}"
        elif platform.system() == "Darwin":
            os.system("""osascript -e 'tell app "Ollama" to quit'""")
            return f"(Hopefully) Stopped Ollama server using osascript..."
    except ProcessLookupError:
        return f"No process found with PID {pid}"
    except Exception as e:
        return f"Error stopping Ollama server: {e}"


def create_ollama_tab():
    with gr.Tab("Ollama Model Serving"):
        gr.Markdown("# Ollama Model Serving")

        with gr.Row():
            model_list = gr.Dropdown(label="Available Models", choices=get_ollama_models())
            refresh_button = gr.Button("Refresh Model List")

        with gr.Row():
            new_model_name = gr.Textbox(label="Model to Pull")
            pull_button = gr.Button("Pull Model")

        pull_output = gr.Textbox(label="Pull Status")

        with gr.Row():
            # FIXME - Update to update config.txt file
            serve_model = gr.Dropdown(label="Model to Serve", choices=get_ollama_models())
            port = gr.Number(label="Port", value=11434, precision=0)
            serve_button = gr.Button("Start Server")

        serve_output = gr.Textbox(label="Server Status")

        with gr.Row():
            pid = gr.Number(label="Server Process ID", precision=0)
            stop_button = gr.Button("Stop Server")

        stop_output = gr.Textbox(label="Stop Status")

        def update_model_lists():
            models = get_ollama_models()
            return gr.update(choices=models), gr.update(choices=models)

        refresh_button.click(update_model_lists, outputs=[model_list, serve_model])
        pull_button.click(pull_ollama_model, inputs=[new_model_name], outputs=[pull_output])
        serve_button.click(serve_ollama_model, inputs=[serve_model, port], outputs=[serve_output])
        stop_button.click(stop_ollama_server, inputs=[pid], outputs=[stop_output])