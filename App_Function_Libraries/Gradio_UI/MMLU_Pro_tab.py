import threading
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import toml
import logging
import os

from tqdm import tqdm

from App_Function_Libraries.Benchmarks_Evaluations.MMLU_Pro.MMLU_Pro_rewritten import (
    load_config, initialize_client, load_mmlu_pro, run_single_question,
    process_and_save_results, generate_final_report
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_benchmark(config_file, url, api_key, model, timeout, category, parallel, verbosity, log_prompt):
    try:
        # Load and update configuration
        config = load_config()
        config["server"]["url"] = url
        config["server"]["api_key"] = api_key
        config["server"]["model"] = model
        config["server"]["timeout"] = float(timeout)
        config["test"]["categories"] = [category] if category else config["test"]["categories"]
        config["test"]["parallel"] = int(parallel)
        config["log"]["verbosity"] = int(verbosity)
        config["log"]["log_prompt"] = log_prompt

        # Initialize client
        client = initialize_client(config)

        # Load MMLU-Pro dataset
        test_data, dev_data = load_mmlu_pro()
        if test_data is None or dev_data is None:
            return "Failed to load dataset. Please check the logs for more information."

        output_dir = os.path.join("eval_results", config["server"]["model"].replace("/", "-"))
        os.makedirs(output_dir, exist_ok=True)

        results = []
        category_record = {}
        lock = threading.Lock()

        # Run evaluation
        for subject in config["test"]["categories"]:
            logger.info(f"Processing subject: {subject}")
            questions = test_data[subject]
            cot_examples = dev_data[subject]

            with ThreadPoolExecutor(max_workers=config["test"]["parallel"]) as executor:
                futures = []
                for question in questions:
                    future = executor.submit(run_single_question, question, cot_examples, client, config)
                    futures.append((future, question))

                for future, question in tqdm(futures, total=len(futures)):
                    prompt, response, pred, usage = future.result()
                    results, category_record = process_and_save_results(
                        question, pred, client, config, results, category_record, output_dir, lock
                    )

        # Generate final report
        generate_final_report(category_record, output_dir)

        # Read and return the report
        report_path = os.path.join(output_dir, "final_report.txt")
        with open(report_path, "r") as f:
            report = f.read()

        return report

    except Exception as e:
        logger.error(f"An error occurred during benchmark execution: {e}")
        return f"An error occurred during benchmark execution. Please check the logs for more information. Error: {str(e)}"

def list_categories():
    try:
        test_data, _ = load_mmlu_pro()
        return list(test_data.keys())
    except Exception as e:
        logger.error(f"Failed to load categories: {e}")
        return []

def create_mmlu_pro_tab():
    with gr.Tab("MMLU-Pro Benchmark"):
        gr.Markdown("## Run MMLU-Pro Benchmark")

        with gr.Row():
            with gr.Column():
                config_file = gr.Textbox(label="Config File", value="config.toml")
                url = gr.Textbox(label="Server URL")
                api_key = gr.Textbox(label="API Key", type="password")
                model = gr.Textbox(label="Model Name")
                timeout = gr.Number(label="Timeout (seconds)", value=30)
                category = gr.Dropdown(label="Category", choices=list_categories())
                parallel = gr.Slider(label="Parallel Requests", minimum=1, maximum=10, step=1, value=1)
                verbosity = gr.Slider(label="Verbosity Level", minimum=0, maximum=2, step=1, value=1)
                log_prompt = gr.Checkbox(label="Log Prompt")

            with gr.Column():
                run_button = gr.Button("Run Benchmark")
                output = gr.Textbox(label="Benchmark Results", lines=20)

        run_button.click(
            run_benchmark,
            inputs=[config_file, url, api_key, model, timeout, category, parallel, verbosity, log_prompt],
            outputs=output
        )

    return [config_file, url, api_key, model, timeout, category, parallel, verbosity, log_prompt, run_button, output]