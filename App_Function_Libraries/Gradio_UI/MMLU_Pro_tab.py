# MMLU_Pro_tab.py
# is a library that contains the Gradio UI code for the MMLU-Pro benchmarking tool.
#
##############################################################################################################
# Imports
import gradio as gr
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
#
# External Imports
from tqdm import tqdm
# Local Imports
from App_Function_Libraries.Benchmarks_Evaluations.MMLU_Pro.MMLU_Pro_rewritten import (
    load_mmlu_pro_config, initialize_client, load_mmlu_pro, run_single_question,
    process_and_save_results, generate_final_report
)
#
##############################################################################################################
#
# Functions:

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_categories():
    try:
        test_data, _ = load_mmlu_pro()
        return list(test_data.keys())
    except Exception as e:
        logger.error(f"Failed to load categories: {e}")
        return ["Error loading categories"]

def load_categories():
    try:
        test_data, _ = load_mmlu_pro()
        categories = list(test_data.keys())
        return categories, gr.Dropdown(choices=categories, label="Category")
    except Exception as e:
        logger.error(f"Failed to load categories: {e}")
        return ["Error loading categories"], gr.Dropdown(choices=["Error loading categories"], label="Category")

def run_benchmark(config_file, url, api_key, model, timeout, category, parallel, verbosity, log_prompt, categories):
    if category == "Load categories...":
        return "Please load and select a category before running the benchmark."

    try:
        # Load and update configuration
        config = load_mmlu_pro_config(
            url=url,
            api_key=api_key,
            model=model,
            timeout=float(timeout),
            categories=[category] if category else None,
            parallel=int(parallel),
            verbosity=int(verbosity),
            log_prompt=log_prompt
        )

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

                for future, question in futures:
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

def create_mmlu_pro_tab():
    def get_categories():
        try:
            test_data, _ = load_mmlu_pro()
            return list(test_data.keys())
        except Exception as e:
            logger.error(f"Failed to load categories: {e}")
            return ["Error loading categories"]

    with gr.Tab("MMLU-Pro Benchmark"):
        gr.Markdown("## Run MMLU-Pro Benchmark")

        with gr.Row():
            with gr.Column():
                url = gr.Textbox(label="Server URL")
                api_key = gr.Textbox(label="API Key", type="password")
                model = gr.Textbox(label="Model Name")
                timeout = gr.Number(label="Timeout (seconds)", value=30)
                category = gr.Dropdown(label="Category", choices=["Load categories..."])
                load_categories_btn = gr.Button("Load Categories")
                parallel = gr.Slider(label="Parallel Requests", minimum=1, maximum=10, step=1, value=1)
                verbosity = gr.Slider(label="Verbosity Level", minimum=0, maximum=2, step=1, value=1)
                log_prompt = gr.Checkbox(label="Log Prompt")

            with gr.Column():
                run_button = gr.Button("Run Benchmark")
                output = gr.Textbox(label="Benchmark Results", lines=20)

        def load_categories():
            new_categories = get_categories()
            return gr.update(choices=new_categories)

        load_categories_btn.click(
            load_categories,
            outputs=category
        )

        run_button.click(
            run_benchmark,
            inputs=[url, api_key, model, timeout, category, parallel, verbosity, log_prompt],
            outputs=output
        )

    return [url, api_key, model, timeout, category, parallel, verbosity, log_prompt, run_button, output]
