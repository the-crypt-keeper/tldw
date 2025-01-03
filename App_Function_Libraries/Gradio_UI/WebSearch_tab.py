# WebSearch_tab.py
# Gradio UI for performing web searches with aggregated results
#
# Imports
import asyncio
import logging
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.Web_Scraping.WebSearch_APIs import generate_and_search, analyze_and_aggregate
#
########################################################################################################################
#
# Functions:

def create_websearch_tab():
    with gr.TabItem("Web Search & Review", visible=True):
        with gr.Blocks() as perplexity_interface:
            # Add CSS
            gr.HTML("""
                <style>
                    .status-display {
                        padding: 10px;
                        border-radius: 5px;
                        margin: 10px 0;
                    }
                    .status-normal { background-color: #f0f0f0; }
                    .status-processing { background-color: #fff3cd; }
                    .status-error { background-color: #f8d7da; }
                    .status-success { background-color: #d4edda; }
                    
                    .result-card {
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 15px;
                        margin: 10px 0;
                        background-color: white;
                    }
                    
                    .result-title {
                        font-size: 1.2em;
                        font-weight: bold;
                        margin-bottom: 8px;
                    }
                    
                    .result-url {
                        color: #0066cc;
                        margin-bottom: 8px;
                        word-break: break-all;
                    }
                    
                    .result-preview {
                        color: #666;
                        margin-top: 8px;
                    }
                </style>
            """)

            gr.Markdown("# Web Search Interface")

            # State for managing the review process
            state = gr.State({
                "phase1_results": None,
                "search_params": None,
                "selected_indices": []
            })

            with gr.Row():
                with gr.Column():
                    # Input components
                    question_input = gr.Textbox(
                        label="Enter your question",
                        placeholder="What would you like to know?",
                        lines=2
                    )

                    with gr.Row():
                        search_engine = gr.Dropdown(
                            choices=["google", "bing", "duckduckgo", "brave"],
                            value="google",
                            label="Search Engine"
                        )
                        result_count = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Number of Results"
                        )

                    with gr.Row():
                        content_country = gr.Dropdown(
                            choices=["US", "UK", "CA", "AU"],
                            value="US",
                            label="Content Country"
                        )
                        search_lang = gr.Dropdown(
                            choices=["en", "es", "fr", "de"],
                            value="en",
                            label="Search Language"
                        )

            # Status and progress displays
            with gr.Row():
                search_btn = gr.Button("Search", variant="primary")
                status_display = gr.Markdown("Ready", elem_classes=["status-display", "status-normal"])
                progress_display = gr.HTML(visible=False)

            # Results review section
            with gr.Column(visible=False) as review_column:
                gr.Markdown("### Search Results")
                results_container = gr.HTML()  # Container for results
                confirm_selection_btn = gr.Button("Generate Answer from Selected Results")

            # Final output section
            with gr.Column(visible=False) as output_column:
                answer_output = gr.Markdown(label="Generated Answer")
                sources_output = gr.JSON(label="Sources")

            def update_status(message, status_type="normal"):
                """Update the status display with the given message and type."""
                status_classes = {
                    "normal": "status-normal",
                    "processing": "status-processing",
                    "error": "status-error",
                    "success": "status-success"
                }
                return (
                    gr.Markdown(value=message, elem_classes=["status-display", status_classes[status_type]]),
                    gr.HTML(visible=(status_type == "processing"))
                )

            def format_results_html(results):
                """Format search results as HTML with checkboxes."""
                html = ""
                for idx, result in enumerate(results):
                    html += f"""
                    <div class="result-card">
                        <div class="result-checkbox">
                            <input type="checkbox" id="result-{idx}" checked>
                        </div>
                        <div class="result-title">{result.get('title', 'No title')}</div>
                        <div class="result-url">{result.get('url', 'No URL')}</div>
                        <div class="result-preview">{result.get('content', 'No content')[:200]}...</div>
                    </div>
                    """
                return html

            def initial_search(question, engine, count, country, lang, state):
                try:
                    status, progress = update_status("Initializing search...", "processing")
                    yield status, progress, state, "", gr.Column(visible=False), gr.Column(visible=False)

                    search_params = {
                        "engine": engine,
                        "content_country": country,
                        "search_lang": lang,
                        "output_lang": lang,
                        "result_count": count
                    }

                    # Generate and search
                    phase1_results = generate_and_search(question, search_params)

                    # Get results list
                    results_list = phase1_results["web_search_results_dict"]["results"]

                    # Format results as HTML
                    results_html = format_results_html(results_list)

                    # Update state with the list indices
                    state = {
                        "phase1_results": phase1_results,
                        "search_params": search_params,
                        "selected_indices": list(range(len(results_list)))
                    }

                    logging.info(f"Search completed. Results count: {len(results_list)}")
                    logging.info(f"Selected indices: {state['selected_indices']}")

                    status, progress = update_status("Search completed successfully!", "success")
                    yield status, progress, state, results_html, gr.Column(visible=True), gr.Column(visible=False)

                except Exception as e:
                    error_message = f"Error during search: {str(e)}"
                    logging.error(f"Search error: {error_message}")
                    logging.error("Traceback: ", exc_info=True)
                    status, progress = update_status(error_message, "error")
                    yield status, progress, state, "", gr.Column(visible=False), gr.Column(visible=False)

            def generate_final_answer(state):
                try:
                    status, progress = update_status("Generating final answer...", "processing")
                    yield status, progress, "Processing...", {}, gr.Column(visible=False)

                    if not state["phase1_results"] or not state["search_params"]:
                        raise ValueError("No search results available")

                    # Get selected results
                    filtered_results = {}
                    web_search_results = state["phase1_results"]["web_search_results_dict"]["results"]

                    logging.info(f"Processing web search results: {web_search_results}")
                    logging.info(f"Selected indices: {state['selected_indices']}")

                    # Convert list results to dictionary format expected by analyze_and_aggregate
                    for idx in state["selected_indices"]:
                        if idx < len(web_search_results):
                            result = web_search_results[idx]
                            filtered_results[str(idx)] = {
                                'content': result.get('content', ''),
                                'url': result.get('url', ''),
                                'title': result.get('title', ''),
                                'metadata': result.get('metadata', {})
                            }

                    if not filtered_results:
                        raise ValueError("No results selected")

                    logging.info(f"Filtered results prepared for analysis: {filtered_results}")

                    # Create the input structure expected by analyze_and_aggregate
                    input_data = {
                        "results": filtered_results
                    }

                    # Generate final answer
                    phase2_results = asyncio.run(analyze_and_aggregate(
                        input_data,
                        state["phase1_results"].get("sub_query_dict", {}),
                        state["search_params"]
                    ))

                    status, progress = update_status("Answer generated successfully!", "success")
                    yield status, progress, phase2_results["final_answer"]["Report"], phase2_results["final_answer"][
                        "evidence"], gr.Column(visible=True)

                except Exception as e:
                    error_message = f"Error generating answer: {str(e)}"
                    logging.error(f"Error in generate_final_answer: {error_message}")
                    logging.error(f"State: {state}")
                    logging.error(f"Traceback: ", exc_info=True)  # Add full traceback
                    status, progress = update_status(error_message, "error")
                    yield status, progress, "Error occurred while generating answer", {}, gr.Column(visible=False)

            # Connect event handlers
            search_btn.click(
                fn=initial_search,
                inputs=[
                    question_input,
                    search_engine,
                    result_count,
                    content_country,
                    search_lang,
                    state
                ],
                outputs=[
                    status_display,
                    progress_display,
                    state,
                    results_container,
                    review_column,
                    output_column
                ]
            )

            confirm_selection_btn.click(
                fn=generate_final_answer,
                inputs=[state],
                outputs=[
                    status_display,
                    progress_display,
                    answer_output,
                    sources_output,
                    output_column
                ]
            )

    return perplexity_interface

#
# End of File
########################################################################################################################
