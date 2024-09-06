# Website_scraping_tab.py
# Gradio UI for scraping websites
import json
from typing import Optional

# Imports
#
# External Imports
import gradio as gr

from App_Function_Libraries.Article_Extractor_Lib import scrape_from_sitemap, scrape_by_url_level
from App_Function_Libraries.Article_Summarization_Lib import scrape_and_summarize_multiple
from App_Function_Libraries.DB.DB_Manager import load_preset_prompts
from App_Function_Libraries.Gradio_UI.Chat_ui import update_user_prompt


#
# Local Imports
#
#
########################################################################################################################
#
# Functions:

def create_website_scraping_tab():
    with gr.TabItem("Website Scraping"):
        gr.Markdown("# Scrape Websites & Summarize Articles using a Headless Chrome Browser!")
        with gr.Row():
            with gr.Column():
                scrape_method = gr.Radio(
                    ["Individual URLs", "Sitemap", "URL Level"],
                    label="Scraping Method",
                    value="Individual URLs"
                )
                url_input = gr.Textbox(
                    label="Article URLs or Base URL",
                    placeholder="Enter article URLs here, one per line, or base URL for sitemap/URL level scraping",
                    lines=5
                )
                url_level = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    label="URL Level (for URL Level scraping)",
                    value=2,
                    visible=False
                )
                custom_article_title_input = gr.Textbox(
                    label="Custom Article Titles (Optional, one per line)",
                    placeholder="Enter custom titles for the articles, one per line",
                    lines=5
                )
                with gr.Row():
                    custom_prompt_checkbox = gr.Checkbox(label="Use a Custom Prompt", value=False, visible=True)
                    preset_prompt_checkbox = gr.Checkbox(label="Use a pre-set Prompt", value=False, visible=True)
                with gr.Row():
                    preset_prompt = gr.Dropdown(
                        label="Select Preset Prompt",
                        choices=load_preset_prompts(),
                        visible=False
                    )
                with gr.Row():
                    website_custom_prompt_input = gr.Textbox(
                        label="Custom Prompt",
                        placeholder="Enter custom prompt here",
                        lines=3,
                        visible=False
                    )
                with gr.Row():
                    system_prompt_input = gr.Textbox(
                        label="System Prompt",
                        value="""<s>You are a bulleted notes specialist. [INST]```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.[/INST]
**Bulleted Note Creation Guidelines**

**Headings**:
- Based on referenced topics, not categories like quotes or terms
- Surrounded by **bold** formatting 
- Not listed as bullet points
- No space between headings and list items underneath

**Emphasis**:
- **Important terms** set in bold font
- **Text ending in a colon**: also bolded

**Review**:
- Ensure adherence to specified format
- Do not reference these instructions in your response.</s>[INST] {{ .Prompt }} [/INST]
""",
                        lines=3,
                        visible=False
                    )

                api_name_input = gr.Dropdown(
                    choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                             "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM","ollama", "HuggingFace"],
                    value=None,
                    label="API Name (Mandatory for Summarization)"
                )
                api_key_input = gr.Textbox(
                    label="API Key (Mandatory if API Name is specified)",
                    placeholder="Enter your API key here; Ignore if using Local API or Built-in API",
                    type="password"
                )
                keywords_input = gr.Textbox(
                    label="Keywords",
                    placeholder="Enter keywords here (comma-separated)",
                    value="default,no_keyword_set",
                    visible=True
                )

                scrape_button = gr.Button("Scrape and Summarize")
            with gr.Column():
                result_output = gr.Textbox(label="Result", lines=20)

        def update_url_input_label(method):
            if method == "Individual URLs":
                return gr.update(label="Article URLs", placeholder="Enter article URLs here, one per line")
            else:
                return gr.update(label="Base URL", placeholder="Enter the base URL for sitemap/URL level scraping")

        def toggle_url_level_visibility(method):
            return gr.update(visible=(method == "URL Level"))

        scrape_method.change(
            fn=update_url_input_label,
            inputs=[scrape_method],
            outputs=[url_input]
        )

        scrape_method.change(
            fn=toggle_url_level_visibility,
            inputs=[scrape_method],
            outputs=[url_level]
        )

        custom_prompt_checkbox.change(
            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[custom_prompt_checkbox],
            outputs=[website_custom_prompt_input, system_prompt_input]
        )
        preset_prompt_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[preset_prompt_checkbox],
            outputs=[preset_prompt]
        )

        def update_prompts(preset_name):
            prompts = update_user_prompt(preset_name)
            return (
                gr.update(value=prompts["user_prompt"], visible=True),
                gr.update(value=prompts["system_prompt"], visible=True)
            )

        preset_prompt.change(
            update_prompts,
            inputs=preset_prompt,
            outputs=[website_custom_prompt_input, system_prompt_input]
        )

        def scrape_and_summarize_wrapper(
                scrape_method: str,
                url_input: str,
                url_level: Optional[int],
                custom_prompt: Optional[str],
                api_name: Optional[str],
                api_key: Optional[str],
                keywords: str,
                custom_titles: Optional[str],
                system_prompt: Optional[str]
        ) -> str:
            """
            Wrapper function to handle different scraping methods and summarization.
            Returns markdown-formatted string of the website collection data.

            Args:
                scrape_method (str): The method of scraping ('Individual URLs', 'Sitemap', or 'URL Level')
                url_input (str): The input URL(s) or base URL
                url_level (Optional[int]): The URL level for URL Level scraping
                custom_prompt (Optional[str]): Custom prompt for summarization
                api_name (Optional[str]): Name of the API to use for summarization
                api_key (Optional[str]): API key for the chosen API
                keywords (str): Keywords for summarization
                custom_titles (Optional[str]): Custom titles for articles
                system_prompt (Optional[str]): System prompt for summarization

            Returns:
                str: Markdown-formatted string containing concatenated content and metadata for the entire website
            """
            try:
                if scrape_method == "Individual URLs":
                    result = scrape_and_summarize_multiple(url_input, custom_prompt, api_name, api_key, keywords,
                                                           custom_titles, system_prompt)
                elif scrape_method == "Sitemap":
                    result = scrape_from_sitemap(url_input)
                elif scrape_method == "URL Level":
                    if url_level is None:
                        return convert_json_to_markdown(
                            json.dumps({"error": "URL level is required for URL Level scraping."}))
                    result = scrape_by_url_level(url_input, url_level)
                else:
                    return convert_json_to_markdown(json.dumps({"error": f"Unknown scraping method: {scrape_method}"}))

                # Ensure result is always a list
                if not isinstance(result, list):
                    result = [result]

                # Concatenate all content
                all_content = "\n\n".join(
                    [f"# {article.get('title', 'Untitled')}\n\n{article.get('content', '')}" for article in result])

                # Collect all unique URLs
                all_urls = list(set(article.get('url', '') for article in result if article.get('url')))

                # Structure the output for the entire website collection
                website_collection = {
                    "base_url": url_input,
                    "scrape_method": scrape_method,
                    "api_used": api_name,
                    "keywords": keywords,
                    "url_level": url_level if scrape_method == "URL Level" else None,
                    "total_articles_scraped": len(result),
                    "urls_scraped": all_urls,
                    "content": all_content
                }

                # Convert the JSON to markdown and return
                return convert_json_to_markdown(json.dumps(website_collection, indent=2))
            except Exception as e:
                return convert_json_to_markdown(json.dumps({"error": f"An error occurred: {str(e)}"}))

        scrape_button.click(
            fn=scrape_and_summarize_wrapper,
            inputs=[scrape_method, url_input, url_level, website_custom_prompt_input, api_name_input, api_key_input,
                    keywords_input, custom_article_title_input, system_prompt_input],
            outputs=result_output
        )


def convert_json_to_markdown(json_str: str) -> str:
    """
    Converts the JSON output from the scraping process into a markdown format.

    Args:
        json_str (str): JSON-formatted string containing the website collection data

    Returns:
        str: Markdown-formatted string of the website collection data
    """
    try:
        # Parse the JSON string
        data = json.loads(json_str)

        # Check if there's an error in the JSON
        if "error" in data:
            return f"# Error\n\n{data['error']}"

        # Start building the markdown string
        markdown = f"# Website Collection: {data['base_url']}\n\n"

        # Add metadata
        markdown += "## Metadata\n\n"
        markdown += f"- **Scrape Method:** {data['scrape_method']}\n"
        markdown += f"- **API Used:** {data['api_used']}\n"
        markdown += f"- **Keywords:** {data['keywords']}\n"
        if data['url_level'] is not None:
            markdown += f"- **URL Level:** {data['url_level']}\n"
        markdown += f"- **Total Articles Scraped:** {data['total_articles_scraped']}\n\n"

        # Add URLs scraped
        markdown += "## URLs Scraped\n\n"
        for url in data['urls_scraped']:
            markdown += f"- {url}\n"
        markdown += "\n"

        # Add the content
        markdown += "## Content\n\n"
        markdown += data['content']

        return markdown

    except json.JSONDecodeError:
        return "# Error\n\nInvalid JSON string provided."
    except KeyError as e:
        return f"# Error\n\nMissing key in JSON data: {str(e)}"
    except Exception as e:
        return f"# Error\n\nAn unexpected error occurred: {str(e)}"


# Old
# def create_website_scraping_tab():
#     with gr.TabItem("Website Scraping"):
#         gr.Markdown("# Scrape Websites & Summarize Articles using a Headless Chrome Browser!")
#         with gr.Row():
#             with gr.Column():
#                 url_input = gr.Textbox(label="Article URLs", placeholder="Enter article URLs here, one per line", lines=5)
#                 custom_article_title_input = gr.Textbox(label="Custom Article Titles (Optional, one per line)",
#                                                         placeholder="Enter custom titles for the articles, one per line",
#                                                         lines=5)
#                 with gr.Row():
#                     custom_prompt_checkbox = gr.Checkbox(label="Use a Custom Prompt",
#                                                      value=False,
#                                                      visible=True)
#                     preset_prompt_checkbox = gr.Checkbox(label="Use a pre-set Prompt",
#                                                      value=False,
#                                                      visible=True)
#                 with gr.Row():
#                     preset_prompt = gr.Dropdown(label="Select Preset Prompt",
#                                                 choices=load_preset_prompts(),
#                                                 visible=False)
#                 with gr.Row():
#                     website_custom_prompt_input = gr.Textbox(label="Custom Prompt",
#                                                      placeholder="Enter custom prompt here",
#                                                      lines=3,
#                                                      visible=False)
#                 with gr.Row():
#                     system_prompt_input = gr.Textbox(label="System Prompt",
#                                                      value="""<s>You are a bulleted notes specialist. [INST]```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.[/INST]
# **Bulleted Note Creation Guidelines**
#
# **Headings**:
# - Based on referenced topics, not categories like quotes or terms
# - Surrounded by **bold** formatting
# - Not listed as bullet points
# - No space between headings and list items underneath
#
# **Emphasis**:
# - **Important terms** set in bold font
# - **Text ending in a colon**: also bolded
#
# **Review**:
# - Ensure adherence to specified format
# - Do not reference these instructions in your response.</s>[INST] {{ .Prompt }} [/INST]
# """,
#                                                      lines=3,
#                                                      visible=False)
#
#                 custom_prompt_checkbox.change(
#                     fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
#                     inputs=[custom_prompt_checkbox],
#                     outputs=[website_custom_prompt_input, system_prompt_input]
#                 )
#                 preset_prompt_checkbox.change(
#                     fn=lambda x: gr.update(visible=x),
#                     inputs=[preset_prompt_checkbox],
#                     outputs=[preset_prompt]
#                 )
#
#                 def update_prompts(preset_name):
#                     prompts = update_user_prompt(preset_name)
#                     return (
#                         gr.update(value=prompts["user_prompt"], visible=True),
#                         gr.update(value=prompts["system_prompt"], visible=True)
#                     )
#
#                 preset_prompt.change(
#                     update_prompts,
#                     inputs=preset_prompt,
#                     outputs=[website_custom_prompt_input, system_prompt_input]
#                 )
#
#                 api_name_input = gr.Dropdown(
#                     choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
#                              "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM","ollama", "HuggingFace"], value=None, label="API Name (Mandatory for Summarization)")
#                 api_key_input = gr.Textbox(label="API Key (Mandatory if API Name is specified)",
#                                            placeholder="Enter your API key here; Ignore if using Local API or Built-in API", type="password")
#                 keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated)",
#                                             value="default,no_keyword_set", visible=True)
#
#                 scrape_button = gr.Button("Scrape and Summarize")
#             with gr.Column():
#                 result_output = gr.Textbox(label="Result", lines=20)
#
#                 scrape_button.click(
#                     fn=scrape_and_summarize_multiple,
#                     inputs=[url_input, website_custom_prompt_input, api_name_input, api_key_input, keywords_input,
#                             custom_article_title_input, system_prompt_input],
#                     outputs=result_output
#                 )


