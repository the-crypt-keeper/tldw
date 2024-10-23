# XML_Ingestion.py
import logging
import os
import xml.etree.ElementTree as ET
import gradio as gr

from App_Function_Libraries.Gradio_UI.Import_Functionality import import_data
from App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name


def xml_to_text(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Extract text content recursively
        text_content = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text_content.append(elem.text.strip())
        return '\n'.join(text_content)
    except ET.ParseError as e:
        logging.error(f"Error parsing XML file: {str(e)}")
        return None


def import_xml_handler(import_file, title, author, keywords, system_prompt,
                       custom_prompt, auto_summarize, api_name, api_key):
    if not import_file:
        return "Please upload an XML file"

    try:
        xml_text = xml_to_text(import_file.name)
        if not xml_text:
            return "Failed to extract text from XML file"

        # Use your existing import_data function
        result = import_data(xml_text, title, author, keywords, system_prompt,
                             custom_prompt, auto_summarize, api_name, api_key)
        return result
    except Exception as e:
        logging.error(f"Error processing XML file: {str(e)}")
        return f"Error processing XML file: {str(e)}"

#
# End of XML_Ingestion_Lib.py
#######################################################################################################################
