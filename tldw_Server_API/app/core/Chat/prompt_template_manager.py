# File: /tldw_Server_API/app/core/Chat/prompt_template_manager.py
#
# Imports
import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

from jinja2.sandbox import SandboxedEnvironment
#
# Third-party Libraries
from pydantic import BaseModel, Field
from loguru import logger
#
# Local Imports
#
########################################################################################################################
#
# Constants:
#
# Securely define the templates directory with validation
_BASE_DIR = Path(__file__).parent.resolve()
PROMPT_TEMPLATES_DIR = (_BASE_DIR / "prompt_templates").resolve()

# Ensure the templates directory exists and is a directory
if PROMPT_TEMPLATES_DIR.exists() and not PROMPT_TEMPLATES_DIR.is_dir():
    raise RuntimeError(f"Expected directory but found file at: {PROMPT_TEMPLATES_DIR}")

# Create the directory if it doesn't exist
PROMPT_TEMPLATES_DIR.mkdir(exist_ok=True)
#
#
# Functions:

class PromptTemplatePlaceholders(BaseModel):
    system: Optional[List[str]] = None
    user: Optional[List[str]] = None
    assistant: Optional[List[str]] = None

class PromptTemplate(BaseModel):
    name: str
    description: Optional[str] = None
    system_message_template: Optional[str] = None
    user_message_content_template: str = "{message_content}" # Default passthrough
    assistant_message_content_template: Optional[str] = None
    placeholders: Optional[PromptTemplatePlaceholders] = None

_loaded_templates: Dict[str, PromptTemplate] = {}

def load_template(template_name: str) -> Optional[PromptTemplate]:
    """Loads a single prompt template from a JSON file.
    
    Security: Validates template name to prevent path traversal attacks.
    """
    # Check cache first
    if template_name in _loaded_templates:
        return _loaded_templates[template_name]
    
    # Security validation: Only allow alphanumeric, underscore, and hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', template_name):
        logger.warning(f"Invalid template name format attempted: {template_name}")
        return None
    
    # Additional security: Check for path traversal attempts
    if '/' in template_name or '\\' in template_name or '..' in template_name:
        logger.warning(f"Potential path traversal attempt detected in template name: {template_name}")
        return None
    
    # Construct the path
    template_file = PROMPT_TEMPLATES_DIR / f"{template_name}.json"

    # Security: Resolve and validate the path is within expected directory
    try:
        resolved_path = template_file.resolve()
        expected_dir = PROMPT_TEMPLATES_DIR.resolve()
        
        # Check if the resolved path is within the templates directory using commonpath
        if os.path.commonpath([str(resolved_path), str(expected_dir)]) != str(expected_dir):
            logger.warning(f"Path traversal attempt blocked - resolved path outside template directory: {template_name}")
            return None
    except (ValueError, OSError) as e:
        logger.warning(f"Invalid path resolution for template name: {template_name}, error: {e}")
        return None
    
    if not resolved_path.exists():
        logger.warning(f"Prompt template '{template_name}' not found at {resolved_path}")
        return None
    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            template = PromptTemplate(**data)
            _loaded_templates[template_name] = template
            logger.info(f"Successfully loaded prompt template: {template_name}")
            return template
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON for template: {template_name}")
    except Exception as e:
        error_message_str = str(e)
        logger.error("Error loading template {}: {}", template_name, error_message_str, exc_info=True)
    return None


_SANDBOX = SandboxedEnvironment(
    autoescape=True,                # HTML-safe by default
    enable_async=False,             # no await, no async callables
)

def safe_render(template_str: str, data: dict[str, Any]) -> str:
    """Render with a locked-down Jinja sandbox."""
    try:
        tmpl = _SANDBOX.from_string(template_str)
        return tmpl.render(**data)
    except Exception as exc:
        logger.error("Template render error %s", exc, exc_info=False)
        return template_str      # fail closed: return raw


def apply_template_to_string(template_string: Optional[str], data: Dict[str, Any]) -> Optional[str]:
    """
    Applies data to a template string using Jinja2 safe rendering.
    Missing placeholders will typically render as empty strings by Jinja2 default.
    """
    if template_string is None:
        return "" # Returns an empty string if the template_string itself is None
    try:
        # The original was: template_string = safe_render(template_string, data)
        # This needs to assign to a new variable and return it.
        rendered_string = safe_render(template_string, data)
        return rendered_string
    except KeyError as e: # This exception type might not be commonly raised by Jinja's render for missing vars
        logger.warning(f"Placeholder {e} not found in data for template string: '{template_string}'")
        return template_string # Fallback to original
    except Exception as e:
        logger.error(f"Error applying template string '{template_string}': {e}", exc_info=True)
        return template_string # Return original on error


def get_available_templates() -> List[str]:
    """Returns a list of available template names (without .json extension).
    
    Security: Only returns files that are actually within the templates directory.
    """
    if not PROMPT_TEMPLATES_DIR.exists():
        return []
    
    templates = []
    expected_dir = PROMPT_TEMPLATES_DIR.resolve()
    
    for f in PROMPT_TEMPLATES_DIR.glob("*.json"):
        try:
            # Verify each file is actually within the templates directory
            file_path = f.resolve()
            if os.path.commonpath([str(file_path), str(expected_dir)]) == str(expected_dir):
                templates.append(f.stem)
            else:
                logger.warning(f"Skipping file outside templates directory: {f}")
        except (ValueError, OSError) as e:
            logger.warning(f"Error processing template file {f}: {e}")
    
    return templates

# Load a default passthrough template on module load for safety
DEFAULT_RAW_PASSTHROUGH_TEMPLATE = PromptTemplate(
    name="raw_passthrough",
    description="Default template that makes no changes to the prompts.",
    system_message_template="{{original_system_message_from_request}}",
    user_message_content_template="{{message_content}}",
    assistant_message_content_template="{{message_content}}"
)
_loaded_templates["raw_passthrough"] = DEFAULT_RAW_PASSTHROUGH_TEMPLATE

logger.info(f"Prompt templates directory: {PROMPT_TEMPLATES_DIR.resolve()}")
logger.info(f"Available templates found: {get_available_templates()}")

#
# End of prompt_template_manager.py
#######################################################################################################################
