# config_info.py - Endpoint for serving configuration information for documentation
"""
API endpoint to provide configuration information for auto-populating documentation.
Only exposes non-sensitive configuration suitable for documentation.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
import configparser
import os
from pathlib import Path

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_user
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from loguru import logger

router = APIRouter()


def get_config_path() -> Path:
    """Get the path to the config file."""
    # Try environment variable first
    config_path = os.getenv("TLDW_CONFIG_PATH")
    if config_path:
        return Path(config_path)
    
    # Default location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    return Path(project_root) / "Config_Files" / "config.txt"


def load_safe_config() -> Dict:
    """
    Load configuration that is safe to expose for documentation.
    Excludes sensitive information like actual API keys.
    """
    config_path = get_config_path()
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}")
        return {
            "configured": False,
            "message": "Configuration file not found"
        }
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Get authentication mode
    auth_mode = config.get('Authentication', 'auth_mode', fallback='single_user')
    
    # Determine what to expose based on auth mode
    safe_config = {
        "configured": True,
        "auth_mode": auth_mode,
        "server": {
            "host": config.get('Server', 'host', fallback='127.0.0.1'),
            "port": config.getint('Server', 'port', fallback=8000)
        }
    }
    
    # In single-user mode, we can expose the API key for documentation
    if auth_mode == 'single_user':
        api_key = config.get('Authentication', 'single_user_api_key', 
                           fallback='default-secret-key-for-single-user')
        safe_config["api_key_for_docs"] = api_key
    else:
        # In multi-user mode, provide a placeholder
        safe_config["api_key_for_docs"] = "YOUR_API_KEY_HERE"
    
    # Check which LLM providers are configured (without exposing keys)
    configured_providers = []
    if config.has_section('API'):
        provider_keys = {
            'openai_api_key': 'OpenAI',
            'anthropic_api_key': 'Anthropic',
            'groq_api_key': 'Groq',
            'google_api_key': 'Google',
            'cohere_api_key': 'Cohere',
            'mistral_api_key': 'Mistral',
            'deepseek_api_key': 'DeepSeek',
            'huggingface_api_key': 'HuggingFace',
            'openrouter_api_key': 'OpenRouter'
        }
        
        for key_name, provider_name in provider_keys.items():
            value = config.get('API', key_name, fallback='')
            if value and value not in ['', 'your_api_key_here', 'YOUR_API_KEY_HERE']:
                configured_providers.append(provider_name)
    
    safe_config["configured_llm_providers"] = configured_providers
    
    return safe_config


@router.get("/config/docs-info")
async def get_documentation_config():
    """
    Get configuration information suitable for auto-populating documentation.
    
    This endpoint returns non-sensitive configuration that can be used to:
    - Auto-populate API keys in documentation (single-user mode only)
    - Show which LLM providers are configured
    - Provide the correct base URL for examples
    
    Returns:
        Dict with configuration information safe for documentation
    """
    config = load_safe_config()
    
    # Build base URL
    host = config.get("server", {}).get("host", "127.0.0.1")
    port = config.get("server", {}).get("port", 8000)
    
    # Convert 0.0.0.0 to localhost for documentation
    if host == "0.0.0.0":
        host = "localhost"
    
    base_url = f"http://{host}:{port}"
    
    return {
        "configured": config.get("configured", False),
        "auth_mode": config.get("auth_mode", "single_user"),
        "api_key": config.get("api_key_for_docs", "default-secret-key-for-single-user"),
        "base_url": base_url,
        "configured_providers": config.get("configured_llm_providers", []),
        "examples": {
            "python": generate_python_example(
                config.get("api_key_for_docs", "default-secret-key-for-single-user"),
                base_url
            ),
            "curl": generate_curl_example(
                config.get("api_key_for_docs", "default-secret-key-for-single-user"),
                base_url
            ),
            "javascript": generate_js_example(
                config.get("api_key_for_docs", "default-secret-key-for-single-user"),
                base_url
            )
        }
    }


def generate_python_example(api_key: str, base_url: str) -> str:
    """Generate a Python code example with the provided configuration."""
    return f"""import requests

API_KEY = "{api_key}"
BASE_URL = "{base_url}"

# Create evaluation
eval_data = {{
    "name": "test_evaluation",
    "eval_type": "exact_match",
    "eval_spec": {{"threshold": 1.0}},
    "dataset": [
        {{"input": {{"output": "test"}}, "expected": {{"output": "test"}}}}
    ]
}}

response = requests.post(
    f"{{BASE_URL}}/api/v1/evals",
    json=eval_data,
    headers={{"Authorization": f"Bearer {{API_KEY}}"}}
)

print(f"Created evaluation: {{response.json()['id']}}")"""


def generate_curl_example(api_key: str, base_url: str) -> str:
    """Generate a cURL example with the provided configuration."""
    return f"""curl -X POST {base_url}/api/v1/evals \\
  -H "Authorization: Bearer {api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "name": "test_eval",
    "eval_type": "exact_match",
    "eval_spec": {{"threshold": 1.0}},
    "dataset": [
      {{"input": {{"output": "test"}}, "expected": {{"output": "test"}}}}
    ]
  }}'"""


def generate_js_example(api_key: str, base_url: str) -> str:
    """Generate a JavaScript example with the provided configuration."""
    return f"""const API_KEY = '{api_key}';
const BASE_URL = '{base_url}';

const evalData = {{
    name: 'test_evaluation',
    eval_type: 'exact_match',
    eval_spec: {{ threshold: 1.0 }},
    dataset: [
        {{ input: {{ output: 'test' }}, expected: {{ output: 'test' }} }}
    ]
}};

const response = await fetch(`${{BASE_URL}}/api/v1/evals`, {{
    method: 'POST',
    headers: {{
        'Authorization': `Bearer ${{API_KEY}}`,
        'Content-Type': 'application/json'
    }},
    body: JSON.stringify(evalData)
}});

const data = await response.json();
console.log('Created evaluation:', data.id);"""


@router.get("/config/quickstart")
async def get_quickstart_html():
    """
    Get an HTML quickstart page with auto-populated configuration.
    
    Returns an HTML page that can be opened directly in a browser with
    interactive examples using the current configuration.
    """
    from fastapi.responses import HTMLResponse
    
    config = load_safe_config()
    api_key = config.get("api_key_for_docs", "default-secret-key-for-single-user")
    
    # Build base URL
    host = config.get("server", {}).get("host", "127.0.0.1")
    port = config.get("server", {}).get("port", 8000)
    if host == "0.0.0.0":
        host = "localhost"
    base_url = f"http://{host}:{port}"
    
    providers = config.get("configured_llm_providers", [])
    providers_str = ", ".join(providers) if providers else "No LLM providers configured"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>tldw_server API Quick Start</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .config-box {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .code-block {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .copy-btn {{ margin-left: 10px; padding: 5px 10px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }}
        .test-btn {{ padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }}
        .result {{ margin-top: 15px; padding: 15px; border-radius: 5px; }}
        .success {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
        .error {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
        code {{ background: #e9ecef; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
    </style>
</head>
<body>
    <h1>🚀 tldw_server Evaluation API - Quick Start</h1>
    
    <div class="config-box">
        <h2>Your Configuration</h2>
        <p><strong>API Key:</strong> <code id="api-key">{api_key}</code> <button class="copy-btn" onclick="copyText('api-key')">Copy</button></p>
        <p><strong>Base URL:</strong> <code id="base-url">{base_url}</code> <button class="copy-btn" onclick="copyText('base-url')">Copy</button></p>
        <p><strong>Auth Mode:</strong> {config.get("auth_mode", "single_user")}</p>
        <p><strong>Configured LLM Providers:</strong> {providers_str}</p>
    </div>
    
    <div class="config-box">
        <h2>Test Your Configuration</h2>
        <p>Click the button below to test creating an evaluation with your current configuration:</p>
        <button class="test-btn" onclick="testAPI()">Test API Connection</button>
        <div id="result"></div>
    </div>
    
    <div class="config-box">
        <h2>Example Code</h2>
        <h3>Python</h3>
        <div class="code-block">
            <pre>{generate_python_example(api_key, base_url)}</pre>
        </div>
        
        <h3>cURL</h3>
        <div class="code-block">
            <pre>{generate_curl_example(api_key, base_url)}</pre>
        </div>
    </div>
    
    <script>
        function copyText(elementId) {{
            const text = document.getElementById(elementId).textContent;
            navigator.clipboard.writeText(text);
            event.target.textContent = 'Copied!';
            setTimeout(() => {{ event.target.textContent = 'Copy'; }}, 2000);
        }}
        
        async function testAPI() {{
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<p>Testing...</p>';
            
            try {{
                // First test health endpoint
                const healthResponse = await fetch('{base_url}/health');
                
                if (!healthResponse.ok) {{
                    throw new Error('Server is not responding');
                }}
                
                // Then test evaluation creation
                const evalData = {{
                    name: 'test_eval_' + Date.now(),
                    eval_type: 'exact_match',
                    eval_spec: {{ threshold: 1.0 }},
                    dataset: [
                        {{ input: {{ output: 'test' }}, expected: {{ output: 'test' }} }}
                    ]
                }};
                
                const response = await fetch('{base_url}/api/v1/evals', {{
                    method: 'POST',
                    headers: {{
                        'Authorization': 'Bearer {api_key}',
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify(evalData)
                }});
                
                const data = await response.json();
                
                if (response.ok) {{
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = `
                        <strong>✅ Success!</strong><br>
                        Created evaluation with ID: <code>${{data.id}}</code><br>
                        The API is working correctly with your configuration.
                    `;
                }} else {{
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `
                        <strong>❌ API Error</strong><br>
                        Status: ${{response.status}}<br>
                        Message: ${{JSON.stringify(data, null, 2)}}
                    `;
                }}
            }} catch (error) {{
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `
                    <strong>❌ Connection Error</strong><br>
                    ${{error.message}}<br><br>
                    Make sure the server is running:<br>
                    <code>python -m uvicorn tldw_Server_API.app.main:app --reload</code>
                `;
            }}
        }}
    </script>
</body>
</html>"""
    
    return HTMLResponse(content=html_content)