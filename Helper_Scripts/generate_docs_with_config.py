#!/usr/bin/env python3
"""
Generate documentation with auto-populated configuration values.
This script reads the actual config.txt and populates documentation templates.
"""

import os
import sys
import configparser
from pathlib import Path
import json
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config() -> Dict[str, Dict[str, str]]:
    """Load configuration from config.txt file."""
    config_path = get_project_root() / "tldw_Server_API" / "Config_Files" / "config.txt"
    
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        return {}
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Convert to dictionary
    config_dict = {}
    for section in config.sections():
        config_dict[section] = dict(config.items(section))
    
    return config_dict


def get_api_key_for_docs() -> str:
    """
    Get the appropriate API key for documentation.
    Returns the actual key if in single-user mode, otherwise a placeholder.
    """
    config = load_config()
    
    # Check authentication mode
    auth_config = config.get('Authentication', {})
    auth_mode = auth_config.get('auth_mode', 'single_user').lower()
    
    if auth_mode == 'single_user':
        # In single-user mode, return the actual API key
        api_key = auth_config.get('single_user_api_key', 'default-secret-key-for-single-user')
        return api_key
    else:
        # In multi-user mode, return a placeholder
        return "YOUR_API_KEY_HERE"


def get_base_url() -> str:
    """Get the base URL for the API."""
    config = load_config()
    server_config = config.get('Server', {})
    
    host = server_config.get('host', '127.0.0.1')
    port = server_config.get('port', '8000')
    
    # Use localhost for documentation
    if host == '0.0.0.0':
        host = 'localhost'
    
    return f"http://{host}:{port}"


def generate_quick_start_examples() -> str:
    """Generate code examples with auto-populated values."""
    api_key = get_api_key_for_docs()
    base_url = get_base_url()
    config = load_config()
    
    # Determine available LLM providers
    available_providers = []
    api_config = config.get('API', {})
    
    provider_mapping = {
        'openai_api_key': 'openai',
        'anthropic_api_key': 'anthropic',
        'groq_api_key': 'groq',
        'google_api_key': 'google',
        'cohere_api_key': 'cohere',
        'mistral_api_key': 'mistral',
        'deepseek_api_key': 'deepseek'
    }
    
    for key_name, provider in provider_mapping.items():
        if api_config.get(key_name, '').strip() and api_config[key_name] != 'your_api_key_here':
            available_providers.append(provider)
    
    # Python example
    python_example = f'''```python
import requests
import json

# Configuration (auto-populated from your config.txt)
API_KEY = "{api_key}"
BASE_URL = "{base_url}"

# Available LLM providers based on your configuration:
# {', '.join(available_providers) if available_providers else 'No LLM providers configured yet'}

# Create evaluation
eval_data = {{
    "name": "my_evaluation",
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

if response.status_code == 201:
    print(f"✅ Created evaluation: {{response.json()['id']}}")
else:
    print(f"❌ Error: {{response.status_code}} - {{response.json()}}")
```'''

    # cURL example
    curl_example = f'''```bash
# Set your API key (from config.txt)
export API_KEY="{api_key}"
export BASE_URL="{base_url}"

# Create evaluation
curl -X POST $BASE_URL/api/v1/evals \\
  -H "Authorization: Bearer $API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "name": "test_eval",
    "eval_type": "exact_match",
    "eval_spec": {{"threshold": 1.0}},
    "dataset": [
      {{"input": {{"output": "test"}}, "expected": {{"output": "test"}}}}
    ]
  }}'
```'''

    # JavaScript example
    js_example = f'''```javascript
// Configuration (auto-populated from your config.txt)
const API_KEY = '{api_key}';
const BASE_URL = '{base_url}';

// Available providers: {', '.join(available_providers) if available_providers else 'None configured'}

async function createEvaluation() {{
    const response = await fetch(`${{BASE_URL}}/api/v1/evals`, {{
        method: 'POST',
        headers: {{
            'Authorization': `Bearer ${{API_KEY}}`,
            'Content-Type': 'application/json'
        }},
        body: JSON.stringify({{
            name: 'js_eval',
            eval_type: 'exact_match',
            eval_spec: {{ threshold: 1.0 }},
            dataset: [
                {{ input: {{ output: 'test' }}, expected: {{ output: 'test' }} }}
            ]
        }})
    }});
    
    const data = await response.json();
    console.log('Created evaluation:', data.id);
}}
```'''

    return f"""
## Auto-Generated Examples

These examples use values from your current configuration:
- **API Key**: `{api_key}`
- **Base URL**: `{base_url}`
- **Available LLM Providers**: {', '.join(available_providers) if available_providers else 'None configured yet'}

### Python Example
{python_example}

### cURL Example
{curl_example}

### JavaScript Example
{js_example}
"""


def generate_config_status() -> str:
    """Generate a configuration status summary."""
    config = load_config()
    
    if not config:
        return """
## Configuration Status

⚠️ **No configuration file found!**

Please create `tldw_Server_API/Config_Files/config.txt` with your settings.
"""
    
    auth_config = config.get('Authentication', {})
    api_config = config.get('API', {})
    
    auth_mode = auth_config.get('auth_mode', 'single_user')
    
    # Check which APIs are configured
    configured_apis = []
    for key, value in api_config.items():
        if key.endswith('_api_key') and value and value != 'your_api_key_here':
            api_name = key.replace('_api_key', '').replace('_', ' ').title()
            configured_apis.append(api_name)
    
    status = f"""
## Configuration Status

- **Authentication Mode**: {auth_mode}
- **API Key**: {'Configured' if auth_config.get('single_user_api_key') else 'Not configured'}
- **LLM Providers Configured**: {len(configured_apis)}
  {chr(10).join(f'  - ✅ {api}' for api in configured_apis) if configured_apis else '  - ❌ No providers configured'}
"""
    
    return status


def create_html_quickstart() -> str:
    """Create an HTML page with auto-populated configuration."""
    api_key = get_api_key_for_docs()
    base_url = get_base_url()
    config = load_config()
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>tldw_server - Evaluation API Quick Start</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .config-box {{
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .config-item {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .config-label {{
            font-weight: bold;
            min-width: 150px;
        }}
        .config-value {{
            font-family: 'Courier New', monospace;
            background: #f0f0f0;
            padding: 5px 10px;
            border-radius: 4px;
            user-select: all;
        }}
        .copy-button {{
            margin-left: 10px;
            padding: 5px 10px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .copy-button:hover {{
            background: #0056b3;
        }}
        .status-ok {{
            color: #28a745;
        }}
        .status-warning {{
            color: #ffc107;
        }}
        .code-block {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            position: relative;
        }}
        .code-block pre {{
            margin: 0;
            font-family: 'Courier New', monospace;
        }}
        .run-button {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 15px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .result-box {{
            margin-top: 10px;
            padding: 10px;
            background: #e8f5e9;
            border: 1px solid #4caf50;
            border-radius: 4px;
            display: none;
        }}
        .error-box {{
            background: #ffebee;
            border-color: #f44336;
        }}
    </style>
</head>
<body>
    <h1>🚀 tldw_server Evaluation API - Quick Start</h1>
    
    <div class="config-box">
        <h2>Your Current Configuration</h2>
        <div class="config-item">
            <span class="config-label">API Key:</span>
            <span class="config-value" id="api-key">{api_key}</span>
            <button class="copy-button" onclick="copyToClipboard('api-key')">Copy</button>
        </div>
        <div class="config-item">
            <span class="config-label">Base URL:</span>
            <span class="config-value" id="base-url">{base_url}</span>
            <button class="copy-button" onclick="copyToClipboard('base-url')">Copy</button>
        </div>
        <div class="config-item">
            <span class="config-label">Server Status:</span>
            <span id="server-status">Checking...</span>
        </div>
    </div>

    <div class="config-box">
        <h2>Interactive Test</h2>
        <p>Click "Run Test" to create a simple evaluation using your current configuration:</p>
        
        <div class="code-block">
            <button class="run-button" onclick="runTest()">Run Test</button>
            <pre id="test-code">
// Test evaluation creation
const evalData = {{
    name: "test_eval_" + Date.now(),
    eval_type: "exact_match",
    eval_spec: {{ threshold: 1.0 }},
    dataset: [
        {{ input: {{ output: "Paris" }}, expected: {{ output: "Paris" }} }},
        {{ input: {{ output: "London" }}, expected: {{ output: "London" }} }}
    ]
}};

fetch('{base_url}/api/v1/evals', {{
    method: 'POST',
    headers: {{
        'Authorization': 'Bearer {api_key}',
        'Content-Type': 'application/json'
    }},
    body: JSON.stringify(evalData)
}})
.then(response => response.json())
.then(data => console.log('Success:', data))
.catch(error => console.error('Error:', error));
            </pre>
        </div>
        
        <div id="test-result" class="result-box"></div>
    </div>

    <script>
        // Check server status on load
        window.onload = function() {{
            checkServerStatus();
        }};
        
        function checkServerStatus() {{
            fetch('{base_url}/health')
                .then(response => {{
                    if (response.ok) {{
                        document.getElementById('server-status').innerHTML = 
                            '<span class="status-ok">✅ Server is running</span>';
                    }} else {{
                        document.getElementById('server-status').innerHTML = 
                            '<span class="status-warning">⚠️ Server not responding</span>';
                    }}
                }})
                .catch(error => {{
                    document.getElementById('server-status').innerHTML = 
                        '<span class="status-warning">❌ Server offline - Start with: python -m uvicorn tldw_Server_API.app.main:app</span>';
                }});
        }}
        
        function copyToClipboard(elementId) {{
            const element = document.getElementById(elementId);
            const text = element.textContent;
            navigator.clipboard.writeText(text).then(() => {{
                const button = element.nextElementSibling;
                const originalText = button.textContent;
                button.textContent = 'Copied!';
                setTimeout(() => {{
                    button.textContent = originalText;
                }}, 2000);
            }});
        }}
        
        async function runTest() {{
            const resultBox = document.getElementById('test-result');
            resultBox.style.display = 'block';
            resultBox.className = 'result-box';
            resultBox.innerHTML = 'Running test...';
            
            const evalData = {{
                name: "test_eval_" + Date.now(),
                eval_type: "exact_match",
                eval_spec: {{ threshold: 1.0 }},
                dataset: [
                    {{ input: {{ output: "Paris" }}, expected: {{ output: "Paris" }} }},
                    {{ input: {{ output: "London" }}, expected: {{ output: "London" }} }}
                ]
            }};
            
            try {{
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
                    resultBox.innerHTML = `
                        <strong>✅ Success!</strong><br>
                        Evaluation ID: ${{data.id}}<br>
                        Name: ${{data.name}}<br>
                        Type: ${{data.eval_type}}<br>
                        <br>
                        <strong>Next step:</strong> Run the evaluation with:<br>
                        <code>POST {base_url}/api/v1/evals/${{data.id}}/runs</code>
                    `;
                }} else {{
                    resultBox.className = 'result-box error-box';
                    resultBox.innerHTML = `
                        <strong>❌ Error:</strong><br>
                        Status: ${{response.status}}<br>
                        Message: ${{JSON.stringify(data, null, 2)}}
                    `;
                }}
            }} catch (error) {{
                resultBox.className = 'result-box error-box';
                resultBox.innerHTML = `
                    <strong>❌ Connection Error:</strong><br>
                    ${{error.message}}<br>
                    <br>
                    Make sure the server is running:<br>
                    <code>python -m uvicorn tldw_Server_API.app.main:app</code>
                `;
            }}
        }}
    </script>
</body>
</html>"""
    
    return html_content


def main():
    """Main function to generate documentation with config."""
    print("Generating documentation with auto-populated configuration...")
    
    # Generate configuration status
    config_status = generate_config_status()
    print(config_status)
    
    # Generate code examples
    examples = generate_quick_start_examples()
    
    # Save to a markdown file
    output_path = get_project_root() / "Docs" / "Evaluations_Quick_Start_Generated.md"
    with open(output_path, 'w') as f:
        f.write("# Evaluation API Quick Start (Auto-Generated)\n\n")
        f.write("This document was auto-generated from your current configuration.\n\n")
        f.write(config_status)
        f.write(examples)
    
    print(f"\n✅ Generated markdown documentation: {output_path}")
    
    # Generate HTML version
    html_content = create_html_quickstart()
    html_path = get_project_root() / "Docs" / "quickstart.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Generated HTML documentation: {html_path}")
    print(f"\nOpen {html_path} in your browser for an interactive quick start guide!")


if __name__ == "__main__":
    main()