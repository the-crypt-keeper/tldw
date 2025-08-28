/**
 * Auto-configuration script for tldw_server documentation
 * 
 * This script automatically fetches configuration from the running server
 * and populates documentation examples with the correct values.
 * 
 * Usage:
 * 1. Include this script in your HTML documentation
 * 2. Add class="auto-config" to elements that should be auto-populated
 * 3. Use data-config attributes to specify what to populate
 * 
 * Example:
 * <span class="auto-config" data-config="api_key">loading...</span>
 * <pre class="auto-config" data-config="python_example">loading...</pre>
 */

(function() {
    'use strict';
    
    // Default configuration (fallback values)
    const DEFAULT_CONFIG = {
        api_key: 'default-secret-key-for-single-user',
        base_url: 'http://localhost:8000',
        configured: false,
        auth_mode: 'single_user',
        configured_providers: []
    };
    
    // Try to detect the server URL from the current page
    function detectServerUrl() {
        // If running from file:// protocol, use localhost
        if (window.location.protocol === 'file:') {
            return 'http://localhost:8000';
        }
        
        // If running from a server, use the same host
        const port = window.location.port || (window.location.protocol === 'https:' ? 443 : 80);
        
        // Check if we're on the docs server (usually port 8000)
        if (port === 8000 || port === 8080) {
            return `${window.location.protocol}//${window.location.hostname}:8000`;
        }
        
        // Default to localhost
        return 'http://localhost:8000';
    }
    
    // Fetch configuration from the server
    async function fetchConfig() {
        const serverUrl = detectServerUrl();
        
        try {
            const response = await fetch(`${serverUrl}/api/v1/config/docs-info`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            if (!response.ok) {
                console.warn('Could not fetch configuration from server:', response.status);
                return DEFAULT_CONFIG;
            }
            
            const config = await response.json();
            return config;
        } catch (error) {
            console.warn('Server not reachable, using default configuration:', error.message);
            return DEFAULT_CONFIG;
        }
    }
    
    // Generate example code with the configuration
    function generateExamples(config) {
        const examples = {};
        
        // Python example
        examples.python = `import requests

API_KEY = "${config.api_key}"
BASE_URL = "${config.base_url}"

# Create evaluation
eval_data = {
    "name": "test_evaluation",
    "eval_type": "exact_match",
    "eval_spec": {"threshold": 1.0},
    "dataset": [
        {"input": {"output": "test"}, "expected": {"output": "test"}}
    ]
}

response = requests.post(
    f"{BASE_URL}/api/v1/evals",
    json=eval_data,
    headers={"Authorization": f"Bearer {API_KEY}"}
)

print(f"Created evaluation: {response.json()['id']}")`;
        
        // cURL example
        examples.curl = `curl -X POST ${config.base_url}/api/v1/evals \\
  -H "Authorization: Bearer ${config.api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "test_eval",
    "eval_type": "exact_match",
    "eval_spec": {"threshold": 1.0},
    "dataset": [
      {"input": {"output": "test"}, "expected": {"output": "test"}}
    ]
  }'`;
        
        // JavaScript example
        examples.javascript = `const API_KEY = '${config.api_key}';
const BASE_URL = '${config.base_url}';

const evalData = {
    name: 'test_evaluation',
    eval_type: 'exact_match',
    eval_spec: { threshold: 1.0 },
    dataset: [
        { input: { output: 'test' }, expected: { output: 'test' } }
    ]
};

const response = await fetch(\`\${BASE_URL}/api/v1/evals\`, {
    method: 'POST',
    headers: {
        'Authorization': \`Bearer \${API_KEY}\`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(evalData)
});

const data = await response.json();
console.log('Created evaluation:', data.id);`;
        
        return examples;
    }
    
    // Update all elements with the auto-config class
    function updateElements(config) {
        const elements = document.querySelectorAll('.auto-config');
        const examples = config.examples || generateExamples(config);
        
        elements.forEach(element => {
            const configKey = element.getAttribute('data-config');
            
            switch(configKey) {
                case 'api_key':
                    element.textContent = config.api_key;
                    break;
                case 'base_url':
                    element.textContent = config.base_url;
                    break;
                case 'auth_mode':
                    element.textContent = config.auth_mode;
                    break;
                case 'providers':
                    element.textContent = config.configured_providers.length > 0 
                        ? config.configured_providers.join(', ')
                        : 'No LLM providers configured';
                    break;
                case 'python_example':
                    element.textContent = examples.python;
                    break;
                case 'curl_example':
                    element.textContent = examples.curl;
                    break;
                case 'javascript_example':
                case 'js_example':
                    element.textContent = examples.javascript;
                    break;
                case 'status':
                    if (config.configured) {
                        element.innerHTML = '✅ Configuration loaded from server';
                        element.style.color = 'green';
                    } else {
                        element.innerHTML = '⚠️ Using default configuration (server not reachable)';
                        element.style.color = 'orange';
                    }
                    break;
                default:
                    // Try to get the value directly from config
                    if (config[configKey] !== undefined) {
                        element.textContent = config[configKey];
                    }
            }
            
            // Add a copy button if the element has data-copyable="true"
            if (element.getAttribute('data-copyable') === 'true') {
                addCopyButton(element);
            }
        });
    }
    
    // Add a copy button next to an element
    function addCopyButton(element) {
        const button = document.createElement('button');
        button.textContent = 'Copy';
        button.style.marginLeft = '10px';
        button.style.padding = '2px 8px';
        button.style.fontSize = '12px';
        button.style.cursor = 'pointer';
        
        button.onclick = function() {
            const text = element.textContent;
            navigator.clipboard.writeText(text).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy:', err);
            });
        };
        
        // Insert the button after the element
        element.parentNode.insertBefore(button, element.nextSibling);
    }
    
    // Initialize when DOM is ready
    function init() {
        // Check if we're in a documentation environment
        const isDocumentation = 
            document.querySelector('.auto-config') !== null ||
            window.location.pathname.includes('/docs') ||
            window.location.pathname.includes('/Docs');
        
        if (!isDocumentation) {
            return; // Don't run if not in documentation
        }
        
        // Add loading indicators
        document.querySelectorAll('.auto-config').forEach(element => {
            if (element.textContent === '') {
                element.textContent = 'Loading configuration...';
            }
        });
        
        // Fetch and apply configuration
        fetchConfig().then(config => {
            updateElements(config);
            
            // Store in global for debugging
            window.tldwConfig = config;
            
            // Dispatch event for other scripts
            window.dispatchEvent(new CustomEvent('tldw-config-loaded', { detail: config }));
        });
    }
    
    // Run initialization
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    // Export for use in other scripts
    window.TldwAutoConfig = {
        fetchConfig,
        updateElements,
        generateExamples,
        detectServerUrl
    };
})();