/**
 * Endpoint Helper Module
 * Provides common functionality for API endpoint interactions
 */

class EndpointHelper {
    constructor() {
        this.retryConfig = {
            maxRetries: 3,
            retryDelay: 1000,
            retryableStatuses: [408, 429, 500, 502, 503, 504]
        };
        
        this.timeoutConfig = {
            default: 30000,  // 30 seconds
            long: 120000,    // 2 minutes for long operations
            upload: 300000   // 5 minutes for file uploads
        };
    }

    /**
     * Create a standard endpoint section
     */
    createEndpointSection(config) {
        const {
            id,
            method = 'GET',
            path,
            description,
            fields = [],
            bodyType = 'none',
            hasFileUpload = false,
            timeout = 'default'
        } = config;

        const section = document.createElement('div');
        section.className = 'endpoint-section';
        section.id = `section_${id}`;

        // Build HTML
        let html = `
            <h2>
                <span class="endpoint-method ${method.toLowerCase()}">${method}</span>
                <span class="endpoint-path">${path}</span>
            </h2>
            ${description ? `<p>${description}</p>` : ''}
        `;

        // Add form fields
        fields.forEach(field => {
            html += this.createFormField(field, id);
        });

        // Add request button
        const buttonClass = method === 'DELETE' ? 'btn-danger' : '';
        const confirmDelete = method === 'DELETE' ? `if(confirm('Are you sure?')) ` : '';
        
        html += `
            <button class="api-button ${buttonClass}" 
                    onclick="${confirmDelete}endpointHelper.executeRequest('${id}', '${method}', '${path}', '${bodyType}', '${timeout}')">
                ${this.getButtonText(method)}
            </button>
            <button class="btn btn-secondary" onclick="endpointHelper.showCurl('${id}', '${method}', '${path}', '${bodyType}')" style="margin-left: 10px;">
                Show cURL
            </button>
            <pre id="${id}_response"></pre>
            <pre id="${id}_curl" style="display: none;"></pre>
        `;

        section.innerHTML = html;
        return section;
    }

    /**
     * Create a form field based on type
     */
    createFormField(field, endpointId) {
        const {
            name,
            type = 'text',
            label,
            placeholder = '',
            required = false,
            defaultValue = '',
            options = [],
            description = ''
        } = field;

        const fieldId = `${endpointId}_${name}`;
        const requiredMark = required ? '<span class="required">*</span>' : '';

        let html = '<div class="form-group">';

        switch (type) {
            case 'textarea':
                html += `
                    <label for="${fieldId}">${label} ${requiredMark}:</label>
                    <textarea id="${fieldId}" rows="5" placeholder="${placeholder}">${defaultValue}</textarea>
                `;
                break;

            case 'select':
                html += `
                    <label for="${fieldId}">${label} ${requiredMark}:</label>
                    <select id="${fieldId}">
                        ${options.map(opt => 
                            `<option value="${opt.value}" ${opt.value === defaultValue ? 'selected' : ''}>
                                ${opt.label}
                            </option>`
                        ).join('')}
                    </select>
                `;
                break;

            case 'checkbox':
                html += `
                    <label>
                        <input type="checkbox" id="${fieldId}" ${defaultValue ? 'checked' : ''}>
                        ${label}
                    </label>
                `;
                break;

            case 'file':
                const accept = field.accept || '';
                const multiple = field.multiple ? 'multiple' : '';
                html += `
                    <label for="${fieldId}">${label} ${requiredMark}:</label>
                    <input type="file" id="${fieldId}" ${multiple} ${accept ? `accept="${accept}"` : ''}>
                `;
                break;

            case 'number':
                html += `
                    <label for="${fieldId}">${label} ${requiredMark}:</label>
                    <input type="number" id="${fieldId}" 
                           value="${defaultValue}" 
                           placeholder="${placeholder}"
                           ${field.min !== undefined ? `min="${field.min}"` : ''}
                           ${field.max !== undefined ? `max="${field.max}"` : ''}
                           ${field.step !== undefined ? `step="${field.step}"` : ''}>
                `;
                break;

            default: // text, password, etc.
                html += `
                    <label for="${fieldId}">${label} ${requiredMark}:</label>
                    <input type="${type}" id="${fieldId}" 
                           value="${defaultValue}" 
                           placeholder="${placeholder}">
                `;
        }

        if (description) {
            html += `<small>${description}</small>`;
        }

        html += '</div>';
        return html;
    }

    /**
     * Execute request with enhanced error handling and retry logic
     */
    async executeRequest(endpointId, method, path, bodyType = 'none', timeoutType = 'default') {
        const responseEl = document.getElementById(`${endpointId}_response`);
        if (!responseEl) {
            console.error(`Response element not found for ${endpointId}`);
            return;
        }

        try {
            Loading.show(responseEl.parentElement, 'Sending request...');
            responseEl.textContent = '';

            // Build the request
            const { body, query, processedPath } = this.buildRequest(endpointId, path, bodyType);

            // Set timeout
            const timeout = this.timeoutConfig[timeoutType] || this.timeoutConfig.default;

            // Execute with retry logic
            const response = await this.executeWithRetry(
                () => apiClient.makeRequest(method, processedPath, { body, query, timeout }),
                endpointId
            );

            // Display response
            this.displayResponse(responseEl, response, true);
            Toast.success('Request completed successfully');

        } catch (error) {
            this.displayError(responseEl, error);
            Toast.error(`Request failed: ${error.message}`);
        } finally {
            Loading.hide(responseEl.parentElement);
        }
    }

    /**
     * Execute request with retry logic
     */
    async executeWithRetry(requestFn, endpointId, attempt = 1) {
        try {
            return await requestFn();
        } catch (error) {
            const shouldRetry = this.shouldRetry(error, attempt);
            
            if (shouldRetry) {
                const delay = this.retryConfig.retryDelay * attempt;
                Toast.warning(`Request failed. Retrying in ${delay}ms... (Attempt ${attempt + 1}/${this.retryConfig.maxRetries})`);
                
                await this.delay(delay);
                return this.executeWithRetry(requestFn, endpointId, attempt + 1);
            }
            
            throw error;
        }
    }

    /**
     * Determine if request should be retried
     */
    shouldRetry(error, attempt) {
        if (attempt >= this.retryConfig.maxRetries) {
            return false;
        }

        // Check if error has a retryable status code
        const errorMessage = error.message || '';
        const statusMatch = errorMessage.match(/HTTP (\d+)/);
        
        if (statusMatch) {
            const status = parseInt(statusMatch[1]);
            return this.retryConfig.retryableStatuses.includes(status);
        }

        // Retry on network errors
        return errorMessage.includes('NetworkError') || 
               errorMessage.includes('Failed to fetch') ||
               errorMessage.includes('timeout');
    }

    /**
     * Build request body and query parameters
     */
    buildRequest(endpointId, path, bodyType) {
        let body = null;
        let query = {};
        let processedPath = path;

        // Process path parameters
        const pathParams = path.match(/\{([^}]+)\}/g);
        if (pathParams) {
            pathParams.forEach(param => {
                const paramName = param.slice(1, -1);
                const input = document.getElementById(`${endpointId}_${paramName}`);
                if (input && input.value) {
                    processedPath = processedPath.replace(param, input.value);
                }
            });
        }

        // Build body based on type
        if (bodyType === 'json' || bodyType === 'json_with_query') {
            const payloadEl = document.getElementById(`${endpointId}_payload`);
            if (payloadEl) {
                try {
                    body = JSON.parse(payloadEl.value);
                } catch (e) {
                    throw new Error(`Invalid JSON: ${e.message}`);
                }
            } else {
                // Build from form fields
                body = this.buildJsonFromFields(endpointId);
            }
        }

        if (bodyType === 'form') {
            body = this.buildFormData(endpointId);
        }

        if (bodyType === 'query' || bodyType === 'json_with_query') {
            query = this.buildQueryParams(endpointId);
        }

        return { body, query, processedPath };
    }

    /**
     * Build JSON from form fields
     */
    buildJsonFromFields(endpointId) {
        const data = {};
        const section = document.getElementById(`section_${endpointId}`);
        
        if (!section) return data;

        // Find all inputs in this section
        const inputs = section.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            const name = input.id.replace(`${endpointId}_`, '');
            
            // Skip path parameters and special fields
            if (name.includes('response') || name.includes('curl') || name.includes('payload')) {
                return;
            }

            if (input.type === 'checkbox') {
                data[name] = input.checked;
            } else if (input.type === 'number') {
                if (input.value) data[name] = parseFloat(input.value);
            } else if (input.value) {
                data[name] = input.value;
            }
        });

        return data;
    }

    /**
     * Build FormData from form fields
     */
    buildFormData(endpointId) {
        const formData = new FormData();
        const section = document.getElementById(`section_${endpointId}`);
        
        if (!section) return formData;

        const inputs = section.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            const name = input.id.replace(`${endpointId}_`, '');
            
            if (input.type === 'file' && input.files.length > 0) {
                Array.from(input.files).forEach(file => {
                    formData.append(name, file);
                });
            } else if (input.type === 'checkbox') {
                formData.append(name, input.checked);
            } else if (input.value) {
                formData.append(name, input.value);
            }
        });

        return formData;
    }

    /**
     * Build query parameters
     */
    buildQueryParams(endpointId) {
        const params = {};
        const section = document.getElementById(`section_${endpointId}`);
        
        if (!section) return params;

        // Look for fields marked as query parameters
        const queryInputs = section.querySelectorAll('[data-query="true"]');
        queryInputs.forEach(input => {
            const name = input.id.replace(`${endpointId}_`, '');
            if (input.value) {
                params[name] = input.value;
            }
        });

        return params;
    }

    /**
     * Display response with enhanced formatting
     */
    displayResponse(element, response, success = true) {
        if (typeof response === 'object') {
            // Use JSON viewer for objects
            const viewer = new JSONViewer(element, response, {
                expanded: 2,
                theme: document.documentElement.getAttribute('data-theme') || 'light',
                enableCopy: true,
                enableCollapse: true
            });
        } else {
            element.textContent = response;
        }

        // Add success/error styling
        element.className = success ? 'response-success' : 'response-error';
    }

    /**
     * Display detailed error information
     */
    displayError(element, error) {
        const errorInfo = {
            message: error.message,
            timestamp: new Date().toISOString(),
            details: {}
        };

        // Extract additional error information
        if (error.response) {
            errorInfo.details.status = error.response.status;
            errorInfo.details.statusText = error.response.statusText;
            errorInfo.details.headers = error.response.headers;
        }

        if (error.stack) {
            errorInfo.stack = error.stack.split('\n').slice(0, 5).join('\n');
        }

        this.displayResponse(element, errorInfo, false);
    }

    /**
     * Show cURL command for request
     */
    showCurl(endpointId, method, path, bodyType) {
        const curlEl = document.getElementById(`${endpointId}_curl`);
        if (!curlEl) return;

        try {
            const { body, query, processedPath } = this.buildRequest(endpointId, path, bodyType);
            const curlCommand = apiClient.generateCurl(method, processedPath, { body, query });
            
            curlEl.textContent = curlCommand;
            curlEl.style.display = curlEl.style.display === 'none' ? 'block' : 'none';
            
            // Copy to clipboard button
            if (!curlEl.nextElementSibling || !curlEl.nextElementSibling.classList.contains('copy-curl')) {
                const copyBtn = document.createElement('button');
                copyBtn.className = 'btn btn-sm btn-secondary copy-curl';
                copyBtn.textContent = 'Copy cURL';
                copyBtn.onclick = () => {
                    Utils.copyToClipboard(curlCommand);
                    Toast.success('cURL command copied to clipboard');
                };
                curlEl.parentNode.insertBefore(copyBtn, curlEl.nextSibling);
            }
        } catch (error) {
            curlEl.textContent = `Error generating cURL: ${error.message}`;
            curlEl.style.display = 'block';
        }
    }

    /**
     * Get appropriate button text based on method
     */
    getButtonText(method) {
        const texts = {
            'GET': 'Fetch',
            'POST': 'Submit',
            'PUT': 'Update',
            'DELETE': 'Delete',
            'PATCH': 'Patch'
        };
        return texts[method] || 'Send Request';
    }

    /**
     * Utility: delay function for retries
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Batch operation helper
     */
    async executeBatch(operations, options = {}) {
        const {
            parallel = false,
            stopOnError = false,
            onProgress = null
        } = options;

        const results = [];
        const total = operations.length;

        if (parallel) {
            // Execute all operations in parallel
            const promises = operations.map((op, index) => 
                this.executeBatchOperation(op, index, total, onProgress)
            );
            
            if (stopOnError) {
                return Promise.all(promises);
            } else {
                return Promise.allSettled(promises);
            }
        } else {
            // Execute operations sequentially
            for (let i = 0; i < operations.length; i++) {
                try {
                    const result = await this.executeBatchOperation(operations[i], i, total, onProgress);
                    results.push({ status: 'fulfilled', value: result });
                } catch (error) {
                    results.push({ status: 'rejected', reason: error });
                    if (stopOnError) {
                        throw error;
                    }
                }
            }
            return results;
        }
    }

    /**
     * Execute single batch operation
     */
    async executeBatchOperation(operation, index, total, onProgress) {
        if (onProgress) {
            onProgress({
                current: index + 1,
                total,
                operation,
                status: 'processing'
            });
        }

        const result = await operation.execute();

        if (onProgress) {
            onProgress({
                current: index + 1,
                total,
                operation,
                status: 'completed',
                result
            });
        }

        return result;
    }
}

// Create global instance
const endpointHelper = new EndpointHelper();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EndpointHelper;
}