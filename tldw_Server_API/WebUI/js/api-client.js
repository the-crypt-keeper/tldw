/**
 * API Client for making requests to the backend
 */

class APIClient {
    constructor() {
        this.baseUrl = 'http://localhost:8000';
        this.token = 'default-secret-key-for-single-user';
        this.requestHistory = [];
        this.maxHistorySize = 50;
        this.init();
    }

    init() {
        // Load saved configuration
        const savedConfig = Utils.getFromStorage('api-config');
        if (savedConfig) {
            this.baseUrl = savedConfig.baseUrl || this.baseUrl;
            this.token = savedConfig.token || this.token;
        }

        // Load request history
        const savedHistory = Utils.getFromStorage('request-history');
        if (savedHistory && Array.isArray(savedHistory)) {
            this.requestHistory = savedHistory;
        }
    }

    setBaseUrl(url) {
        this.baseUrl = url;
        this.saveConfig();
    }

    setToken(token) {
        this.token = token;
        this.saveConfig();
    }

    saveConfig() {
        Utils.saveToStorage('api-config', {
            baseUrl: this.baseUrl,
            token: this.token
        });
    }

    async makeRequest(method, path, options = {}) {
        const {
            body = null,
            query = {},
            headers = {},
            streaming = false,
            onProgress = null
        } = options;

        // Build URL with query parameters
        const url = new URL(`${this.baseUrl}${path}`);
        Object.keys(query).forEach(key => {
            if (query[key] !== undefined && query[key] !== null && query[key] !== '') {
                url.searchParams.append(key, query[key]);
            }
        });

        // Prepare fetch options
        const fetchOptions = {
            method,
            headers: {
                'Accept': streaming ? 'text/event-stream' : 'application/json',
                'Token': this.token,
                ...headers
            }
        };

        // Add body if present
        if (body) {
            if (body instanceof FormData) {
                fetchOptions.body = body;
            } else {
                fetchOptions.headers['Content-Type'] = 'application/json';
                fetchOptions.body = typeof body === 'string' ? body : JSON.stringify(body);
            }
        }

        // Record request start time
        const startTime = Date.now();

        try {
            const response = await fetch(url.toString(), fetchOptions);
            const duration = Date.now() - startTime;

            // Save to history
            this.addToHistory({
                method,
                path,
                url: url.toString(),
                timestamp: startTime,
                duration,
                status: response.status,
                success: response.ok
            });

            if (!response.ok) {
                let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.detail || errorData.message || JSON.stringify(errorData);
                } catch (e) {
                    // If response is not JSON, use statusText
                }
                throw new Error(errorMessage);
            }

            // Handle streaming responses
            if (streaming && response.body) {
                return this.handleStreamingResponse(response, onProgress);
            }

            // Handle regular JSON responses
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }

            return await response.text();
        } catch (error) {
            // Record error in history
            this.addToHistory({
                method,
                path,
                url: url.toString(),
                timestamp: startTime,
                duration: Date.now() - startTime,
                error: error.message,
                success: false
            });
            throw error;
        }
    }

    async handleStreamingResponse(response, onProgress) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        const chunks = [];

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.substring(6).trim();
                    if (data === '[DONE]') {
                        return chunks;
                    }
                    try {
                        const parsed = JSON.parse(data);
                        chunks.push(parsed);
                        if (onProgress) {
                            onProgress(parsed);
                        }
                    } catch (e) {
                        console.warn('Failed to parse SSE data:', data);
                    }
                }
            }
        }

        return chunks;
    }

    addToHistory(request) {
        this.requestHistory.unshift(request);
        if (this.requestHistory.length > this.maxHistorySize) {
            this.requestHistory = this.requestHistory.slice(0, this.maxHistorySize);
        }
        Utils.saveToStorage('request-history', this.requestHistory);
    }

    getHistory() {
        return this.requestHistory;
    }

    clearHistory() {
        this.requestHistory = [];
        Utils.saveToStorage('request-history', []);
    }

    // Convenience methods for common HTTP methods
    async get(path, query = {}, options = {}) {
        return this.makeRequest('GET', path, { query, ...options });
    }

    async post(path, body = null, options = {}) {
        return this.makeRequest('POST', path, { body, ...options });
    }

    async put(path, body = null, options = {}) {
        return this.makeRequest('PUT', path, { body, ...options });
    }

    async delete(path, options = {}) {
        return this.makeRequest('DELETE', path, options);
    }

    async patch(path, body = null, options = {}) {
        return this.makeRequest('PATCH', path, { body, ...options });
    }

    // Generate cURL command for a request
    generateCurl(method, path, options = {}) {
        const { body = null, query = {}, headers = {} } = options;

        // Build URL with query parameters
        const url = new URL(`${this.baseUrl}${path}`);
        Object.keys(query).forEach(key => {
            if (query[key] !== undefined && query[key] !== null && query[key] !== '') {
                url.searchParams.append(key, query[key]);
            }
        });

        let curl = `curl -X ${method} "${url.toString()}"`;

        // Add headers
        curl += ` \\\n  -H "Accept: application/json"`;
        curl += ` \\\n  -H "Token: ${this.token}"`;
        
        Object.keys(headers).forEach(key => {
            curl += ` \\\n  -H "${key}: ${headers[key]}"`;
        });

        // Add body
        if (body) {
            if (body instanceof FormData) {
                for (let [key, value] of body.entries()) {
                    if (value instanceof File) {
                        curl += ` \\\n  -F "${key}=@${value.name}"`;
                    } else {
                        curl += ` \\\n  -F "${key}=${value}"`;
                    }
                }
            } else {
                curl += ` \\\n  -H "Content-Type: application/json"`;
                const bodyStr = typeof body === 'string' ? body : JSON.stringify(body, null, 2);
                curl += ` \\\n  -d '${Utils.escapeCurlData(bodyStr)}'`;
            }
        }

        return curl;
    }

    // Check API health/status
    async checkHealth() {
        try {
            const response = await this.get('/health');
            return { online: true, ...response };
        } catch (error) {
            return { online: false, error: error.message };
        }
    }
}

// Create global instance
const apiClient = new APIClient();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { APIClient, apiClient };
}