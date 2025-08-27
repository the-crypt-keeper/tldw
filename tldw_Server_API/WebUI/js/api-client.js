/**
 * API Client for making requests to the backend
 */

class APIClient {
    constructor() {
        this.baseUrl = 'http://localhost:8000';
        this.token = '';  // Token should be set by user, not hardcoded
        this.requestHistory = [];
        this.maxHistorySize = 50;
        this.configLoaded = false;
        this.authMode = 'unknown'; // 'single-user', 'multi-user', or 'unknown'
        this.websockets = new Map(); // Store active WebSocket connections
        this.init();
    }

    async init() {
        // Check if we're served from the same origin
        const currentUrl = window.location.href;
        if (currentUrl.includes('/webui/')) {
            // We're being served from the FastAPI server - use same origin
            this.baseUrl = window.location.origin;
            console.log('WebUI served from same origin, using:', this.baseUrl);
            
            // Try to load dynamic configuration from the server
            try {
                const response = await fetch('/webui/config.json');
                if (response.ok) {
                    const config = await response.json();
                    
                    // Use apiUrl if provided, otherwise keep same origin
                    if (config.apiUrl) {
                        this.baseUrl = config.apiUrl;
                    }
                    
                    // Store the authentication mode
                    this.authMode = config.mode || 'unknown';
                    
                    // Check for API key (will be present in single user mode)
                    if (config.apiKey && config.apiKey !== '') {
                        this.token = config.apiKey;
                        this.configLoaded = true;
                        console.log(`✅ Auto-configured for ${this.authMode} mode`);
                        
                        // Show success message if we got an API key
                        if (this.authMode === 'single-user') {
                            console.log('🔑 API key automatically configured from server');
                            
                            // Update the UI to show auto-configuration
                            setTimeout(() => {
                                const apiKeyInput = document.getElementById('apiKeyInput');
                                if (apiKeyInput) {
                                    apiKeyInput.value = '(Auto-configured)';
                                    apiKeyInput.disabled = true;
                                    apiKeyInput.style.backgroundColor = '#e8f5e9';
                                }
                                
                                // Update any status message
                                const statusElement = document.querySelector('.api-status-text');
                                if (statusElement) {
                                    statusElement.textContent = 'Connected (Single User Mode)';
                                }
                            }, 100);
                        }
                    } else if (this.authMode === 'multi-user') {
                        console.log('🔐 Multi-user mode - manual authentication required');
                    }
                }
            } catch (error) {
                console.log('Could not load dynamic config:', error.message);
            }
        } else {
            // Try to load static configuration from webui-config.json
            try {
                const configPath = '/webui-config.json';
                const response = await fetch(configPath);
                if (response.ok) {
                    const config = await response.json();
                    if (config.apiUrl) {
                        this.baseUrl = config.apiUrl;
                    }
                    if (config.apiKey && config.apiKey !== '') {
                        this.token = config.apiKey;
                        this.configLoaded = true;
                        console.log('Loaded API configuration from webui-config.json');
                    }
                }
            } catch (error) {
                // Config file not found or error reading it, that's okay
                console.log('No webui-config.json found, using defaults');
            }
        }

        // Then load any saved configuration (user overrides)
        const savedConfig = Utils.getFromStorage('api-config');
        if (savedConfig) {
            // Only override if user has explicitly saved settings
            if (savedConfig.baseUrl) {
                this.baseUrl = savedConfig.baseUrl;
            }
            if (savedConfig.token) {
                this.token = savedConfig.token;
            }
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
            onProgress = null,
            timeout = 30000  // Default 30 second timeout
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
                ...headers
            }
        };
        
        // Add appropriate authentication header based on mode
        if (this.token) {
            // Use X-API-KEY header for single-user mode, Authorization Bearer for multi-user mode
            if (this.authMode === 'single-user') {
                fetchOptions.headers['X-API-KEY'] = this.token;
            } else if (this.authMode === 'multi-user') {
                // Multi-user mode uses Bearer token
                fetchOptions.headers['Authorization'] = `Bearer ${this.token}`;
            } else {
                // Unknown mode - try X-API-KEY first (most common for manual setup)
                fetchOptions.headers['X-API-KEY'] = this.token;
            }
        }

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
            // Create abort controller for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);
            
            fetchOptions.signal = controller.signal;
            
            const response = await fetch(url.toString(), fetchOptions);
            clearTimeout(timeoutId);
            
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
                let errorDetails = null;
                
                try {
                    const contentType = response.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        errorDetails = await response.json();
                        // Handle different error response formats
                        if (typeof errorDetails === 'object') {
                            // OpenAI-style error format (used by evaluations endpoint)
                            if (errorDetails.error && typeof errorDetails.error === 'object') {
                                errorMessage = errorDetails.error.message || errorDetails.error.detail || errorMessage;
                            }
                            // FastAPI default error format
                            else if (errorDetails.detail) {
                                // detail can be a string or an array of validation errors
                                if (typeof errorDetails.detail === 'string') {
                                    errorMessage = errorDetails.detail;
                                } else if (Array.isArray(errorDetails.detail)) {
                                    // Validation errors array
                                    errorMessage = errorDetails.detail.map(err => 
                                        err.msg || err.message || JSON.stringify(err)
                                    ).join(', ');
                                } else if (typeof errorDetails.detail === 'object') {
                                    // Complex error object in detail
                                    errorMessage = errorDetails.detail.message || JSON.stringify(errorDetails.detail);
                                }
                            }
                            // Simple message field
                            else if (errorDetails.message) {
                                errorMessage = errorDetails.message;
                            }
                        }
                    } else {
                        const errorText = await response.text();
                        if (errorText) {
                            errorMessage = `${errorMessage}: ${errorText}`;
                        }
                    }
                } catch (e) {
                    // If response parsing fails, use original message
                    console.warn('Failed to parse error response:', e);
                }
                
                const error = new Error(errorMessage);
                error.status = response.status;
                error.statusText = response.statusText;
                error.details = errorDetails;
                error.response = response;
                throw error;
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
            // Enhanced error handling
            const duration = Date.now() - startTime;
            
            // Check if it's a timeout error
            if (error.name === 'AbortError') {
                error.message = `Request timeout after ${timeout}ms`;
                error.isTimeout = true;
            }
            
            // Record error in history with more details
            this.addToHistory({
                method,
                path,
                url: url.toString(),
                timestamp: startTime,
                duration,
                error: error.message,
                errorStatus: error.status,
                success: false
            });
            
            // Add request context to error
            error.request = {
                method,
                path,
                url: url.toString(),
                duration
            };
            
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

    // ============================================================
    // WebSocket Support
    // ============================================================

    /**
     * Create a WebSocket connection
     * @param {string} path - WebSocket endpoint path (e.g., '/api/v1/mcp/ws')
     * @param {Object} options - WebSocket options
     * @returns {WebSocketManager} WebSocket manager instance
     */
    createWebSocket(path, options = {}) {
        const {
            id = `ws_${Date.now()}`,
            protocols = [],
            reconnect = true,
            reconnectDelay = 1000,
            maxReconnectDelay = 30000,
            reconnectDecay = 1.5,
            maxReconnectAttempts = null,
            onOpen = null,
            onMessage = null,
            onError = null,
            onClose = null,
            onReconnecting = null,
            heartbeatInterval = 30000,
            heartbeatMessage = JSON.stringify({ type: 'ping' })
        } = options;

        // Convert HTTP URL to WebSocket URL
        const wsUrl = this.baseUrl.replace(/^http/, 'ws') + path;
        
        // Add token to URL if available
        const url = new URL(wsUrl);
        if (this.token) {
            url.searchParams.append('token', this.token);
        }

        // Create WebSocket manager
        const manager = new WebSocketManager({
            url: url.toString(),
            protocols,
            reconnect,
            reconnectDelay,
            maxReconnectDelay,
            reconnectDecay,
            maxReconnectAttempts,
            onOpen,
            onMessage,
            onError,
            onClose,
            onReconnecting,
            heartbeatInterval,
            heartbeatMessage
        });

        // Store the WebSocket manager
        this.websockets.set(id, manager);

        return manager;
    }

    /**
     * Get existing WebSocket connection
     * @param {string} id - WebSocket ID
     * @returns {WebSocketManager|null} WebSocket manager or null if not found
     */
    getWebSocket(id) {
        return this.websockets.get(id) || null;
    }

    /**
     * Close WebSocket connection
     * @param {string} id - WebSocket ID
     */
    closeWebSocket(id) {
        const manager = this.websockets.get(id);
        if (manager) {
            manager.close();
            this.websockets.delete(id);
        }
    }

    /**
     * Close all WebSocket connections
     */
    closeAllWebSockets() {
        this.websockets.forEach(manager => manager.close());
        this.websockets.clear();
    }

    /**
     * Get all active WebSocket connections
     * @returns {Array} Array of WebSocket IDs and their states
     */
    getActiveWebSockets() {
        const active = [];
        this.websockets.forEach((manager, id) => {
            active.push({
                id,
                url: manager.url,
                readyState: manager.readyState,
                readyStateText: manager.readyStateText,
                reconnectCount: manager.reconnectCount
            });
        });
        return active;
    }
}

/**
 * WebSocket Manager class with auto-reconnect and heartbeat support
 */
class WebSocketManager {
    constructor(options) {
        this.url = options.url;
        this.protocols = options.protocols;
        this.reconnect = options.reconnect;
        this.reconnectDelay = options.reconnectDelay;
        this.maxReconnectDelay = options.maxReconnectDelay;
        this.reconnectDecay = options.reconnectDecay;
        this.maxReconnectAttempts = options.maxReconnectAttempts;
        this.onOpen = options.onOpen;
        this.onMessage = options.onMessage;
        this.onError = options.onError;
        this.onClose = options.onClose;
        this.onReconnecting = options.onReconnecting;
        this.heartbeatInterval = options.heartbeatInterval;
        this.heartbeatMessage = options.heartbeatMessage;

        this.ws = null;
        this.reconnectCount = 0;
        this.reconnectTimeout = null;
        this.heartbeatTimeout = null;
        this.messageQueue = [];
        this.isIntentionallyClosed = false;

        this.connect();
    }

    connect() {
        try {
            this.ws = new WebSocket(this.url, this.protocols);
            this.setupEventHandlers();
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            if (this.onError) this.onError(error);
            this.handleReconnect();
        }
    }

    setupEventHandlers() {
        this.ws.onopen = (event) => {
            console.log('WebSocket connected:', this.url);
            this.reconnectCount = 0;
            this.reconnectDelay = this.constructor.reconnectDelay;
            
            // Process queued messages
            while (this.messageQueue.length > 0) {
                const message = this.messageQueue.shift();
                this.send(message);
            }

            // Start heartbeat
            this.startHeartbeat();

            if (this.onOpen) this.onOpen(event);
        };

        this.ws.onmessage = (event) => {
            // Reset heartbeat on any message
            this.startHeartbeat();

            // Try to parse JSON messages
            let data = event.data;
            try {
                data = JSON.parse(event.data);
            } catch (e) {
                // Not JSON, use as-is
            }

            if (this.onMessage) this.onMessage(data, event);
        };

        this.ws.onerror = (event) => {
            console.error('WebSocket error:', event);
            if (this.onError) this.onError(event);
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            this.stopHeartbeat();
            
            if (this.onClose) this.onClose(event);

            if (!this.isIntentionallyClosed && this.reconnect) {
                this.handleReconnect();
            }
        };
    }

    handleReconnect() {
        if (this.maxReconnectAttempts && this.reconnectCount >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }

        this.reconnectCount++;
        const delay = Math.min(
            this.reconnectDelay * Math.pow(this.reconnectDecay, this.reconnectCount - 1),
            this.maxReconnectDelay
        );

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectCount})`);
        
        if (this.onReconnecting) {
            this.onReconnecting(this.reconnectCount, delay);
        }

        this.reconnectTimeout = setTimeout(() => {
            this.connect();
        }, delay);
    }

    startHeartbeat() {
        this.stopHeartbeat();
        
        if (this.heartbeatInterval && this.heartbeatMessage) {
            this.heartbeatTimeout = setTimeout(() => {
                if (this.isConnected()) {
                    this.send(this.heartbeatMessage);
                    this.startHeartbeat();
                }
            }, this.heartbeatInterval);
        }
    }

    stopHeartbeat() {
        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
        }
    }

    /**
     * Send a message through WebSocket
     * @param {string|Object} message - Message to send (will be JSON stringified if object)
     */
    send(message) {
        if (typeof message === 'object') {
            message = JSON.stringify(message);
        }

        if (this.isConnected()) {
            try {
                this.ws.send(message);
                return true;
            } catch (error) {
                console.error('Failed to send message:', error);
                this.messageQueue.push(message);
                return false;
            }
        } else {
            // Queue message for sending when connected
            this.messageQueue.push(message);
            return false;
        }
    }

    /**
     * Close the WebSocket connection
     * @param {number} code - Close code
     * @param {string} reason - Close reason
     */
    close(code = 1000, reason = 'Normal closure') {
        this.isIntentionallyClosed = true;
        
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }

        this.stopHeartbeat();

        if (this.ws) {
            this.ws.close(code, reason);
            this.ws = null;
        }
    }

    /**
     * Check if WebSocket is connected
     * @returns {boolean} Connection status
     */
    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }

    /**
     * Get WebSocket ready state
     * @returns {number} Ready state
     */
    get readyState() {
        return this.ws ? this.ws.readyState : WebSocket.CLOSED;
    }

    /**
     * Get human-readable ready state
     * @returns {string} Ready state text
     */
    get readyStateText() {
        if (!this.ws) return 'CLOSED';
        switch (this.ws.readyState) {
            case WebSocket.CONNECTING: return 'CONNECTING';
            case WebSocket.OPEN: return 'OPEN';
            case WebSocket.CLOSING: return 'CLOSING';
            case WebSocket.CLOSED: return 'CLOSED';
            default: return 'UNKNOWN';
        }
    }

    /**
     * Add event listener
     * @param {string} event - Event name (open, message, error, close)
     * @param {Function} handler - Event handler
     */
    on(event, handler) {
        switch (event) {
            case 'open': this.onOpen = handler; break;
            case 'message': this.onMessage = handler; break;
            case 'error': this.onError = handler; break;
            case 'close': this.onClose = handler; break;
            case 'reconnecting': this.onReconnecting = handler; break;
        }
    }

    /**
     * Remove event listener
     * @param {string} event - Event name
     */
    off(event) {
        switch (event) {
            case 'open': this.onOpen = null; break;
            case 'message': this.onMessage = null; break;
            case 'error': this.onError = null; break;
            case 'close': this.onClose = null; break;
            case 'reconnecting': this.onReconnecting = null; break;
        }
    }
}

// Create global instance
const apiClient = new APIClient();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { APIClient, apiClient, WebSocketManager };
}