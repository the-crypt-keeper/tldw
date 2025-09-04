/**
 * Utility functions for the API WebUI
 */

const Utils = {
    /**
     * Debounce function to limit how often a function can be called
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Format JSON with syntax highlighting
     */
    formatJSON(json, indent = 2) {
        if (typeof json === 'string') {
            try {
                json = JSON.parse(json);
            } catch (e) {
                return json;
            }
        }
        return JSON.stringify(json, null, indent);
    },

    /**
     * Copy text to clipboard
     */
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (err) {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            try {
                document.execCommand('copy');
                document.body.removeChild(textarea);
                return true;
            } catch (err) {
                document.body.removeChild(textarea);
                return false;
            }
        }
    },

    /**
     * Format bytes to human readable string
     */
    formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    },

    /**
     * Format duration in milliseconds to human readable string
     */
    formatDuration(ms) {
        if (ms < 1000) return `${ms}ms`;
        if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
        return `${Math.floor(ms / 60000)}m ${((ms % 60000) / 1000).toFixed(0)}s`;
    },

    /**
     * Generate a unique ID
     */
    generateId(prefix = 'id') {
        return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    },

    /**
     * Escape HTML to prevent XSS - SECURE VERSION
     */
    escapeHtml(text) {
        if (typeof text !== 'string') {
            return '';
        }
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;'
        };
        return text.replace(/[&<>"'/]/g, (char) => map[char]);
    },
    
    /**
     * Safely set HTML content with proper escaping
     */
    safeSetHTML(element, html, isPreEscaped = false) {
        if (!element) return;
        if (isPreEscaped) {
            // Only use innerHTML if content is already escaped/safe
            element.innerHTML = html;
        } else {
            // Default to safe textContent
            element.textContent = html;
        }
    },
    
    /**
     * Create DOM elements safely from untrusted content
     */
    createSafeElement(tag, attrs = {}, textContent = '') {
        const elem = document.createElement(tag);
        
        // Whitelist safe attributes
        const safeAttrs = ['className', 'id', 'type', 'placeholder', 'aria-label', 'aria-describedby', 'role'];
        for (const [key, value] of Object.entries(attrs)) {
            if (safeAttrs.includes(key) || key.startsWith('data-')) {
                if (key === 'className') {
                    elem.className = value;
                } else {
                    elem.setAttribute(key, this.escapeHtml(String(value)));
                }
            }
        }
        
        if (textContent) {
            elem.textContent = textContent;
        }
        return elem;
    },

    /**
     * Parse query parameters from URL
     */
    parseQueryParams(url = window.location.href) {
        const params = new URLSearchParams(new URL(url).search);
        const result = {};
        for (const [key, value] of params) {
            result[key] = value;
        }
        return result;
    },

    /**
     * Deep merge objects
     */
    deepMerge(target, ...sources) {
        if (!sources.length) return target;
        const source = sources.shift();

        if (this.isObject(target) && this.isObject(source)) {
            for (const key in source) {
                if (this.isObject(source[key])) {
                    if (!target[key]) Object.assign(target, { [key]: {} });
                    this.deepMerge(target[key], source[key]);
                } else {
                    Object.assign(target, { [key]: source[key] });
                }
            }
        }
        return this.deepMerge(target, ...sources);
    },

    /**
     * Check if value is a plain object
     */
    isObject(item) {
        return item && typeof item === 'object' && !Array.isArray(item);
    },

    /**
     * Save data to localStorage with optional expiry
     */
    saveToStorage(key, data, expiryMinutes = null) {
        const item = {
            value: data,
            timestamp: Date.now()
        };
        if (expiryMinutes) {
            item.expiry = Date.now() + (expiryMinutes * 60 * 1000);
        }
        try {
            localStorage.setItem(key, JSON.stringify(item));
            return true;
        } catch (e) {
            console.error('Failed to save to localStorage:', e);
            return false;
        }
    },

    /**
     * Get data from localStorage with expiry check
     */
    getFromStorage(key) {
        try {
            const itemStr = localStorage.getItem(key);
            if (!itemStr) return null;
            
            const item = JSON.parse(itemStr);
            if (item.expiry && Date.now() > item.expiry) {
                localStorage.removeItem(key);
                return null;
            }
            return item.value;
        } catch (e) {
            console.error('Failed to get from localStorage:', e);
            return null;
        }
    },

    /**
     * Validate JSON string
     */
    isValidJSON(str) {
        try {
            JSON.parse(str);
            return true;
        } catch (e) {
            return false;
        }
    },

    /**
     * Create a download link for data
     */
    downloadData(data, filename, type = 'application/json') {
        const blob = new Blob([typeof data === 'string' ? data : JSON.stringify(data, null, 2)], { type });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    },

    /**
     * Escape string for use in cURL command
     */
    escapeCurlData(data) {
        return data.replace(/'/g, "'\\''");
    },

    /**
     * Validate character/conversation IDs
     */
    validateId(id, type = 'any') {
        if (!id) return false;
        
        // Check for basic ID format (alphanumeric with underscores/hyphens)
        const idPattern = /^[a-zA-Z0-9_-]+$/;
        if (!idPattern.test(id)) return false;
        
        // Type-specific validation
        switch(type) {
            case 'character':
                return /^(char_)?[a-zA-Z0-9_-]+$/.test(id);
            case 'conversation':
                return /^(conv_)?[a-zA-Z0-9_-]+$/.test(id);
            case 'numeric':
                return /^\d+$/.test(id);
            default:
                return true;
        }
    },
    
    /**
     * Validate and parse JSON safely
     */
    parseJSONSafely(jsonStr, defaultValue = null) {
        try {
            // Check for basic JSON structure
            if (typeof jsonStr !== 'string') {
                return defaultValue;
            }
            
            // Trim whitespace
            jsonStr = jsonStr.trim();
            
            // Check for empty strings
            if (!jsonStr || jsonStr === '{}' || jsonStr === '[]') {
                return defaultValue || (jsonStr === '[]' ? [] : {});
            }
            
            // Parse and validate
            const parsed = JSON.parse(jsonStr);
            
            // Additional security checks
            const jsonString = JSON.stringify(parsed);
            if (jsonString.length > 1000000) { // 1MB limit
                console.warn('JSON too large, rejecting');
                return defaultValue;
            }
            
            return parsed;
        } catch (e) {
            console.error('Failed to parse JSON:', e.message);
            return defaultValue;
        }
    },
    
    /**
     * Validate file upload
     */
    validateFileUpload(file, options = {}) {
        const {
            maxSize = 10 * 1024 * 1024, // 10MB default
            allowedTypes = [],
            allowedExtensions = []
        } = options;
        
        if (!file) return { valid: false, error: 'No file provided' };
        
        // Check file size
        if (file.size > maxSize) {
            return { valid: false, error: `File too large. Maximum size: ${this.formatBytes(maxSize)}` };
        }
        
        // Check file type
        if (allowedTypes.length > 0 && !allowedTypes.includes(file.type)) {
            return { valid: false, error: `Invalid file type. Allowed: ${allowedTypes.join(', ')}` };
        }
        
        // Check file extension
        if (allowedExtensions.length > 0) {
            const ext = file.name.split('.').pop().toLowerCase();
            if (!allowedExtensions.includes(ext)) {
                return { valid: false, error: `Invalid file extension. Allowed: ${allowedExtensions.join(', ')}` };
            }
        }
        
        return { valid: true };
    },

    /**
     * Retry a function with exponential backoff
     */
    async retryWithBackoff(fn, options = {}) {
        const {
            maxRetries = 3,
            initialDelay = 1000,
            maxDelay = 10000,
            backoffFactor = 2,
            shouldRetry = (error) => true
        } = options;
        
        let lastError;
        
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return await fn();
            } catch (error) {
                lastError = error;
                
                // Check if we should retry
                if (attempt === maxRetries || !shouldRetry(error)) {
                    throw error;
                }
                
                // Calculate delay with exponential backoff
                const delay = Math.min(
                    initialDelay * Math.pow(backoffFactor, attempt),
                    maxDelay
                );
                
                console.log(`Retry attempt ${attempt + 1}/${maxRetries} after ${delay}ms`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
        
        throw lastError;
    },
    
    /**
     * Error handler with user-friendly messages
     */
    handleError(error, context = '') {
        console.error(`Error in ${context}:`, error);
        
        // Determine error type and provide user-friendly message
        let userMessage = 'An unexpected error occurred';
        let technicalDetails = error.message || 'Unknown error';
        let recoveryAction = null;
        
        if (error.name === 'AbortError') {
            userMessage = 'Request timed out';
            recoveryAction = 'Try again with a shorter request or check your connection';
        } else if (error.message?.includes('Failed to fetch')) {
            userMessage = 'Connection failed';
            recoveryAction = 'Check your internet connection and API server status';
        } else if (error.message?.includes('401') || error.message?.includes('Unauthorized')) {
            userMessage = 'Authentication failed';
            recoveryAction = 'Check your API key or login credentials';
        } else if (error.message?.includes('403') || error.message?.includes('Forbidden')) {
            userMessage = 'Access denied';
            recoveryAction = 'You don\'t have permission for this action';
        } else if (error.message?.includes('404')) {
            userMessage = 'Resource not found';
            recoveryAction = 'The requested item may have been deleted or moved';
        } else if (error.message?.includes('429')) {
            userMessage = 'Rate limit exceeded';
            recoveryAction = 'Please wait a moment before trying again';
        } else if (error.message?.includes('500') || error.message?.includes('Internal Server')) {
            userMessage = 'Server error';
            recoveryAction = 'The server encountered an error. Try again later';
        } else if (error.message?.includes('JSON')) {
            userMessage = 'Invalid data format';
            recoveryAction = 'Check your input data and try again';
        }
        
        return {
            userMessage,
            technicalDetails,
            recoveryAction,
            originalError: error
        };
    },
    
    /**
     * Format timestamp to readable date
     */
    formatDate(timestamp, includeTime = true) {
        const date = new Date(timestamp);
        const options = {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        };
        if (includeTime) {
            options.hour = '2-digit';
            options.minute = '2-digit';
        }
        return date.toLocaleString(undefined, options);
    },

    /**
     * Highlight JSON syntax
     */
    syntaxHighlightJSON(json) {
        if (typeof json !== 'string') {
            json = JSON.stringify(json, null, 2);
        }
        json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            return '<span class="' + cls + '">' + match + '</span>';
        });
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Utils;
}