/**
 * Reusable UI Components for the API WebUI
 */

class ToastManager {
    constructor() {
        this.container = null;
        this.init();
    }

    init() {
        if (!document.getElementById('toast-container')) {
            this.container = document.createElement('div');
            this.container.id = 'toast-container';
            this.container.className = 'toast-container';
            document.body.appendChild(this.container);
        } else {
            this.container = document.getElementById('toast-container');
        }
    }

    show(message, type = 'info', duration = 5000, title = null) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };

        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <div class="toast-content">
                ${title ? `<div class="toast-title">${title}</div>` : ''}
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" aria-label="Close">×</button>
        `;

        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.onclick = () => this.remove(toast);

        this.container.appendChild(toast);

        // Auto remove after duration
        if (duration > 0) {
            setTimeout(() => this.remove(toast), duration);
        }

        return toast;
    }

    remove(toast) {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }

    success(message, duration = 5000) {
        return this.show(message, 'success', duration, 'Success');
    }

    error(message, duration = 5000) {
        return this.show(message, 'error', duration, 'Error');
    }

    warning(message, duration = 5000) {
        return this.show(message, 'warning', duration, 'Warning');
    }

    info(message, duration = 5000) {
        return this.show(message, 'info', duration, 'Info');
    }
}

class LoadingIndicator {
    constructor() {
        this.activeLoaders = new Map();
    }

    show(element, message = 'Loading...') {
        if (!element) return;

        const loaderId = Utils.generateId('loader');
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.id = loaderId;
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-message">${message}</div>
            </div>
        `;

        element.style.position = 'relative';
        element.appendChild(overlay);
        this.activeLoaders.set(element, loaderId);

        return loaderId;
    }

    hide(element) {
        if (!element || !this.activeLoaders.has(element)) return;

        const loaderId = this.activeLoaders.get(element);
        const overlay = document.getElementById(loaderId);
        if (overlay) {
            overlay.remove();
        }
        this.activeLoaders.delete(element);
    }

    hideAll() {
        this.activeLoaders.forEach((loaderId, element) => {
            const overlay = document.getElementById(loaderId);
            if (overlay) {
                overlay.remove();
            }
        });
        this.activeLoaders.clear();
    }
}

class Modal {
    constructor(options = {}) {
        this.options = {
            title: 'Modal',
            content: '',
            size: 'medium', // small, medium, large, full
            closeButton: true,
            backdrop: true,
            keyboard: true,
            ...options
        };
        this.modal = null;
        this.backdrop = null;
        this.create();
    }

    create() {
        // Create backdrop
        if (this.options.backdrop) {
            this.backdrop = document.createElement('div');
            this.backdrop.className = 'modal-backdrop';
            this.backdrop.onclick = () => {
                if (this.options.backdrop === 'static') return;
                this.close();
            };
        }

        // Create modal
        this.modal = document.createElement('div');
        this.modal.className = `modal modal-${this.options.size}`;
        this.modal.innerHTML = `
            <div class="modal-header">
                <h2 class="modal-title">${this.options.title}</h2>
                ${this.options.closeButton ? '<button class="modal-close" aria-label="Close">×</button>' : ''}
            </div>
            <div class="modal-body">
                ${this.options.content}
            </div>
            ${this.options.footer ? `<div class="modal-footer">${this.options.footer}</div>` : ''}
        `;

        if (this.options.closeButton) {
            const closeBtn = this.modal.querySelector('.modal-close');
            closeBtn.onclick = () => this.close();
        }

        // Keyboard events
        if (this.options.keyboard) {
            document.addEventListener('keydown', this.handleKeydown.bind(this));
        }
    }

    handleKeydown(e) {
        if (e.key === 'Escape') {
            this.close();
        }
    }

    show() {
        if (this.backdrop) {
            document.body.appendChild(this.backdrop);
        }
        document.body.appendChild(this.modal);
        document.body.style.overflow = 'hidden';
        
        // Focus management
        const focusable = this.modal.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (focusable.length) {
            focusable[0].focus();
        }
    }

    close() {
        if (this.backdrop && this.backdrop.parentNode) {
            this.backdrop.parentNode.removeChild(this.backdrop);
        }
        if (this.modal && this.modal.parentNode) {
            this.modal.parentNode.removeChild(this.modal);
        }
        document.body.style.overflow = '';
        
        if (this.options.keyboard) {
            document.removeEventListener('keydown', this.handleKeydown.bind(this));
        }

        if (this.options.onClose) {
            this.options.onClose();
        }
    }

    setContent(content) {
        const body = this.modal.querySelector('.modal-body');
        if (body) {
            body.innerHTML = content;
        }
    }
}

class JSONViewer {
    constructor(container, json, options = {}) {
        this.container = container;
        this.json = json;
        this.options = {
            expanded: 1, // Levels to expand by default
            theme: 'light',
            enableCopy: true,
            enableCollapse: true,
            ...options
        };
        this.render();
    }

    render() {
        this.container.innerHTML = '';
        const wrapper = document.createElement('div');
        wrapper.className = `json-viewer json-viewer-${this.options.theme}`;
        
        if (this.options.enableCopy) {
            const toolbar = document.createElement('div');
            toolbar.className = 'json-viewer-toolbar';
            toolbar.innerHTML = `
                <button class="btn btn-sm" onclick="Utils.copyToClipboard('${Utils.escapeHtml(JSON.stringify(this.json, null, 2))}')">
                    Copy JSON
                </button>
                <button class="btn btn-sm" onclick="Utils.downloadData(${Utils.escapeHtml(JSON.stringify(this.json))}, 'data.json')">
                    Download
                </button>
            `;
            wrapper.appendChild(toolbar);
        }

        const content = document.createElement('div');
        content.className = 'json-viewer-content';
        content.innerHTML = this.renderValue(this.json, 0);
        wrapper.appendChild(content);

        this.container.appendChild(wrapper);

        // Add collapse/expand functionality
        if (this.options.enableCollapse) {
            this.attachCollapseHandlers();
        }
    }

    renderValue(value, depth) {
        if (value === null) {
            return '<span class="json-null">null</span>';
        }
        if (typeof value === 'boolean') {
            return `<span class="json-boolean">${value}</span>`;
        }
        if (typeof value === 'number') {
            return `<span class="json-number">${value}</span>`;
        }
        if (typeof value === 'string') {
            return `<span class="json-string">"${Utils.escapeHtml(value)}"</span>`;
        }
        if (Array.isArray(value)) {
            return this.renderArray(value, depth);
        }
        if (typeof value === 'object') {
            return this.renderObject(value, depth);
        }
        return Utils.escapeHtml(String(value));
    }

    renderArray(arr, depth) {
        if (arr.length === 0) {
            return '<span class="json-bracket">[]</span>';
        }

        const expanded = depth < this.options.expanded;
        let html = `<span class="json-toggle ${expanded ? 'expanded' : 'collapsed'}" data-type="array">▼</span>`;
        html += '<span class="json-bracket">[</span>';
        html += `<div class="json-content" ${expanded ? '' : 'style="display:none"'}>`;
        
        arr.forEach((item, index) => {
            html += '<div class="json-item">';
            html += this.renderValue(item, depth + 1);
            if (index < arr.length - 1) {
                html += '<span class="json-comma">,</span>';
            }
            html += '</div>';
        });
        
        html += '</div>';
        html += '<span class="json-bracket">]</span>';
        return html;
    }

    renderObject(obj, depth) {
        const keys = Object.keys(obj);
        if (keys.length === 0) {
            return '<span class="json-bracket">{}</span>';
        }

        const expanded = depth < this.options.expanded;
        let html = `<span class="json-toggle ${expanded ? 'expanded' : 'collapsed'}" data-type="object">▼</span>`;
        html += '<span class="json-bracket">{</span>';
        html += `<div class="json-content" ${expanded ? '' : 'style="display:none"'}>`;
        
        keys.forEach((key, index) => {
            html += '<div class="json-item">';
            html += `<span class="json-key">"${Utils.escapeHtml(key)}"</span>`;
            html += '<span class="json-colon">:</span> ';
            html += this.renderValue(obj[key], depth + 1);
            if (index < keys.length - 1) {
                html += '<span class="json-comma">,</span>';
            }
            html += '</div>';
        });
        
        html += '</div>';
        html += '<span class="json-bracket">}</span>';
        return html;
    }

    attachCollapseHandlers() {
        const toggles = this.container.querySelectorAll('.json-toggle');
        toggles.forEach(toggle => {
            toggle.onclick = (e) => {
                e.stopPropagation();
                const content = toggle.nextElementSibling.nextElementSibling;
                if (toggle.classList.contains('expanded')) {
                    toggle.classList.remove('expanded');
                    toggle.classList.add('collapsed');
                    toggle.textContent = '▶';
                    content.style.display = 'none';
                } else {
                    toggle.classList.remove('collapsed');
                    toggle.classList.add('expanded');
                    toggle.textContent = '▼';
                    content.style.display = 'block';
                }
            };
        });
    }
}

// Initialize global instances
const Toast = new ToastManager();
const Loading = new LoadingIndicator();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ToastManager, LoadingIndicator, Modal, JSONViewer, Toast, Loading };
}