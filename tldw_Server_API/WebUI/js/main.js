/**
 * Main JavaScript file for API WebUI
 */

class WebUI {
    constructor() {
        this.loadedContentGroups = new Set();
        this.activeTopTabButton = null;
        this.activeSubTabButton = null;
        this.theme = 'light';
        this.apiStatusCheckInterval = null;
        this.init();
    }

    async init() {
        // Load saved theme
        this.loadTheme();

        // Initialize tabs
        this.initTabs();

        // Initialize global settings
        this.initGlobalSettings();

        // Start API status check
        this.startApiStatusCheck();

        // Initialize keyboard shortcuts
        this.initKeyboardShortcuts();

        // Load default tab
        this.loadDefaultTab();

        // Initialize search functionality
        this.initSearch();

        console.log('WebUI initialized successfully');
    }

    loadTheme() {
        const savedTheme = Utils.getFromStorage('theme') || 'light';
        this.setTheme(savedTheme);
    }

    setTheme(theme) {
        this.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        Utils.saveToStorage('theme', theme);
        
        // Update theme toggle button
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.innerHTML = theme === 'dark' ? '☀️' : '🌙';
            themeToggle.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
        }
    }

    toggleTheme() {
        this.setTheme(this.theme === 'dark' ? 'light' : 'dark');
    }

    initTabs() {
        // Top level tabs
        const topTabButtons = document.querySelectorAll('.top-tab-button');
        topTabButtons.forEach(btn => {
            btn.addEventListener('click', () => this.activateTopTab(btn));
        });

        // Sub-level tabs
        const subTabButtons = document.querySelectorAll('.sub-tab-button');
        subTabButtons.forEach(btn => {
            btn.addEventListener('click', () => this.activateSubTab(btn));
        });
    }

    async activateTopTab(tabButton) {
        try {
            // Remove active class from previous tab
            if (this.activeTopTabButton) {
                this.activeTopTabButton.classList.remove('active');
            }

            // Set new active tab
            this.activeTopTabButton = tabButton;
            this.activeTopTabButton.classList.add('active');

            // Get tab name
            const topTabName = tabButton.dataset.toptab;

            // Hide all sub-tab rows
            document.querySelectorAll('.sub-tab-row').forEach(row => {
                row.classList.remove('active');
            });

            // Show corresponding sub-tab row
            const subTabRow = document.getElementById(`${topTabName}-subtabs`);
            if (subTabRow) {
                subTabRow.classList.add('active');

                // Activate first sub-tab
                const firstSubTab = subTabRow.querySelector('.sub-tab-button');
                if (firstSubTab) {
                    await this.activateSubTab(firstSubTab);
                }
            } else {
                // Handle tabs without sub-tabs (like Global Settings)
                this.showContent(topTabName);
            }

            // Save active tab to storage
            Utils.saveToStorage('active-top-tab', topTabName);
        } catch (error) {
            console.error('Error activating top tab:', error);
            // Try to show Global Settings as fallback
            const globalSettings = document.getElementById('tabGlobalSettings');
            if (globalSettings) {
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                globalSettings.classList.add('active');
            }
        }
    }

    async activateSubTab(tabButton) {
        const parentRow = tabButton.closest('.sub-tab-row');
        if (!parentRow) return;

        // Remove active class from all sub-tabs in this row
        parentRow.querySelectorAll('.sub-tab-button').forEach(btn => {
            btn.classList.remove('active');
        });

        // Set new active sub-tab
        this.activeSubTabButton = tabButton;
        this.activeSubTabButton.classList.add('active');

        // Get content ID and load group
        const contentId = tabButton.dataset.contentId;
        const loadGroup = tabButton.dataset.loadGroup;

        // Load content if not already loaded
        if (loadGroup && !this.loadedContentGroups.has(loadGroup)) {
            try {
                if (typeof Loading !== 'undefined' && Loading) {
                    Loading.show(document.querySelector('.content-container'), 'Loading content...');
                }
                await this.loadContentGroup(loadGroup, contentId);
                this.loadedContentGroups.add(loadGroup);
            } catch (error) {
                console.error(`Failed to load content group ${loadGroup}:`, error);
                if (typeof Toast !== 'undefined' && Toast) {
                    Toast.error(`Failed to load content: ${error.message}`);
                } else {
                    // Fallback alert if Toast not available
                    alert(`Failed to load content: ${error.message}`);
                }
            } finally {
                if (typeof Loading !== 'undefined' && Loading) {
                    Loading.hide(document.querySelector('.content-container'));
                }
            }
        }

        // Show the content
        this.showContent(contentId);

        // Save active sub-tab to storage
        Utils.saveToStorage('active-sub-tab', contentId);

        // Initialize specific tab functionality if needed
        if (contentId === 'tabChatCompletions' && typeof initializeChatCompletionsTab === 'function') {
            initializeChatCompletionsTab();
        }
        
        if (contentId === 'tabEvalsOpenAI' || contentId === 'tabEvalsGEval') {
            if (typeof initializeEvaluationsTab === 'function') {
                initializeEvaluationsTab();
            }
        }
        
        // Initialize model dropdowns for tabs that have LLM selection
        // This includes chat, media processing, and evaluation tabs
        const tabsWithModelSelection = [
            'tabChatCompletions', 'tabCharacterChat', 'tabConversations',
            'tabMediaIngestion', 'tabMediaProcessingNoDB', 
            'tabEvalsOpenAI', 'tabEvalsGEval'
        ];
        
        if (tabsWithModelSelection.includes(contentId)) {
            // Small delay to ensure DOM is ready
            setTimeout(() => {
                if (typeof populateModelDropdowns === 'function') {
                    populateModelDropdowns();
                }
            }, 100);
        }
    }

    async loadContentGroup(groupName, targetContentId) {
        const response = await fetch(`tabs/${groupName}_content.html`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} for tabs/${groupName}_content.html`);
        }

        const html = await response.text();
        const mainContentArea = document.getElementById('main-content-area');
        mainContentArea.insertAdjacentHTML('beforeend', html);

        // Re-initialize form handlers for newly loaded content
        this.initFormHandlers();
        
        // After loading content, ensure all newly loaded tabs are hidden initially
        // This is important when loading multiple tabs from a single HTML file
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // Initialize model dropdowns for groups that contain LLM-using tabs
        const groupsWithModelSelection = ['chat', 'media', 'evaluations'];
        if (groupsWithModelSelection.includes(groupName)) {
            // Populate dropdowns after DOM is updated
            setTimeout(() => {
                if (typeof populateModelDropdowns === 'function') {
                    populateModelDropdowns();
                }
            }, 100);
        }
    }

    showContent(contentId) {
        // Hide all tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        // Show selected content
        const content = document.getElementById(contentId);
        if (content) {
            content.classList.add('active');
            console.log(`Showing tab: ${contentId}`);
        } else {
            console.warn(`Tab content not found: ${contentId}`);
        }
    }

    loadDefaultTab() {
        // Try to restore previously active tab
        const savedTopTab = Utils.getFromStorage('active-top-tab');
        const savedSubTab = Utils.getFromStorage('active-sub-tab');

        if (savedTopTab) {
            const tabButton = document.querySelector(`.top-tab-button[data-toptab="${savedTopTab}"]`);
            if (tabButton) {
                this.activateTopTab(tabButton);
                return;
            }
        }

        // Default to General tab (which has Global Settings)
        const defaultTab = document.querySelector('.top-tab-button[data-toptab="general"]');
        if (defaultTab) {
            this.activateTopTab(defaultTab);
        } else {
            // Fallback to first available tab
            const firstTab = document.querySelector('.top-tab-button');
            if (firstTab) {
                this.activateTopTab(firstTab);
            }
        }
        
        // Ensure at least one content tab is visible
        setTimeout(() => {
            const activeTabs = document.querySelectorAll('.tab-content.active');
            if (activeTabs.length === 0) {
                // Force show Global Settings as fallback
                const globalSettings = document.getElementById('tabGlobalSettings');
                if (globalSettings) {
                    globalSettings.classList.add('active');
                    console.log('Forced Global Settings tab to be visible');
                }
            }
        }, 100);
    }

    async initGlobalSettings() {
        // Wait for API client to load configuration
        if (apiClient.init) {
            await apiClient.init();
        }
        
        // Load saved API configuration
        const baseUrlInput = document.getElementById('baseUrl');
        const apiKeyInput = document.getElementById('apiKeyInput');

        if (baseUrlInput) {
            baseUrlInput.value = apiClient.baseUrl;
            baseUrlInput.addEventListener('change', (e) => {
                apiClient.setBaseUrl(e.target.value);
                if (typeof Toast !== 'undefined' && Toast) {
                    Toast.success('API base URL updated');
                }
                this.checkApiStatus();
            });
        }

        if (apiKeyInput) {
            apiKeyInput.value = apiClient.token;
            
            // Show indicator if config was auto-loaded
            if (apiClient.configLoaded && apiClient.token) {
                apiKeyInput.placeholder = 'Auto-configured from server';
                // Add visual indicator
                const label = apiKeyInput.previousElementSibling;
                if (label && label.tagName === 'LABEL') {
                    label.innerHTML = 'API Token: <span style="color: green;">✓ Auto-configured</span>';
                }
            }
            
            apiKeyInput.addEventListener('change', (e) => {
                apiClient.setToken(e.target.value);
                if (typeof Toast !== 'undefined' && Toast) {
                    Toast.success('API token updated');
                }
            });
        }

        // Add theme toggle handler
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }
    }

    async checkApiStatus() {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.api-status-text');

        try {
            const startTime = Date.now();
            const health = await apiClient.checkHealth();
            const responseTime = Date.now() - startTime;
            
            if (health.online) {
                if (statusDot) {
                    statusDot.classList.remove('offline');
                    statusDot.classList.remove('slow');
                    // Add slow indicator if response is slow
                    if (responseTime > 1000) {
                        statusDot.classList.add('slow');
                    }
                }
                if (statusText) {
                    if (responseTime > 1000) {
                        statusText.textContent = `Connected (${responseTime}ms)`;
                    } else {
                        statusText.textContent = 'Connected';
                    }
                    statusText.title = `API: ${apiClient.baseUrl}\nResponse time: ${responseTime}ms`;
                }
            } else {
                if (statusDot) {
                    statusDot.classList.add('offline');
                    statusDot.classList.remove('slow');
                }
                if (statusText) {
                    statusText.textContent = 'API Offline';
                    statusText.title = `Cannot reach API at ${apiClient.baseUrl}`;
                }
            }
        } catch (error) {
            if (statusDot) {
                statusDot.classList.add('offline');
                statusDot.classList.remove('slow');
            }
            if (statusText) {
                // More descriptive error messages
                if (error.message.includes('Failed to fetch')) {
                    statusText.textContent = 'API Unreachable';
                    statusText.title = `Cannot connect to ${apiClient.baseUrl}\nCheck if the API server is running`;
                } else if (error.status === 401) {
                    statusText.textContent = 'Auth Failed';
                    statusText.title = 'Authentication failed - check your API token';
                } else {
                    statusText.textContent = `Error (${error.status || 'Network'})`;
                    statusText.title = error.message;
                }
            }
        }
    }

    startApiStatusCheck() {
        // Initial check
        this.checkApiStatus();

        // Set up periodic check every 30 seconds
        this.apiStatusCheckInterval = setInterval(() => {
            this.checkApiStatus();
        }, 30000);
    }

    stopApiStatusCheck() {
        if (this.apiStatusCheckInterval) {
            clearInterval(this.apiStatusCheckInterval);
            this.apiStatusCheckInterval = null;
        }
    }

    initKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K: Focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.getElementById('endpoint-search');
                if (searchInput) {
                    searchInput.focus();
                }
            }

            // Ctrl/Cmd + Shift + D: Toggle theme
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'D') {
                e.preventDefault();
                this.toggleTheme();
            }

            // Ctrl/Cmd + Shift + H: Show history
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'H') {
                e.preventDefault();
                this.showRequestHistory();
            }

            // Escape: Close modals
            if (e.key === 'Escape') {
                // This will be handled by individual modals
            }
        });
    }

    initSearch() {
        const searchInput = document.getElementById('endpoint-search');
        if (!searchInput) return;

        const searchHandler = Utils.debounce((e) => {
            const query = e.target.value.toLowerCase();
            this.filterEndpoints(query);
        }, 300);

        searchInput.addEventListener('input', searchHandler);
    }

    filterEndpoints(query) {
        const endpoints = document.querySelectorAll('.endpoint-section');
        let visibleCount = 0;

        endpoints.forEach(endpoint => {
            const title = endpoint.querySelector('h2')?.textContent.toLowerCase() || '';
            const path = endpoint.querySelector('.endpoint-path')?.textContent.toLowerCase() || '';
            
            if (title.includes(query) || path.includes(query)) {
                endpoint.style.display = 'block';
                visibleCount++;
            } else {
                endpoint.style.display = 'none';
            }
        });

        // Show message if no results
        const noResultsMsg = document.getElementById('no-search-results');
        if (noResultsMsg) {
            noResultsMsg.style.display = visibleCount === 0 ? 'block' : 'none';
        }
    }

    showRequestHistory() {
        const history = apiClient.getHistory();
        
        if (history.length === 0) {
            if (typeof Toast !== 'undefined' && Toast) {
                Toast.info('No request history available');
            } else {
                alert('No request history available');
            }
            return;
        }

        let historyHtml = `
            <div class="history-list">
                <div class="history-controls mb-3">
                    <button class="btn btn-sm btn-danger" onclick="webUI.clearHistory()">Clear History</button>
                </div>
                <div class="history-items">
        `;

        history.forEach((item, index) => {
            const statusClass = item.success ? 'success' : 'error';
            const timestamp = Utils.formatDate(item.timestamp);
            const duration = Utils.formatDuration(item.duration);

            historyHtml += `
                <div class="history-item ${statusClass}">
                    <div class="history-item-header">
                        <span class="endpoint-method ${item.method.toLowerCase()}">${item.method}</span>
                        <span class="history-path">${item.path}</span>
                        <span class="history-status">${item.status || 'Error'}</span>
                    </div>
                    <div class="history-item-details">
                        <span class="history-timestamp">${timestamp}</span>
                        <span class="history-duration">${duration}</span>
                    </div>
                    ${item.error ? `<div class="history-error">${item.error}</div>` : ''}
                </div>
            `;
        });

        historyHtml += `
                </div>
            </div>
        `;

        const modal = new Modal({
            title: 'Request History',
            content: historyHtml,
            size: 'large'
        });
        modal.show();
    }

    clearHistory() {
        apiClient.clearHistory();
        if (typeof Toast !== 'undefined' && Toast) {
            Toast.success('Request history cleared');
        }
        // Close any open modals
        document.querySelectorAll('.modal').forEach(modal => {
            modal.remove();
        });
        document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
            backdrop.remove();
        });
    }

    initFormHandlers() {
        // Initialize copy buttons for all pre elements
        document.querySelectorAll('pre').forEach(pre => {
            if (!pre.querySelector('.copy-button')) {
                const copyBtn = document.createElement('button');
                copyBtn.className = 'copy-button';
                copyBtn.textContent = 'Copy';
                copyBtn.onclick = async () => {
                    const text = pre.textContent.replace('Copy', '').trim();
                    const success = await Utils.copyToClipboard(text);
                    if (success) {
                        copyBtn.textContent = 'Copied!';
                        setTimeout(() => {
                            copyBtn.textContent = 'Copy';
                        }, 2000);
                    }
                };
                pre.appendChild(copyBtn);
            }
        });

        // Initialize file inputs with better UI
        document.querySelectorAll('input[type="file"]').forEach(input => {
            if (!input.parentElement.classList.contains('file-input-wrapper')) {
                const wrapper = document.createElement('div');
                wrapper.className = 'file-input-wrapper';
                
                const label = document.createElement('label');
                label.className = 'file-input-label';
                label.innerHTML = `
                    <span class="file-input-icon">📁</span>
                    <span class="file-input-text">Choose file or drag here</span>
                `;

                input.parentNode.insertBefore(wrapper, input);
                wrapper.appendChild(input);
                wrapper.appendChild(label);

                // Handle file selection
                input.addEventListener('change', (e) => {
                    if (e.target.files.length > 0) {
                        const file = e.target.files[0];
                        wrapper.classList.add('has-file');
                        label.querySelector('.file-input-text').textContent = file.name;
                        label.querySelector('.file-input-icon').textContent = '📄';
                    } else {
                        wrapper.classList.remove('has-file');
                        label.querySelector('.file-input-text').textContent = 'Choose file or drag here';
                        label.querySelector('.file-input-icon').textContent = '📁';
                    }
                });

                // Handle drag and drop
                wrapper.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    wrapper.classList.add('drag-over');
                });

                wrapper.addEventListener('dragleave', () => {
                    wrapper.classList.remove('drag-over');
                });

                wrapper.addEventListener('drop', (e) => {
                    e.preventDefault();
                    wrapper.classList.remove('drag-over');
                    if (e.dataTransfer.files.length > 0) {
                        input.files = e.dataTransfer.files;
                        input.dispatchEvent(new Event('change'));
                    }
                });
            }
        });
    }
}

// Initialize WebUI when DOM is ready
let webUI;
document.addEventListener('DOMContentLoaded', () => {
    webUI = new WebUI();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebUI;
}