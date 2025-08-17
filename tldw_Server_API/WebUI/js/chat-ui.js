/**
 * Chat UI specific functionality
 */

class ChatUI {
    constructor() {
        this.messageIdCounter = 0;
        this.messageTemplates = {
            system: 'You are a helpful assistant that provides concise answers.',
            user: 'What is the capital of France?',
            assistant: 'The capital of France is Paris.',
            tool: ''
        };
        this.presets = [];
        this.loadPresets();
    }

    loadPresets() {
        // Load saved presets
        const saved = Utils.getFromStorage('chat-presets');
        if (saved) {
            this.presets = saved;
        } else {
            // Default presets
            this.presets = [
                {
                    name: 'Simple Q&A',
                    messages: [
                        { role: 'system', content: 'You are a helpful assistant.' },
                        { role: 'user', content: 'Hello!' }
                    ],
                    temperature: 0.7,
                    max_tokens: 1024
                },
                {
                    name: 'Code Assistant',
                    messages: [
                        { role: 'system', content: 'You are a programming assistant. Provide clear, concise code examples.' },
                        { role: 'user', content: 'How do I sort an array in Python?' }
                    ],
                    temperature: 0.2,
                    max_tokens: 2048
                },
                {
                    name: 'Creative Writing',
                    messages: [
                        { role: 'system', content: 'You are a creative writing assistant. Be imaginative and descriptive.' },
                        { role: 'user', content: 'Write a short story opening.' }
                    ],
                    temperature: 0.9,
                    max_tokens: 2048
                }
            ];
        }
    }

    savePresets() {
        Utils.saveToStorage('chat-presets', this.presets);
    }

    addMessage(prefix = 'chatCompletions', role = 'user', content = '') {
        const container = document.getElementById(`${prefix}_messagesContainer`);
        if (!container) {
            console.error('Messages container not found for prefix:', prefix);
            return;
        }

        const id = this.messageIdCounter++;
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message-entry';
        messageDiv.id = `${prefix}_message_entry_${id}`;
        messageDiv.dataset.messageId = id;

        messageDiv.innerHTML = `
            <div class="message-header">
                <span class="message-role-badge ${role}">${role}</span>
                <button class="remove-message-btn" onclick="chatUI.removeMessage('${prefix}', ${id})" aria-label="Remove message">
                    ×
                </button>
            </div>
            <div class="message-body">
                <div class="form-group">
                    <label for="${prefix}_message_role_${id}">Role:</label>
                    <select id="${prefix}_message_role_${id}" class="message-role-select" onchange="chatUI.handleRoleChange('${prefix}', ${id})">
                        <option value="system" ${role === 'system' ? 'selected' : ''}>System</option>
                        <option value="user" ${role === 'user' ? 'selected' : ''}>User</option>
                        <option value="assistant" ${role === 'assistant' ? 'selected' : ''}>Assistant</option>
                        <option value="tool" ${role === 'tool' ? 'selected' : ''}>Tool</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="${prefix}_message_content_${id}">Content:</label>
                    <div class="message-content-wrapper">
                        <textarea 
                            id="${prefix}_message_content_${id}" 
                            class="message-content-area" 
                            rows="3" 
                            placeholder="Enter message content..."
                            oninput="chatUI.handleContentChange('${prefix}', ${id})"
                        >${content}</textarea>
                        <div class="message-content-toolbar">
                            <button class="btn btn-sm" onclick="chatUI.formatJSON('${prefix}', ${id})" title="Format as JSON">
                                { }
                            </button>
                            <button class="btn btn-sm" onclick="chatUI.clearContent('${prefix}', ${id})" title="Clear">
                                Clear
                            </button>
                        </div>
                    </div>
                </div>

                <div class="message-image-section" id="${prefix}_message_image_section_${id}" style="${role === 'user' ? 'display: block' : 'display: none'}">
                    <div class="form-group">
                        <label for="${prefix}_message_image_${id}">Image (Optional):</label>
                        <div class="file-input-wrapper">
                            <input type="file" id="${prefix}_message_image_${id}" accept="image/*" onchange="chatUI.handleImageUpload('${prefix}', ${id})">
                            <label class="file-input-label">
                                <span class="file-input-icon">🖼️</span>
                                <span class="file-input-text">Choose image or drag here</span>
                            </label>
                        </div>
                        <div id="${prefix}_message_image_preview_${id}" class="image-preview-container" style="display: none;">
                            <img class="message-image-preview" alt="Image preview">
                            <button class="btn btn-sm btn-danger" onclick="chatUI.clearImage('${prefix}', ${id})">Remove Image</button>
                        </div>
                    </div>
                </div>

                <div class="message-optional-fields">
                    <div class="form-group">
                        <label for="${prefix}_message_name_${id}">Name (Optional):</label>
                        <input type="text" id="${prefix}_message_name_${id}" placeholder="e.g., function name for assistant tool calls">
                    </div>

                    <div id="${prefix}_message_tool_fields_${id}" class="tool-fields" style="${role === 'tool' ? 'display: block' : 'display: none'}">
                        <div class="form-group">
                            <label for="${prefix}_message_tool_call_id_${id}">Tool Call ID <span class="required">*</span>:</label>
                            <input type="text" id="${prefix}_message_tool_call_id_${id}" placeholder="Required for tool role" required>
                        </div>
                    </div>
                </div>
            </div>
        `;

        container.appendChild(messageDiv);

        // Initialize drag and drop for the image input
        this.initImageDragDrop(prefix, id);

        // Auto-save after adding message
        this.autoSaveMessages(prefix);
    }

    removeMessage(prefix, id) {
        const messageDiv = document.getElementById(`${prefix}_message_entry_${id}`);
        if (messageDiv) {
            messageDiv.remove();
            this.autoSaveMessages(prefix);
            Toast.success('Message removed');
        }
    }

    handleRoleChange(prefix, id) {
        const roleSelect = document.getElementById(`${prefix}_message_role_${id}`);
        const imageSection = document.getElementById(`${prefix}_message_image_section_${id}`);
        const toolFields = document.getElementById(`${prefix}_message_tool_fields_${id}`);
        const roleBadge = document.querySelector(`#${prefix}_message_entry_${id} .message-role-badge`);

        if (roleSelect && roleBadge) {
            const role = roleSelect.value;
            roleBadge.textContent = role;
            roleBadge.className = `message-role-badge ${role}`;

            // Show/hide image section for user role
            if (imageSection) {
                imageSection.style.display = role === 'user' ? 'block' : 'none';
                if (role !== 'user') {
                    this.clearImage(prefix, id);
                }
            }

            // Show/hide tool fields for tool role
            if (toolFields) {
                toolFields.style.display = role === 'tool' ? 'block' : 'none';
            }
        }

        this.autoSaveMessages(prefix);
    }

    handleContentChange(prefix, id) {
        this.autoSaveMessages(prefix);
    }

    handleImageUpload(prefix, id) {
        const fileInput = document.getElementById(`${prefix}_message_image_${id}`);
        const previewContainer = document.getElementById(`${prefix}_message_image_preview_${id}`);
        const fileLabel = fileInput.parentElement.querySelector('.file-input-text');

        if (fileInput && fileInput.files && fileInput.files[0]) {
            const file = fileInput.files[0];
            const reader = new FileReader();

            reader.onload = (e) => {
                if (previewContainer) {
                    const img = previewContainer.querySelector('img');
                    img.src = e.target.result;
                    previewContainer.style.display = 'block';
                    fileInput.dataset.imageDataUrl = e.target.result;
                }
                if (fileLabel) {
                    fileLabel.textContent = file.name;
                }
                fileInput.parentElement.classList.add('has-file');
            };

            reader.readAsDataURL(file);
        }
    }

    clearImage(prefix, id) {
        const fileInput = document.getElementById(`${prefix}_message_image_${id}`);
        const previewContainer = document.getElementById(`${prefix}_message_image_preview_${id}`);
        const fileLabel = fileInput?.parentElement.querySelector('.file-input-text');

        if (fileInput) {
            fileInput.value = '';
            delete fileInput.dataset.imageDataUrl;
            fileInput.parentElement.classList.remove('has-file');
        }
        if (previewContainer) {
            previewContainer.style.display = 'none';
        }
        if (fileLabel) {
            fileLabel.textContent = 'Choose image or drag here';
        }
    }

    initImageDragDrop(prefix, id) {
        const wrapper = document.querySelector(`#${prefix}_message_image_${id}`).parentElement;
        if (!wrapper) return;

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
            
            const fileInput = document.getElementById(`${prefix}_message_image_${id}`);
            if (e.dataTransfer.files.length > 0 && e.dataTransfer.files[0].type.startsWith('image/')) {
                fileInput.files = e.dataTransfer.files;
                this.handleImageUpload(prefix, id);
            }
        });
    }

    formatJSON(prefix, id) {
        const textarea = document.getElementById(`${prefix}_message_content_${id}`);
        if (!textarea) return;

        try {
            const json = JSON.parse(textarea.value);
            textarea.value = JSON.stringify(json, null, 2);
            Toast.success('JSON formatted');
        } catch (e) {
            Toast.error('Invalid JSON');
        }
    }

    clearContent(prefix, id) {
        const textarea = document.getElementById(`${prefix}_message_content_${id}`);
        if (textarea) {
            textarea.value = '';
            this.autoSaveMessages(prefix);
        }
    }

    buildPayload(prefix = 'chatCompletions') {
        const payload = {};

        // Get API provider
        const apiProviderEl = document.getElementById(`${prefix}_api_provider`);
        if (apiProviderEl?.value) {
            payload.api_provider = apiProviderEl.value;
        }

        // Get model
        const modelEl = document.getElementById(`${prefix}_model`);
        if (modelEl?.value) {
            payload.model = modelEl.value;
        }

        // Build messages array
        const messages = [];
        const container = document.getElementById(`${prefix}_messagesContainer`);
        if (container) {
            const messageEntries = container.querySelectorAll('.message-entry');
            messageEntries.forEach(entry => {
                const id = entry.dataset.messageId;
                const role = document.getElementById(`${prefix}_message_role_${id}`)?.value;
                const content = document.getElementById(`${prefix}_message_content_${id}`)?.value;
                const name = document.getElementById(`${prefix}_message_name_${id}`)?.value?.trim();
                const imageInput = document.getElementById(`${prefix}_message_image_${id}`);
                const imageDataUrl = imageInput?.dataset.imageDataUrl;

                if (!role) return;

                const message = { role };

                // Handle content
                const contentParts = [];
                if (content?.trim()) {
                    contentParts.push({ type: 'text', text: content });
                }

                if (role === 'user' && imageDataUrl) {
                    contentParts.push({
                        type: 'image_url',
                        image_url: { url: imageDataUrl }
                    });
                }

                if (contentParts.length === 1 && contentParts[0].type === 'text') {
                    message.content = contentParts[0].text;
                } else if (contentParts.length > 0) {
                    message.content = contentParts;
                } else if (role === 'assistant') {
                    message.content = null;
                } else {
                    message.content = '';
                }

                // Add optional fields
                if (name) {
                    message.name = name;
                }

                if (role === 'tool') {
                    const toolCallId = document.getElementById(`${prefix}_message_tool_call_id_${id}`)?.value?.trim();
                    if (!toolCallId) {
                        throw new Error(`Tool Call ID is required for message with role 'tool'`);
                    }
                    message.tool_call_id = toolCallId;
                }

                messages.push(message);
            });
        }

        if (messages.length === 0) {
            throw new Error('At least one message is required');
        }

        payload.messages = messages;

        // Add other parameters
        const params = [
            { key: 'temperature', type: 'float' },
            { key: 'max_tokens', type: 'int' },
            { key: 'stream', type: 'boolean' },
            { key: 'top_p', type: 'float' },
            { key: 'frequency_penalty', type: 'float' },
            { key: 'presence_penalty', type: 'float' },
            { key: 'seed', type: 'int' },
            { key: 'n', type: 'int' }
        ];

        params.forEach(param => {
            const el = document.getElementById(`${prefix}_${param.key}`);
            if (!el) return;

            let value;
            if (param.type === 'boolean') {
                value = el.checked;
            } else if (param.type === 'float') {
                value = el.value ? parseFloat(el.value) : undefined;
            } else if (param.type === 'int') {
                value = el.value ? parseInt(el.value) : undefined;
            } else {
                value = el.value || undefined;
            }

            if (value !== undefined) {
                payload[param.key] = value;
            }
        });

        return payload;
    }

    autoSaveMessages(prefix) {
        try {
            const payload = this.buildPayload(prefix);
            Utils.saveToStorage(`${prefix}_autosave`, payload);
        } catch (e) {
            // Silent fail for auto-save
        }
    }

    loadAutoSavedMessages(prefix) {
        const saved = Utils.getFromStorage(`${prefix}_autosave`);
        if (!saved) return;

        // Clear existing messages
        const container = document.getElementById(`${prefix}_messagesContainer`);
        if (container) {
            container.innerHTML = '';
        }

        // Load saved messages
        if (saved.messages) {
            saved.messages.forEach(msg => {
                this.addMessage(prefix, msg.role, msg.content);
            });
        }

        // Load parameters
        Object.keys(saved).forEach(key => {
            if (key !== 'messages') {
                const el = document.getElementById(`${prefix}_${key}`);
                if (el) {
                    if (el.type === 'checkbox') {
                        el.checked = saved[key];
                    } else {
                        el.value = saved[key];
                    }
                }
            }
        });
    }

    async sendChatRequest() {
        const prefix = 'chatCompletions';
        const responseArea = document.getElementById('chatCompletions_response');
        
        if (!responseArea) {
            console.error('Response area not found');
            return;
        }

        try {
            // Build payload
            const payload = this.buildPayload(prefix);
            
            // Show loading
            Loading.show(responseArea.parentElement, 'Sending request...');
            responseArea.textContent = '';

            // Generate and display cURL command
            const curlCommand = apiClient.generateCurl('POST', '/api/v1/chat/completions', { body: payload });
            const curlEl = document.getElementById('chatCompletions_curl');
            if (curlEl) {
                curlEl.textContent = curlCommand;
            }

            // Handle streaming vs non-streaming
            if (payload.stream) {
                await this.handleStreamingResponse(responseArea, payload);
            } else {
                const response = await apiClient.post('/api/v1/chat/completions', payload);
                
                // Display response with JSON viewer
                const viewer = new JSONViewer(responseArea, response, {
                    expanded: 2,
                    theme: webUI.theme,
                    enableCopy: true,
                    enableCollapse: true
                });

                // Update conversation ID if present
                if (response.tldw_conversation_id) {
                    const convIdEl = document.getElementById(`${prefix}_conversation_id`);
                    if (convIdEl) {
                        convIdEl.value = response.tldw_conversation_id;
                        Toast.info(`Conversation ID: ${response.tldw_conversation_id}`);
                    }
                }
            }

            Toast.success('Request completed successfully');
        } catch (error) {
            console.error('Chat request error:', error);
            responseArea.textContent = `Error: ${error.message}`;
            Toast.error(`Request failed: ${error.message}`);
        } finally {
            Loading.hide(responseArea.parentElement);
        }
    }

    async handleStreamingResponse(responseArea, payload) {
        let fullContent = '';
        let metadata = null;

        const onProgress = (chunk) => {
            // Handle content chunks
            if (chunk.choices?.[0]?.delta?.content) {
                fullContent += chunk.choices[0].delta.content;
                responseArea.textContent = fullContent;
            }

            // Handle metadata
            if (chunk.tldw_metadata) {
                metadata = chunk.tldw_metadata;
                if (metadata.conversation_id) {
                    const convIdEl = document.getElementById('chatCompletions_conversation_id');
                    if (convIdEl) {
                        convIdEl.value = metadata.conversation_id;
                    }
                }
            }

            // Auto-scroll to bottom
            responseArea.scrollTop = responseArea.scrollHeight;
        };

        await apiClient.post('/api/v1/chat/completions', payload, {
            streaming: true,
            onProgress
        });

        // Add final message
        responseArea.textContent += '\n\n[Stream completed]';
        
        if (metadata) {
            responseArea.textContent += `\n[Conversation ID: ${metadata.conversation_id}]`;
        }
    }

    loadPreset(presetName) {
        const preset = this.presets.find(p => p.name === presetName);
        if (!preset) return;

        const prefix = 'chatCompletions';

        // Clear existing messages
        const container = document.getElementById(`${prefix}_messagesContainer`);
        if (container) {
            container.innerHTML = '';
            this.messageIdCounter = 0;
        }

        // Load preset messages
        preset.messages.forEach(msg => {
            this.addMessage(prefix, msg.role, msg.content);
        });

        // Load preset parameters
        if (preset.temperature !== undefined) {
            const tempEl = document.getElementById(`${prefix}_temperature`);
            if (tempEl) tempEl.value = preset.temperature;
        }

        if (preset.max_tokens !== undefined) {
            const maxTokensEl = document.getElementById(`${prefix}_max_tokens`);
            if (maxTokensEl) maxTokensEl.value = preset.max_tokens;
        }

        Toast.success(`Loaded preset: ${presetName}`);
    }

    saveCurrentAsPreset(name) {
        const prefix = 'chatCompletions';
        
        try {
            const payload = this.buildPayload(prefix);
            const preset = {
                name,
                messages: payload.messages,
                temperature: payload.temperature,
                max_tokens: payload.max_tokens
            };

            // Check if preset with same name exists
            const existingIndex = this.presets.findIndex(p => p.name === name);
            if (existingIndex >= 0) {
                this.presets[existingIndex] = preset;
            } else {
                this.presets.push(preset);
            }

            this.savePresets();
            Toast.success(`Saved preset: ${name}`);
        } catch (error) {
            Toast.error(`Failed to save preset: ${error.message}`);
        }
    }
}

// Initialize chat UI
const chatUI = new ChatUI();

// Initialize chat completions tab when loaded
function initializeChatCompletionsTab() {
    const prefix = 'chatCompletions';
    
    // Check if already initialized
    const container = document.getElementById(`${prefix}_messagesContainer`);
    if (!container || container.children.length > 0) {
        return;
    }

    // Load auto-saved messages or defaults
    const hasSaved = Utils.getFromStorage(`${prefix}_autosave`);
    if (hasSaved) {
        chatUI.loadAutoSavedMessages(prefix);
    } else {
        // Add default messages
        chatUI.addMessage(prefix, 'system', 'You are a helpful assistant that provides concise answers.');
        chatUI.addMessage(prefix, 'user', 'What is the capital of France?');
    }

    // Set default values
    const defaults = {
        temperature: { el: `${prefix}_temperature`, value: '0.7' },
        max_tokens: { el: `${prefix}_max_tokens`, value: '1024' },
        top_p: { el: `${prefix}_top_p`, value: '1.0' },
        n: { el: `${prefix}_n`, value: '1' }
    };

    Object.values(defaults).forEach(({ el, value }) => {
        const element = document.getElementById(el);
        if (element && !element.value) {
            element.placeholder = value;
        }
    });
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ChatUI, chatUI, initializeChatCompletionsTab };
}