/**
 * Tab-specific functions for the WebUI
 * This file contains all functions that are called from onclick handlers in dynamically loaded tabs
 */

// ============================================================================
// Chat Tab Functions
// ============================================================================

function toggleLogprobs() {
    const logprobsChecked = document.getElementById('chatCompletions_logprobs').checked;
    document.getElementById('top_logprobs_group').style.display = logprobsChecked ? 'block' : 'none';
}

function toggleToolChoiceJSON() {
    const toolChoice = document.getElementById('chatCompletions_tool_choice').value;
    document.getElementById('tool_choice_json_group').style.display = toolChoice === 'specific' ? 'block' : 'none';
}

async function makeChatCompletionsRequest() {
    const responseEl = document.getElementById('chatCompletions_response');
    
    try {
        // Build the payload with all parameters
        const payload = {};
        
        // Basic Parameters
        const provider = document.getElementById('chatCompletions_provider').value;
        if (provider) payload.api_provider = provider;
        
        const model = document.getElementById('chatCompletions_model').value;
        if (model) payload.model = model;
        
        const messagesText = document.getElementById('chatCompletions_messages').value;
        try {
            const parsedMessages = JSON.parse(messagesText);
            if (!Array.isArray(parsedMessages)) {
                throw new Error('Messages must be an array');
            }
            payload.messages = parsedMessages;
        } catch (e) {
            throw new Error('Invalid messages JSON format: ' + e.message);
        }
        
        const temperature = parseFloat(document.getElementById('chatCompletions_temperature').value);
        if (!isNaN(temperature)) payload.temperature = temperature;
        
        const maxTokens = parseInt(document.getElementById('chatCompletions_max_tokens').value);
        if (!isNaN(maxTokens)) payload.max_tokens = maxTokens;
        
        payload.stream = document.getElementById('chatCompletions_stream').checked;
        
        // Sampling Parameters
        const frequencyPenalty = parseFloat(document.getElementById('chatCompletions_frequency_penalty').value);
        if (!isNaN(frequencyPenalty)) payload.frequency_penalty = frequencyPenalty;
        
        const presencePenalty = parseFloat(document.getElementById('chatCompletions_presence_penalty').value);
        if (!isNaN(presencePenalty)) payload.presence_penalty = presencePenalty;
        
        const topP = parseFloat(document.getElementById('chatCompletions_top_p').value);
        if (!isNaN(topP)) payload.top_p = topP;
        
        const topK = parseInt(document.getElementById('chatCompletions_top_k').value);
        if (!isNaN(topK)) payload.topk = topK;
        
        const minP = parseFloat(document.getElementById('chatCompletions_min_p').value);
        if (!isNaN(minP)) payload.minp = minP;
        
        const seed = parseInt(document.getElementById('chatCompletions_seed').value);
        if (!isNaN(seed)) payload.seed = seed;
        
        const n = parseInt(document.getElementById('chatCompletions_n').value);
        if (!isNaN(n)) payload.n = n;
        
        // Response Control
        const responseFormat = document.querySelector('input[name="chatCompletions_response_format"]:checked').value;
        if (responseFormat === 'json_object') {
            payload.response_format = { type: 'json_object' };
        }
        
        const stopSequences = document.getElementById('chatCompletions_stop').value;
        if (stopSequences) {
            payload.stop = stopSequences.split(',').map(s => s.trim()).filter(s => s);
        }
        
        const user = document.getElementById('chatCompletions_user').value;
        if (user) payload.user = user;
        
        const logprobs = document.getElementById('chatCompletions_logprobs').checked;
        if (logprobs) {
            payload.logprobs = true;
            const topLogprobs = parseInt(document.getElementById('chatCompletions_top_logprobs').value);
            if (!isNaN(topLogprobs)) payload.top_logprobs = topLogprobs;
        }
        
        const logitBiasText = document.getElementById('chatCompletions_logit_bias').value;
        if (logitBiasText && logitBiasText !== '{}') {
            try {
                const parsed = JSON.parse(logitBiasText);
                if (parsed && typeof parsed === 'object') {
                    payload.logit_bias = parsed;
                }
            } catch (e) {
                console.warn('Invalid logit bias JSON:', e);
            }
        }
        
        // Context & Templates
        const promptTemplate = document.getElementById('chatCompletions_prompt_template').value;
        if (promptTemplate) payload.prompt_template_name = promptTemplate;
        
        const characterIdStr = document.getElementById('chatCompletions_character_id').value;
        if (characterIdStr) {
            const characterId = parseInt(characterIdStr);
            if (!isNaN(characterId) && characterId > 0) {
                payload.character_id = characterId;
            } else {
                console.warn('Invalid character ID:', characterIdStr);
            }
        }
        
        const conversationId = document.getElementById('chatCompletions_conversation_id').value;
        if (conversationId) {
            // Basic validation for conversation ID
            if (/^[a-zA-Z0-9_-]+$/.test(conversationId)) {
                payload.conversation_id = conversationId;
            } else {
                console.warn('Invalid conversation ID format:', conversationId);
            }
        }
        
        // Function Calling
        const toolsText = document.getElementById('chatCompletions_tools').value;
        if (toolsText && toolsText !== '[]') {
            try {
                const parsedTools = JSON.parse(toolsText);
                if (parsedTools && Array.isArray(parsedTools)) {
                    payload.tools = parsedTools;
                }
            } catch (e) {
                console.warn('Invalid tools JSON:', e);
            }
        }
        
        const toolChoice = document.getElementById('chatCompletions_tool_choice').value;
        if (toolChoice === 'specific') {
            const toolChoiceJSON = document.getElementById('chatCompletions_tool_choice_json').value;
            if (toolChoiceJSON && toolChoiceJSON !== '{}') {
                try {
                    const parsed = JSON.parse(toolChoiceJSON);
                    if (parsed) {
                        payload.tool_choice = parsed;
                    }
                } catch (e) {
                    console.warn('Invalid tool choice JSON:', e);
                }
            }
        } else if (toolChoice !== 'auto') {
            payload.tool_choice = toolChoice;
        }
        
        // Display the request payload for debugging
        console.log('Request payload:', payload);
        responseEl.textContent = 'Sending request with parameters:\n' + JSON.stringify(payload, null, 2) + '\n\n';
        
        if (payload.stream) {
            // Handle streaming response
            responseEl.textContent += 'Streaming response:\n';
            const response = await apiClient.post('/api/v1/chat/completions', payload, {
                streaming: true,
                onProgress: (chunk) => {
                    if (chunk.choices && chunk.choices[0] && chunk.choices[0].delta && chunk.choices[0].delta.content) {
                        responseEl.textContent += chunk.choices[0].delta.content;
                    }
                }
            });
            responseEl.textContent += '\n\n[Stream Complete]';
        } else {
            // Handle regular response
            const response = await apiClient.post('/api/v1/chat/completions', payload);
            responseEl.textContent += '\nResponse:\n' + JSON.stringify(response, null, 2);
        }
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        console.error('Chat completions error:', error);
    }
}

// Interactive chat interface
const MAX_CHAT_MESSAGES = 100;
let chatMessages = [
    {role: 'system', content: 'You are a helpful assistant.'}
];

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const messagesDiv = document.getElementById('chat-messages');
    const model = document.getElementById('chat-model').value;
    
    if (!input.value.trim()) return;
    
    const userMessage = input.value;
    input.value = '';
    
    // Add user message to display
    const userDiv = document.createElement('div');
    userDiv.className = 'chat-message user';
    // Create user message elements safely
    const userLabel = document.createElement('strong');
    userLabel.textContent = 'User:';
    const userContent = document.createElement('span');
    userContent.textContent = userMessage;
    userDiv.appendChild(userLabel);
    userDiv.appendChild(document.createTextNode(' '));
    userDiv.appendChild(userContent);
    messagesDiv.appendChild(userDiv);
    
    // Add to messages array with history limit
    chatMessages.push({role: 'user', content: userMessage});
    if (chatMessages.length > MAX_CHAT_MESSAGES) {
        const systemMsg = chatMessages[0];
        chatMessages = [systemMsg, ...chatMessages.slice(-(MAX_CHAT_MESSAGES - 1))];
    }
    
    // Create assistant message placeholder
    const assistantDiv = document.createElement('div');
    assistantDiv.className = 'chat-message assistant';
    const assistantLabel = document.createElement('strong');
    assistantLabel.textContent = 'Assistant:';
    const assistantContent = document.createElement('span');
    assistantContent.className = 'typing';
    assistantContent.textContent = 'Thinking...';
    assistantDiv.appendChild(assistantLabel);
    assistantDiv.appendChild(document.createTextNode(' '));
    assistantDiv.appendChild(assistantContent);
    messagesDiv.appendChild(assistantDiv);
    
    // Smooth scrolling
    requestAnimationFrame(() => {
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    });
    
    try {
        const response = await apiClient.post('/api/v1/chat/completions', {
            model: model,
            messages: chatMessages,
            temperature: 0.7,
            max_tokens: 1000
        });
        
        if (response.choices && response.choices[0] && response.choices[0].message) {
            const assistantMessage = response.choices[0].message.content;
            chatMessages.push({role: 'assistant', content: assistantMessage});
            
            // Limit chat history
            if (chatMessages.length > MAX_CHAT_MESSAGES) {
                const systemMsg = chatMessages[0];
                chatMessages = [systemMsg, ...chatMessages.slice(-(MAX_CHAT_MESSAGES - 1))];
            }
            
            assistantDiv.innerHTML = '';
            const label = document.createElement('strong');
            label.textContent = 'Assistant:';
            const content = document.createElement('span');
            content.textContent = assistantMessage;
            assistantDiv.appendChild(label);
            assistantDiv.appendChild(document.createTextNode(' '));
            assistantDiv.appendChild(content);
        } else {
            assistantDiv.innerHTML = '';
            const label2 = document.createElement('strong');
            label2.textContent = 'Assistant:';
            const error = document.createElement('em');
            error.textContent = 'No response received';
            assistantDiv.appendChild(label2);
            assistantDiv.appendChild(document.createTextNode(' '));
            assistantDiv.appendChild(error);
        }
    } catch (error) {
        assistantDiv.innerHTML = '';
        const errorLabel = document.createElement('strong');
        errorLabel.textContent = 'Assistant:';
        const errorMsg = document.createElement('em');
        errorMsg.textContent = `Error: ${error.message}`;
        assistantDiv.appendChild(errorLabel);
        assistantDiv.appendChild(document.createTextNode(' '));
        assistantDiv.appendChild(errorMsg);
        console.error('Chat error:', error);
    }
    
    requestAnimationFrame(() => {
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    });
}

function clearChat() {
    chatMessages = [
        {role: 'system', content: 'You are a helpful assistant.'}
    ];
    const messagesDiv = document.getElementById('chat-messages');
    // Use DocumentFragment for better performance
    const fragment = document.createDocumentFragment();
    const systemDiv = document.createElement('div');
    systemDiv.className = 'chat-message system';
    systemDiv.textContent = 'System: You are a helpful assistant.';
    fragment.appendChild(systemDiv);
    messagesDiv.innerHTML = '';
    messagesDiv.appendChild(fragment);
}

async function exportCharacter() {
    const characterId = document.getElementById('exportCharacter_character_id').value;
    const format = document.getElementById('exportCharacter_format').value;
    const responseEl = document.getElementById('exportCharacter_response');
    
    try {
        responseEl.textContent = 'Exporting...';
        
        const response = await apiClient.get(`/api/v1/characters/${characterId}/export`, {
            format: format
        });
        
        if (format === 'json' || format === 'markdown') {
            responseEl.textContent = typeof response === 'string' ? response : JSON.stringify(response, null, 2);
        } else {
            // For PNG export, we'd need to handle binary data differently
            responseEl.textContent = 'PNG export successful. Binary data received.';
        }
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
    }
}

// ============================================================================
// Character/Conversation Tab Functions
// ============================================================================

async function createCharacter() {
    const responseEl = document.getElementById('charactersCreate_response');
    try {
        responseEl.textContent = 'Creating character...';
        
        // Collect all form values
        const body = {};
        
        // Required field
        const name = document.getElementById('charactersCreate_name').value;
        if (!name) {
            throw new Error('Character name is required');
        }
        body.name = name;
        
        // Basic Information
        const description = document.getElementById('charactersCreate_description').value;
        if (description) body.description = description;
        
        const personality = document.getElementById('charactersCreate_personality').value;
        if (personality) body.personality = personality;
        
        const scenario = document.getElementById('charactersCreate_scenario').value;
        if (scenario) body.scenario = scenario;
        
        // Conversation Settings
        const systemPrompt = document.getElementById('charactersCreate_system_prompt').value;
        if (systemPrompt) body.system_prompt = systemPrompt;
        
        const postHistoryInstructions = document.getElementById('charactersCreate_post_history_instructions').value;
        if (postHistoryInstructions) body.post_history_instructions = postHistoryInstructions;
        
        const firstMessage = document.getElementById('charactersCreate_first_message').value;
        if (firstMessage) body.first_message = firstMessage;
        
        const messageExample = document.getElementById('charactersCreate_message_example').value;
        if (messageExample) body.message_example = messageExample;
        
        // Handle alternate_greetings
        const alternateGreetingsValue = document.getElementById('charactersCreate_alternate_greetings').value;
        if (alternateGreetingsValue) {
            try {
                body.alternate_greetings = JSON.parse(alternateGreetingsValue);
            } catch (e) {
                body.alternate_greetings = alternateGreetingsValue.split(',').map(g => g.trim()).filter(g => g);
            }
        }
        
        // Metadata
        const creator = document.getElementById('charactersCreate_creator').value;
        if (creator) body.creator = creator;
        
        const creatorNotes = document.getElementById('charactersCreate_creator_notes').value;
        if (creatorNotes) body.creator_notes = creatorNotes;
        
        const characterVersion = document.getElementById('charactersCreate_character_version').value;
        if (characterVersion) body.character_version = characterVersion;
        
        const tags = document.getElementById('charactersCreate_tags').value;
        if (tags) {
            body.tags = tags.split(',').map(t => t.trim()).filter(t => t);
        }
        
        const extensionsValue = document.getElementById('charactersCreate_extensions').value;
        if (extensionsValue && extensionsValue !== '{}') {
            try {
                body.extensions = JSON.parse(extensionsValue);
            } catch (e) {
                throw new Error('Extensions must be valid JSON');
            }
        }
        
        const imageBase64 = document.getElementById('charactersCreate_image_base64').value;
        if (imageBase64) body.image_base64 = imageBase64;
        
        const response = await apiClient.makeRequest('POST', '/api/v1/characters', { body });
        responseEl.textContent = JSON.stringify(response, null, 2);
        Toast.success('Character created successfully');
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to create character: ${error.message}`);
    }
}

async function listCharacters() {
    const responseEl = document.getElementById('charactersList_response');
    try {
        responseEl.textContent = 'Loading characters...';
        const response = await apiClient.makeRequest('GET', '/api/v1/characters');
        responseEl.textContent = JSON.stringify(response, null, 2);
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to list characters: ${error.message}`);
    }
}

async function getCharacter() {
    const responseEl = document.getElementById('charactersGet_response');
    try {
        const characterId = document.getElementById('charactersGet_id').value;
        if (!characterId) {
            throw new Error('Character ID is required');
        }
        
        responseEl.textContent = 'Loading character...';
        const response = await apiClient.makeRequest('GET', `/api/v1/characters/${characterId}`);
        responseEl.textContent = JSON.stringify(response, null, 2);
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to get character: ${error.message}`);
    }
}

async function updateCharacter() {
    const responseEl = document.getElementById('charactersUpdate_response');
    try {
        const characterId = document.getElementById('charactersUpdate_id').value;
        if (!characterId) {
            throw new Error('Character ID is required');
        }
        
        const payload = document.getElementById('charactersUpdate_payload').value;
        const body = JSON.parse(payload);
        
        responseEl.textContent = 'Updating character...';
        const response = await apiClient.makeRequest('PUT', `/api/v1/characters/${characterId}`, { body });
        responseEl.textContent = JSON.stringify(response, null, 2);
        Toast.success('Character updated successfully');
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to update character: ${error.message}`);
    }
}

async function deleteCharacter() {
    const responseEl = document.getElementById('charactersDelete_response');
    try {
        const characterId = document.getElementById('charactersDelete_id').value;
        if (!characterId) {
            throw new Error('Character ID is required');
        }
        
        responseEl.textContent = 'Deleting character...';
        const response = await apiClient.makeRequest('DELETE', `/api/v1/characters/${characterId}`);
        responseEl.textContent = response ? JSON.stringify(response, null, 2) : 'Character deleted successfully';
        Toast.success('Character deleted successfully');
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to delete character: ${error.message}`);
    }
}

// Conversation functions
async function createConversation() {
    const responseEl = document.getElementById('conversationsCreate_response');
    try {
        responseEl.textContent = 'Creating conversation...';
        
        const metadata = document.getElementById('conversationsCreate_metadata').value;
        
        const body = {
            title: document.getElementById('conversationsCreate_title').value,
            initial_message: document.getElementById('conversationsCreate_initial_message').value
        };
        
        const characterId = document.getElementById('conversationsCreate_character_id').value;
        if (characterId) body.character_id = characterId;
        
        const systemPrompt = document.getElementById('conversationsCreate_system_prompt').value;
        if (systemPrompt) body.system_prompt = systemPrompt;
        
        if (metadata && metadata.trim() !== '{}') {
            body.metadata = JSON.parse(metadata);
        }
        
        const response = await apiClient.makeRequest('POST', '/api/v1/conversations', { body });
        responseEl.textContent = JSON.stringify(response, null, 2);
        Toast.success('Conversation created successfully');
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to create conversation: ${error.message}`);
    }
}

async function listConversations() {
    const responseEl = document.getElementById('conversationsList_response');
    try {
        responseEl.textContent = 'Loading conversations...';
        
        const params = new URLSearchParams();
        const characterId = document.getElementById('conversationsList_character_id').value;
        if (characterId) params.append('character_id', characterId);
        
        const limit = document.getElementById('conversationsList_limit').value;
        if (limit) params.append('limit', limit);
        
        const queryString = params.toString();
        const url = queryString ? `/api/v1/conversations?${queryString}` : '/api/v1/conversations';
        
        const response = await apiClient.makeRequest('GET', url);
        responseEl.textContent = JSON.stringify(response, null, 2);
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to list conversations: ${error.message}`);
    }
}

async function getConversationDetails() {
    const responseEl = document.getElementById('conversationsGet_response');
    try {
        const conversationId = document.getElementById('conversationsGet_id').value;
        if (!conversationId) {
            throw new Error('Conversation ID is required');
        }
        
        responseEl.textContent = 'Loading conversation...';
        
        const includeMessages = document.getElementById('conversationsGet_include_messages').checked;
        const params = includeMessages ? '?include_messages=true' : '';
        
        const response = await apiClient.makeRequest('GET', `/api/v1/conversations/${conversationId}${params}`);
        responseEl.textContent = JSON.stringify(response, null, 2);
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to get conversation: ${error.message}`);
    }
}

async function sendConversationMessage() {
    const responseEl = document.getElementById('conversationsChat_response');
    try {
        const conversationId = document.getElementById('conversationsChat_id').value;
        if (!conversationId) {
            throw new Error('Conversation ID is required');
        }
        
        const message = document.getElementById('conversationsChat_message').value;
        if (!message) {
            throw new Error('Message is required');
        }
        
        responseEl.textContent = 'Sending message...';
        
        const body = { message };
        
        const model = document.getElementById('conversationsChat_model').value;
        if (model) body.model = model;
        
        const temperature = document.getElementById('conversationsChat_temperature').value;
        if (temperature) body.temperature = parseFloat(temperature);
        
        const stream = document.getElementById('conversationsChat_stream').checked;
        body.stream = stream;
        
        if (stream) {
            // Handle streaming response
            responseEl.textContent = 'Streaming response...\n';
            const response = await fetch(`${apiClient.baseUrl}/api/v1/conversations/${conversationId}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-KEY': apiClient.token
                },
                body: JSON.stringify(body)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') continue;
                        
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.choices?.[0]?.delta?.content) {
                                responseEl.textContent += parsed.choices[0].delta.content;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }
            }
        } else {
            const response = await apiClient.makeRequest('POST', `/api/v1/conversations/${conversationId}/chat`, { body });
            responseEl.textContent = JSON.stringify(response, null, 2);
        }
        
        Toast.success('Message sent successfully');
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to send message: ${error.message}`);
    }
}

async function updateConversation() {
    const responseEl = document.getElementById('conversationsUpdate_response');
    try {
        const conversationId = document.getElementById('conversationsUpdate_id').value;
        if (!conversationId) {
            throw new Error('Conversation ID is required');
        }
        
        const payload = document.getElementById('conversationsUpdate_payload').value;
        const body = JSON.parse(payload);
        
        responseEl.textContent = 'Updating conversation...';
        const response = await apiClient.makeRequest('PUT', `/api/v1/conversations/${conversationId}`, { body });
        responseEl.textContent = JSON.stringify(response, null, 2);
        Toast.success('Conversation updated successfully');
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to update conversation: ${error.message}`);
    }
}

async function deleteConversation() {
    const responseEl = document.getElementById('conversationsDelete_response');
    try {
        const conversationId = document.getElementById('conversationsDelete_id').value;
        if (!conversationId) {
            throw new Error('Conversation ID is required');
        }
        
        responseEl.textContent = 'Deleting conversation...';
        const response = await apiClient.makeRequest('DELETE', `/api/v1/conversations/${conversationId}`);
        responseEl.textContent = response ? JSON.stringify(response, null, 2) : 'Conversation deleted successfully';
        Toast.success('Conversation deleted successfully');
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to delete conversation: ${error.message}`);
    }
}

async function exportConversation() {
    const responseEl = document.getElementById('conversationsExport_response');
    try {
        const conversationId = document.getElementById('conversationsExport_id').value;
        if (!conversationId) {
            throw new Error('Conversation ID is required');
        }
        
        const format = document.getElementById('conversationsExport_format').value;
        
        responseEl.textContent = 'Exporting conversation...';
        const response = await apiClient.makeRequest('GET', `/api/v1/conversations/${conversationId}/export?format=${format}`);
        
        if (format === 'json') {
            responseEl.textContent = JSON.stringify(response, null, 2);
        } else {
            responseEl.textContent = response;
        }
        
        Toast.success('Conversation exported successfully');
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        Toast.error(`Failed to export conversation: ${error.message}`);
    }
}

// ============================================================================
// Initialization Functions
// ============================================================================

function initializeChatCompletionsTab() {
    console.log('Chat Completions tab initialized');
    // Populate model dropdowns when tab is initialized
    if (typeof populateModelDropdowns === 'function') {
        populateModelDropdowns();
    }
}

async function populateModelDropdowns() {
    try {
        // Get available providers from API
        const providersInfo = await apiClient.getAvailableProviders();
        
        if (!providersInfo || !providersInfo.providers || providersInfo.providers.length === 0) {
            console.warn('No LLM providers configured');
            document.querySelectorAll('.llm-model-select').forEach(select => {
                select.innerHTML = '<option value="">No models available - check configuration</option>';
            });
            return;
        }
        
        // Build options HTML
        let optionsHtml = '';
        const defaultProvider = providersInfo.default_provider;
        let defaultModel = null;
        
        const sortedProviders = providersInfo.providers.sort((a, b) => {
            if (a.type === 'commercial' && b.type === 'local') return -1;
            if (a.type === 'local' && b.type === 'commercial') return 1;
            return a.display_name.localeCompare(b.display_name);
        });
        
        sortedProviders.forEach(provider => {
            if (provider.models && provider.models.length > 0) {
                optionsHtml += `<optgroup label="${provider.display_name}">`;
                
                provider.models.forEach(model => {
                    const value = `${provider.name}/${model}`;
                    const displayName = model;
                    const isDefault = provider.name === defaultProvider && provider.default_model === model;
                    
                    if (isDefault) {
                        defaultModel = value;
                    }
                    
                    optionsHtml += `<option value="${value}"${isDefault ? ' data-default="true"' : ''}>${displayName}${isDefault ? ' (default)' : ''}</option>`;
                });
                
                optionsHtml += '</optgroup>';
            }
        });
        
        // Update all model select dropdowns
        document.querySelectorAll('.llm-model-select').forEach(select => {
            const currentValue = select.value;
            const hasUseDefault = select.querySelector('option[value=""]');
            
            let html = '';
            if (hasUseDefault && hasUseDefault.textContent.includes('Use default')) {
                html = '<option value="">Use default</option>';
            }
            html += optionsHtml;
            
            select.innerHTML = html;
            
            if (currentValue) {
                select.value = currentValue;
            } else if (defaultModel && !hasUseDefault) {
                select.value = defaultModel;
            }
        });
        
        console.log(`Populated model dropdowns with ${providersInfo.total_configured} providers`);
        
    } catch (error) {
        console.error('Failed to populate model dropdowns:', error);
        document.querySelectorAll('.llm-model-select').forEach(select => {
            select.innerHTML = '<option value="">Error loading models</option>';
        });
    }
}

// Make sure functions are globally available
console.log('Tab functions loaded successfully');