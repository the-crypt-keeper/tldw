// Media Analysis Tab Functionality

class MediaAnalysisManager {
    constructor() {
        this.selectedMediaId = null;
        this.selectedPromptId = null;
        this.availablePrompts = [];
        this.currentAnalyses = [];
        this.searchTimeout = null;
        this.isAnalyzing = false;
    }

    async initialize() {
        console.log('Initializing Media Analysis Manager');
        
        // Load saved prompts
        await this.loadAvailablePrompts();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Initialize model dropdown
        if (typeof window.populateModelDropdowns === 'function') {
            await window.populateModelDropdowns();
        }
    }

    setupEventListeners() {
        // Search input with debounce
        const searchInput = document.getElementById('analysisMediaSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                clearTimeout(this.searchTimeout);
                this.searchTimeout = setTimeout(() => {
                    this.searchMediaForAnalysis(e.target.value);
                }, 300);
            });
        }

        // Prompt source toggle
        const promptSourceRadios = document.querySelectorAll('input[name="promptSource"]');
        promptSourceRadios.forEach(radio => {
            radio.addEventListener('change', () => this.togglePromptSource());
        });

        // Run analysis button
        const runButton = document.getElementById('runAnalysisBtn');
        if (runButton) {
            runButton.addEventListener('click', () => this.runMediaAnalysis());
        }

        // Clear analysis button
        const clearButton = document.getElementById('clearAnalysisBtn');
        if (clearButton) {
            clearButton.addEventListener('click', () => this.clearAnalysis());
        }

        // Model parameters
        const tempSlider = document.getElementById('analysisTemperature');
        const tempValue = document.getElementById('analysisTemperatureValue');
        if (tempSlider && tempValue) {
            tempSlider.addEventListener('input', (e) => {
                tempValue.textContent = e.target.value;
            });
        }

        const maxTokensSlider = document.getElementById('analysisMaxTokens');
        const maxTokensValue = document.getElementById('analysisMaxTokensValue');
        if (maxTokensSlider && maxTokensValue) {
            maxTokensSlider.addEventListener('input', (e) => {
                maxTokensValue.textContent = e.target.value;
            });
        }
    }

    async searchMediaForAnalysis(query) {
        if (!query || query.trim().length < 2) {
            document.getElementById('analysisMediaResults').innerHTML = '';
            return;
        }

        const resultsDiv = document.getElementById('analysisMediaResults');
        resultsDiv.innerHTML = '<div class="loading">Searching...</div>';

        try {
            const response = await fetch(`/api/v1/media/search?query=${encodeURIComponent(query)}&limit=10`);
            if (!response.ok) throw new Error('Search failed');

            const data = await response.json();
            this.displaySearchResults(data.items || []);
        } catch (error) {
            console.error('Error searching media:', error);
            resultsDiv.innerHTML = '<div class="error">Failed to search media</div>';
        }
    }

    displaySearchResults(items) {
        const resultsDiv = document.getElementById('analysisMediaResults');
        
        if (items.length === 0) {
            resultsDiv.innerHTML = '<div class="no-results">No media items found</div>';
            return;
        }

        const html = items.map(item => `
            <div class="search-result-item" onclick="mediaAnalysisManager.loadMediaForAnalysis(${item.id})">
                <h4>${this.escapeHtml(item.title || 'Untitled')}</h4>
                <p class="item-type">${item.media_type || 'Unknown'}</p>
                ${item.author ? `<p class="item-author">By: ${this.escapeHtml(item.author)}</p>` : ''}
                ${item.description ? `<p class="item-description">${this.escapeHtml(item.description).substring(0, 150)}...</p>` : ''}
            </div>
        `).join('');

        resultsDiv.innerHTML = html;
    }

    async loadMediaForAnalysis(mediaId) {
        this.selectedMediaId = mediaId;
        const selectedMediaDiv = document.getElementById('selectedMediaForAnalysis');
        
        selectedMediaDiv.innerHTML = '<div class="loading">Loading media details...</div>';

        try {
            const response = await fetch(`/api/v1/media/${mediaId}`);
            if (!response.ok) throw new Error('Failed to load media');

            const media = await response.json();
            
            selectedMediaDiv.innerHTML = `
                <div class="selected-media-card">
                    <h3>${this.escapeHtml(media.title || 'Untitled')}</h3>
                    <p><strong>Type:</strong> ${media.media_type || 'Unknown'}</p>
                    ${media.author ? `<p><strong>Author:</strong> ${this.escapeHtml(media.author)}</p>` : ''}
                    ${media.description ? `<p><strong>Description:</strong> ${this.escapeHtml(media.description)}</p>` : ''}
                    <p><strong>Content Length:</strong> ${media.content ? media.content.length : 0} characters</p>
                </div>
            `;

            // Load existing analyses for this media
            await this.loadExistingAnalyses(mediaId);
            
            // Clear search results
            document.getElementById('analysisMediaResults').innerHTML = '';
            document.getElementById('analysisMediaSearch').value = '';
        } catch (error) {
            console.error('Error loading media:', error);
            selectedMediaDiv.innerHTML = '<div class="error">Failed to load media details</div>';
        }
    }

    async loadAvailablePrompts() {
        try {
            const response = await fetch('/api/v1/prompts/list?page=1&per_page=100');
            if (!response.ok) throw new Error('Failed to load prompts');

            const data = await response.json();
            this.availablePrompts = data.prompts || [];
            this.updatePromptDropdown();
        } catch (error) {
            console.error('Error loading prompts:', error);
            this.availablePrompts = [];
        }
    }

    updatePromptDropdown() {
        const select = document.getElementById('savedPromptSelect');
        if (!select) return;

        const html = ['<option value="">Select a saved prompt...</option>'];
        
        this.availablePrompts.forEach(prompt => {
            html.push(`<option value="${prompt.id}">${this.escapeHtml(prompt.name)}</option>`);
        });

        select.innerHTML = html.join('');
    }

    togglePromptSource() {
        const customPromptDiv = document.getElementById('customPromptSection');
        const savedPromptDiv = document.getElementById('savedPromptSection');
        const useCustom = document.getElementById('useCustomPrompt').checked;

        if (useCustom) {
            customPromptDiv.style.display = 'block';
            savedPromptDiv.style.display = 'none';
        } else {
            customPromptDiv.style.display = 'none';
            savedPromptDiv.style.display = 'block';
        }
    }

    async loadPromptDetails() {
        const promptId = document.getElementById('savedPromptSelect').value;
        if (!promptId) {
            this.selectedPromptId = null;
            return;
        }

        try {
            const prompt = this.availablePrompts.find(p => p.id == promptId);
            if (prompt) {
                this.selectedPromptId = promptId;
                // Could display prompt details here if needed
            }
        } catch (error) {
            console.error('Error loading prompt details:', error);
        }
    }

    async runMediaAnalysis() {
        if (!this.selectedMediaId) {
            alert('Please select a media item first');
            return;
        }

        if (this.isAnalyzing) {
            alert('Analysis already in progress');
            return;
        }

        const useCustom = document.getElementById('useCustomPrompt').checked;
        let systemPrompt = '';
        let userPrompt = '';

        if (useCustom) {
            systemPrompt = document.getElementById('analysisSystemPrompt').value;
            userPrompt = document.getElementById('analysisUserPrompt').value;
        } else {
            const promptId = document.getElementById('savedPromptSelect').value;
            if (!promptId) {
                alert('Please select a saved prompt');
                return;
            }
            const prompt = this.availablePrompts.find(p => p.id == promptId);
            if (!prompt) {
                alert('Selected prompt not found');
                return;
            }
            systemPrompt = prompt.system_prompt || '';
            userPrompt = prompt.user_prompt || '';
        }

        const model = document.getElementById('analysisModelSelect').value;
        if (!model) {
            alert('Please select a model');
            return;
        }

        const temperature = parseFloat(document.getElementById('analysisTemperature').value);
        const maxTokens = parseInt(document.getElementById('analysisMaxTokens').value);
        const stream = document.getElementById('analysisStreamResponse').checked;

        this.isAnalyzing = true;
        const runButton = document.getElementById('runAnalysisBtn');
        const resultsDiv = document.getElementById('analysisResults');
        
        runButton.disabled = true;
        runButton.textContent = 'Analyzing...';
        resultsDiv.innerHTML = '<div class="loading">Running analysis...</div>';

        try {
            const payload = {
                system_prompt: systemPrompt,
                user_prompt: userPrompt,
                model: model,
                temperature: temperature,
                max_tokens: maxTokens,
                stream: stream
            };

            const response = await fetch(`/api/v1/media/${this.selectedMediaId}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Analysis failed');
            }

            if (stream) {
                await this.handleStreamingResponse(response, resultsDiv);
            } else {
                const result = await response.json();
                this.displayAnalysisResult(result);
            }

            // Reload existing analyses to show the new one
            await this.loadExistingAnalyses(this.selectedMediaId);
        } catch (error) {
            console.error('Error running analysis:', error);
            resultsDiv.innerHTML = `<div class="error">Analysis failed: ${error.message}</div>`;
        } finally {
            this.isAnalyzing = false;
            runButton.disabled = false;
            runButton.textContent = 'Run Analysis';
        }
    }

    async handleStreamingResponse(response, resultsDiv) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullContent = '';

        resultsDiv.innerHTML = '<div class="streaming-output"></div>';
        const outputDiv = resultsDiv.querySelector('.streaming-output');

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);
                        if (data === '[DONE]') continue;
                        
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.choices && parsed.choices[0].delta?.content) {
                                fullContent += parsed.choices[0].delta.content;
                                outputDiv.textContent = fullContent;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error reading stream:', error);
            throw error;
        }

        // Display final result
        this.displayAnalysisResult({
            analysis: fullContent,
            created_at: new Date().toISOString()
        });
    }

    displayAnalysisResult(result) {
        const resultsDiv = document.getElementById('analysisResults');
        resultsDiv.innerHTML = `
            <div class="analysis-result">
                <div class="result-header">
                    <h3>Analysis Complete</h3>
                    <span class="timestamp">${new Date(result.created_at).toLocaleString()}</span>
                </div>
                <div class="result-content">
                    ${this.formatAnalysisContent(result.analysis || result.content || '')}
                </div>
                ${result.version_id ? `<div class="result-meta">Version ID: ${result.version_id}</div>` : ''}
            </div>
        `;
    }

    formatAnalysisContent(content) {
        // Basic markdown-like formatting
        return content
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>');
    }

    clearAnalysis() {
        document.getElementById('analysisResults').innerHTML = '';
        document.getElementById('analysisSystemPrompt').value = '';
        document.getElementById('analysisUserPrompt').value = '';
    }

    async loadExistingAnalyses(mediaId) {
        const container = document.getElementById('existingAnalyses');
        container.innerHTML = '<div class="loading">Loading analyses...</div>';

        try {
            const response = await fetch(`/api/v1/media/${mediaId}/analyses`);
            if (!response.ok) throw new Error('Failed to load analyses');

            const analyses = await response.json();
            this.currentAnalyses = analyses;
            this.displayExistingAnalyses(analyses);
        } catch (error) {
            console.error('Error loading analyses:', error);
            container.innerHTML = '<div class="error">Failed to load existing analyses</div>';
        }
    }

    displayExistingAnalyses(analyses) {
        const container = document.getElementById('existingAnalyses');
        
        if (!analyses || analyses.length === 0) {
            container.innerHTML = '<div class="no-analyses">No analyses found for this media item</div>';
            return;
        }

        const html = analyses.map(analysis => `
            <div class="analysis-card" data-version-id="${analysis.version_id}">
                <div class="analysis-header">
                    <h4>${analysis.version_name || 'Analysis'}</h4>
                    <span class="analysis-date">${new Date(analysis.created_at).toLocaleDateString()}</span>
                </div>
                <div class="analysis-preview">
                    ${this.escapeHtml(analysis.content).substring(0, 200)}...
                </div>
                <div class="analysis-actions">
                    <button onclick="mediaAnalysisManager.viewAnalysis('${analysis.version_id}')" class="btn-small">View</button>
                    <button onclick="mediaAnalysisManager.editAnalysis('${analysis.version_id}')" class="btn-small">Edit</button>
                    <button onclick="mediaAnalysisManager.deleteAnalysis('${analysis.version_id}')" class="btn-small btn-danger">Delete</button>
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    viewAnalysis(versionId) {
        const analysis = this.currentAnalyses.find(a => a.version_id === versionId);
        if (!analysis) return;

        const resultsDiv = document.getElementById('analysisResults');
        resultsDiv.innerHTML = `
            <div class="analysis-result">
                <div class="result-header">
                    <h3>${analysis.version_name || 'Analysis'}</h3>
                    <span class="timestamp">${new Date(analysis.created_at).toLocaleString()}</span>
                </div>
                <div class="result-content">
                    ${this.formatAnalysisContent(analysis.content || '')}
                </div>
                <div class="result-meta">Version ID: ${analysis.version_id}</div>
            </div>
        `;
    }

    async editAnalysis(versionId) {
        const analysis = this.currentAnalyses.find(a => a.version_id === versionId);
        if (!analysis) return;

        const newContent = prompt('Edit analysis content:', analysis.content);
        if (newContent === null || newContent === analysis.content) return;

        try {
            const response = await fetch(`/api/v1/media/${this.selectedMediaId}/analyses/${versionId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: newContent,
                    version_name: analysis.version_name
                })
            });

            if (!response.ok) throw new Error('Failed to update analysis');

            await this.loadExistingAnalyses(this.selectedMediaId);
            alert('Analysis updated successfully');
        } catch (error) {
            console.error('Error updating analysis:', error);
            alert('Failed to update analysis');
        }
    }

    async deleteAnalysis(versionId) {
        if (!confirm('Are you sure you want to delete this analysis?')) return;

        try {
            const response = await fetch(`/api/v1/media/${this.selectedMediaId}/analyses/${versionId}`, {
                method: 'DELETE'
            });

            if (!response.ok) throw new Error('Failed to delete analysis');

            await this.loadExistingAnalyses(this.selectedMediaId);
            alert('Analysis deleted successfully');
        } catch (error) {
            console.error('Error deleting analysis:', error);
            alert('Failed to delete analysis');
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize when the tab is loaded
let mediaAnalysisManager;

function initializeMediaAnalysis() {
    if (!mediaAnalysisManager) {
        mediaAnalysisManager = new MediaAnalysisManager();
        mediaAnalysisManager.initialize();
    }
}

// Auto-initialize if the tab is already visible
if (document.getElementById('tabMediaAnalysis')?.style.display !== 'none') {
    initializeMediaAnalysis();
}

// Export functions for HTML onclick handlers
window.searchMediaForAnalysis = (query) => mediaAnalysisManager?.searchMediaForAnalysis(query);
window.loadMediaForAnalysis = (mediaId) => mediaAnalysisManager?.loadMediaForAnalysis(mediaId);
window.togglePromptSource = () => mediaAnalysisManager?.togglePromptSource();
window.loadPromptDetails = () => mediaAnalysisManager?.loadPromptDetails();
window.runMediaAnalysis = () => mediaAnalysisManager?.runMediaAnalysis();
window.clearAnalysis = () => mediaAnalysisManager?.clearAnalysis();