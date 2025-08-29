/**
 * Real-Time Voice Translation Application
 * Frontend JavaScript for WebSocket communication and audio processing
 */

class VoiceTranslator {
    constructor() {
        this.websocket = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.isRealtimeMode = false;
        this.clientId = this.generateClientId();
        this.translationHistory = [];
        this.currentAudio = null;
        
        this.initializeElements();
        this.bindEvents();
        this.loadHistory();
        this.connectWebSocket();
    }
    
    initializeElements() {
        // Language selectors
        this.sourceLanguage = document.getElementById('sourceLanguage');
        this.targetLanguage = document.getElementById('targetLanguage');
        this.sourceFlag = document.getElementById('sourceFlag');
        this.targetFlag = document.getElementById('targetFlag');
        
        // Text areas
        this.sourceText = document.getElementById('sourceText');
        this.translationText = document.getElementById('translationText');
        
        // Buttons
        this.startRecordingBtn = document.getElementById('startRecording');
        this.stopRecordingBtn = document.getElementById('stopRecording');
        this.swapLanguagesBtn = document.getElementById('swapLanguages');
        this.clearSourceBtn = document.getElementById('clearSource');
        this.copySourceBtn = document.getElementById('copySource');
        this.copyTranslationBtn = document.getElementById('copyTranslation');
        this.playTranslationBtn = document.getElementById('playTranslation');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.helpBtn = document.getElementById('helpBtn');
        this.clearHistoryBtn = document.getElementById('clearHistory');
        
        // Controls
        this.realtimeMode = document.getElementById('realtimeMode');
        this.translationStatus = document.getElementById('translationStatus');
        
        // Modals
        this.settingsModal = document.getElementById('settingsModal');
        this.helpModal = document.getElementById('helpModal');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        
        // Toasts
        this.errorToast = document.getElementById('errorToast');
        this.successToast = document.getElementById('successToast');
        
        // History
        this.historyList = document.getElementById('historyList');
    }
    
    bindEvents() {
        // Language selection
        this.sourceLanguage.addEventListener('change', () => this.updateLanguageFlags());
        this.targetLanguage.addEventListener('change', () => this.updateLanguageFlags());
        this.swapLanguagesBtn.addEventListener('click', () => this.swapLanguages());
        
        // Recording
        this.startRecordingBtn.addEventListener('click', () => this.startRecording());
        this.stopRecordingBtn.addEventListener('click', () => this.stopRecording());
        
        // Text controls
        this.clearSourceBtn.addEventListener('click', () => this.clearSourceText());
        this.copySourceBtn.addEventListener('click', () => this.copyText(this.sourceText.value));
        this.copyTranslationBtn.addEventListener('click', () => this.copyText(this.translationText.textContent));
        this.playTranslationBtn.addEventListener('click', () => this.playTranslation());
        
        // Text input
        this.sourceText.addEventListener('input', () => this.handleTextInput());
        
        // Real-time mode
        this.realtimeMode.addEventListener('change', () => this.toggleRealtimeMode());
        
        // Modals
        this.settingsBtn.addEventListener('click', () => this.showModal(this.settingsModal));
        this.helpBtn.addEventListener('click', () => this.showModal(this.helpModal));
        
        // Modal close buttons
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => this.closeModal(e.target.closest('.modal')));
        });
        
        // Click outside modal to close
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModal(e.target);
            }
        });
        
        // History
        this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        
        // Toast close buttons
        document.querySelectorAll('.toast-close').forEach(btn => {
            btn.addEventListener('click', (e) => this.hideToast(e.target.closest('.toast')));
        });
        
        // Initialize language flags
        this.updateLanguageFlags();
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/translate/${this.clientId}`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('Connected', 'success');
        };
        
        this.websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateStatus('Disconnected', 'error');
            // Try to reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Connection Error', 'error');
        };
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'translation_result':
                this.handleTranslationResult(message);
                break;
            case 'text_translation_result':
                this.handleTextTranslationResult(message);
                break;
            case 'languages_switched':
                this.handleLanguagesSwitched(message);
                break;
            case 'error':
                this.showError(message.message);
                break;
            case 'pong':
                // Keep alive response
                break;
        }
    }
    
    handleTranslationResult(message) {
        this.translationText.innerHTML = message.translated_text;
        this.currentAudio = message.translated_audio;
        
        // Add to history
        this.addToHistory({
            sourceText: message.source_text,
            translatedText: message.translated_text,
            sourceLanguage: message.source_language,
            targetLanguage: message.target_language,
            timestamp: new Date().toISOString()
        });
        
        this.updateStatus('Translation Complete', 'success');
        
        // Auto-play if real-time mode is enabled
        if (this.isRealtimeMode && this.currentAudio) {
            this.playAudio(this.currentAudio);
        }
    }
    
    handleTextTranslationResult(message) {
        this.translationText.innerHTML = message.translated_text;
        this.currentAudio = message.translated_audio;
        
        // Add to history
        this.addToHistory({
            sourceText: message.source_text,
            translatedText: message.translated_text,
            sourceLanguage: message.source_language,
            targetLanguage: message.target_language,
            timestamp: new Date().toISOString()
        });
        
        this.updateStatus('Translation Complete', 'success');
    }
    
    handleLanguagesSwitched(message) {
        this.updateLanguageFlags();
        this.showSuccess('Languages switched successfully');
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processAudioChunks();
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start(1000); // Collect data every second
            this.isRecording = true;
            
            this.startRecordingBtn.style.display = 'none';
            this.stopRecordingBtn.style.display = 'flex';
            this.updateStatus('Recording...', 'processing');
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.showError('Could not access microphone. Please check permissions.');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            this.startRecordingBtn.style.display = 'flex';
            this.stopRecordingBtn.style.display = 'none';
            this.updateStatus('Processing...', 'processing');
        }
    }
    
    async processAudioChunks() {
        if (this.audioChunks.length === 0) return;
        
        try {
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            const arrayBuffer = await audioBlob.arrayBuffer();
            const base64Audio = this.arrayBufferToBase64(arrayBuffer);
            
            // Send audio chunk to server
            this.sendWebSocketMessage({
                type: 'audio_chunk',
                audio: base64Audio
            });
            
        } catch (error) {
            console.error('Error processing audio:', error);
            this.showError('Error processing audio');
        }
    }
    
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
    
    handleTextInput() {
        const text = this.sourceText.value.trim();
        
        if (text.length > 0) {
            // Debounce text translation
            clearTimeout(this.textTranslationTimeout);
            this.textTranslationTimeout = setTimeout(() => {
                this.translateText(text);
            }, 1000);
        } else {
            this.translationText.innerHTML = '<div class="placeholder">Translation will appear here...</div>';
        }
    }
    
    translateText(text) {
        if (!text.trim()) return;
        
        this.updateStatus('Translating...', 'processing');
        
        this.sendWebSocketMessage({
            type: 'text_translate',
            text: text,
            source_lang: this.sourceLanguage.value,
            target_lang: this.targetLanguage.value
        });
    }
    
    swapLanguages() {
        const tempLang = this.sourceLanguage.value;
        const tempFlag = this.sourceFlag.textContent;
        
        this.sourceLanguage.value = this.targetLanguage.value;
        this.sourceFlag.textContent = this.targetFlag.textContent;
        
        this.targetLanguage.value = tempLang;
        this.targetFlag.textContent = tempFlag;
        
        // Send language switch message
        this.sendWebSocketMessage({
            type: 'switch_languages',
            source_lang: this.sourceLanguage.value,
            target_lang: this.targetLanguage.value
        });
    }
    
    updateLanguageFlags() {
        const flags = {
            'en': '🇺🇸',
            'es': '🇪🇸'
        };
        
        this.sourceFlag.textContent = flags[this.sourceLanguage.value] || '🌐';
        this.targetFlag.textContent = flags[this.targetLanguage.value] || '🌐';
    }
    
    toggleRealtimeMode() {
        this.isRealtimeMode = this.realtimeMode.checked;
        
        if (this.isRealtimeMode) {
            this.showSuccess('Real-time mode enabled');
        } else {
            this.showSuccess('Real-time mode disabled');
        }
    }
    
    clearSourceText() {
        this.sourceText.value = '';
        this.translationText.innerHTML = '<div class="placeholder">Translation will appear here...</div>';
    }
    
    copyText(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showSuccess('Text copied to clipboard');
        }).catch(() => {
            this.showError('Failed to copy text');
        });
    }
    
    playTranslation() {
        if (this.currentAudio) {
            this.playAudio(this.currentAudio);
        } else {
            this.showError('No audio available to play');
        }
    }
    
    playAudio(base64Audio) {
        try {
            const audioBlob = this.base64ToBlob(base64Audio, 'audio/wav');
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            
            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
            };
            
            audio.play().catch(error => {
                console.error('Error playing audio:', error);
                this.showError('Error playing audio');
            });
            
        } catch (error) {
            console.error('Error creating audio:', error);
            this.showError('Error playing audio');
        }
    }
    
    base64ToBlob(base64, mimeType) {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }
    
    updateStatus(message, type = 'info') {
        const statusElement = this.translationStatus;
        const icon = statusElement.querySelector('i');
        const text = statusElement.querySelector('span');
        
        text.textContent = message;
        statusElement.className = `status-indicator ${type}`;
        
        // Remove processing class after delay
        if (type === 'processing') {
            setTimeout(() => {
                statusElement.classList.remove('processing');
            }, 3000);
        }
    }
    
    sendWebSocketMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        } else {
            this.showError('Connection lost. Trying to reconnect...');
            this.connectWebSocket();
        }
    }
    
    showModal(modal) {
        modal.style.display = 'block';
    }
    
    closeModal(modal) {
        modal.style.display = 'none';
    }
    
    showLoading() {
        this.loadingOverlay.style.display = 'flex';
    }
    
    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }
    
    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        this.errorToast.style.display = 'flex';
        setTimeout(() => this.hideToast(this.errorToast), 5000);
    }
    
    showSuccess(message) {
        document.getElementById('successMessage').textContent = message;
        this.successToast.style.display = 'flex';
        setTimeout(() => this.hideToast(this.successToast), 3000);
    }
    
    hideToast(toast) {
        toast.style.display = 'none';
    }
    
    addToHistory(item) {
        this.translationHistory.unshift(item);
        
        // Keep only last 50 items
        if (this.translationHistory.length > 50) {
            this.translationHistory = this.translationHistory.slice(0, 50);
        }
        
        this.saveHistory();
        this.renderHistory();
    }
    
    renderHistory() {
        this.historyList.innerHTML = '';
        
        this.translationHistory.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            
            const time = new Date(item.timestamp).toLocaleTimeString();
            const languages = `${item.sourceLanguage.toUpperCase()} → ${item.targetLanguage.toUpperCase()}`;
            
            historyItem.innerHTML = `
                <div class="history-item-header">
                    <span class="history-item-languages">${languages}</span>
                    <span class="history-item-time">${time}</span>
                </div>
                <div class="history-item-text">${item.sourceText}</div>
                <div class="history-item-translation">${item.translatedText}</div>
            `;
            
            // Add click to copy functionality
            historyItem.addEventListener('click', () => {
                this.sourceText.value = item.sourceText;
                this.translationText.innerHTML = item.translatedText;
                this.showSuccess('Translation loaded');
            });
            
            this.historyList.appendChild(historyItem);
        });
    }
    
    saveHistory() {
        localStorage.setItem('translationHistory', JSON.stringify(this.translationHistory));
    }
    
    loadHistory() {
        const saved = localStorage.getItem('translationHistory');
        if (saved) {
            this.translationHistory = JSON.parse(saved);
            this.renderHistory();
        }
    }
    
    clearHistory() {
        this.translationHistory = [];
        this.saveHistory();
        this.renderHistory();
        this.showSuccess('History cleared');
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new VoiceTranslator();
    
    // Make app globally available for debugging
    window.voiceTranslator = app;
    
    // Show loading initially
    app.showLoading();
    
    // Hide loading after a short delay (simulating model loading)
    setTimeout(() => {
        app.hideLoading();
    }, 2000);
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, stop recording if active
        if (window.voiceTranslator && window.voiceTranslator.isRecording) {
            window.voiceTranslator.stopRecording();
        }
    }
});

// Handle beforeunload
window.addEventListener('beforeunload', () => {
    if (window.voiceTranslator && window.voiceTranslator.websocket) {
        window.voiceTranslator.websocket.close();
    }
}); 