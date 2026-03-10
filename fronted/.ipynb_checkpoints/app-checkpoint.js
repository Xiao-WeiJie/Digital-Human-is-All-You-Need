/**
 * Digital Human Phase 2 - Frontend Application
 *
 * 功能：
 * - WebSocket 实时通信
 * - 文本/语音交互
 * - 数字人切换
 * - 状态显示
 */

// ============================================================
// 配置
// ============================================================

const CONFIG = {
    // 自动使用当前页面的主机名
    get wsUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}/ws`;
    },
    get apiUrl() {
        return `${window.location.protocol}//${window.location.host}`;
    },
    reconnectInterval: 3000,
    maxReconnectAttempts: 5,
};

// ============================================================
// 状态管理
// ============================================================

const state = {
    connected: false,
    reconnectAttempts: 0,
    currentAvatar: null,
    avatars: [],
    isRecording: false,
    isProcessing: false,
    uploadedFile: null,
    processingStep: null,
};

// ============================================================
// DOM 元素
// ============================================================

const elements = {
    // 连接状态
    connectionStatus: document.getElementById('connectionStatus'),
    statusDot: document.querySelector('.status-dot'),
    statusText: document.querySelector('.status-text'),

    // 数字人显示
    avatarDisplay: document.getElementById('avatarDisplay'),
    avatarVideo: document.getElementById('avatarVideo'),
    avatarPlaceholder: document.getElementById('avatarPlaceholder'),
    stateIndicator: document.getElementById('stateIndicator'),
    avatarOptions: document.getElementById('avatarOptions'),

    // 会话
    conversationHistory: document.getElementById('conversationHistory'),
    resetSessionBtn: document.getElementById('resetSessionBtn'),

    // 输入
    textInput: document.getElementById('textInput'),
    sendTextBtn: document.getElementById('sendTextBtn'),
    recordBtn: document.getElementById('recordBtn'),
    recordingIndicator: document.getElementById('recordingIndicator'),
    audioFileInput: document.getElementById('audioFileInput'),
    uploadedFileName: document.getElementById('uploadedFileName'),
    sendAudioBtn: document.getElementById('sendAudioBtn'),

    // 进度
    progressPanel: document.getElementById('progressPanel'),

    // 设置
    settingsPanel: document.getElementById('settingsPanel'),
    settingsToggle: document.getElementById('settingsToggle'),
    closeSettingsBtn: document.getElementById('closeSettingsBtn'),
    serverUrlInput: document.getElementById('serverUrlInput'),
    apiUrlInput: document.getElementById('apiUrlInput'),
};

// ============================================================
// WebSocket 管理
// ============================================================

let ws = null;

function connectWebSocket() {
    const wsUrl = elements.serverUrlInput?.value || CONFIG.wsUrl;

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('[WS] 连接成功');
            state.connected = true;
            state.reconnectAttempts = 0;
            updateConnectionStatus('connected', '已连接');
        };

        ws.onclose = (event) => {
            console.log('[WS] 连接关闭', event);
            state.connected = false;
            updateConnectionStatus('disconnected', '已断开');

            // 自动重连
            if (state.reconnectAttempts < CONFIG.maxReconnectAttempts) {
                state.reconnectAttempts++;
                console.log(`[WS] 尝试重连 (${state.reconnectAttempts}/${CONFIG.maxReconnectAttempts})`);
                setTimeout(connectWebSocket, CONFIG.reconnectInterval);
            }
        };

        ws.onerror = (error) => {
            console.error('[WS] 错误', error);
            updateConnectionStatus('error', '连接错误');
        };

        ws.onmessage = (event) => {
            handleWebSocketMessage(JSON.parse(event.data));
        };

    } catch (error) {
        console.error('[WS] 连接失败', error);
        updateConnectionStatus('error', '连接失败');
    }
}

function sendWebSocketMessage(type, data = {}) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type, data }));
    } else {
        console.warn('[WS] 未连接，无法发送消息');
        // 回退到 HTTP API
        fallbackToHttpApi(type, data);
    }
}

function updateConnectionStatus(status, text) {
    const { statusDot, statusText } = elements;

    statusDot.className = 'status-dot';
    if (status === 'connected') {
        statusDot.classList.add('connected');
    } else if (status === 'connecting') {
        statusDot.classList.add('connecting');
    }

    statusText.textContent = text;
}

// ============================================================
// HTTP API 回退
// ============================================================

async function fallbackToHttpApi(type, data) {
    const apiUrl = elements.apiUrlInput?.value || CONFIG.apiUrl;

    try {
        if (type === 'text_input') {
            setProcessing(true);
            showProgress();

            // Step 1: 发送文本
            const formData = new FormData();
            formData.append('text', data.text);
            if (state.currentAvatar) {
                formData.append('avatar_id', state.currentAvatar);
            }

            updateProgress('llm', 'active');
            const response = await fetch(`${apiUrl}/text`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            // 获取视频
            const blob = await response.blob();
            const videoUrl = URL.createObjectURL(blob);

            updateProgress('video', 'completed');
            hideProgress();
            setProcessing(false);

            // 播放视频
            playVideo(videoUrl);

            // 添加消息
            addMessage('user', data.text);
            addMessage('ai', '已生成回复视频');

        } else if (type === 'voice_input') {
            setProcessing(true);
            showProgress();

            const formData = new FormData();
            formData.append('audio', data.audioBlob, 'recording.wav');
            if (state.currentAvatar) {
                formData.append('avatar_id', state.currentAvatar);
            }

            updateProgress('asr', 'active');
            const response = await fetch(`${apiUrl}/voice`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const blob = await response.blob();
            const videoUrl = URL.createObjectURL(blob);

            updateProgress('video', 'completed');
            hideProgress();
            setProcessing(false);

            playVideo(videoUrl);
            addMessage('ai', '已生成回复视频');

        } else if (type === 'switch_avatar') {
            const response = await fetch(`${apiUrl}/avatar/set`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ avatar_id: data.avatar_id }),
            });

            if (response.ok) {
                const result = await response.json();
                state.currentAvatar = data.avatar_id;
                updateAvatarDisplay(result);
            }

        } else if (type === 'reset_session') {
            await fetch(`${apiUrl}/session/reset`, { method: 'POST' });
            clearConversation();
        }

    } catch (error) {
        console.error('[API] 请求失败', error);
        showError(`请求失败: ${error.message}`);
        hideProgress();
        setProcessing(false);
    }
}

// ============================================================
// 消息处理
// ============================================================

function handleWebSocketMessage(message) {
    console.log('[WS] 收到消息', message);

    const { type, data } = message;

    switch (type) {
        case 'avatar_list':
            handleAvatarList(data.avatars);
            break;

        case 'avatar_switched':
            handleAvatarSwitched(data);
            break;

        case 'state_update':
            handleStateUpdate(data);
            break;

        case 'asr_result':
            handleAsrResult(data);
            break;

        case 'llm_response':
            handleLlmResponse(data);
            break;

        case 'tts_progress':
            handleTtsProgress(data);
            break;

        case 'video_ready':
            handleVideoReady(data);
            break;

        case 'error':
            handleError(data);
            break;

        case 'pong':
            // 心跳响应
            break;

        default:
            console.log('[WS] 未知消息类型', type);
    }
}

function handleAvatarList(avatars) {
    state.avatars = avatars;
    renderAvatarOptions();

    // 选择第一个作为默认
    if (avatars.length > 0 && !state.currentAvatar) {
        sendWebSocketMessage('switch_avatar', { avatar_id: avatars[0].avatar_id });
    }
}

function handleAvatarSwitched(data) {
    state.currentAvatar = data.avatar_id;
    updateAvatarSelection(data.avatar_id);
    updateAvatarDisplay(data);
}

function handleStateUpdate(data) {
    const { state: newState, message } = data;
    updateStateIndicator(newState, message);

    if (newState === 'thinking') {
        updateProgress('llm', 'active');
    } else if (newState === 'talking') {
        updateProgress('video', 'active');
    }
}

function handleAsrResult(data) {
    updateProgress('asr', 'completed');
    addMessage('user', data.transcript);
    updateProgress('llm', 'active');
}

function handleLlmResponse(data) {
    updateProgress('llm', 'completed');
    updateProgress('tts', 'active');
    // 先显示文本回复
    addMessage('ai', data.response);
}

function handleTtsProgress(data) {
    if (data.status === 'completed') {
        updateProgress('tts', 'completed');
        updateProgress('video', 'active');
    }
}

function handleVideoReady(data) {
    console.log('[WS] 收到视频就绪消息:', data);
    updateProgress('video', 'completed');
    hideProgress();
    setProcessing(false);

    // 播放视频
    if (data.video_url) {
        console.log('[WS] 使用 video_url 播放:', data.video_url);
        playVideo(data.video_url);
    } else if (data.video_path) {
        // 使用 API 获取视频
        const apiUrl = elements.apiUrlInput?.value || CONFIG.apiUrl;
        const videoUrl = `${apiUrl}/video/${encodeURIComponent(data.video_path)}`;
        console.log('[WS] 使用 video_path 构建URL:', videoUrl);
        playVideo(videoUrl);
    } else {
        console.error('[WS] 视频就绪消息中没有 video_url 或 video_path');
    }
}

function handleError(data) {
    console.error('[WS] 服务器错误', data);
    showError(data.error);
    hideProgress();
    setProcessing(false);
}

// ============================================================
// UI 更新
// ============================================================

function renderAvatarOptions() {
    const container = elements.avatarOptions;
    container.innerHTML = '';

    state.avatars.forEach(avatar => {
        const option = document.createElement('div');
        option.className = 'avatar-option';
        option.dataset.avatarId = avatar.avatar_id;

        if (avatar.avatar_id === state.currentAvatar) {
            option.classList.add('active');
        }

        option.innerHTML = `
            <div class="avatar-thumb"></div>
            <span class="avatar-name">${avatar.name}</span>
        `;

        option.addEventListener('click', () => {
            sendWebSocketMessage('switch_avatar', { avatar_id: avatar.avatar_id });
        });

        container.appendChild(option);
    });
}

function updateAvatarSelection(avatarId) {
    const options = elements.avatarOptions.querySelectorAll('.avatar-option');
    options.forEach(opt => {
        opt.classList.toggle('active', opt.dataset.avatarId === avatarId);
    });
}

function updateAvatarDisplay(data) {
    elements.avatarPlaceholder.classList.add('hidden');
    elements.avatarVideo.classList.remove('hidden');

    if (data.source_image) {
        // 显示源图像作为静态图
        const apiUrl = elements.apiUrlInput?.value || CONFIG.apiUrl;
        elements.avatarVideo.poster = `${apiUrl}/image/${encodeURIComponent(data.source_image)}`;
    }
}

function updateStateIndicator(state, message) {
    const indicator = elements.stateIndicator;
    const icon = indicator.querySelector('.state-icon');
    const text = indicator.querySelector('.state-text');

    icon.className = 'state-icon';

    switch (state) {
        case 'idle':
            text.textContent = '待机';
            break;
        case 'listening':
            icon.classList.add('thinking');
            text.textContent = '聆听中';
            break;
        case 'thinking':
            icon.classList.add('thinking');
            text.textContent = '思考中';
            break;
        case 'talking':
            icon.classList.add('talking');
            text.textContent = '回复中';
            break;
        case 'error':
            icon.classList.add('error');
            text.textContent = message || '错误';
            break;
        default:
            text.textContent = message || state;
    }
}

function showProgress() {
    elements.progressPanel.classList.remove('hidden');
    // 重置所有步骤
    elements.progressPanel.querySelectorAll('.progress-step').forEach(step => {
        step.className = 'progress-step';
    });
}

function hideProgress() {
    setTimeout(() => {
        elements.progressPanel.classList.add('hidden');
    }, 1000);
}

function updateProgress(step, status) {
    const stepEl = elements.progressPanel.querySelector(`[data-step="${step}"]`);
    if (stepEl) {
        stepEl.className = 'progress-step ' + status;
    }
}

function addMessage(role, content) {
    const container = elements.conversationHistory;

    // 移除欢迎消息
    const welcome = container.querySelector('.welcome-message');
    if (welcome) {
        welcome.remove();
    }

    const message = document.createElement('div');
    message.className = `message ${role}`;

    const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

    message.innerHTML = `
        <div class="message-content">
            <p>${escapeHtml(content)}</p>
            <span class="message-time">${time}</span>
        </div>
    `;

    container.appendChild(message);
    container.scrollTop = container.scrollHeight;
}

function clearConversation() {
    elements.conversationHistory.innerHTML = `
        <div class="welcome-message">
            <p>你好！我是你的数字人助手。</p>
            <p>你可以通过文字或语音与我交流。</p>
        </div>
    `;
}

function showError(message) {
    // 简单的错误提示
    alert(`错误: ${message}`);
}

function setProcessing(processing) {
    state.isProcessing = processing;
    elements.sendTextBtn.disabled = processing;
    elements.recordBtn.disabled = processing;
}

function playVideo(url) {
    console.log('[Video] 播放视频:', url);
    elements.avatarVideo.src = url;
    elements.avatarVideo.muted = false;  // 确保取消静音
    elements.avatarVideo.load();  // 重新加载视频

    elements.avatarVideo.onloadeddata = () => {
        console.log('[Video] 视频加载完成，开始播放');
        elements.avatarVideo.play().catch(e => {
            console.log('[Video] 自动播放被阻止，尝试静音播放', e);
            // 如果自动播放被阻止，尝试静音播放
            elements.avatarVideo.muted = true;
            elements.avatarVideo.play().catch(e2 => console.log('[Video] 静音播放也失败', e2));
        });
    };

    elements.avatarVideo.onerror = (e) => {
        console.error('[Video] 视频加载错误', e);
    };

    elements.avatarPlaceholder.classList.add('hidden');
    elements.avatarVideo.classList.remove('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// 事件绑定
// ============================================================

function bindEvents() {
    // 文本发送
    elements.sendTextBtn.addEventListener('click', sendTextMessage);
    elements.textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTextMessage();
        }
    });

    // 语音录制
    elements.recordBtn.addEventListener('mousedown', startRecording);
    elements.recordBtn.addEventListener('mouseup', stopRecording);
    elements.recordBtn.addEventListener('mouseleave', stopRecording);
    elements.recordBtn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startRecording();
    });
    elements.recordBtn.addEventListener('touchend', stopRecording);

    // 文件上传
    elements.audioFileInput.addEventListener('change', handleFileUpload);
    elements.sendAudioBtn.addEventListener('click', sendUploadedAudio);

    // 重置会话
    elements.resetSessionBtn.addEventListener('click', () => {
        sendWebSocketMessage('reset_session');
        clearConversation();
    });

    // 设置面板
    elements.settingsToggle.addEventListener('click', () => {
        elements.settingsPanel.classList.toggle('open');
    });

    elements.closeSettingsBtn.addEventListener('click', () => {
        elements.settingsPanel.classList.remove('open');
    });

    // 服务器配置变更
    elements.serverUrlInput?.addEventListener('change', () => {
        if (ws) {
            ws.close();
        }
        connectWebSocket();
    });
}

function sendTextMessage() {
    const text = elements.textInput.value.trim();
    if (!text || state.isProcessing) return;

    elements.textInput.value = '';
    addMessage('user', text);
    setProcessing(true);
    showProgress();

    sendWebSocketMessage('text_input', { text });
}

// ============================================================
// 录音功能
// ============================================================

let mediaRecorder = null;
let audioChunks = [];

async function startRecording() {
    if (state.isProcessing || state.isRecording) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            sendVoiceMessage(audioBlob);
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        state.isRecording = true;

        elements.recordBtn.classList.add('recording');
        elements.recordingIndicator.classList.add('active');

    } catch (error) {
        console.error('录音失败', error);
        showError('无法访问麦克风');
    }
}

function stopRecording() {
    if (!state.isRecording || !mediaRecorder) return;

    mediaRecorder.stop();
    state.isRecording = false;

    elements.recordBtn.classList.remove('recording');
    elements.recordingIndicator.classList.remove('active');
}

function sendVoiceMessage(audioBlob) {
    setProcessing(true);
    showProgress();
    updateProgress('asr', 'active');

    // WebSocket 发送二进制数据
    if (ws && ws.readyState === WebSocket.OPEN) {
        const reader = new FileReader();
        reader.onload = () => {
            const base64 = reader.result.split(',')[1];
            sendWebSocketMessage('voice_input', { audio_data: base64 });
        };
        reader.readAsDataURL(audioBlob);
    } else {
        // HTTP 回退
        fallbackToHttpApi('voice_input', { audioBlob });
    }
}

// ============================================================
// 文件上传
// ============================================================

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    state.uploadedFile = file;
    elements.uploadedFileName.textContent = file.name;
    elements.sendAudioBtn.style.display = 'inline-block';
}

function sendUploadedAudio() {
    if (!state.uploadedFile) return;

    const reader = new FileReader();
    reader.onload = () => {
        const base64 = reader.result.split(',')[1];
        sendWebSocketMessage('voice_input', { audio_data: base64 });
    };
    reader.readAsDataURL(state.uploadedFile);

    // 重置
    state.uploadedFile = null;
    elements.uploadedFileName.textContent = '';
    elements.sendAudioBtn.style.display = 'none';
    elements.audioFileInput.value = '';
}

// ============================================================
// 初始化
// ============================================================

function init() {
    console.log('Digital Human Phase 2 - Frontend');
    bindEvents();
    connectWebSocket();
}

// 启动
document.addEventListener('DOMContentLoaded', init);
