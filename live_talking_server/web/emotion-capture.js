/**
 * EmotionCapture - 情绪捕捉组件
 *
 * 通过摄像头捕捉用户表情，调用后端情绪检测 API，返回情绪分析结果
 *
 * 使用示例:
 *   const capture = new EmotionCapture({
 *     serverUrl: '',
 *     sampleInterval: 1000,
 *     videoElement: document.getElementById('camera'),
 *     sessionId: 'default'
 *   });
 *
 *   capture.on(EmotionCapture.Events.DETECTED, (emotion) => {
 *     console.log('检测到情绪:', emotion.dominant_emotion);
 *   });
 *
 *   capture.start();
 */

class EmotionCapture {
    /**
     * 默认配置
     */
    static DEFAULT_OPTIONS = {
        serverUrl: '',                  // API 基础路径（空字符串表示相对路径）
        sampleInterval: 15000,           // 采样间隔（毫秒）
        imageQuality: 0.8,              // JPEG 压缩质量
        videoConstraints: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
        },
        sessionId: 'default',           // 会话 ID
        enabled: true,                  // 是否启用
        autoStart: false,               // 是否自动开始
        maxRetries: 3,                  // 最大重试次数
        retryDelay: 30000,              // 重试延迟（毫秒）
    };

    /**
     * 事件类型
     */
    static Events = {
        STARTED: 'started',             // 摄像头已启动
        STOPPED: 'stopped',             // 摄像头已停止
        DETECTED: 'detected',           // 检测到情绪
        ERROR: 'error',                 // 发生错误
        NO_FACE: 'no_face',             // 未检测到人脸
        PERMISSION_DENIED: 'permission_denied', // 权限被拒绝
        DISABLED: 'disabled',           // 功能被禁用
    };

    constructor(options = {}) {
        this.options = { ...EmotionCapture.DEFAULT_OPTIONS, ...options };
        this.videoElement = this.options.videoElement || null;

        // 状态
        this.isRunning = false;
        this.isPaused = false;
        this.currentEmotion = null;
        this.consecutiveErrors = 0;
        this.retryTimeout = null;

        // 事件监听器
        this.eventListeners = new Map();

        // 内部资源
        this.stream = null;
        this.canvas = null;
        this.ctx = null;
        this.sampleIntervalId = null;

        // 检查浏览器支持
        this.checkBrowserSupport();
    }

    /**
     * 检查浏览器支持
     */
    checkBrowserSupport() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.warn('[EmotionCapture] Browser does not support getUserMedia');
            this.browserSupported = false;
        } else {
            this.browserSupported = true;
        }
    }

    /**
     * 设置视频元素
     */
    setVideoElement(element) {
        this.videoElement = element;
    }

    /**
     * 设置会话 ID
     */
    setSessionId(sessionId) {
        this.options.sessionId = sessionId;
    }

    /**
     * 注册事件监听器
     */
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
        return this;
    }

    /**
     * 移除事件监听器
     */
    off(event, callback) {
        if (!this.eventListeners.has(event)) return this;
        const listeners = this.eventListeners.get(event);
        const index = listeners.indexOf(callback);
        if (index > -1) {
            listeners.splice(index, 1);
        }
        return this;
    }

    /**
     * 触发事件
     */
    emit(event, data) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.forEach(callback => {
                try {
                    callback(data);
                } catch (e) {
                    console.error(`[EmotionCapture] Event handler error for ${event}:`, e);
                }
            });
        }
    }

    /**
     * 启动捕捉
     */
    async start() {
        if (this.isRunning) return;
        if (!this.browserSupported) {
            this.emit(EmotionCapture.Events.ERROR, {
                message: '浏览器不支持摄像头访问',
                code: 'BROWSER_NOT_SUPPORTED'
            });
            return;
        }

        console.log('[EmotionCapture] Starting... videoElement:', this.videoElement ? 'exists' : 'null');

        try {
            // 获取摄像头权限
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: this.options.videoConstraints,
                audio: false
            });

            console.log('[EmotionCapture] Camera stream obtained');

            // 绑定到视频元素
            if (this.videoElement) {
                this.videoElement.srcObject = this.stream;
                await this.videoElement.play();
                console.log('[EmotionCapture] Video element playing, readyState:', this.videoElement.readyState);

                // 等待视频元数据加载
                if (this.videoElement.readyState < 1) {
                    await new Promise((resolve) => {
                        this.videoElement.onloadedmetadata = () => {
                            console.log('[EmotionCapture] Video metadata loaded');
                            resolve();
                        };
                    });
                }
            } else {
                console.warn('[EmotionCapture] No videoElement provided, preview will not show');
            }

            // 创建离屏 canvas
            this.canvas = document.createElement('canvas');
            this.canvas.width = this.options.videoConstraints.width.ideal || 640;
            this.canvas.height = this.options.videoConstraints.height.ideal || 480;
            this.ctx = this.canvas.getContext('2d');

            this.isRunning = true;
            this.isPaused = false;
            this.consecutiveErrors = 0;

            // 开始定时采样
            this.startSampling();

            this.emit(EmotionCapture.Events.STARTED);
            console.log('[EmotionCapture] Started successfully');

        } catch (error) {
            console.error('[EmotionCapture] Failed to start:', error);

            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                this.emit(EmotionCapture.Events.PERMISSION_DENIED, {
                    message: '摄像头权限被拒绝',
                    originalError: error
                });
            } else {
                this.emit(EmotionCapture.Events.ERROR, {
                    message: '无法启动摄像头: ' + error.message,
                    originalError: error
                });
            }
        }
    }

    /**
     * 停止捕捉
     */
    stop() {
        if (!this.isRunning) return;

        // 停止采样
        this.stopSampling();

        // 停止所有轨道
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        // 清理视频元素
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }

        // 清理资源
        this.canvas = null;
        this.ctx = null;
        this.isRunning = false;
        this.isPaused = false;

        // 清除重试定时器
        if (this.retryTimeout) {
            clearTimeout(this.retryTimeout);
            this.retryTimeout = null;
        }

        this.emit(EmotionCapture.Events.STOPPED);
        console.log('[EmotionCapture] Stopped');
    }

    /**
     * 暂停捕捉
     */
    pause() {
        if (!this.isRunning || this.isPaused) return;
        this.isPaused = true;
        this.stopSampling();
        console.log('[EmotionCapture] Paused');
    }

    /**
     * 恢复捕捉
     */
    resume() {
        if (!this.isRunning || !this.isPaused) return;
        this.isPaused = false;
        this.startSampling();
        console.log('[EmotionCapture] Resumed');
    }

    /**
     * 获取当前情绪
     */
    getCurrentEmotion() {
        return this.currentEmotion;
    }

    /**
     * 开始定时采样
     */
    startSampling() {
        if (this.sampleIntervalId) return;

        // 立即执行一次
        this.captureAndDetect();

        // 定时采样
        this.sampleIntervalId = setInterval(() => {
            this.captureAndDetect();
        }, this.options.sampleInterval);
    }

    /**
     * 停止定时采样
     */
    stopSampling() {
        if (this.sampleIntervalId) {
            clearInterval(this.sampleIntervalId);
            this.sampleIntervalId = null;
        }
    }

    /**
     * 捕获帧并检测
     */
    async captureAndDetect() {
        if (!this.isRunning || this.isPaused) return;
        if (!this.videoElement || !this.ctx) return;

        // 检查视频是否有足够的数据
        if (this.videoElement.readyState < 2) {
            console.debug('[EmotionCapture] Video not ready, readyState:', this.videoElement.readyState);
            return;
        }

        // 检查视频尺寸
        const vw = this.videoElement.videoWidth;
        const vh = this.videoElement.videoHeight;
        if (!vw || !vh || vw < 10 || vh < 10) {
            console.warn('[EmotionCapture] Invalid video dimensions:', vw, 'x', vh);
            return;
        }

        try {
            // 绘制当前帧到 canvas
            this.ctx.drawImage(
                this.videoElement,
                0, 0,
                this.canvas.width,
                this.canvas.height
            );

            // 转换为 base64
            const base64Image = this.canvas.toDataURL('image/jpeg', this.options.imageQuality);

            // 调用后端 API
            const result = await this.detectEmotion(base64Image);

            if (result) {
                this.consecutiveErrors = 0;

                // 检查是否被禁用
                if (result.disabled) {
                    this.emit(EmotionCapture.Events.DISABLED, result);
                    this.pause();
                    return;
                }

                this.currentEmotion = result;
                this.emit(EmotionCapture.Events.DETECTED, result);
            }

        } catch (error) {
            console.error('[EmotionCapture] Capture error:', error);
            this.consecutiveErrors++;

            if (this.consecutiveErrors >= this.options.maxRetries) {
                this.emit(EmotionCapture.Events.ERROR, {
                    message: `连续 ${this.consecutiveErrors} 次检测失败，暂停采样`,
                    consecutiveErrors: this.consecutiveErrors
                });

                // 暂停采样，延迟后自动恢复
                this.pause();
                this.retryTimeout = setTimeout(() => {
                    if (this.isRunning) {
                        this.resume();
                        this.consecutiveErrors = 0;
                    }
                }, this.options.retryDelay);
            }
        }
    }

    /**
     * 调用后端情绪检测 API
     */
    async detectEmotion(base64Image) {
        const url = this.options.serverUrl + '/api/v1/emotion/detect';

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: base64Image,
                session_id: this.options.sessionId,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        if (result.code !== 0) {
            throw new Error(result.msg || 'Detection failed');
        }

        return result.data;
    }

    /**
     * 获取情绪上下文（历史统计）
     */
    async getEmotionContext() {
        const url = this.options.serverUrl + '/api/v1/emotion/context?session_id=' + encodeURIComponent(this.options.sessionId);

        const response = await fetch(url, {
            method: 'GET',
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        if (result.code !== 0) {
            throw new Error(result.msg || 'Failed to get context');
        }

        return result.data;
    }

    /**
     * 销毁实例
     */
    destroy() {
        this.stop();
        this.eventListeners.clear();
        this.videoElement = null;
        console.log('[EmotionCapture] Destroyed');
    }
}

// 导出（兼容模块和全局）
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EmotionCapture;
} else if (typeof window !== 'undefined') {
    window.EmotionCapture = EmotionCapture;
}
