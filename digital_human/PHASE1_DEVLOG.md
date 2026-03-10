# Digital Human Phase 1 - 开发日志

## 概述

本文档记录数字人第一阶段闭环的开发过程和联调说明。

## 已完成内容

### 1. 模块适配器层 (`digital_human/`)

| 文件 | 功能 | 状态 |
|------|------|------|
| `config.py` | 统一配置管理 | ✅ 完成 |
| `sensevoice_adapter.py` | SenseVoice ASR 适配器 | ✅ 完成 |
| `rag_adapter.py` | RAG/LLM 对话适配器 | ✅ 完成 |
| `tts_adapter.py` | TTS 适配器 (GPT-SoVITS + Kokoro) | ✅ 完成 |
| `liveportrait_adapter.py` | FasterLivePortrait 动画适配器 | ✅ 完成 |
| `orchestrator.py` | 主编排器 | ✅ 完成 |
| `cli.py` | 命令行入口 | ✅ 完成 |
| `api_server.py` | FastAPI 服务 | ✅ 完成 |

### 2. 核心链路

#### 主链路（语音输入）
```
用户语音 → SenseVoice ASR → RAG/LLM → TTS → FasterLivePortrait → 视频
```

#### 调试链路（文本输入）
```
用户文本 → RAG/LLM → TTS → FasterLivePortrait → 视频
```

### 3. 复用的现有能力

| 模块 | 复用内容 |
|------|----------|
| **SenseVoice** | `funasr.AutoModel` + `SenseVoiceSmall` 模型 |
| **RAG** | `PsyMindSystem.process_message()` 完整流程 |
| **TTS** | `Kokoro-82M` 本地 TTS（备选 GPT-SoVITS API）|
| **LivePortrait** | `GradioLivePortraitPipeline.run_text_driving()` 和 `run_audio_driving()` |
| **JoyVASA** | `JoyVASAAudio2MotionPipeline.gen_motion_sequence()` |

## 环境配置

### 必需环境变量

```bash
# 阿里云 DashScope API Key（用于 RAG/LLM）
export DASHSCOPE_API_KEY="sk-your-key"
```

### 可选环境变量

```bash
# GPT-SoVITS API 地址（可选，默认使用 Kokoro）
export GPT_SOVITS_API_URL="http://127.0.0.1:9880"

# SenseVoice 设备（默认 cuda:0）
export SENSEVOICE_DEVICE="cuda:0"

# 默认源图像（可选）
export DEFAULT_SOURCE_IMAGE="/path/to/avatar.jpg"
```

### 依赖安装

```bash
# 核心依赖
pip install funasr httpx omegaconf opencv-python torch torchaudio

# RAG 依赖
pip install langchain-openai langchain-core chromadb sqlite3

# FasterLivePortrait 依赖
pip install onnx onnxruntime-gpu tensorrt  # 或 onnxruntime

# Kokoro TTS（可选，用于本地 TTS）
pip install kokoro soundfile phonemizer
# Windows 需要安装 eSpeak NG: https://github.com/espeak-ng/espeak-ng/releases
```

## 使用方式

### 1. 命令行工具

```bash
# 文本输入模式
python -m digital_human.cli --mode text --source path/to/avatar.jpg --input "你好"

# 语音输入模式
python -m digital_human.cli --mode voice --source path/to/avatar.jpg --input user.wav

# 交互式模式
python -m digital_human.cli --mode interactive --source path/to/avatar.jpg
```

### 2. Python API

```python
from digital_human import create_orchestrator

# 创建编排器
orchestrator = create_orchestrator(
    source_image="path/to/avatar.jpg",
    dashscope_api_key="your-key",
)

# 文本输入
result = orchestrator.process_text_sync("你好，请介绍一下自己")
print(f"视频: {result.video_path}")

# 语音输入
result = orchestrator.process_voice_sync("path/to/user_audio.wav")
print(f"识别: {result.transcript}")
print(f"回复: {result.response}")
print(f"视频: {result.video_path}")
```

### 3. REST API 服务

```bash
# 启动服务
uvicorn digital_human.api_server:app --host 0.0.0.0 --port 8000

# 文本输入
curl -X POST "http://localhost:8000/text" \
    -F "text=你好" \
    -F "source_image=@avatar.jpg" \
    --output output.mp4

# 语音输入
curl -X POST "http://localhost:8000/voice" \
    -F "audio=@user.wav" \
    -F "source_image=@avatar.jpg" \
    --output output.mp4
```

## 模型文件

### 必需模型

| 模型 | 路径 | 获取方式 |
|------|------|----------|
| SenseVoice-Small | `iic/SenseVoiceSmall` | 自动下载 (ModelScope) |
| FasterLivePortrait | `FasterLivePortrait/checkpoints/` | `huggingface-cli download warmshao/FasterLivePortrait --local-dir checkpoints` |
| JoyVASA | `FasterLivePortrait/checkpoints/joyvasa_*.pth` | 包含在 FasterLivePortrait 模型中 |
| Kokoro-82M | `FasterLivePortrait/checkpoints/Kokoro-82M/` | 可选，用于本地 TTS |

### RAG 知识库

RAG 系统首次运行时会自动初始化知识库，数据来自 `Psychology_Rag/data/`。

## 已知限制

### 1. 首次运行延迟
- 各模块采用延迟加载，首次调用会初始化模型（约 30-60 秒）
- 后续调用会显著加快

### 2. GPT-SoVITS 外部依赖
- GPT-SoVITS 需要单独部署
- 如不可用，会自动回退到 Kokoro 本地 TTS

### 3. JoyVASA 处理时间
- 长音频的 motion 生成较慢（约 2-5 秒/秒音频）
- 可考虑后续优化为流式处理

### 4. 视频渲染
- 当前为离线渲染，生成完整视频后返回
- 第二阶段可考虑实时流式输出

## 联调状态

| 环节 | 状态 | 说明 |
|------|------|------|
| SenseVoice ASR | ✅ 可用 | 需下载模型 |
| RAG/LLM | ✅ 可用 | 需配置 API Key |
| Kokoro TTS | ⚠️ 部分 | 需要 eSpeak NG (Windows) |
| GPT-SoVITS TTS | ⚠️ 外部依赖 | 需单独部署服务 |
| JoyVASA Motion | ✅ 可用 | 需下载模型 |
| LivePortrait Render | ✅ 可用 | 需下载模型 |

## 下一步（第二阶段）

1. **实时流式输出**：WebSocket 帧推流
2. **双形象切换**：待机/说话状态
3. **待机微动作**：眨眼、呼吸等
4. **点头/手势**：增强动作系统
5. **WebRTC**：低延迟实时通信

---

*最后更新: 2026-03-07*
