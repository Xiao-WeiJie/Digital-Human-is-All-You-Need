# Digital Human Phase 2 - Frontend

数字人第二阶段前端界面。

## 功能

- 数字人展示与切换
- 文本输入交互
- 语音输入（浏览器录音/文件上传）
- 实时状态显示
- WebSocket 实时通信

## 使用方式

### 1. 启动后端服务

```bash
# 在项目根目录
python start_server.py
```

### 2. 访问前端

打开浏览器访问: http://localhost:8000/

## 文件结构

```
frontend/
├── index.html    # 主页面
├── style.css     # 样式
├── app.js        # 前端逻辑
└── README.md     # 本文件
```

## 配置

点击页面右下角的设置按钮可以配置：

- **服务器地址**: WebSocket 连接地址 (默认: ws://localhost:8000/ws)
- **API 地址**: REST API 地址 (默认: http://localhost:8000)

## 功能说明

### 数字人切换

- 顶部显示当前选中的数字人
- 点击底部的选项卡切换数字人
- 切换后后续交互使用新的数字人

### 文本输入

- 在输入框中输入文字
- 点击"发送"或按 Enter 键提交
- 等待 AI 生成回复视频

### 语音输入

**方式一：浏览器录音**

- 按住"按住说话"按钮进行录音
- 松开按钮自动发送

**方式二：上传音频**

- 点击"上传音频"选择本地音频文件
- 点击"发送音频"提交

### 状态指示

页面显示当前处理状态：

- 🎤 语音识别
- 🧠 AI 思考
- 🔊 语音合成
- 🎬 视频生成

## 开发说明

### 依赖

- 纯原生 HTML/CSS/JS，无需构建
- 需要现代浏览器支持 WebSocket 和 MediaRecorder API

### API 接口

前端通过以下方式与后端通信：

1. **WebSocket** (`/ws`): 实时状态更新
2. **REST API**:
   - `GET /avatars`: 获取数字人列表
   - `POST /avatar/set`: 设置当前数字人
   - `POST /text`: 文本输入
   - `POST /voice`: 语音输入
   - `GET /idle/{avatar_id}`: 获取待机视频

### 消息格式

WebSocket 消息采用 JSON 格式：

```json
{
    "type": "text_input",
    "data": { "text": "你好" },
    "timestamp": 1234567890.123
}
```

消息类型：
- 客户端 -> 服务器: `text_input`, `voice_input`, `switch_avatar`, `reset_session`, `ping`
- 服务器 -> 客户端: `state_update`, `asr_result`, `llm_response`, `tts_progress`, `video_ready`, `avatar_list`, `avatar_switched`, `error`, `pong`

## 浏览器兼容性

- Chrome 80+
- Firefox 75+
- Safari 14+
- Edge 80+

需要支持：
- WebSocket
- MediaRecorder API
- ES6+ JavaScript
