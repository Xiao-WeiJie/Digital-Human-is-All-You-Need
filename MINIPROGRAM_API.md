# 微信小程序 / uniapp 对接接口文档

本文档面向 uniapp 开发的微信小程序前端，说明当前已通过 FRP + Caddy 暴露的数字人后端接口，以及项目内另一个独立 Agent 后端的现状与对接方式。

## 1. 对外访问地址

- 主域名: https://yiharmony.top
- www 域名: https://www.yiharmony.top
- 内网服务源端口: 8010
- 对外暴露链路: 8010 -> FRP -> 47.239.128.254:8080 -> Caddy HTTPS 反代

建议前端固定使用一个域名作为 `API_BASE`，避免缓存和跨域配置混乱。推荐:

```js
const API_BASE = 'https://yiharmony.top'
```

## 2. 微信小程序配置要求

在微信公众平台后台把以下域名加入合法域名:

- request 合法域名: https://yiharmony.top
- uploadFile 合法域名: https://yiharmony.top
- downloadFile 合法域名: https://yiharmony.top

如果你还要兼容 `www` 入口，再额外加入:

- https://www.yiharmony.top

说明:

- 所有接口都必须走 HTTPS，不能直接访问 47.239.128.254:8080。
- 当前 `https://www.yiharmony.top/*` 会 302 跳转到 `https://yiharmony.top/*`，因此小程序基地址统一建议使用裸域名。
- 服务端返回的视频地址通常是相对路径，例如 `/videos/abc.mp4`，小程序必须拼成绝对地址后再播放。

## 3. 当前建议的小程序接入方式

### 3.1 推荐使用的接口

小程序建议对接以下 HTTP 接口:

- GET `/api/v1/health`
- GET `/api/v1/avatars`
- POST `/api/v1/generate`
- GET `/videos/{filename}`
- DELETE `/api/v1/video/{filename}`
- POST `/api/v1/emotion/detect`
- GET `/api/v1/emotion/context`
- POST `/api/v1/tongue/detect`
- POST `/api/auth/register`
- POST `/api/auth/login`
- POST `/api/auth/logout`
- GET `/api/auth/me`
- POST `/api/auth/password/reset/request`
- POST `/api/auth/password/reset/confirm`
- PUT `/api/user/preferences`
- GET `/api/agent/ride/suggest-pickup`
- GET `/api/agent/ride/search-destination`
- POST `/api/agent/ride/estimate`
- POST `/api/agent/ride/create-preview`
- POST `/api/agent/ticket/tasks`
- GET `/api/agent/ticket/tasks/{taskId}`
- GET `/api/agent/ticket/tasks/{taskId}/events`
- POST `/api/agent/ticket/tasks/{taskId}/actions/{action}`

### 3.2 不建议小程序直接使用的接口

以下接口主要给浏览器版页面或 WebRTC 页面使用，不建议微信小程序直接接:

- POST `/offer`
- POST `/human`
- POST `/humanaudio`
- POST `/interrupt_talk`
- POST `/is_speaking`
- POST `/switch_avatar`

原因:

- 这组接口围绕标准浏览器 WebRTC 会话和页面内实时播放器设计。
- 微信小程序不是标准浏览器环境，直接复用这套 SDP / RTCPeerConnection 流程成本高，兼容性也不稳定。
- 小程序更适合走“请求生成 -> 返回视频 URL -> 播放视频”的批量模式。

## 4. 通用响应约定

数字人服务整体遵循如下响应风格:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {}
}
```

约定:

- `code = 0` 表示成功。
- `code != 0` 表示失败。
- 失败时通常返回 `msg`。
- 某些接口成功时没有 `msg`，前端以 `code === 0` 为准。

## 5. 数字人主服务接口

### 5.1 健康检查

请求:

- 方法: GET
- 路径: `/api/v1/health`
- Content-Type: 无要求

成功响应:

```json
{
  "code": 0,
  "data": {
    "status": "ok",
    "service": "digital-human-server"
  }
}
```

用途:

- 小程序启动时探测服务是否可用。
- 反向代理、运维探活。

### 5.2 获取数字人列表

请求:

- 方法: GET
- 路径: `/api/v1/avatars`

说明:

- 这是给小程序新增的推荐接口。
- 旧页面还在使用 POST `/list_avatars`，两者返回结构一致。

成功响应示例:

```json
{
  "code": 0,
  "data": {
    "avatars": [
      {
        "id": "human_1",
        "name": "小暖",
        "description": "默认数字人形象",
        "source_image": "/human_choice/Human_1/qys.png",
        "idle_video": "/human_choice/Human_1/qys.mp4",
        "listening_video": "/human_choice/listening1.mp4",
        "emotion_reaction_videos": {
          "happy": "/human_choice/happy1.mp4",
          "sad": "/human_choice/sad1.mp4"
        }
      },
      {
        "id": "human_2",
        "name": "小安",
        "description": "备用数字人形象",
        "source_image": "/human_choice/Human_2/wzx.png",
        "idle_video": "/human_choice/Human_2/wzx.mp4",
        "listening_video": "/human_choice/listening2.mp4",
        "emotion_reaction_videos": {
          "happy": "/human_choice/happy2.mp4",
          "sad": "/human_choice/sad2.mp4"
        }
      }
    ],
    "default": "human_1",
    "emotion_reaction": {
      "enabled": true,
      "delay_min_ms": 2000,
      "delay_max_ms": 3000,
      "happy_reply": "你今天看起来心情不错哦，是有什么开心的事吗？你愿意和我分享一下喜悦吗？",
      "sad_reply": "你今天面色看起来不太好，是有什么心事吗？你愿意跟我聊聊发生了什么吗？有时候把心里的想法说出来，会让自己感觉轻松一些。"
    }
  }
}
```

字段说明:

- `avatars[].id`: 数字人 ID。当前可用值为 `human_1`、`human_2`。
- `avatars[].source_image`: 头像预览图，可在小程序里用于列表展示。
- `avatars[].idle_video`: 待机视频。
- `avatars[].listening_video`: 录音时可播放的聆听动画。
- `default`: 默认数字人。
- `emotion_reaction`: 首次识别到情绪时的固定反馈配置。

### 5.3 文本生成数字人视频

这是小程序最核心的接口。

请求:

- 方法: POST
- 路径: `/api/v1/generate`
- Content-Type: `application/json`

请求体:

```json
{
  "text": "我今天有点焦虑，想和你聊聊。",
  "avatar_id": "human_1",
  "session_id": "miniapp_user_1001"
}
```

字段说明:

- `text`: 用户输入文本，必填。
- `avatar_id`: 数字人 ID，选填，不传则使用默认值 `human_1`。
- `session_id`: 会话 ID，选填。建议前端自行传入用户维度或会话维度值，便于情绪上下文连续追踪。

成功响应示例:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "video_url": "/videos/batch_1746958123000.mp4",
    "response_text": "听起来你今天承受了不少压力，我们可以先慢慢把让你焦虑的事情理一理。",
    "user_text": "我今天有点焦虑，想和你聊聊。",
    "total_time": 18.6,
    "metrics": {
      "llm_time": 2.3,
      "tts_time": 1.7,
      "video_time": 13.2,
      "joyvasa_time": 7.8,
      "render_time": 3.4,
      "ffmpeg_time": 1.2,
      "motion_post_time": 0.4,
      "generate_video_wall_time": 13.5,
      "total_time": 18.6
    },
    "audio_duration": 6.9
  }
}
```

字段说明:

- `video_url`: 相对路径，前端必须拼成绝对地址后使用。
- `response_text`: 数字人实际回复文案，可直接渲染到聊天气泡。
- `user_text`: 服务端最终识别或接收的用户文本。
- `total_time`: 总耗时，单位秒。
- `audio_duration`: 合成音频时长，单位秒。
- `metrics`: 各阶段耗时。

`metrics` 常见字段:

- `asr_time`: 语音识别耗时，只有音频输入时才可能出现。
- `llm_time`: 大模型生成回复耗时。
- `tts_time`: 语音合成耗时。
- `video_time`: 视频生成器内部统计耗时。
- `joyvasa_time`: 驱动动作生成耗时。
- `render_time`: 渲染耗时。
- `ffmpeg_time`: 音视频封装耗时。
- `motion_post_time`: 动作后处理耗时。
- `generate_video_wall_time`: 生成视频阶段整体墙钟耗时。
- `total_time`: 整体耗时。

失败响应示例:

```json
{
  "code": -1,
  "msg": "请提供文本或语音输入"
}
```

### 5.4 语音生成数字人视频

请求:

- 方法: POST
- 路径: `/api/v1/generate`
- Content-Type: `multipart/form-data`

表单字段:

- `audio`: 音频文件，必填。
- `avatar_id`: 数字人 ID，选填。
- `session_id`: 会话 ID，选填。

兼容字段:

- 历史页面使用 `source_image` 传数字人 ID。
- 新前端统一推荐传 `avatar_id`。

说明:

- 当前服务端会把上传文件按二进制写入临时文件后交给 ASR。
- 为保证稳定，建议小程序上传标准 WAV 音频，优先使用 16kHz、单声道。
- 如果上传其他容器格式，可能出现识别失败或结果不稳定。

成功响应结构与文本模式完全一致。

### 5.5 获取生成视频

请求:

- 方法: GET
- 路径: `/videos/{filename}`

示例:

```text
GET /videos/batch_1746958123000.mp4
```

返回:

- 直接返回 MP4 文件流。
- 响应头包含 `Content-Type: video/mp4`。

前端使用建议:

- 直接将 `video_url` 拼接为绝对 URL 后赋给小程序 `video` 组件。
- 拼接规则示例: `https://yiharmony.top/videos/batch_1746958123000.mp4`

### 5.6 删除已播放视频

请求:

- 方法: DELETE
- 路径: `/api/v1/video/{filename}`

成功响应:

```json
{
  "code": 0,
  "msg": "ok"
}
```

补充说明:

- 如果服务端启动时使用了 `--save-videos`，删除请求会被跳过，并返回成功提示。
- 如果文件已经不存在，也会按成功处理。

建议:

- 小程序在视频播放结束后再调用删除接口。
- 若你希望保留历史视频，可不调删除接口。

### 5.7 单帧情绪检测

请求:

- 方法: POST
- 路径: `/api/v1/emotion/detect`
- Content-Type: `application/json`

请求体:

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
  "session_id": "miniapp_user_1001"
}
```

字段说明:

- `image`: Base64 图片字符串，可带 `data:image/...;base64,` 前缀。
- `session_id`: 选填，建议与对话会话复用，便于累计情绪上下文。

成功响应示例:

```json
{
  "code": 0,
  "data": {
    "dominant_emotion": "happy",
    "emotion_scores": {
      "happy": 85.5,
      "sad": 5.2,
      "angry": 2.1,
      "neutral": 7.2
    },
    "sad_score": 5.2,
    "risk_level": "normal",
    "confidence": 85.5,
    "timestamp": 1746958123.123
  }
}
```

特殊情况:

- 如果后端禁用了情绪识别，仍会返回 `code: 0`，但 `data.disabled = true`，且情绪固定为 `neutral`。

风险等级说明:

- `normal`: 正常
- `warning`: 需要关注
- `critical`: 需要重点关怀

### 5.8 获取情绪上下文

请求:

- 方法: GET
- 路径: `/api/v1/emotion/context`
- Query: `session_id`

示例:

```text
GET /api/v1/emotion/context?session_id=miniapp_user_1001
```

成功响应示例:

```json
{
  "code": 0,
  "data": {
    "has_data": true,
    "dominant_emotions": {
      "happy": 5,
      "neutral": 3
    },
    "avg_sad_score": 15.5,
    "latest_risk_level": "normal",
    "trend": "stable",
    "sample_count": 8,
    "prompt_context": "[用户情绪状态]\n当前主要情绪: happy\n"
  }
}
```

字段说明:

- `has_data`: 是否已有上下文。
- `dominant_emotions`: 最近窗口内主要情绪分布。
- `avg_sad_score`: 最近窗口悲伤平均分。
- `latest_risk_level`: 当前风险等级。
- `trend`: `rising`、`falling`、`stable`、`unknown`。
- `prompt_context`: 后端拼给大模型的情绪提示文本。

### 5.9 单帧舌诊识别

请求:

- 方法: POST
- 路径: `/api/v1/tongue/detect`
- Content-Type: `application/json`

请求体:

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
  "session_id": "miniapp_user_1001"
}
```

成功响应示例:

```json
{
  "code": 0,
  "data": {
    "tongue_color": "淡红",
    "coat_color": "白",
    "thickness": "薄",
    "greasiness": "不腻",
    "confidence": 0.9214,
    "advice": "本次结果仅供辅助参考",
    "timestamp": 1746958123.123
  }
}
```

业务错误码:

- `-1`: 缺少 `image`
- `-2`: 图片无效
- `-3`: 未检测到舌体
- `-4`: 检测到多个舌体，请重新取景
- `-5`: 模型加载失败或推理失败

### 5.10 Mock 配置接口

请求:

- 方法: GET
- 路径: `/api/v1/mock/config`

用途:

- 仅供页面联调或演示模式使用。
- 小程序生产接入通常不需要依赖该接口。

## 6. 将预设视频迁移到本地 uniapp 小程序

本节解决的是“数字人预设素材本地化”问题，不替代服务端生成接口。

迁移后建议按下面的职责拆分:

- 预设图片、待机视频、聆听视频、情绪反馈视频: 放到 uniapp 小程序本地静态资源。
- 文本生成视频、语音生成视频、情绪识别、舌诊识别: 继续调用远端接口。

这样做的好处是:

- 进入页面时展示数字人待机态，不依赖后端静态文件服务。
- 本地开发联调时，即使外网静态资源不可用，也能稳定预览预设素材。
- 小程序只把“生成结果视频”走网络播放，职责更清晰。

### 6.1 当前需要迁移的素材来源

服务端当前的预设素材目录是:

```text
/root/autodl-tmp/Human_Choice
```

真正需要迁移的是 [autodl-tmp/Human_Choice/avatar_config.json](autodl-tmp/Human_Choice/avatar_config.json#L1) 里已经被引用的文件，而不是把整个目录无差别复制过去。

当前配置涉及的素材如下:

- `human_1`
  - `Human_1/qys.png`
  - `Human_1/qys.mp4`
  - `listening1.mp4`
  - `happy1.mp4`
  - `sad1.mp4`
- `human_2`
  - `Human_2/wzx.png`
  - `Human_2/wzx.mp4`
  - `listening2.mp4`
  - `happy2.mp4`
  - `sad2.mp4`

当前目录里还存在 `wait1.mp4`、`wait2.mp4`、`qys_hevc.mp4`、`wzx_hevc.mp4`，但它们没有被 [autodl-tmp/Human_Choice/avatar_config.json](autodl-tmp/Human_Choice/avatar_config.json#L1) 引用。除非你计划在小程序里启用这些素材，否则这次迁移可以先不带上。

### 6.2 推荐迁移后的 uniapp 目录结构

建议在 uniapp 项目里保留和服务端一致的相对路径，这样最省心，后续配置也不用二次改写。

```text
uniapp-project/
  static/
    digital-human/
      avatar-config.local.json
      Human_1/
        qys.png
        qys.mp4
      Human_2/
        wzx.png
        wzx.mp4
      listening1.mp4
      listening2.mp4
      happy1.mp4
      happy2.mp4
      sad1.mp4
      sad2.mp4
```

建议不要把服务端原始文件名改掉，尤其不要改 `Human_1/qys.mp4` 这类配置里已经使用的相对路径。路径保持一致后，迁移逻辑只需要替换资源根目录。

### 6.3 建议保留一份前端专用配置文件

不建议小程序直接读取服务端那份完整配置，因为其中包含服务端专用字段，例如绝对路径形式的 `tts.ref_audio_dir`。小程序本地建议保留一份裁剪版配置文件，例如:

```json
{
  "avatars": {
    "human_1": {
      "name": "数字人1",
      "source_image": "Human_1/qys.png",
      "idle_video": "Human_1/qys.mp4",
      "listening_video": "listening1.mp4",
      "description": "默认数字人形象",
      "emotion_reaction_videos": {
        "happy": "happy1.mp4",
        "sad": "sad1.mp4"
      }
    },
    "human_2": {
      "name": "数字人2",
      "source_image": "Human_2/wzx.png",
      "idle_video": "Human_2/wzx.mp4",
      "listening_video": "listening2.mp4",
      "description": "备用数字人形象",
      "emotion_reaction_videos": {
        "happy": "happy2.mp4",
        "sad": "sad2.mp4"
      }
    }
  },
  "default_avatar": "human_1"
}
```

推荐文件名:

- `static/digital-human/avatar-config.local.json`

### 6.4 uniapp 本地资源映射方式

推荐在小程序前端封装一个“相对路径 -> 本地静态资源路径”的工具函数:

```js
const LOCAL_PRESET_ROOT = '/static/digital-human'

function buildLocalAssetUrl(relativePath) {
  if (!relativePath) return ''
  return `${LOCAL_PRESET_ROOT}/${relativePath.replace(/^\/+/, '')}`
}
```

再把本地配置转换成前端实际消费的结构:

```js
import avatarConfig from '@/static/digital-human/avatar-config.local.json'

function buildEmotionVideos(map = {}) {
  return Object.keys(map).reduce((result, key) => {
    result[key] = buildLocalAssetUrl(map[key])
    return result
  }, {})
}

export function getLocalAvatarPresets() {
  const avatars = avatarConfig.avatars || {}

  return {
    default: avatarConfig.default_avatar,
    avatars: Object.keys(avatars).map((id) => {
      const item = avatars[id]
      return {
        id,
        name: item.name,
        description: item.description,
        source_image: buildLocalAssetUrl(item.source_image),
        idle_video: buildLocalAssetUrl(item.idle_video),
        listening_video: buildLocalAssetUrl(item.listening_video),
        emotion_reaction_videos: buildEmotionVideos(item.emotion_reaction_videos)
      }
    })
  }
}
```

这样页面层拿到的就是可以直接绑定给 `image` 或 `video` 组件的本地地址。

### 6.5 推荐对接策略

本地迁移后，有两种对接方式。

方案 A，预设素材完全本地化，推荐优先采用:

- 页面初始化时直接读取本地 `avatar-config.local.json`。
- 小程序展示头像、待机视频、聆听视频、情绪反馈视频时，一律使用本地静态资源。
- 真正需要生成回复视频时，再调用 `/api/v1/generate`。

这种方式最稳定，也最适合本地开发。

方案 B，接口元数据继续走远端，素材地址改写为本地:

- 仍然调用 `/api/v1/avatars` 获取数字人列表。
- 以前端本地配置为准，把返回结果中的 `source_image`、`idle_video`、`listening_video`、`emotion_reaction_videos` 覆盖成本地路径。
- 其余字段如 `id`、`name`、`description`、默认数字人 ID 可以继续使用服务端响应。

如果后续服务端还会新增数字人但小程序发布时间滞后，这种“远端元数据 + 本地素材覆盖”的混合方式更容易兼容。

### 6.6 页面使用示例

```js
import { getLocalAvatarPresets } from '@/common/avatar-presets'
import { generateByText } from '@/common/api'

const presetState = getLocalAvatarPresets()
const currentAvatar = presetState.avatars.find((item) => item.id === presetState.default)

export default {
  data() {
    return {
      avatars: presetState.avatars,
      currentAvatar,
      idleVideoUrl: currentAvatar?.idle_video || '',
      listeningVideoUrl: currentAvatar?.listening_video || '',
      responseVideoUrl: '',
      responseText: ''
    }
  },
  methods: {
    async submitText(text) {
      const result = await generateByText({
        text,
        avatarId: this.currentAvatar.id,
        sessionId: `miniapp_${Date.now()}`
      })

      this.responseText = result.response_text
      this.responseVideoUrl = result.video_url
    },
    onStartRecord() {
      this.idleVideoUrl = this.currentAvatar.listening_video
    },
    onStopRecord() {
      this.idleVideoUrl = this.currentAvatar.idle_video
    },
    playEmotionReaction(emotion) {
      const url = this.currentAvatar.emotion_reaction_videos?.[emotion]
      if (url) {
        this.responseVideoUrl = url
      }
    }
  }
}
```

核心点只有一个: “预设态素材” 和 “生成结果视频” 必须分开管理，不要都混成同一套 URL 来源。

### 6.7 包体与发布注意事项

当前已启用的这批素材总大小大约为 52.9MB，主要由以下几个 MP4 构成:

- `Human_1/qys.mp4`: 约 12.5MB
- `Human_2/wzx.mp4`: 约 12.5MB
- `listening2.mp4`: 约 12.4MB
- `listening1.mp4`: 约 7.6MB

这意味着:

- 本地开发调试时可以先按上面的目录结构迁移。
- 正式发布微信小程序时，不要默认把全部视频直接塞进主包。
- 如果打包受限，优先考虑分包、首次启动下载到用户文件目录、或改为云存储/CDN 托管。

如果你只是为了本地开发联调，最小迁移集建议先带上:

- 默认数字人的 `source_image`
- 默认数字人的 `idle_video`
- 默认数字人的 `listening_video`
- 默认数字人的 `happy` / `sad` 情绪反馈视频

这样可以先把主流程跑通，再决定是否把第二个数字人也一起本地化。

## 7. uniapp 调用示例

### 6.1 公共工具

```js
const API_BASE = 'https://yiharmony.top'

function buildAbsoluteUrl(path) {
  if (!path) return ''
  if (/^https?:\/\//.test(path)) return path
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`
}
```

### 6.2 拉取数字人列表

```js
export function fetchAvatars() {
  return new Promise((resolve, reject) => {
    uni.request({
      url: `${API_BASE}/api/v1/avatars`,
      method: 'GET',
      success: (res) => {
        const body = res.data || {}
        if (body.code === 0) {
          resolve(body.data)
        } else {
          reject(new Error(body.msg || '获取数字人列表失败'))
        }
      },
      fail: reject
    })
  })
}
```

### 6.3 文本生成视频

```js
export function generateByText({ text, avatarId, sessionId }) {
  return new Promise((resolve, reject) => {
    uni.request({
      url: `${API_BASE}/api/v1/generate`,
      method: 'POST',
      header: {
        'Content-Type': 'application/json'
      },
      data: {
        text,
        avatar_id: avatarId,
        session_id: sessionId
      },
      success: (res) => {
        const body = res.data || {}
        if (body.code === 0) {
          body.data.video_url = buildAbsoluteUrl(body.data.video_url)
          resolve(body.data)
        } else {
          reject(new Error(body.msg || '生成失败'))
        }
      },
      fail: reject
    })
  })
}
```

### 6.4 音频生成视频

```js
export function generateByAudio({ filePath, avatarId, sessionId }) {
  return new Promise((resolve, reject) => {
    uni.uploadFile({
      url: `${API_BASE}/api/v1/generate`,
      filePath,
      name: 'audio',
      formData: {
        avatar_id: avatarId,
        session_id: sessionId
      },
      success: (res) => {
        const body = JSON.parse(res.data || '{}')
        if (body.code === 0) {
          body.data.video_url = buildAbsoluteUrl(body.data.video_url)
          resolve(body.data)
        } else {
          reject(new Error(body.msg || '生成失败'))
        }
      },
      fail: reject
    })
  })
}
```

### 6.5 播放结束后删除视频

```js
export function deleteVideoByUrl(videoUrl) {
  const filename = videoUrl.split('/').pop().split('?')[0]
  return new Promise((resolve, reject) => {
    uni.request({
      url: `${API_BASE}/api/v1/video/${encodeURIComponent(filename)}`,
      method: 'DELETE',
      success: (res) => {
        const body = res.data || {}
        if (body.code === 0) {
          resolve(body)
        } else {
          reject(new Error(body.msg || '删除失败'))
        }
      },
      fail: reject
    })
  })
}
```

### 6.6 情绪识别

```js
export function detectEmotion({ imageBase64, sessionId }) {
  return new Promise((resolve, reject) => {
    uni.request({
      url: `${API_BASE}/api/v1/emotion/detect`,
      method: 'POST',
      header: {
        'Content-Type': 'application/json'
      },
      data: {
        image: imageBase64,
        session_id: sessionId
      },
      success: (res) => {
        const body = res.data || {}
        if (body.code === 0) {
          resolve(body.data)
        } else {
          reject(new Error(body.msg || '情绪识别失败'))
        }
      },
      fail: reject
    })
  })
}
```

### 6.7 舌诊识别

```js
export function detectTongue({ imageBase64, sessionId }) {
  return new Promise((resolve, reject) => {
    uni.request({
      url: `${API_BASE}/api/v1/tongue/detect`,
      method: 'POST',
      header: {
        'Content-Type': 'application/json'
      },
      data: {
        image: imageBase64,
        session_id: sessionId
      },
      success: (res) => {
        const body = res.data || {}
        if (body.code === 0) {
          resolve(body.data)
        } else {
          reject(new Error(body.msg || '舌诊识别失败'))
        }
      },
      fail: reject
    })
  })
}
```

## 8. 小程序页面交互建议

推荐流程:

1. 小程序启动后请求 `/api/v1/health`。
2. 进入数字人页后请求 `/api/v1/avatars`。
3. 用户输入文本时直接调用 JSON 版 `/api/v1/generate`。
4. 用户录音时调用 `uni.uploadFile` 上传音频到 `/api/v1/generate`。
5. 拿到 `video_url` 后转绝对地址，绑定到 `video` 组件。
6. 视频播放结束后，根据你的产品策略决定是否调删除接口。

## 9. 当前独立 Agent 后端说明

项目里还有一套独立的 Agent 订票任务流后端，代码位于 `live_talking_server/web/backend`。它不是当前 8010 这个 aiohttp 服务的一部分，也没有通过你现在的 Caddy 配置对外公开。

当前前端页面里写死的默认地址是:

```text
http://localhost:3010
```

这意味着:

- 当前公网域名 `https://yiharmony.top` 默认只能直接访问数字人主服务。
- 如果小程序还要接 Agent 订票流，必须单独部署 Fastify backend，并再配一个 HTTPS 域名或路径代理。

### 8.1 Agent 后端基础信息

- 本地默认地址: `http://localhost:3010`
- 技术栈: Fastify
- CORS: 已开启
- 健康检查: GET `/health`
- 用户标识头: 可选 `x-user-id`

### 8.2 Agent 创建任务

请求:

- 方法: POST
- 路径: `/api/assistant/tasks`
- Content-Type: `application/json`

请求体:

```json
{
  "taskType": "book_train_ticket",
  "rawText": "帮我订明天上午从杭州到上海的高铁，二等座。",
  "idempotencyKey": "create-task-001"
}
```

成功响应示例:

```json
{
  "success": true,
  "requestId": "req-001",
  "data": {
    "taskId": "task_xxx",
    "status": "awaiting_intent_confirmation",
    "intent": {
      "rawText": "帮我订明天上午从杭州到上海的高铁，二等座。",
      "travelDate": "2026-05-12",
      "origin": "杭州",
      "destination": "上海",
      "departureTimeLowerBound": "08:00",
      "departureTimeUpperBound": "12:00",
      "trainTypes": ["G"],
      "seatPreferences": ["二等座"],
      "passengerNames": [],
      "allowWaitlist": false,
      "allowFallbackTime": false,
      "allowFallbackSeat": false,
      "queryOnly": false
    },
    "missingSlots": [],
    "ambiguousSlots": []
  }
}
```

### 8.3 Agent 查询任务详情

- GET `/api/tasks/{taskId}`

返回内容包含:

- `task`: 当前任务实体
- `recentEvents`: 最近事件
- `candidates`: 候选车次
- `pendingHumanAction`: 当前待人工动作

### 8.4 Agent 查询任务事件

- GET `/api/tasks/{taskId}/events?afterSequence=0`

适合小程序使用轮询方式增量拉取事件。

### 8.5 Agent SSE 订阅

- GET `/api/tasks/{taskId}/stream`

说明:

- 浏览器页面使用的是 SSE `EventSource`。
- 微信小程序不适合直接依赖原生 SSE。
- 如果以后要给小程序接这套 Agent 流，建议优先使用轮询接口 `/api/tasks/{taskId}` 或 `/events`。

SSE 事件名:

- `task_event`

SSE 数据体结构:

```json
{
  "taskId": "task_xxx",
  "sequenceId": 12,
  "timestamp": "2026-05-11T10:00:00.000Z",
  "type": "task_completed",
  "status": "completed",
  "checkpointId": "cp_xxx",
  "message": "任务已完成",
  "requiresAck": false,
  "payload": {}
}
```

### 8.6 Agent 动作接口

以下接口都要求:

- 方法: POST
- Content-Type: `application/json`
- Body 内带 `idempotencyKey`

接口列表:

- `/api/tasks/{taskId}/confirm-intent`
- `/api/tasks/{taskId}/confirm-candidate`
- `/api/tasks/{taskId}/human-step-done`
- `/api/tasks/{taskId}/confirm-submit`
- `/api/tasks/{taskId}/complete-payment`
- `/api/tasks/{taskId}/cancel`

#### confirm-intent

```json
{
  "idempotencyKey": "action-001",
  "confirmed": true
}
```

#### confirm-candidate

```json
{
  "idempotencyKey": "action-002",
  "candidateTrainNo": "G1234",
  "seatType": "二等座",
  "allowWaitlist": false
}
```

#### human-step-done

```json
{
  "idempotencyKey": "action-003",
  "checkpointId": "cp_task_xxx_human_verification",
  "actionType": "complete_qr_login"
}
```

#### confirm-submit

```json
{
  "idempotencyKey": "action-004",
  "checkpointId": "cp_task_xxx_final_submit",
  "confirmed": true
}
```

#### complete-payment

```json
{
  "idempotencyKey": "action-005",
  "checkpointId": "cp_task_xxx_payment"
}
```

#### cancel

```json
{
  "idempotencyKey": "action-006",
  "reason": "用户主动取消"
}
```

### 8.7 Agent 任务状态值

常见状态:

- `draft`
- `parsing_intent`
- `awaiting_slot_clarification`
- `awaiting_intent_confirmation`
- `searching`
- `awaiting_candidate_confirmation`
- `login_required`
- `awaiting_human_verification`
- `awaiting_final_submit_confirmation`
- `awaiting_payment`
- `completed`
- `failed`
- `cancelled`

## 10. 二期新增接口契约

本节对应 2026 年 5 月 26 日小程序二期前端当前实现，包含模式切换、认证、打车 Agent、订票 Agent 所需的新接口契约。

### 10.1 用户偏好

小程序本地设置对象新增:

```json
{
  "alertPush": true,
  "healthDigest": true,
  "quietMode": false,
  "mode": "family"
}
```

字段说明:

- `mode`: 取值为 `family` 或 `senior`
- `family`: 家属模式，保留完整说明和辅助提示
- `senior`: 老年模式，隐藏悬浮提示、放大字号、压缩次要文案

服务端推荐数据结构:

```json
{
  "preferences": {
    "alertPush": true,
    "healthDigest": true,
    "quietMode": false,
    "mode": "senior"
  }
}
```

### 10.2 认证接口

#### 10.2.1 注册

- 方法: POST
- 路径: `/api/auth/register`

请求体:

```json
{
  "phone": "13800138000",
  "username": "hefengxiyu",
  "password": "123456"
}
```

#### 10.2.2 登录

- 方法: POST
- 路径: `/api/auth/login`

请求体:

```json
{
  "account": "13800138000",
  "password": "123456"
}
```

说明:

- `account` 同时支持手机号或用户名

#### 10.2.3 当前登录态

- 方法: GET
- 路径: `/api/auth/me`
- Header: `Authorization: Bearer <token>`

#### 10.2.4 退出登录

- 方法: POST
- 路径: `/api/auth/logout`

#### 10.2.5 开发态忘记密码

- 方法: POST
- 路径: `/api/auth/password/reset/request`

请求体:

```json
{
  "account": "13800138000"
}
```

开发联调阶段建议返回:

```json
{
  "code": 0,
  "data": {
    "resetCode": "834251"
  }
}
```

确认重置:

- 方法: POST
- 路径: `/api/auth/password/reset/confirm`

请求体:

```json
{
  "account": "13800138000",
  "resetCode": "834251",
  "password": "new-password"
}
```

#### 10.2.6 更新偏好

- 方法: PUT
- 路径: `/api/user/preferences`

请求体:

```json
{
  "alertPush": true,
  "healthDigest": true,
  "quietMode": false,
  "mode": "senior"
}
```

### 10.3 打车 Agent 接口

#### 10.3.1 推荐上车点

- 方法: GET
- 路径: `/api/agent/ride/suggest-pickup`

请求参数:

- `keyword`: 当前需求文本，可选

建议返回:

```json
{
  "code": 0,
  "data": {
    "items": [
      {
        "id": "pickup-1",
        "name": "颐和苑北门",
        "address": "小区北门靠近主路",
        "lat": 31.2304,
        "lng": 121.4737
      }
    ]
  }
}
```

#### 10.3.2 搜索目的地

- 方法: GET
- 路径: `/api/agent/ride/search-destination`

请求参数:

- `keyword`: 目的地关键词

#### 10.3.3 费用和路线估算

- 方法: POST
- 路径: `/api/agent/ride/estimate`

请求体:

```json
{
  "pickup": {
    "id": "pickup-1",
    "name": "颐和苑北门"
  },
  "destination": {
    "id": "dest-1",
    "name": "市人民医院门诊楼"
  },
  "demand": "帮我叫车去医院门诊"
}
```

建议返回字段:

- `distanceText`
- `durationText`
- `priceText`
- `vehicleText`

#### 10.3.4 拟真呼叫结果

- 方法: POST
- 路径: `/api/agent/ride/create-preview`

建议返回字段:

- `summary`
- `driverText`
- `arrivalText`

### 10.4 订票 Agent 接口

#### 10.4.1 创建任务

- 方法: POST
- 路径: `/api/agent/ticket/tasks`

请求体:

```json
{
  "demand": "帮我查明天上午去医院的高铁票"
}
```

建议返回:

```json
{
  "code": 0,
  "data": {
    "taskId": "ticket-task-001",
    "candidates": [
      {
        "id": "G1234",
        "trainNo": "G1234",
        "from_station_name": "上海",
        "to_station_name": "杭州东",
        "start_time": "08:20",
        "arrive_time": "09:32",
        "priceText": "二等座 73 元"
      }
    ]
  }
}
```

#### 10.4.2 查询任务

- 方法: GET
- 路径: `/api/agent/ticket/tasks/{taskId}`

建议返回:

- `status`
- `statusText`
- `candidates`

#### 10.4.3 查询事件流

- 方法: GET
- 路径: `/api/agent/ticket/tasks/{taskId}/events`

建议返回:

```json
{
  "code": 0,
  "data": {
    "events": [
      {
        "title": "任务创建完成",
        "desc": "已开始查询候选车次",
        "time": "2026-05-26 14:30:00"
      }
    ]
  }
}
```

#### 10.4.4 任务动作

小程序当前使用以下动作名:

- `confirm-candidate`
- `submit-preview`
- `cancel`

调用方式:

- 方法: POST
- 路径: `/api/agent/ticket/tasks/{taskId}/actions/{action}`

### 10.5 点外卖前端说明

点外卖当前为纯前端拟真流程，不调用真实平台接口，也不依赖新增服务端路由。

## 11. 前端联调注意事项

### 11.1 关于视频 URL

服务端返回的是相对路径，不是完整域名。例如:

```json
{
  "video_url": "/videos/batch_xxx.mp4"
}
```

前端必须转成:

```text
https://yiharmony.top/videos/batch_xxx.mp4
```

### 11.2 关于文本模式

已支持使用 JSON 直接调 `/api/v1/generate`，这就是给 uniapp / 小程序准备的推荐接法。

### 11.3 关于音频模式

推荐上传标准 WAV 音频，避免服务端 ASR 因容器格式不一致导致识别失败。

### 11.4 关于会话 ID

建议前端自己生成并稳定传递 `session_id`，例如:

- `openid`
- `openid + 页面会话时间戳`
- 业务侧用户 ID

这样情绪识别和上下文接口可以更准确地串联同一用户会话。

### 11.5 关于 Agent 模式

如果你的小程序只做数字人问答、情绪识别、舌诊识别，那么当前 `https://yiharmony.top` 已够用。

如果你还要接认证、打车和订票 Agent 流程，下一步需要:

1. 单独部署 `live_talking_server/web/backend`。
2. 给它配置 HTTPS 域名或 Nginx/Caddy 反代路径。
3. 小程序优先用轮询接口，不要直接依赖 SSE。

### 11.6 2026-05-26 联调现状

以下状态基于 2026 年 5 月 26 日直接请求 `https://yiharmony.top` 的结果:

- `GET /api/v1/health`: `200`
- `GET /api/v1/avatars`: `200`
- `GET /api/auth/me`: `404`
- `GET /api/agent/ride/suggest-pickup`: `404`
- `GET /api/agent/ride/search-destination`: `404`
- `POST /api/agent/ticket/tasks`: `405`

说明:

- 当前线上域名已挂数字人基础服务。
- 认证与打车接口尚未对外暴露，或反代规则尚未配置到 Fastify backend。
- 订票路径已经被上游识别到，但当前方法或路由挂载状态仍与前端约定不一致，需要服务端复核。
