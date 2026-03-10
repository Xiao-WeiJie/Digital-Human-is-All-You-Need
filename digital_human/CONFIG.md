# 数字人 TTS 配置说明

## 配置文件位置

`digital_human/avatars.json`

## 配置步骤

### 1. 准备参考音频

每个数字人需要一个参考音频文件（.wav 格式），用于 GPT-SoVITS 声音克隆。

参考音频要求：
- 格式：WAV（推荐 16kHz 或更高采样率）
- 时长：3-10 秒
- 内容：清晰的语音，无背景噪音
- 语言：与数字人使用的语言一致

### 2. 上传参考音频到 GPT-SoVITS 服务器

将参考音频上传到 GPT-SoVITS 服务器上的路径，例如：
- `/autodl-tmp/digital_human/ref_audio/human_1.wav`
- `/autodl-tmp/digital_human/ref_audio/human_2.wav`

### 3. 编辑配置文件

编辑 `digital_human/avatars.json`，为每个数字人配置独立的 TTS Profile：

```json
{
  "avatars": [
    {
      "avatar_id": "human_1",
      "name": "数字人 1",
      "source_image": "/autodl-tmp/digital_human/Human_Choice/Human_1/qys.png",
      "description": "第一个数字人",
      "tts_profile": {
        "ref_audio_path": "/autodl-tmp/digital_human/ref_audio/human_1.wav",
        "prompt_text": "这里是参考音频对应的文本内容，需要与音频内容完全一致",
        "prompt_lang": "zh",
        "speed_factor": 1.0
      }
    },
    {
      "avatar_id": "human_2",
      "name": "数字人 2",
      "source_image": "/autodl-tmp/digital_human/Human_Choice/Human_2/dy.jpg",
      "description": "第二个数字人",
      "tts_profile": {
        "ref_audio_path": "/autodl-tmp/digital_human/ref_audio/human_2.wav",
        "prompt_text": "这是第二个数字人参考音频对应的文本内容",
        "prompt_lang": "zh",
        "speed_factor": 1.0
      }
    }
  ]
}
```

### 4. TTS Profile 字段说明

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `ref_audio_path` | 参考音频文件路径（服务器上的路径） | 必填 |
| `prompt_text` | 参考音频对应的文本内容 | 必填 |
| `prompt_lang` | 参考文本语言（zh/en/ja 等） | zh |
| `speed_factor` | 语速因子（0.5-2.0） | 1.0 |

### 5. 环境变量配置

也可以通过环境变量配置默认值：

```bash
# GPT-SoVITS API 地址
export GPT_SOVITS_API_URL="http://your-gpt-sovits-server:9880"

# 默认参考音频（如果 avatar 配置中没有指定）
export GPT_SOVITS_REF_AUDIO="/path/to/default.wav"
export GPT_SOVITS_PROMPT_TEXT="默认参考音频文本"
```

## 当前配置示例

当前配置文件中，两个数字人分别使用：

**Human_1:**
- 源图像: `/autodl-tmp/digital_human/Human_Choice/Human_1/qys.png`
- 参考音频: `/autodl-tmp/digital_human/ref_audio/human_1.wav`

**Human_2:**
- 源图像: `/autodl-tmp/digital_human/Human_Choice/Human_2/dy.jpg`
- 参考音频: `/autodl-tmp/digital_human/ref_audio/human_2.wav`

## 注意事项

1. **参考音频路径**: 必须是 GPT-SoVITS 服务器上的路径，不是当前项目路径
2. **文本匹配**: `prompt_text` 必须与参考音频内容完全一致
3. **音频质量**: 参考音频质量直接影响合成效果
4. **共用配置**: 如果两个数字人使用相同声音，可以指向同一个参考音频文件

## 验证配置

启动服务后，可以通过 API 验证配置：

```bash
# 获取所有数字人配置
curl http://localhost:8000/avatars

# 获取指定数字人配置
curl http://localhost:8000/avatar/human_1
```
