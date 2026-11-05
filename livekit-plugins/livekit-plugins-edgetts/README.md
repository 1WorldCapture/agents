# livekit-plugins-edgetts

基于 `edge-tts` 的 Microsoft Edge 在线语音合成 LiveKit Agents 插件。无需申请 API Key，支持流式返回 24 kHz 单声道 MP3，并由 LiveKit 解码为 PCM 音频帧。

> 此插件使用 Edge 浏览器的在线朗读服务，不是有服务等级承诺的 Azure TTS API。微软可能调整协议、限制访问或中断服务；生产环境建议配置备用 TTS。文本会发送到微软服务器。

## 安装

```bash
uv add livekit-plugins-edgetts
```

## 使用

```python
from livekit.agents import AgentSession
from livekit.plugins import edgetts

session = AgentSession(
    tts=edgetts.TTS(
        voice="zh-CN-XiaoxiaoNeural",
        rate="+0%",
        volume="+0%",
        pitch="+0Hz",
    ),
    # stt=...
    # llm=...
)
```

获取当前可用的声音列表：

```python
speech = edgetts.TTS()
voices = await speech.list_voices()
```

可用配置：

| 参数 | 环境变量 | 默认值 |
| --- | --- | --- |
| `voice` | — | `en-US-EmmaMultilingualNeural` |
| `rate` | — | `+0%` |
| `volume` | — | `+0%` |
| `pitch` | — | `+0Hz` |
| `proxy` | `EDGE_TTS_PROXY` | 无 |
| `receive_timeout` | — | 每个文本块的 WSS 接收超时为 60 秒 |

`update_options()` 可以为后续合成修改 `voice`、`rate`、`volume` 和 `pitch`。

Edge TTS 支持流式音频输出，但不支持持续输入文本 token，因此插件声明 `streaming=False`。`AgentSession` 会自动使用 LiveKit 的句子级 `StreamAdapter`：LLM 输出按句合成，每句的 MP3 音频仍会边接收边解码播放。长文本拆分和鉴权由上游 `edge-tts` 库处理，不支持任意自定义 SSML。
