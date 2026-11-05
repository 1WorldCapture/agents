# livekit-plugins-doubao

豆包在线语音识别的 LiveKit Agents 插件。参考 VoiceKey 的豆包输入法接入，实现基于 WSS、Protobuf 和 Opus 的流式 STT；不加载离线模型，也不使用火山引擎离线识别。

> 此插件使用豆包输入法在线协议，并非火山引擎公开 ASR API。上游协议或服务策略变化可能影响可用性。

## 安装

```bash
uv add livekit-plugins-doubao
```

## 使用

```python
from livekit.agents import AgentSession
from livekit.plugins import doubao

session = AgentSession(
    stt=doubao.STT(language="zh-CN"),
    # llm=...
    # tts=...
)
```

默认使用在线设备注册，并将设备 ID 缓存到 `~/.livekit/doubao-device.json`。已有设备 ID 时可以跳过注册：

```python
stt = doubao.STT(device_id="your-device-id")
```

可用配置：

| 参数 | 环境变量 | 默认值 |
| --- | --- | --- |
| `language` | — | `zh-CN`，仅用于识别事件的语言标记 |
| `interim_results` | — | `True` |
| `punctuation` | — | `True` |
| `speech_rejection` | — | `True` |
| `app_key` | `DOUBAO_APP_KEY` | 参考客户端的 ASR app key |
| `device_id` | `DOUBAO_DEVICE_ID` | 未提供时在线注册 |
| `device_cache_path` | `DOUBAO_DEVICE_CACHE_PATH` | `~/.livekit/doubao-device.json` |
| `base_url` | `DOUBAO_WS_URL` | 豆包输入法在线 WSS 地址 |

`recognize()` 和 `stream()` 共用同一套在线流式协议。输入音频会重采样到 16 kHz、转成单声道，再编码为 20ms Opus 帧。服务端按句返回临时和最终结果；`end_input()` 会发送结束帧与 `FinishSession`，并等待剩余定稿。
