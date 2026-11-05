# livekit-plugins-weixin

微信输入法在线语音识别的 LiveKit Agents 插件。参考 VoiceKey 的微信渠道，实现基于 WSS、Protobuf、AES、Snappy 和 Opus 的流式 STT；不加载微信离线语音模型，也不是微信消息语音转文字接口。

> 此插件使用微信输入法在线协议，并非腾讯公开的 ASR API。上游协议或服务策略变化可能影响可用性；音频会发送到腾讯服务器。

## 安装

```bash
uv add livekit-plugins-weixin
```

## 使用

```python
from livekit.agents import AgentSession
from livekit.plugins import weixin

session = AgentSession(
    stt=weixin.STT(language="zh-CN"),
    # llm=...
    # tts=...
)
```

首次连接会在线生成本地设备身份并换取 UIN，缓存到 `~/.livekit/weixin-identity.json`。后续连接复用该身份；已有身份时也可以显式提供：

```python
stt = weixin.STT(device="your-device", uin="your-uin")
```

可用配置：

| 参数 | 环境变量 | 默认值 |
| --- | --- | --- |
| `language` | — | `zh-CN`，仅用于识别事件的语言标记 |
| `interim_results` | — | `True` |
| `device` | `WEIXIN_DEVICE` | 未提供时从缓存读取或在线生成 |
| `uin` | `WEIXIN_UIN` | 必须与 `device` 同时提供 |
| `identity_cache_path` | `WEIXIN_IDENTITY_CACHE_PATH` | `~/.livekit/weixin-identity.json` |
| `base_url` | `WEIXIN_WS_URL` | 微信输入法在线 WSS 地址 |

`recognize()` 和 `stream()` 共用同一套在线流式协议。输入音频会重采样到 16 kHz、转成单声道，再编码为 20ms Opus 帧，每 6 帧批量上传。服务端返回整句覆盖式临时结果；`end_input()` 会发送结束标记并轮询整理后的定稿，最后发出 `FINAL_TRANSCRIPT` 和 `END_OF_SPEECH`。
