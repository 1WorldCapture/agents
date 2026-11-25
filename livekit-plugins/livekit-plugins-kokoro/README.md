# livekit-plugins-kokoro

基于 [Kokoro-82M v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh) 的本地语音合成 LiveKit Agents 插件。模型用 ONNX Runtime 在 CPU 上运行（[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)），中文用 misaki 转音素，夹在中文里的英文单词用 espeak-ng 转音素。不需要 API Key，文本不离开本机，输出 24 kHz 单声道 PCM。

## 安装

```bash
uv add livekit-plugins-kokoro
```

模型文件（`kokoro-v1.1-zh.onnx` 约 326 MB，`voices-v1.1-zh.bin` 约 54 MB）在第一次合成时从 kokoro-onnx 的 GitHub Release 下载，也可以提前下载：

```bash
python agent.py download-files
```

## 使用

```python
from livekit.agents import AgentSession
from livekit.plugins import kokoro

session = AgentSession(
    tts=kokoro.TTS(voice="zf_001", speed=1.0),
    # stt=...
    # llm=...
)
```

| 参数 | 环境变量 | 默认值 |
| --- | --- | --- |
| `voice` | — | `zf_001` |
| `speed` | — | `1.0`（0.5 到 2.0） |
| `model_dir` | `KOKORO_MODEL_DIR` | `~/.cache/kokoro` |

声音：`zf_001` 到 `zf_099` 是女声，`zm_009` 到 `zm_100` 是男声（编号不连续），`await kokoro.TTS().list_voices()` 返回全部 103 个。`update_options()` 可以为后续合成修改 `voice` 和 `speed`。

加载模型约需几秒。`AgentSession` 启动时会调用 `prewarm()` 在后台加载；同一进程里的所有 `TTS` 实例共用一份模型，加载、转音素和推理都在同一个后台线程上依次执行。

Kokoro 不支持持续输入文本 token，因此插件声明 `streaming=False`，`AgentSession` 会自动用句子级 `StreamAdapter` 逐句合成。较长的一句会按音素分段，边合成边播放。
