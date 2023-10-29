# 辅助工具

## 语音识别（Speech Recognition）

- Whisper 46.8k：<https://github.com/openai/whisper>

Whisper 是一种自动**语音识别** （ASR） 系统，根据从网络收集的 680,000 小时的多语言和多任务监督数据进行训练。我们表明，使用如此庞大而多样化的数据集可以提高对口音、背景噪音和技术语言的鲁棒性。此外，它可以转录多种语言，以及从这些语言翻译成英语。我们正在开源模型和推理代码，作为构建有用应用程序和进一步研究健壮语音处理的基础。

## bug

```py
Traceback (most recent call last):
  File "whisper.py", line 1, in <module>
    import whisper
  File "/home/gy/proj/llm-notes/gpt/whisper.py", line 3, in <module>
    model = whisper.load_model("base")
AttributeError: partially initialized module 'whisper' has no attribute 'load_model' (most likely due to a circular import)
```

## 语音合成（Text-to-Speech generation）

- TTS 21,2k：<https://github.com/coqui-ai/TTS>

如果您只对使用已发布的 🐸TTS 模型合成语音感兴趣，那么从 PyPI 安装是最简单的选择。

## bug

hf的网络问题
```py
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /coqui/XTTS-v1/resolve/hifigan/model.pth (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f5ed5c620e0>: Failed to establish a new connection: [Errno 101] Network is unreachable'))
```