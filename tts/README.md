# 语音合成（Text-to-Speech generation）

## TTS 21.2k：<https://github.com/coqui-ai/TTS>

如果您只对使用已发布的 🐸TTS 模型合成语音感兴趣，那么从 PyPI 安装是最简单的选择。

bug：

hf的网络问题
```py
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /coqui/XTTS-v1/resolve/hifigan/model.pth (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f5ed5c620e0>: Failed to establish a new connection: [Errno 101] Network is unreachable'))
```

没办法，只能在“网络良好”的状态下去玩语音合成了，简单说一下需要用到的模型：

使用`tts --list_models`列出的模型比较多，简单分类的话：
- multilingual类，一般也是多个数据集训练而成，使用的时候必须指明language，以及speaker_wav
- 单语言：例如`en`，`zh-CN`，中文只有一个，en比较多一些，默认是tts_models/en/ljspeech/tacotron2-DDC
- vocoder_models，这个对应一个vocoder的过程，好像zh-CN没有，其它语言都有一个对应的

## GPT-SoVITS 18k：<https://github.com/RVC-Boss/GPT-SoVITS>

有WebUI，很良心的项目，模型需要从huggingface上下载，作者有打包好的Windows整合包，下载可以直接傻瓜式操作。1分钟的素材就可以实现克隆声音，这个微调过程也是在web界面操作，特别方便。

## 辅助工具

### 语音识别（Speech Recognition）

- Whisper 46.8k：<https://github.com/openai/whisper>

Whisper 是一种自动**语音识别** （ASR） 系统，根据从网络收集的 680,000 小时的多语言和多任务监督数据进行训练。我们表明，使用如此庞大而多样化的数据集可以提高对口音、背景噪音和技术语言的鲁棒性。此外，它可以转录多种语言，以及从这些语言翻译成英语。我们正在开源模型和推理代码，作为构建有用应用程序和进一步研究健壮语音处理的基础。

bug：

```py
Traceback (most recent call last):
  File "whisper.py", line 1, in <module>
    import whisper
  File "/home/gy/proj/llm-notes/gpt/whisper.py", line 3, in <module>
    model = whisper.load_model("base")
AttributeError: partially initialized module 'whisper' has no attribute 'load_model' (most likely due to a circular import)
```

这是由于文件是"whisper.py"，和包名相同，产生了重复导入的错误；更改文件名，在Windows下会出现错误：

```py
File "C:\ProgramData\anaconda3\envs\wsp\lib\subprocess.py", line 1456, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
FileNotFoundError: [WinError 2] 系统找不到指定的文件。
```

但是这个找不到的文件，并不是我们给定音频文件名的问题，换到Linux系统下，发现错误变成了：

```py
File "/home/gy/anaconda3/envs/wsp/lib/python3.8/subprocess.py", line 1720, in _execute_child
    raise child_exception_type(errno_num, err_msg, err_filename)
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

成功告知了错误的原因，按照官网的说明安装`sudo apt update && sudo apt install ffmpeg`即可

## 大文件上传（LFS）

```sh
# 预操作
sudo apt install git-lfs
git-lfs install
git lfs track "gpt/tts"

# 常规提交，先推送一次.gitattributes
git add .gitattributes
git commit -m "ready to add lfs"
git push origin main

# 然后提交大文件
git add .
git commit -m "add tts models"
git push
```