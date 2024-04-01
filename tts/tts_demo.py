import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# 中文
# tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST").to(device)
# OUTPUT_PATH = "speech.wav"
# tts.tts_to_file(text="你好，世界。", file_path=OUTPUT_PATH)

# 英文
# tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to(device)
# OUTPUT_PATH = "speech.wav"
# tts.tts_to_file(text="hello world.", file_path=OUTPUT_PATH)

# 多语言
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="good morning!", speaker_wav="speech.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text="good morning!", speaker_wav="speech.wav", language="en", file_path="output.wav")