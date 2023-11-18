import whisper

model = whisper.load_model("base")
result = model.transcribe("录音.m4a")
print(result["text"])