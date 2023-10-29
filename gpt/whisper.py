import whisper

model = whisper.load_model("base")
result = model.transcribe("../../录音.mp3")
print(result["text"])