# VOICE-ASSISTANT-COM-WHISPER-GPT-TEXT-TO-SPEECH
Este projeto implementa um assistente de voz completo: Áudio > Texto > Resposta IA > Áudio de resposta

!pip install -q openai openai-whisper gTTS

import os
from base64 import b64decode
from IPython.display import Audio, display, Javascript
from google.colab import output
import whisper
from openai import OpenAI
from gtts import gTTS

language = "pt"

RECORD = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

def record_audio(sec=5):
    display(Javascript(RECORD))
    js_result = output.eval_js(f"record({sec * 1000})")
    audio_data = b64decode(js_result.split(",")[1])
    file_name = "request_audio.wav"
    with open(file_name, "wb") as f:
        f.write(audio_data)
    return file_name

print("Ouvindo...\n")
record_file = record_audio(5)
display(Audio(record_file, autoplay=False))

model = whisper.load_model("small")
result = model.transcribe(record_file, fp16=False, language=language)
transcription = result["text"]

print("\nTranscrição:")
print(transcription)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": transcription}
    ]
)

chatgpt_response = response.choices[0].message.content

print("\nResposta do modelo:")
print(chatgpt_response)

tts = gTTS(text=chatgpt_response, lang=language, slow=False)
response_audio = "response_audio.wav"
tts.save(response_audio)

display(Audio(response_audio, autoplay=True))
