import sounddevice as sd #import sounddevice library for record_audio, 'pip install sounddevice'
import tempfile #import tempfile library for save_audio_to_wav
import wave #import wave library for save_audio_to_wav
import whisper #import whisper library for transcribe_audio, 'pip install openai-whisper'
import requests #import requests library for query_llm, 'pip install requests'
import pyttsx3 #import pyttsx3 library for speak, 'pip install pyttsx3'

whisper_model = whisper.load_model("base") # Load the Whisper model for transcribe_audio
tts_engine = pyttsx3.init() # Initialize the TTS engine for speak

# Record audio from microphone
def record_audio(duration=5, samplerate=16000):
   print("🎙️ Listening...")
   audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
   sd.wait()
   return audio.flatten()


# Save audio to temp WAV file
def save_audio_to_wav(audio, samplerate=16000):
   with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
       with wave.open(f.name, 'w') as wf:
           wf.setnchannels(1)
           wf.setsampwidth(2)
           wf.setframerate(samplerate)
           wf.writeframes(audio.tobytes())
       return f.name
   
# Transcribe audio to text
def transcribe_audio(file_path):
   print("🗣️ Transcribing...")
   result = whisper_model.transcribe(file_path) 
   return result["text"]

# Load context from text file
def load_context(file_path="context.txt", max_chars=4000):
   """
   Loads context from a text file, optionally truncating to max_chars.
   """
   try:
       with open(file_path, "r", encoding="utf-8") as f:
           content = f.read()
           return content[:max_chars]  # truncate if needed
   except FileNotFoundError:
       print(f"⚠️ Context file not found: {file_path}")
       return ""


# Query local LLM via Ollama
def query_llm(user_input, model="rhysjones/phi-2-orange"):
    print("🧠 Thinking...")


    context = load_context()
    prompt = f"""
You are a strict, literal assistant. Use must only use the information explicitly listed in the contxt below.
Do not guess, infer, or invent anything. If the date is not listed, say "I don't know."
Do not reformat or rephrase the items. Do not add explanations or commentary. Just list the menu items exactly as shown.


Example:
Context:
----------
**August 1 (Friday):**
- BBQ Chicken Sandwich  
- Coleslaw  
- Orange  

**August 4 (Monday):**
- Chicken Florentine  
- Pasta  
- Spinach  
- Apple  


----------
Question:
What is the menu for August 1st?
Answer:
BBQ Chicken Sandwich, Coleslaw, Orange


Now your turn:
Context
----------
{context}
----------


Question:
{user_input}


Answer:
"""


    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}
    )
    output = ""
    for line in response.iter_lines():
        if line:
            chunk = line.decode("utf-8")
            if '"response":"' in chunk:
                output += chunk.split('"response":"')[1].split('"')[0]
    return output

# Speak response
def speak(text):
    print("🔊 Speaking...")
    tts_engine.say(text)
    tts_engine.runAndWait()


def main():
    print("Calling record audio!")

    audio = record_audio() # Save the recorded audio to a variable
    wav_path = save_audio_to_wav(audio) # Call save_audio_to_wav function to save the audio to a WAV file
    text = transcribe_audio(wav_path) # Call transcribe_audio function to transcribe the audio file
    print(f"📝 You said: {text}")
    if text.strip():            
        response = query_llm(text)
        print(f"🤖 LLM says: {response}")
        speak(response)

if __name__ == "__main__":
    main()
