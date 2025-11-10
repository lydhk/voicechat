import sounddevice as sd #import sounddevice library for record_audio, 'pip install sounddevice'
import tempfile #import tempfile library for save_audio_to_wav
import wave #import wave library for save_audio_to_wav
import whisper #import whisper library for transcribe_audio, 'pip install openai-whisper'
import requests #import requests library for query_llm, 'pip install requests'
import platform #import platform library for beep
import pyttsx3 #import pyttsx3 library for speak, 'pip install pyttsx3'
import os #import os library for load_context
import winsound #import winsound library for beep on Windows
import numpy as np #import numpy library for beep on non-Windows
import time #import time library for record_until_silence
import webrtcvad #import webrtcvad library for voice activity detection, 'pip install webrtcvad'
import json #import json library for ollama_respond
import queue #import queue library for detect_voice and record_until_silence


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
CONTEXT_FILE = "context.txt"

SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MAX_RECORD_SECONDS = 15
SILENCE_THRESHOLD_FRAMES = int(0.8 * 1000 / FRAME_DURATION)  # 0.8 sec silence


# Load context from text file
def load_context():
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() 
    else:
        return ""

def build_prompt(context: str, user_input: str) -> str:
    """Construct strict, literal prompt to avoid hallucination."""
    return f"""
You are a strict, literal assistant. You must answer **only** using the information explicitly listed in the context below.
If the requested date or menu is not present, reply exactly with: "I don't know."
Do not guess, infer, or make assumptions.
Do not reformat, rephrase, or add any commentary.
List the menu items exactly as they appear in the context, separated by commas.

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

----------

Now your turn:
Context:
----------
{context}
----------

Question:
{user_input}

Answer:
"""

def ollama_respond(user_input: str) -> str:
    """Send grounded prompt + context to Ollama and return model response."""
    context = load_context()
    prompt = build_prompt(context, user_input)

    payload = {"model": MODEL_NAME, "prompt": prompt}
    try:
        r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60)
        r.raise_for_status()
        full_text = ""
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    full_text += data["response"]
        return full_text.strip()
    except Exception as e:
        print(f"[ERROR] Ollama request failed: {e}")
        return "Sorry, there was an error contacting the language model."

def tts_speak(text: str):
    """Speak text using pyttsx3."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def beep():
    """Beep cue (cross-platform)."""
    if platform.system() == "Windows":
        winsound.Beep(880, 200)
    else:
        duration = 0.2
        freq = 880
        t = np.linspace(0, duration, int(44100 * duration), False)
        tone = np.sin(freq * 2 * np.pi * t)
        sd.play(tone, 44100)
        sd.wait()

def detect_voice(vad, q):
    """Wait until speech is detected."""
    while True:
        audio = sd.rec(FRAME_SIZE, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        is_speech = vad.is_speech(audio.tobytes(), SAMPLE_RATE)
        if is_speech:
            q.put(audio.copy())
            return

def record_until_silence(vad, q):
    """Record voice until silence detected."""
    frames = []
    silent_frames = 0
    start_time = time.time()
    while True:
        audio = sd.rec(FRAME_SIZE, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        frames.append(audio.copy())

        if vad.is_speech(audio.tobytes(), SAMPLE_RATE):
            silent_frames = 0
        else:
            silent_frames += 1

        # Stop if silence > threshold or max time reached
        if (
            silent_frames > SILENCE_THRESHOLD_FRAMES
            or (time.time() - start_time) > MAX_RECORD_SECONDS
        ):
            break

    data = np.concatenate(frames, axis=0)
    # write("last_input.wav", SAMPLE_RATE, data)
    return data

def run_session():
    vad = webrtcvad.Vad(2)
    q = queue.Queue()
    print("Listening for speech...")

    detect_voice(vad, q)
    beep()
    print("Recording...")
    audio_data = record_until_silence(vad, q)

    # STT placeholder: integrate with Whisper/Vosk later
    print("Performing STT (placeholder)...")
    prompt_text = "[voice captured]"  # Replace with real STT output
    print(f"Prompt: {prompt_text}")

    print("Querying Ollama...")
    answer = ollama_respond(prompt_text)
    print(f"Ollama response: {answer}")
    tts_speak(answer)
    print("Done. Ready for next query.\n")

if __name__ == "__main__":
    print("VoiceChat assistant running.")
    while True:
        try:
            run_session()
            # restart modules each session to avoid long-run drift
            time.sleep(2)
        except KeyboardInterrupt:
            print("Stopping assistant.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(5)
