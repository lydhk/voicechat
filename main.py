import sounddevice as sd #import sounddevice library for record_audio, 'pip install sounddevice'
import tempfile #import tempfile library for save_audio_to_wav
import wave #import wave library for save_audio_to_wav
# from faster_whisper import WhisperModel #import whisper library for transcribe_audio, 'pip install openai-whisper'
import whisper #import whisper library for transcribe_audio, 'pip install openai-whisper'
import requests #import requests library for query_llm, 'pip install requests'
import platform #import platform library for beep
import pyttsx3 #import pyttsx3 library for speak, 'pip install pyttsx3'
import os #import os library for load_context
import winsound #import winsound library for beep on Windows
import numpy as np #import numpy library for beep on non-Windows
import time #import time library for record_until_silence
try:
    import webrtcvad
except ImportError:
    import webrtcvad_wheels as webrtcvad
import webrtcvad #import webrtcvad library for voice activity detection, 'pip install webrtcvad'
import json #import json library for ollama_respond
import queue #import queue library for detect_voice and record_until_silence
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
CONTEXT_FILE = "context.txt"

SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MAX_RECORD_SECONDS = 15
SILENCE_THRESHOLD_FRAMES = int(0.8 * 1000 / FRAME_DURATION)  # 0.8 sec silence


# Initialize Whisper model once (reused across calls)
whisper_model = whisper.load_model("base")
# whisper_model = WhisperModel("base", device="cpu")

def record_audio(duration=4.5, samplerate=16000):
   print("🎙️ Listening...")
   audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
   sd.wait()
   return audio.flatten()

def save_audio_to_wav(audio, samplerate=16000):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        with wave.open(f.name, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())
        return f.name

def transcribe_audio(file_path):
    print("🗣️ Transcribing...")
    result = whisper_model.transcribe(file_path) 
    return result["text"]

def transcribe_audio2(audio_data, sample_rate=16000):
    """
    Convert recorded audio (numpy array) to text using faster-whisper.
    Lightweight version without SciPy.
    """
    tmp_path = None
    try:
        # Save to temporary WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_path = tmp_wav.name
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_data.tobytes())

        # Transcribe with Whisper
        segments, info = whisper_model.transcribe(tmp_path)
        text = " ".join([seg.text for seg in segments]).strip()
        print(f"[STT] {text}")
        return text or "[unintelligible]"
    except Exception as e:
        print(f"[STT ERROR] {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

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

def detect_voice(vad, q, sample_rate=16000, frame_size=160, min_speech_ms=250, pre_beep_delay=0.3):
    """
    Wait until sustained speech is detected before triggering.
    - min_speech_ms: minimum duration of continuous speech to confirm (ms)
    - pre_beep_delay: pause before beeping, so user has time to get ready
    """
    speech_start = None
    frame_duration = frame_size / sample_rate  # seconds per frame

    print("[VAD] Listening for voice...")

    while True:
        # Capture a short frame of audio
        audio = sd.rec(frame_size, samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()

        is_speech = vad.is_speech(audio.tobytes(), sample_rate)

        if is_speech:
            if speech_start is None:
                speech_start = time.time()
            elif (time.time() - speech_start) * 1000 >= min_speech_ms:
                # Sustained speech confirmed
                print("[VAD] Sustained voice detected")
                time.sleep(pre_beep_delay)  # allow short pause before beep
                q.put(audio.copy())
                return
        else:
            speech_start = None  # reset if silence

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
    vad = webrtcvad.Vad(3)
    q = queue.Queue()
    print("Listening for speech...")

    detect_voice(vad, q)
    beep()
    time.sleep(0.5)
    print("Recording...")
    # audio_data = record_until_silence(vad, q)

    # trying old way
    audio_data = record_audio() 

    # trying old way
    print("Processing audio...")
    wav_path = save_audio_to_wav(audio_data) # Call save_audio_to_wav function to save the audio to a WAV file
    prompt_text = transcribe_audio(wav_path) # Call transcribe_audio function to transcribe the audio file
    
    # STT placeholder: integrate with Whisper/Vosk later
    #print("Performing STT...")
    #prompt_text = transcribe_audio(audio_data, SAMPLE_RATE)
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
