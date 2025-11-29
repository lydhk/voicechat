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
import re #import re library for extract_date
import sys
import io
import logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
# Use absolute path relative to this script
CONTEXT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "context.txt")

SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MAX_RECORD_SECONDS = 15
SILENCE_THRESHOLD_FRAMES = int(0.8 * 1000 / FRAME_DURATION)  # 0.8 sec silence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("openmic.log"),
        logging.StreamHandler()
    ]
)

# Initialize Whisper model once (reused across calls)
whisper_model = whisper.load_model("base")
# whisper_model = WhisperModel("base", device="cpu")

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
    text = result["text"]
    logging.info(f"[STT] {text}")
    return text

def parse_menu_context():
    """Parse context.txt into a dictionary: {'October 1': 'Item 1, Item 2', ...}"""
    menu_map = {}
    if not os.path.exists(CONTEXT_FILE):
        logging.warning(f"Context file not found: {os.path.abspath(CONTEXT_FILE)}")
        return menu_map
    
    logging.info(f"Reading context file: {os.path.abspath(CONTEXT_FILE)}")
    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to find blocks like "**October 1 (Wednesday):**" followed by items
    # Matches: **Month Day (DayOfWeek):**
    # Then captures everything until the next ** or end of string
    pattern = re.compile(r"\*\*([A-Za-z]+ \d+) \([A-Za-z]+\):\*\*(.*?)(?=\*\*|$)", re.DOTALL)
    
    matches = pattern.findall(content)
    for date_str, items_block in matches:
        # Clean up items
        items = [line.strip().replace("- ", "") for line in items_block.strip().split('\n') if line.strip().startswith("-")]
        menu_map[date_str.lower()] = ", ".join(items)
        
    logging.info(f"Loaded {len(menu_map)} menu items.")
    return menu_map

def extract_date(user_input):
    """Extract date from user input (e.g., 'October 1st' -> 'october 1')."""
    # Normalize input
    text = user_input.lower()
    
    # Regex for "Month Day" (e.g., october 1, oct 1, october 1st)
    # \b(january|february|...)\b \d+(st|nd|rd|th)?
    months = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    pattern = re.compile(rf"\b{months}\s+(\d+)(?:st|nd|rd|th)?\b")
    
    match = pattern.search(text)
    if match:
        month = match.group(1)
        day = match.group(2)
        
        # Normalize month names if needed (optional, but context.txt uses full names)
        # For now, assuming user speaks full month or we match partial.
        # Let's map short months to full if needed, but context.txt seems to have full names.
        # Actually, let's just return what we found and rely on loose matching or normalization if needed.
        # But context.txt has "October", so "oct" needs to be mapped.
        
        full_months = {
            "jan": "january", "feb": "february", "mar": "march", "apr": "april",
            "jun": "june", "jul": "july", "aug": "august", "sep": "september",
            "oct": "october", "nov": "november", "dec": "december"
        }
        if month in full_months:
            month = full_months[month]
            
        return f"{month} {day}"
    return None

def lookup_menu(user_input):
    """Look up menu deterministically."""
    date_key = extract_date(user_input)
    
    if not date_key:
        return "Please specify a date, for example, October 1st."
        
    menu_map = parse_menu_context()
    
    # Try exact match
    if date_key in menu_map:
        return f"The menu for {date_key.title()} is: {menu_map[date_key]}."
        
    return f"I couldn't find a menu for {date_key.title()}."

# Removed ollama_respond and build_prompt as they are no longer needed.

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
    print("🔴 Recording... (speak now)")
    
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
    return data

def record_audio(duration=5.0, samplerate=16000):
    print(f"🔴 Recording for {duration} seconds...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    return audio.flatten()

def run_session():
    vad = webrtcvad.Vad(3)
    q = queue.Queue()
    print("Listening for speech...")

    detect_voice(vad, q)
    beep()
    # Short pause to ensure beep doesn't bleed into recording too much, 
    # though detect_voice already has a pre-beep delay.
    time.sleep(0.2) 
    
    # audio_data = record_until_silence(vad, q)
    audio_data = record_audio()

    print("Processing audio...")
    wav_path = save_audio_to_wav(audio_data)
    prompt_text = transcribe_audio(wav_path)
    
    print(f"Prompt: {prompt_text}")

    if not prompt_text or not prompt_text.strip():
        print("No speech detected. Skipping LLM query.")
        logging.info("[STT] Empty input, skipping.")
        return

    print("Querying Menu...")
    answer = lookup_menu(prompt_text)
    print(f"Response: {answer}")
    tts_speak(answer)
    print("Done. Ready for next query.\n")

if __name__ == "__main__":
    print("VoiceChat assistant running.")
    tts_speak("Voice chat assistant is now running.")
    while True:
        try:
            run_session()
            # restart modules each session to avoid long-run drift
            time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping assistant.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(5)
