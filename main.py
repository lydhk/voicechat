import sounddevice as sd #import sounddevice library for record_audio, 'pip install sounddevice'
import tempfile #import tempfile library for save_audio_to_wav
import wave #import wave library for save_audio_to_wav
import whisper #import whisper library for transcribe_audio, 'pip install openai-whisper'

whisper_model = whisper.load_model("base") # Load the Whisper model for transcribe_audio


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





def main():
    print("Calling record audio!")

    audio = record_audio() # Save the recorded audio to a variable
    wav_path = save_audio_to_wav(audio) # Call save_audio_to_wav function to save the audio to a WAV file
    text = transcribe_audio(wav_path) # Call transcribe_audio function to transcribe the audio file
    print(f"Transcription: {text}") # Print the transcribed text
    print("Recording complete!")
    print(f"Audio saved to: {wav_path}") # Print the path to the saved WAV file


if __name__ == "__main__":
    main()
