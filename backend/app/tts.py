import os
import tempfile
import soundfile as sf
import torch
import glob
from kokoro import KPipeline

# Initialization
# We initialize it once at module level
_pipeline = KPipeline(lang_code='a')
VOICES_DIR = "/models/kokoro/voices"

class KokoroTTS:
    def __init__(self):
        self.pipeline = _pipeline
        self.voices_dir = VOICES_DIR
        self.sample_rate = 24000 # Kokoro standard sample rate

    def get_available_voices(self):
        if not os.path.exists(self.voices_dir):
            return []
        # Find all .pt files and return their names
        pt_files = glob.glob(os.path.join(self.voices_dir, "*.pt"))
        return sorted([os.path.basename(f).replace(".pt", "") for f in pt_files])

    def synthesize(self, text: str, voice: str, speed: float = 1.0):
        # KPipeline can take a voice name (if in its own dir) or a path to .pt
        # We'll try to find the .pt file in our voices_dir
        voice_path = os.path.join(self.voices_dir, f"{voice}.pt")
        
        if not os.path.exists(voice_path):
            # Fallback to a name if it's a built-in Kokoro voice
            # or use the first available one if we have any
            available = self.get_available_voices()
            if available:
                print(f"Voice {voice} not found at {voice_path}, falling back to {available[0]}")
                voice_path = os.path.join(self.voices_dir, f"{available[0]}.pt")
            else:
                # Last resort: just use the name and hope KPipeline finds it internally
                voice_path = voice

        # KPipeline returns a generator of (graphemes, phonemes, audio)
        generator = self.pipeline(text, voice=voice_path, speed=speed)
        
        all_audio = []
        for gs, ps, audio in generator:
            if audio is not None:
                all_audio.append(audio)
        
        if not all_audio:
            return torch.zeros(0), self.sample_rate

        full_audio = torch.cat(all_audio) if len(all_audio) > 1 else all_audio[0]
        return full_audio, self.sample_rate

    def synthesize_to_file(self, text: str, out_path: str, voice: str, speed: float = 1.0) -> str:
        audio, sr = self.synthesize(text, voice, speed)
        if torch.is_tensor(audio):
            audio_np = audio.numpy()
        else:
            audio_np = audio
            
        sf.write(out_path, audio_np, sr)
        return out_path

_tts = KokoroTTS()

def tts_sentence_to_wav(sentence_text: str, out_dir: str, voice_style: str, speed: float = 1.0) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)
    
    # If the call site sends a .json name (legacy), we'll try to get the first one
    if voice_style.endswith(".json"):
        available = _tts.get_available_voices()
        voice_style = available[0] if available else "af_heart"
        
    return _tts.synthesize_to_file(sentence_text, tmp, voice_style, speed)

def list_voice_styles():
    return _tts.get_available_voices()
