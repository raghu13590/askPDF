import os
import tempfile
import soundfile as sf

from app.helper import load_text_to_speech, load_voice_style

# Initialize once at startup
ONNX_DIR = "/models/supertonic/onnx"   # mount or copy the ONNX files here
VOICE_STYLE = ["/models/supertonic/voice_styles/M1.json"]

# Load TTS model and style
text_to_speech = load_text_to_speech(ONNX_DIR, use_gpu=False)
style = load_voice_style(VOICE_STYLE, verbose=False)

class SupertonicTTS:
    def __init__(self):
        self.tts = text_to_speech
        self.style = style
        self.sample_rate = self.tts.sample_rate

    def synthesize(self, text: str):
        wav, duration = self.tts(text, self.style, total_step=5, speed=1.05)
        # Trim to duration
        audio = wav[0, : int(self.sample_rate * duration[0].item())]
        return audio, self.sample_rate

    def synthesize_to_file(self, text: str, out_path: str) -> str:
        audio, sr = self.synthesize(text)
        sf.write(out_path, audio, sr, subtype="PCM_16")
        return out_path

_tts = SupertonicTTS()

def tts_sentence_to_wav(sentence_text: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)
    return _tts.synthesize_to_file(sentence_text, tmp)
