
"""
Text-to-Speech (TTS) module using Kokoro pipeline.
Provides voice synthesis and voice style management utilities.
"""

import os
import glob
import tempfile

import soundfile as sf
import torch
from kokoro import KPipeline


# Initialize the Kokoro pipeline once at module level
_pipeline = KPipeline(lang_code='a')
VOICES_DIR = "/models/kokoro/voices"

class KokoroTTS:
    """
    KokoroTTS provides text-to-speech synthesis using the Kokoro pipeline.
    """
    def __init__(self):
        """Initialize the TTS pipeline and set up voice directory and sample rate."""
        self.pipeline = _pipeline
        self.voices_dir = VOICES_DIR
        self.sample_rate = 24000  # Kokoro standard sample rate

    def get_available_voices(self) -> list[str]:
        """Return a sorted list of available voice names (without .pt extension)."""
        if not os.path.exists(self.voices_dir):
            return []
        pt_files = glob.glob(os.path.join(self.voices_dir, "*.pt"))
        return sorted([os.path.basename(f).replace(".pt", "") for f in pt_files])

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[torch.Tensor, int]:
        """
        Synthesize speech from text using the specified voice and speed.
        Returns a tuple of (audio tensor, sample rate).
        """
        voice_path = os.path.join(self.voices_dir, f"{voice}.pt")
        if not os.path.exists(voice_path):
            available = self.get_available_voices()
            if available:
                print(f"Voice {voice} not found at {voice_path}, falling back to {available[0]}")
                voice_path = os.path.join(self.voices_dir, f"{available[0]}.pt")
            else:
                raise FileNotFoundError(f"No voices found in {self.voices_dir} and voice '{voice}' is not a valid built-in voice.")

        # KPipeline returns a generator of (graphemes, phonemes, audio)
        generator = self.pipeline(text, voice=voice_path, speed=speed)
        all_audio = [audio for _, _, audio in generator if audio is not None]
        if not all_audio:
            return torch.zeros(0), self.sample_rate
        full_audio = torch.cat(all_audio) if len(all_audio) > 1 else all_audio[0]
        return full_audio, self.sample_rate

    def synthesize_to_file(self, text: str, out_path: str, voice: str, speed: float = 1.0) -> str:
        """
        Synthesize speech and write the result to a WAV file.
        Returns the output file path.
        """
        audio, sr = self.synthesize(text, voice, speed)
        audio_np = audio.numpy() if torch.is_tensor(audio) else audio
        sf.write(out_path, audio_np, sr)
        return out_path
        sf.write(out_path, audio_np, sr)
        return out_path


# Module-level TTS instance
_tts = KokoroTTS()
_available_voices = _tts.get_available_voices()
print(f"Kokoro discovered voices: {_available_voices}")


def tts_sentence_to_wav(
    sentence_text: str,
    out_dir: str,
    voice_style: str = None,
    speed: float = 1.0
) -> str:
    """
    Synthesize a sentence to a temporary WAV file in the given directory.
    Returns the path to the generated WAV file.
    """
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)

    # Handle None or empty voice_style
    available = _tts.get_available_voices()
    if not voice_style or (voice_style and voice_style.endswith(".json")):
        if not available:
            raise ValueError("No voices available and no voice_style provided.")
        voice_style = available[0]

    return _tts.synthesize_to_file(sentence_text, tmp, voice_style, speed)


def list_voice_styles() -> list[str]:
    """Return a list of available voice styles."""
    return _tts.get_available_voices()
