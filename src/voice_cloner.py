"""OPTIONAL ENHANCEMENT: voice cloning with Coqui XTTS.

Instead of picking one of a handful of stock edge-tts voices per speaker,
this clones each detected speaker's own voice from a short reference clip of
their original audio, so the English dub sounds like *them*, not a generic
narrator. Only used when main.py is run with --clone-voice.

Requires: pip install TTS  (Coqui TTS; large download, GPU strongly
recommended for reasonable speed on a 2-hour video).
"""

import os
from typing import Dict

from pydub import AudioSegment

from .utils import log
from .transcriber import Segment


def extract_reference_clips(
    audio_path: str, segments: list[Segment], work_dir: str, seconds: float = 8.0
) -> Dict[str, str]:
    """For each speaker, grab a few seconds of their clearest original audio
    to use as the XTTS voice-cloning reference."""
    full_audio = AudioSegment.from_file(audio_path)
    refs: Dict[str, str] = {}

    by_speaker: Dict[str, Segment] = {}
    for seg in segments:
        # Prefer a longer, single segment as a cleaner reference clip.
        if seg.speaker not in by_speaker or (seg.end - seg.start) > (
            by_speaker[seg.speaker].end - by_speaker[seg.speaker].start
        ):
            by_speaker[seg.speaker] = seg

    os.makedirs(work_dir, exist_ok=True)
    for speaker, seg in by_speaker.items():
        start_ms = int(seg.start * 1000)
        end_ms = min(int(seg.end * 1000), start_ms + int(seconds * 1000))
        clip = full_audio[start_ms:end_ms]
        ref_path = os.path.join(work_dir, f"ref_{speaker}.wav")
        clip.export(ref_path, format="wav")
        refs[speaker] = ref_path

    log("clone", f"Extracted {len(refs)} reference voice clip(s)")
    return refs


def synthesize_with_cloned_voice(text_en: str, reference_wav: str, out_path: str) -> str:
    from TTS.api import TTS

    log("clone", "Synthesizing with XTTS (voice-cloned)")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(
        text=text_en,
        speaker_wav=reference_wav,
        language="en",
        file_path=out_path,
    )
    return out_path
