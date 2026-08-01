"""OPTIONAL ENHANCEMENT: speaker diarization.

Tells different speakers apart so each can be dubbed in a distinct voice.
Not required for the core assignment — only imported/used when main.py is
run with --diarize, so the base install doesn't need pyannote.audio.

Requires: pip install pyannote.audio, and a (free) Hugging Face token with
access to pyannote/speaker-diarization-3.1 accepted on huggingface.co.
"""

from typing import List

from .utils import log
from .transcriber import Segment


def diarize_and_assign(audio_path: str, segments: List[Segment], hf_token: str) -> List[Segment]:
    from pyannote.audio import Pipeline

    log("diarize", "Loading pyannote speaker-diarization pipeline")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )

    log("diarize", "Running diarization (identifying who spoke when)")
    diarization = pipeline(audio_path)

    # Build a list of (start, end, speaker_label) turns.
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    def speaker_at(midpoint: float) -> str:
        for start, end, speaker in turns:
            if start <= midpoint <= end:
                return speaker
        return "SPEAKER_00"

    for seg in segments:
        midpoint = (seg.start + seg.end) / 2
        seg.speaker = speaker_at(midpoint)

    n_speakers = len({s.speaker for s in segments})
    log("diarize", f"Identified {n_speakers} distinct speaker(s)")
    return segments
