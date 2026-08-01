"""Step 2: Transcribe — turn the source-language speech into timestamped text.

Uses faster-whisper (a CTranslate2 reimplementation of OpenAI Whisper) rather
than the original openai-whisper package: same accuracy, several times faster
and lower memory, which matters a lot when the assignment explicitly asks us
to process a 2-hour video and report timing.
"""

from dataclasses import dataclass
from typing import List

from .utils import log


@dataclass
class Segment:
    start: float
    end: float
    text: str                # original-language text
    speaker: str = "SPEAKER_00"  # overwritten later if diarization is enabled
    text_en: str = ""        # filled in by the translator step


def transcribe(
    audio_path: str, model_size: str = "medium", beam_size: int = 1, batched: bool = True
) -> (List[Segment], str):
    """Run Whisper over the audio. Returns (segments, detected_language_code).

    model_size: tiny/base/small/medium/large-v3. "medium" is a good default
    accuracy/speed tradeoff for a laptop/GPU; drop to "small" if you need
    the 2-hour video to finish faster and can trade a little accuracy.
    beam_size: 1 = greedy decoding (fast, default here); 5 = Whisper's usual
    beam search (slower, marginally more accurate). On CPU, 1 is a large
    speedup for a small quality cost -- worth it for a first working run.
    batched: use faster-whisper's BatchedInferencePipeline, which processes
    multiple audio chunks in parallel instead of one at a time. Often 2-4x
    faster on CPU. Falls back to the plain pipeline automatically if the
    installed faster-whisper version doesn't support it.
    """
    import os
    from faster_whisper import WhisperModel

    log("transcribe", f"Loading Whisper model '{model_size}'")
    # compute_type="int8" keeps this runnable on CPU; use "float16" on GPU.
    # cpu_threads: use all available cores instead of CTranslate2's
    # conservative default -- meaningfully faster on CPU-only machines
    # (e.g. a Mac mini with no CUDA GPU).
    model = WhisperModel(
        model_size, device="auto", compute_type="auto",
        cpu_threads=os.cpu_count() or 4,
    )

    transcribe_kwargs = dict(
        task="transcribe",       # keep in source language; we translate separately
        vad_filter=True,          # skip silence, improves speed and segment quality
        word_timestamps=False,
        beam_size=beam_size,
    )

    pipeline = model
    if batched:
        try:
            from faster_whisper import BatchedInferencePipeline
            pipeline = BatchedInferencePipeline(model=model)
            transcribe_kwargs["batch_size"] = 16
            log("transcribe", "Using batched inference pipeline")
        except ImportError:
            log("transcribe", "Batched pipeline unavailable in this faster-whisper version, using standard pipeline")

    log("transcribe", "Running speech recognition (this is the slow part)")
    segments_iter, info = pipeline.transcribe(audio_path, **transcribe_kwargs)

    segments: List[Segment] = []
    for seg in segments_iter:
        segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))
        if len(segments) % 20 == 0:
            log("transcribe", f"...{len(segments)} segments so far, at {seg.end:.1f}s")

    log(
        "transcribe",
        f"Done: {len(segments)} segments, detected language = "
        f"{info.language} (p={info.language_probability:.2f})",
    )
    return segments, info.language
