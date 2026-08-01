import asyncio
import os
from typing import List

from pydub import AudioSegment

from .utils import log, run
from .transcriber import Segment

# A small set of Edge neural voices spanning gender/energy, used as the
# "same voice, same energy" approximation (per-speaker matching, not full
# cloning — see voice_cloner.py for the stretch-goal alternative).
VOICE_MALE = "en-US-GuyNeural"
VOICE_FEMALE = "en-US-AriaNeural"
VOICE_MALE_ENERGETIC = "en-US-DavisNeural"
VOICE_FEMALE_ENERGETIC = "en-US-JennyNeural"


def pick_voice(segment: Segment, speaker_gender: str = "unknown", energetic: bool = False) -> str:
    if speaker_gender == "male":
        return VOICE_MALE_ENERGETIC if energetic else VOICE_MALE
    if speaker_gender == "female":
        return VOICE_FEMALE_ENERGETIC if energetic else VOICE_FEMALE
    # Default when we haven't run gender detection.
    return VOICE_FEMALE


async def _edge_tts_to_file(text: str, voice: str, rate: str, out_path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(out_path)


def _rate_string(speedup: float) -> str:
    """edge-tts wants a signed percentage string like '+15%' or '-10%'.

    Clamped to a range that still sounds like natural speech: we allow
    speeding up more than slowing down, because a dub that talks a bit fast
    to keep pace reads as "energetic"; a dub slowed down to fill a gap reads
    as "drugged" and is far more noticeable/distracting to a listener.
    """
    pct = round((speedup - 1.0) * 100)
    pct = max(-15, min(pct, 50))  # slow down at most 15%, speed up at most 50%
    return f"{pct:+d}%"


def synthesize_segment(seg: Segment, work_dir: str, voice: str) -> str:
    """Generate English speech for one segment, adjusted to roughly fit its
    original time slot. Returns the path to the rendered WAV clip.
    """
    target_duration = max(seg.end - seg.start, 0.3)
    raw_path = os.path.join(work_dir, f"tts_raw_{seg.start:.2f}.mp3")
    fitted_path = os.path.join(work_dir, f"tts_fit_{seg.start:.2f}.wav")

    # First pass at normal rate, to measure how long the text naturally takes.
    asyncio.run(_edge_tts_to_file(seg.text_en, voice, rate="+0%", out_path=raw_path))
    natural = AudioSegment.from_file(raw_path)
    natural_duration = len(natural) / 1000.0

    if natural_duration > 0:
        needed_speedup = natural_duration / target_duration
    else:
        needed_speedup = 1.0

    # Only regenerate if we need to meaningfully speed up (translation is
    # longer than its slot) or apply the small allowed slowdown. If the
    # translation is much shorter than the slot, leave it at natural pace
    # and just let it finish early -- do not slow it down to compensate.
    if needed_speedup > 1.03 or (0.85 < needed_speedup < 0.97):
        asyncio.run(
            _edge_tts_to_file(seg.text_en, voice, rate=_rate_string(needed_speedup), out_path=raw_path)
        )
        natural = AudioSegment.from_file(raw_path)
        natural_duration = len(natural) / 1000.0

    # Corrective micro-stretch with ffmpeg's atempo, but only to catch up
    # when we're still running long (residual > 1.05) -- never to slow
    # things down further. A clip finishing early is fine; a clip that
    # still overruns its slot risks talking over the next segment.
    residual = natural_duration / target_duration if target_duration > 0 else 1.0
    natural.export(fitted_path, format="wav")
    if residual > 1.05:
        atempo = min(residual, 1.6)  # cap how much speed-up we'll force
        stretched_path = fitted_path.replace(".wav", "_stretched.wav")
        run(
            ["ffmpeg", "-y", "-i", fitted_path, "-filter:a", f"atempo={atempo}", stretched_path],
            desc=None,
        )
        fitted_path = stretched_path

    return fitted_path


def build_dubbed_track(
    segments: List[Segment], total_duration: float, work_dir: str, out_path: str,
    voice_for_segment=None,
) -> str:
    """Lay every synthesized segment onto a single silent track at the right
    offset, producing one continuous English audio file the length of the
    original video."""
    os.makedirs(work_dir, exist_ok=True)
    log("synthesize", f"Synthesizing {len(segments)} English segments")

    timeline = AudioSegment.silent(duration=int(total_duration * 1000) + 500)

    for i, seg in enumerate(segments):
        voice = voice_for_segment(seg) if voice_for_segment else pick_voice(seg)
        clip_path = synthesize_segment(seg, work_dir, voice)
        clip = AudioSegment.from_file(clip_path)
        offset_ms = int(seg.start * 1000)
        timeline = timeline.overlay(clip, position=offset_ms)

        if (i + 1) % 10 == 0 or i == len(segments) - 1:
            log("synthesize", f"...{i + 1}/{len(segments)} segments rendered")

    timeline.export(out_path, format="wav")
    log("synthesize", f"Dubbed audio track written to {out_path}")
    return out_path
