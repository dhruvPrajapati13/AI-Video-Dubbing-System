"""Step 4b: Remix & Output — swap in the new audio track and ship the final
video. Video stream is copied (-c:v copy), not re-encoded, so we don't lose
quality or burn extra time re-compressing footage we didn't touch.
"""

import os

from .utils import run, log


def mux_video_with_audio(video_path: str, dubbed_audio_path: str, out_path: str) -> str:
    log("remix", "Muxing dubbed audio onto original video (video stream copied, not re-encoded)")
    run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", dubbed_audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            out_path,
        ],
        desc="ffmpeg mux",
    )
    log("remix", f"Final dubbed video written to {out_path}")
    return out_path


def get_duration_seconds(media_path: str) -> float:
    result = run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", media_path,
        ],
        desc=None,
    )
    return float(result.stdout.strip())
