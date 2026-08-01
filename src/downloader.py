"""Step 1: Fetch — download the source video and pull out a clean audio track.

yt-dlp is used because it's far more resilient to YouTube's changing internals
than pytube/pytube-like libraries (this is the hint given in the assignment,
and matches real-world experience: pytube breaks every few months).
"""

import os
from .utils import run, log


def download_video(url: str, out_dir: str) -> str:
    """Download the best video+audio as a single mp4. Returns the file path."""
    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "source.%(ext)s")

    log("download", f"Fetching {url}")
    run(
        [
            "yt-dlp",
            "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
            "--merge-output-format", "mp4",
            "-o", out_template,
            "--no-playlist",
            url,
        ],
        desc="yt-dlp download",
    )

    video_path = os.path.join(out_dir, "source.mp4")
    if not os.path.exists(video_path):
        # Fall back to whatever yt-dlp actually produced.
        candidates = [f for f in os.listdir(out_dir) if f.startswith("source.")]
        if not candidates:
            raise RuntimeError("yt-dlp reported success but no output file was found.")
        video_path = os.path.join(out_dir, candidates[0])

    log("download", f"Saved video to {video_path}")
    return video_path


def extract_audio(video_path: str, out_dir: str) -> str:
    """Pull a mono 16kHz WAV out of the video — the format Whisper wants."""
    audio_path = os.path.join(out_dir, "source_audio.wav")
    log("download", "Extracting audio track for transcription")
    run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            audio_path,
        ],
        desc="ffmpeg extract audio",
    )
    return audio_path
