#!/usr/bin/env python3
"""Automated Video Dubbing System.

Takes a YouTube URL in any language, downloads it, transcribes and
translates the speech into English, synthesizes natural English speech, and
produces a new video with the dubbed audio track.

Usage:
    python main.py "https://www.youtube.com/watch?v=XXXXXXXX"
    python main.py "https://youtu.be/XXXXXXXX" --whisper-model small
    python main.py "https://youtu.be/XXXXXXXX" --diarize --clone-voice

Core pipeline (always runs):
    1. Fetch & Transcribe  -> src/downloader.py, src/transcriber.py
    2. Translate            -> src/translator.py
    3. Synthesize           -> src/synthesizer.py
    4. Remix & Output       -> src/remixer.py

Optional stretch goals (flags below):
    --diarize       multi-speaker detection -> src/diarizer.py
    --clone-voice   per-speaker voice cloning -> src/voice_cloner.py
"""

import argparse
import os
import sys
import time

from src.utils import log, format_hms
from src.downloader import download_video, extract_audio
from src.transcriber import transcribe
from src.translator import translate_segments
from src.synthesizer import build_dubbed_track, pick_voice
from src.remixer import mux_video_with_audio, get_duration_seconds


def parse_args():
    p = argparse.ArgumentParser(description="Dub a YouTube video into English.")
    p.add_argument("url", nargs="?", help="YouTube video URL")
    p.add_argument("--whisper-model", default="medium",
                    help="tiny/base/small/medium/large-v3 (default: medium)")
    p.add_argument("--beam-size", type=int, default=1,
                    help="Whisper decoding beam size. 1=greedy/fast (default), 5=more accurate/slower")
    p.add_argument("--no-batched", action="store_true",
                    help="Disable batched inference (use if it causes issues on your machine)")
    p.add_argument("--out-dir", default="output", help="Where to write intermediate + final files")
    p.add_argument("--diarize", action="store_true",
                    help="[stretch] Detect and separate multiple speakers")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                    help="Hugging Face token, required only for --diarize")
    p.add_argument("--clone-voice", action="store_true",
                    help="[stretch] Clone each speaker's own voice with XTTS instead of stock voices")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.url:
        args.url = input("YouTube URL to dub: ").strip()

    run_dir = os.path.join(args.out_dir, f"run_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()

    log("main", f"Starting dub job for {args.url}")
    log("main", f"Working directory: {run_dir}")

    # 1. Fetch & Transcribe
    video_path = download_video(args.url, run_dir)
    audio_path = extract_audio(video_path, run_dir)
    total_duration = get_duration_seconds(video_path)
    log("main", f"Video duration: {format_hms(total_duration)}")

    segments, source_lang = transcribe(
        audio_path, model_size=args.whisper_model,
        beam_size=args.beam_size, batched=not args.no_batched,
    )
    if not segments:
        log("main", "No speech detected — nothing to dub.")
        sys.exit(1)

    # Optional: multi-speaker detection
    if args.diarize:
        if not args.hf_token:
            log("main", "WARNING: --diarize requires --hf-token (or HF_TOKEN env var). Skipping.")
        else:
            from src.diarizer import diarize_and_assign
            segments = diarize_and_assign(audio_path, segments, args.hf_token)

    # 2. Translate
    segments = translate_segments(segments, source_lang)

    # 3. Synthesize
    def voice_for_segment(seg):
        return pick_voice(seg)  # simple default; refine with real gender detection if desired

    dubbed_audio_path = os.path.join(run_dir, "dubbed_audio.wav")

    if args.clone_voice:
        if not args.diarize:
            log("main", "WARNING: --clone-voice works best with --diarize so each speaker "
                         "gets their own reference clip. Proceeding with a single voice profile.")
        from src.voice_cloner import extract_reference_clips, synthesize_with_cloned_voice
        from pydub import AudioSegment

        refs = extract_reference_clips(audio_path, segments, os.path.join(run_dir, "refs"))
        timeline = AudioSegment.silent(duration=int(total_duration * 1000) + 500)
        for i, seg in enumerate(segments):
            ref = refs.get(seg.speaker, next(iter(refs.values())))
            clip_path = os.path.join(run_dir, f"clone_{i}.wav")
            synthesize_with_cloned_voice(seg.text_en, ref, clip_path)
            clip = AudioSegment.from_file(clip_path)
            timeline = timeline.overlay(clip, position=int(seg.start * 1000))
            log("synthesize", f"...{i + 1}/{len(segments)} cloned segments rendered")
        timeline.export(dubbed_audio_path, format="wav")
    else:
        build_dubbed_track(
            segments, total_duration, os.path.join(run_dir, "tts_parts"),
            dubbed_audio_path, voice_for_segment=voice_for_segment,
        )

    # 4. Remix & Output
    final_path = os.path.join(run_dir, "dubbed_final.mp4")
    mux_video_with_audio(video_path, dubbed_audio_path, final_path)

    elapsed = time.time() - t0
    log("main", f"DONE in {format_hms(elapsed)}. Final video: {final_path}")

    # Write a transcript/translation log alongside the video for the walkthrough.
    transcript_path = os.path.join(run_dir, "transcript_en.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{format_hms(seg.start)} - {format_hms(seg.end)}] ({seg.speaker}) {seg.text_en}\n")
    log("main", f"Transcript written to {transcript_path}")


if __name__ == "__main__":
    main()
