# AI Powered Video Dubbing System

Take any YouTube video — in German, French, Hindi, or dozens of other languages — and get back a fully dubbed English version. Same video, same pacing, new voice.

Built for the IdeaLabs Digital internship assignment.

## Features

- 🌍 **Any source language** — auto-detects the spoken language, no manual config needed
- 🎯 **Timing-aware dubbing** — English audio is paced to match the original speech, not just slapped on
- 🗣️ **Natural voices** — free neural TTS (edge-tts), not robotic text-to-speech
- 🎬 **Lossless video** — original video quality is untouched; only the audio track changes
- 👥 **Multi-speaker support** *(optional)* — speaker diarization + per-speaker voices
- 🎙️ **Voice cloning** *(optional)* — clone each speaker's actual voice with XTTS

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

You'll also need **ffmpeg** on your PATH:
```bash
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Ubuntu/Debian
```

> **GPU recommended.** Whisper transcription and NLLB translation are both much faster with CUDA. On CPU-only machines, use `--whisper-model small` — `medium` will be slow on longer videos.

## Usage

```bash
python main.py "https://www.youtube.com/watch?v=XXXXXXXX"
```

**Common options:**

| Flag | What it does |
|---|---|
| `--whisper-model small` | Faster transcription, small accuracy trade-off (recommended on CPU) |
| `--beam-size 1` | Greedy decoding — fastest option, default |
| `--diarize --hf-token hf_xxx` | Detect multiple speakers, dub each with a distinct voice |
| `--clone-voice --diarize --hf-token hf_xxx` | Clone each speaker's actual voice instead of a stock one |

**Output** lands in `output/run_<timestamp>/`:
- `dubbed_final.mp4` — the finished video
- `transcript_en.txt` — timestamped English transcript
- intermediate files (source audio, per-segment TTS clips) for debugging

## How it works

```
YouTube URL
    │
    ▼
downloader.py    yt-dlp fetches the video; ffmpeg extracts a 16kHz mono WAV
    │
    ▼
transcriber.py   faster-whisper → timestamped segments in the source language
    │
    ▼
translator.py    NLLB-200 translates each segment into English
    │
    ▼
synthesizer.py   edge-tts renders English speech, rate-matched to fit each
    │             segment's original duration
    ▼
remixer.py       ffmpeg swaps in the new audio; video stream is copied,
                  never re-encoded
```

Optional modules, gated behind flags so the base install stays light:
- **`diarizer.py`** — pyannote.audio speaker diarization
- **`voice_cloner.py`** — Coqui XTTS voice cloning

## Design decisions

| Choice | Why |
|---|---|
| yt-dlp over pytube | pytube breaks constantly as YouTube changes internals; yt-dlp is actively maintained |
| faster-whisper over openai-whisper | Same weights, several times faster via CTranslate2 — matters on long videos |
| NLLB-200 over a single-language MT tool | Needed to handle *any* source language, not just Hindi/German/French; translates for meaning, not word-for-word |
| Segment-level timing, not word-level | Keeps sentences grammatical while still anchored to original timestamps |
| Rate-adjusted TTS + capped corrective stretch | Avoids the "chipmunk"/"drugged" artifacts of forcing exact duration matches |
| `-c:v copy` on remux | No quality loss, and remuxing stays fast even on a 2-hour file |

## Known limitations

- Voice picking is gender-heuristic, not true voice matching (see `--clone-voice` for the real thing)
- Long silences/music-only stretches rely on Whisper's built-in VAD, nothing custom
- Segment-by-segment translation can lose some cross-sentence context; a document-level pass would improve coherence
