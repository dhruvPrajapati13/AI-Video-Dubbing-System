# Automated Video Dubbing System

Takes a YouTube URL in any language, downloads it, and produces a new video
dubbed into English — same video, new audio.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# ffmpeg must also be on your PATH (brew install ffmpeg / apt install ffmpeg)
```

GPU (CUDA) is optional but strongly recommended for the 2-hour video —
Whisper transcription and NLLB translation are both much faster on GPU.
On CPU, `--whisper-model small` will be far more practical than `medium`
for long videos.

## Usage

```bash
python main.py "https://www.youtube.com/watch?v=XXXXXXXX"

# Faster/cheaper model for long videos on CPU
python main.py "URL" --whisper-model small

# Stretch goals
python main.py "URL" --diarize --hf-token hf_xxx   # multi-speaker detection
python main.py "URL" --diarize --clone-voice --hf-token hf_xxx  # + voice cloning
```

Output lands in `output/run_<timestamp>/`:
- `dubbed_final.mp4` — the finished video
- `transcript_en.txt` — timestamped English transcript (handy for the walkthrough)
- intermediate files (source video/audio, per-segment TTS clips) kept for debugging

## Pipeline architecture

```
YouTube URL
    |
    v
[downloader.py]   yt-dlp fetches the video; ffmpeg extracts a 16kHz mono WAV
    |
    v
[transcriber.py]  faster-whisper transcribes speech -> timestamped segments
    |              (source language, VAD-filtered to skip silence)
    v
[translator.py]   NLLB-200 translates each segment's text into English,
    |              batched for throughput
    v
[synthesizer.py]  edge-tts renders each English segment as speech, at a rate
    |              estimated to fit the original segment's duration, with a
    |              small corrective ffmpeg atempo pass if still off; segments
    |              are overlaid onto one silent track at their original
    |              timestamps
    v
[remixer.py]      ffmpeg muxes the new audio onto the original video,
                   copying (not re-encoding) the video stream
```

Optional stretch modules:
- `diarizer.py` — pyannote.audio speaker diarization, tags each segment with
  a speaker label so multiple speakers can get distinct voices.
- `voice_cloner.py` — Coqui XTTS clones each speaker's own voice from a
  short reference clip of their original audio, instead of using a stock
  edge-tts voice.

## Key design decisions

- **yt-dlp over pytube**: pytube breaks constantly as YouTube changes its
  internals; yt-dlp is actively maintained and far more reliable.
- **faster-whisper over openai-whisper**: same model weights, several times
  faster and lower memory via CTranslate2 — meaningful on a 2-hour video.
- **NLLB-200 over a literal MT API**: one open-source model covers ~200
  languages including German/French/Hindi, translates for meaning (not
  word-for-word), and runs fully offline once downloaded — no per-request
  cost or rate limits on a 2-hour video's worth of segments. IndicTrans2
  (per the assignment hint) is left as a documented drop-in for
  Indian-language-only workloads, where it outperforms a general model.
- **Segment-level timing over word-level**: translating and synthesizing per
  Whisper segment (rather than per word) keeps sentences grammatical while
  still anchoring each chunk to its original timestamp — a practical middle
  ground between full transcript-level dubbing (loses sync) and word-level
  dubbing (breaks grammar).
- **Rate-adjusted TTS + corrective atempo, not just time-stretching**: edge-tts's
  own `rate` parameter is asked to speak faster/slower first, so the voice
  stays natural; only a small residual mismatch gets fixed with ffmpeg's
  `atempo` filter, to avoid the "chipmunk" artifacts of large pitch-preserving
  stretches.
- **Video stream copied, not re-encoded**: `-c:v copy` in the final mux keeps
  the original video quality intact and makes remuxing the fast part of the
  pipeline, even on a 2-hour file.
- **"Same voice, same energy" as a heuristic, not a hard guarantee**: the
  core assignment picks a fitting stock voice by (rough) gender; true voice
  matching is the `--clone-voice` stretch goal via XTTS, which clones the
  actual speaker's timbre from the source audio.

## Known limitations / what I'd improve next

- Gender/energy detection for stock-voice picking is currently a stub
  (`pick_voice` defaults to one voice); a real implementation would run
  pitch analysis (e.g. via librosa) per speaker segment to choose better.
- Very long silences or music-only stretches aren't specially handled beyond
  Whisper's VAD filtering.
- NLLB translation quality dips on very short, context-free segments (a
  known limitation of segment-by-segment MT); a document-level translation
  pass with segment realignment would improve coherence for a v2.
