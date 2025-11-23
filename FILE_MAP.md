# Project File Map

## Root Directory

| File                  | Description                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `main.py`             | Entry point. Handles Telegram bot, configuration loading, and compute mode selection.            |
| `compiler.py`         | **Multi-Pass AI Engine**. Handles segmentation, transitions, AI enhancement, and merging.        |
| `audio_processing.py` | **Advanced Remix Engine**. Generates beats, shapes spectrum, and syncs effects with transitions. |
| `downloader.py`       | Handles video downloading (yt-dlp) with title-based filenames.                                   |
| `uploader.py`         | Uploads to YouTube (only if enabled).                                                            |
| `requirements.txt`    | Project dependencies.                                                                            |
| `.env`                | Configuration (API keys, `COMPUTE_MODE`, `ENHANCEMENT_LEVEL`, `TRANSITION_SETTINGS`).            |

## Features

- **Multi-Pass Engine**: Robust FFmpeg pipeline (Segment -> Transition -> Merge) for seamless edits.
- **AI Enhancement**:
  - **Real-ESRGAN**: GPU-accelerated upscaling (NCNN Vulkan).
  - **GFPGAN**: Face restoration (optional).
  - **Smart Skip**: Skips enhancement if source > 1440p.
- **GPU Acceleration**:
  - Auto-detects NVIDIA GPUs (`h264_nvenc`).
  - Hybrid mode: GPU for AI, CPU for encoding.
- **Audio Remix**: Granular synthesis, beat accents, and spectral EQ.
- **Configuration**:
  - `COMPUTE_MODE`: `cpu`, `gpu`, `hybrid`, or `auto`.
  - `ENHANCEMENT_LEVEL`: `1x`, `2x`, `3x`, `4x`.
  - `TRANSITION_INTERVAL`: Seconds between transitions.
