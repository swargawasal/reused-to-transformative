import os
import logging
import random
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import subprocess

logger = logging.getLogger("audio_processing")

# Configuration
TRANSITION_INTERVAL = int(os.getenv("TRANSITION_INTERVAL", "10"))
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

def _generate_sine_wave(freq, duration, sr=44100, vol=0.5):
    t = np.linspace(0, duration, int(sr * duration), False)
    return np.sin(freq * 2 * np.pi * t) * vol

def _create_kick_oneshot(sr=44100):
    duration = 0.3
    t = np.linspace(0, duration, int(sr * duration))
    freq = np.linspace(120, 40, len(t))
    envelope = np.exp(-12 * t)
    wave = np.sin(2 * np.pi * freq * t) * envelope
    return wave * 0.9

def _create_impact_oneshot(sr=44100):
    """Cinematic impact sound."""
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration))
    noise = np.random.uniform(-1, 1, len(t)) * np.exp(-3 * t)
    # Lowpass filter for 'boom'
    sos = signal.butter(2, 100, 'low', fs=sr, output='sos')
    boom = signal.sosfilt(sos, noise)
    return boom * 0.8

def _add_beat_accents(y, sr, interval_sec):
    """Add impact/beat every interval."""
    impact = _create_impact_oneshot(sr)
    y_out = y.copy()
    
    total_samples = y.shape[1]
    interval_samples = int(interval_sec * sr)
    
    # Start from 0 or interval? Let's start from interval to avoid boom at 0:00
    for idx in range(interval_samples, total_samples, interval_samples):
        if idx + len(impact) < total_samples:
            # Mix impact
            y_out[0, idx:idx+len(impact)] += impact * 0.4
            y_out[1, idx:idx+len(impact)] += impact * 0.4
            
    return y_out

def _micro_glitch(y, sr):
    """Randomly repeat small chunks."""
    y_out = y.copy()
    num_glitches = random.randint(2, 5)
    
    for _ in range(num_glitches):
        start = random.randint(0, y.shape[1] - 4000)
        length = random.randint(500, 2000) # 10-40ms
        chunk = y[:, start:start+length]
        
        # Repeat 2-4 times
        repeats = random.randint(2, 4)
        for r in range(repeats):
            pos = start + length * r
            if pos + length < y.shape[1]:
                y_out[:, pos:pos+length] = chunk
                
    return y_out

def process_video_audio(video_path: str) -> str:
    """
    Process audio from video:
    1. Extract
    2. Heavy Remix (Granular + Beats)
    3. Return path to new audio
    """
    logger.info(f"🎧 Processing Audio: {video_path}")
    base, _ = os.path.splitext(video_path)
    temp_wav = f"{base}_temp.wav"
    remixed_wav = f"{base}_remixed.wav"
    
    try:
        # Extract
        cmd = f'{FFMPEG_BIN} -y -i "{video_path}" -vn -acodec pcm_s16le "{temp_wav}" -v error'
        os.system(cmd)
        
        if not os.path.exists(temp_wav):
            logger.error("❌ Audio extraction failed.")
            return None
            
        # Remix
        if heavy_remix(temp_wav, remixed_wav):
            os.remove(temp_wav)
            return remixed_wav
        else:
            return temp_wav # Fallback to original extracted

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return None

def heavy_remix(input_path: str, output_path: str) -> bool:
    """
    Robust Audio Remix:
    1. Convert to WAV via FFmpeg (Safe Load)
    2. Load with Librosa
    3. DSP Chain
    4. Save
    """
    logger.info(f"🎛️ Heavy Remix: {input_path}")
    temp_safe_wav = input_path + ".safe.wav"
    
    try:
        # 1. Safe Convert to WAV (Fixes PySoundFile/MP4 issues)
        cmd = [
            FFMPEG_BIN, "-y", "-i", input_path,
            "-vn", "-ac", "2", "-ar", "44100", "-f", "wav",
            temp_safe_wav
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 2. Load
        y, sr = librosa.load(temp_safe_wav, sr=None, mono=False)
        if y.ndim == 1: y = np.stack([y, y])
        
        # 3. Time Stretch (±3%)
        rate = random.uniform(0.97, 1.03)
        y_stretch = librosa.effects.time_stretch(y, rate=rate)
        
        # 4. Pitch Shift (±0.3 semitones)
        steps = random.uniform(-0.3, 0.3)
        y_shifted = librosa.effects.pitch_shift(y_stretch, sr=sr, n_steps=steps)
        
        # 5. Beat Drops (Synced to Interval)
        y_beats = _add_beat_accents(y_shifted, sr, TRANSITION_INTERVAL)
        
        # 6. Micro Glitches
        y_glitch = _micro_glitch(y_beats, sr)
        
        # 7. Normalize (-1dB)
        max_val = np.max(np.abs(y_glitch))
        if max_val > 0:
            y_final = y_glitch / max_val * 0.89 # approx -1dB
        else:
            y_final = y_glitch
            
        sf.write(output_path, y_final.T, sr)
        logger.info("✅ Remix complete.")
        return True
        
    except Exception as e:
        logger.error(f"Remix failed: {e}")
        return False
    finally:
        if os.path.exists(temp_safe_wav):
            os.remove(temp_safe_wav)
