# copyright_checker.py
import os
import subprocess
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("copyright_checker")
logger.setLevel(logging.INFO)

# === Audio Fingerprint ===
from dejavu import Dejavu
from dejavu.recognize import FileRecognizer

# === Chromaprint helper (fpcalc) ===
FP_CALC_BIN = os.getenv("FPCALC_BIN", "fpcalc")  # path to fpcalc binary, if needed

# === Videohash helper ===
from videohash import VideoHash

# === Configuration ===
MATCH_THRESHOLD = float(os.getenv("COPYRIGHT_MATCH_THRESHOLD", "0.85"))

# === Dejavu config ===
DJV_CONFIG = {
    "database": {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "user": os.getenv("MYSQL_USER", "root"),
        "passwd": os.getenv("MYSQL_PASS", ""),
        "db": os.getenv("MYSQL_DB", "dejavu"),
    }
}

# initialize Dejavu
try:
    djv = Dejavu(DJV_CONFIG)
except Exception as e:
    logger.warning("⚠️ Dejavu init failed: %s", e)
    djv = None


# === Audio fingerprint + matching ===
def is_audio_safe(audio_path: str) -> bool:
    """
    Check if audio is copyrighted using Dejavu + local fingerprints.
    Returns True if safe (no match), False if likely copyrighted.
    """
    if not djv:
        logger.warning("⚠️ Dejavu not initialized, skipping audio check.")
        return True

    try:
        result = djv.recognize(FileRecognizer, audio_path)
        # result example: {"song_name": ..., "confidence": 90, ...}
        confidence = result.get("confidence", 0)
        logger.info("🎵 Audio match confidence: %s", confidence)
        return confidence < (MATCH_THRESHOLD * 100)
    except Exception as e:
        logger.warning("⚠️ Audio check failed: %s", e)
        return True


# === Video hash + matching ===
def is_video_safe(video_path: str, known_hashes: Optional[list] = None) -> bool:
    """
    Check video for reused content using Videohash.
    known_hashes: list of previously computed VideoHash objects to compare against.
    Returns True if safe (no match), False if reused content found.
    """
    if not VideoHash:
        logger.warning("⚠️ Videohash not available, skipping video check.")
        return True

    try:
        vh = VideoHash(video_path)
        if not known_hashes:
            return True  # nothing to compare, assume safe

        for kh in known_hashes:
            similarity = vh.similarity(kh)
            logger.info("🎬 Video similarity: %.2f", similarity)
            if similarity >= MATCH_THRESHOLD:
                return False
        return True
    except Exception as e:
        logger.warning("⚠️ Video check failed: %s", e)
        return True


# === Quick combined check ===
def quick_scan(video_path: str, audio_path: Optional[str] = None, known_video_hashes: Optional[list] = None) -> dict:
    """
    Returns a dict with {safe: bool, reason: str}
    """
    if audio_path and not is_audio_safe(audio_path):
        return {"safe": False, "reason": "audio_copyright_detected"}
    if video_path and not is_video_safe(video_path, known_video_hashes):
        return {"safe": False, "reason": "video_reuse_detected"}
    return {"safe": True, "reason": "passed_checks"}
