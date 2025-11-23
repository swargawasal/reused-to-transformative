import os
import logging
import yt_dlp
from datetime import datetime
import re

logger = logging.getLogger("downloader")

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def _sanitize_filename(name: str) -> str:
    """Sanitize filename."""
    clean = re.sub(r'[^\w\s-]', '', name)
    return clean.replace(' ', '_')

def download_video(url: str, custom_title: str = None) -> str:
    """
    Download video from URL synchronously.
    Strategy:
    1. Try without cookies.
    Returns the absolute path to the downloaded file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if custom_title:
        clean_title = _sanitize_filename(custom_title)
        filename = f"{clean_title}__{timestamp}.mp4"
    else:
        filename = f"video__{timestamp}.mp4"
        
    output_path = os.path.join(DOWNLOAD_DIR, filename)
    absolute_path = os.path.abspath(output_path)
    
    # Base options
    ydl_opts = {
        'outtmpl': absolute_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    # Attempt 1: No Cookies
    try:
        logger.info(f"⬇️ Downloading (Attempt 1 - No Cookies): {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if os.path.exists(absolute_path):
            logger.info(f"✅ Download complete: {absolute_path}")
            return absolute_path
    except Exception as e:
        logger.warning(f"⚠️ Attempt 1 failed: {e}")

    # Attempt 2: With Cookies (Only if provided)
    cookies_path = os.getenv("COOKIES_FILE", "").strip('"').strip("'")
    
    if cookies_path and os.path.exists(cookies_path):
        logger.info(f"🔄 Retrying with cookies from file: {cookies_path}")
        ydl_opts['cookiefile'] = cookies_path
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            if os.path.exists(absolute_path):
                logger.info(f"✅ Download complete (with cookies): {absolute_path}")
                return absolute_path
            else:
                logger.error("❌ File not found after retry.")
                return None
        except Exception as e:
            logger.error(f"❌ Download error (Attempt 2): {e}")
            return None
    else:
        logger.warning("⚠️ Download failed and no cookies file provided (COOKIES_FILE). Skipping retry.")
        return None  