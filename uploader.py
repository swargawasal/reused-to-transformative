import os
import time
import logging
from typing import Optional
import asyncio

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRET_FILE = os.environ.get("CLIENT_SECRET_FILE", "client_secret.json")
TOKEN_FILE = os.environ.get("YOUTUBE_TOKEN_FILE", "token.json")

logger = logging.getLogger("uploader")
logger.setLevel(logging.INFO)


def _get_service_sync():
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception:
            logger.warning("Failed to read token file, will run auth flow.")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                logger.warning("Refresh failed; performing new auth flow.")
                creds = None
        if not creds:
            if not os.path.exists(CLIENT_SECRET_FILE):
                raise FileNotFoundError(f"Client secrets not found at {CLIENT_SECRET_FILE}")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    service = build("youtube", "v3", credentials=creds)
    return service


def _upload_sync(
    file_path: str,
    hashtags: str = "",
    title: Optional[str] = None,
    description: Optional[str] = None,
    privacy: str = "public",
) -> Optional[str]:
    # Enforce .mp4 extension
    if not file_path.lower().endswith(".mp4"):
        logger.error("❌ Upload rejected: File must be .mp4")
        return None

    service = _get_service_sync()
    final_title = (title or "Untitled").strip()
    final_description = ((description or "").strip() + ("\n\n" + hashtags if hashtags else "")).strip()

    body = {
        "snippet": {
            "title": final_title,
            "description": final_description,
            "categoryId": "22",  # People & Blogs
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        file_path,
        chunksize=1024 * 1024 * 16,  # 16 MB
        resumable=True,
        mimetype="video/mp4"
    )

    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    logger.info("🚀 Starting upload: %s", file_path)
    retry = 0
    while True:
        try:
            status, response = request.next_chunk()
            if response is not None:
                video_id = response.get("id")
                if video_id:
                    logger.info("✅ Upload complete: %s", video_id)
                    return f"https://youtube.com/watch?v={video_id}"
                return None
            if status and hasattr(status, "progress"):
                progress = int(status.progress() * 100)
                logger.info("Upload progress: %d%%", progress)
        except HttpError as e:
            logger.warning("YouTube API HttpError on chunk: %s", e)
            retry += 1
            if retry > 5:
                logger.error("Max retries reached for upload due to HttpError.")
                return None
            time.sleep(2 ** retry)
        except Exception as e:
            logger.exception("Upload error: %s", e)
            retry += 1
            if retry > 5:
                logger.error("Max retries reached for upload due to Exception.")
                return None
            time.sleep(2 ** retry)


async def upload_to_youtube(
    file_path: str,
    hashtags: str = "",
    title: Optional[str] = None,
    description: Optional[str] = None,
    privacy: str = "public",
) -> Optional[str]:
    return await asyncio.to_thread(_upload_sync, file_path, hashtags, title, description, privacy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_file = os.environ.get("TEST_UPLOAD_FILE", "downloads/final_highres_output.mp4")
    # Create a dummy file for testing if it doesn't exist
    if not os.path.exists(test_file):
        with open(test_file, "wb") as f:
            f.write(b"dummy content")
            
    link = _upload_sync(test_file, hashtags="#example", title="High-Quality Test Upload")
    print("Uploaded:", link)
