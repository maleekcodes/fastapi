from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import mimetypes
import secrets
import subprocess
import time
from urllib.parse import urlparse
import fastapi
import firebase_admin
import requests
from firebase_admin import auth, credentials, firestore, storage
from fastapi import Depends, Header
from google.cloud.firestore_v1.base_query import FieldFilter
import whisper
import asyncio
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
import httpx
from rapidfuzz import fuzz
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

# Load environment variables from .env file
load_dotenv()

app = fastapi.FastAPI()
api = fastapi.FastAPI()

cred = credentials.Certificate("service_account.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

class TranscribeInternalPayload(BaseModel):
    url: str
    user_id: str
    duration: int
    media_type: str = "video"
    fileName: str = None

class TranscribePayload(BaseModel):
    url: str
    user_id: str

class UserCreditsPayload(BaseModel):
    credits: int

class UserInfoResponse(BaseModel):
    credits: int
    email: str
    freeTrial: bool
    fullName: str
    lastLoginAt: str
    provider: str

class UpdateUserInfoPayload(BaseModel):
    credits: Optional[int] = None
    email: Optional[str] = None
    fullName: Optional[str] = None

class TranscriptionHistoryItem(BaseModel):
    id: str
    createdAt: datetime
    durationMinutes: float
    fileName: str
    firebaseUrl: str
    originalUrl: str
    platform: str
    transcription: dict
    userId: str
    mediaType: Optional[str] = None
    thumbnailUrl: Optional[str] = None
    processingDurationSeconds: Optional[float] = None
    title: Optional[str] = None

class TranscriptionHistoryResponse(BaseModel):
    items: list[TranscriptionHistoryItem]
    total: int
    limit: int
    offset: int

class ShareLinkResponse(BaseModel):
    shareToken: str
    shareUrl: str
    videoId: str
    createdAt: datetime
    expiresAt: Optional[datetime] = None

class PublicShareResponse(BaseModel):
    id: str
    title: Optional[str] = None
    fileName: str
    mediaType: Optional[str] = None
    mediaUrl: str
    thumbnailUrl: Optional[str] = None
    durationMinutes: float
    platform: str
    transcription: dict
    createdAt: datetime

def get_current_user_id(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise fastapi.HTTPException(status_code=401, detail="Missing auth token")
    token = authorization.removeprefix("Bearer ").strip()
    if not token or token == "null":
        raise fastapi.HTTPException(status_code=401, detail="Missing auth token")
    try:
        return auth.verify_id_token(token)["uid"]
    except Exception:
        raise fastapi.HTTPException(status_code=401, detail="Invalid auth token")

SAMPLE_TRANSCRIPTION_ID = "sample"
SAMPLE_SOURCE_TYPE = "sample"

def _get_share_content_doc(share: dict):
    if share.get("sourceType") == SAMPLE_SOURCE_TYPE:
        return db.collection("sample_transcription").document(SAMPLE_TRANSCRIPTION_ID).get()
    return db.collection("videos").document(share["videoId"]).get()

def _build_public_share_response(content_doc) -> PublicShareResponse:
    content = content_doc.to_dict()
    media_url = content.get("firebaseUrl") or content.get("originalUrl") or ""
    return PublicShareResponse(
        id=content_doc.id,
        title=content.get("title"),
        fileName=content.get("fileName", "Sample transcription"),
        mediaType=content.get("mediaType"),
        mediaUrl=media_url,
        thumbnailUrl=content.get("thumbnailUrl"),
        durationMinutes=content.get("durationMinutes", 0),
        platform=content.get("platform", "sample"),
        transcription=content.get("transcription", {}),
        createdAt=content.get("createdAt", datetime.now()),
    )

def _create_or_reuse_share(
    *,
    video_id: str,
    user_id: str,
    source_type: str = "video",
) -> ShareLinkResponse:
    existing_query = (
        db.collection("shares")
        .where(filter=FieldFilter("videoId", "==", video_id))
        .where(filter=FieldFilter("isActive", "==", True))
    )
    if source_type != "video":
        existing_query = existing_query.where(
            filter=FieldFilter("sourceType", "==", source_type)
        )

    existing = existing_query.limit(1).get()
    share_base = os.environ.get("SHARE_BASE_URL", "https://share.transcryb.app").rstrip("/")

    if existing:
        share_doc = existing[0]
        share_data = share_doc.to_dict()
        token = share_doc.id
        created_at = share_data.get("createdAt", datetime.now())
        expires_at = share_data.get("expiresAt")
    else:
        token = secrets.token_urlsafe(32)
        created_at = datetime.now()
        expires_at = None
        share_payload = {
            "token": token,
            "videoId": video_id,
            "userId": user_id,
            "createdAt": created_at,
            "isActive": True,
        }
        if source_type != "video":
            share_payload["sourceType"] = source_type
        db.collection("shares").document(token).set(share_payload)

    return ShareLinkResponse(
        shareToken=token,
        shareUrl=f"{share_base}/t/{token}",
        videoId=video_id,
        createdAt=created_at,
        expiresAt=expires_at,
    )

COBALT_API_URL = os.environ.get(
    "COBALT_API_URL",
    "https://cobalt-production-7dd4.up.railway.app/",
).rstrip("/") + "/"

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

RAPIDAPI_HOST = os.environ.get(
    "RAPIDAPI_HOST",
    "social-download-all-in-one.p.rapidapi.com",
)

SOURCES_TO_INDEX = {
    "tiktok": 1,
    "instagram": 0,
    "youtube": 0,
    "facebook": 0,
    "twitter": 0,
    "x": 0,
}

jobs = {}

def create_job(job_id: str, thread: Thread):
    jobs[job_id] = {
        "thread": thread,
        "status": "pending"
    }

def get_job(job_id: str):
    """Get job from in-memory dict and Firestore. Returns (in_memory_job, firestore_job)"""
    in_memory_job = jobs.get(job_id)  # Returns None if not found instead of raising KeyError
    firestore_doc = db.collection("jobs").document(job_id).get()
    firestore_job = firestore_doc.to_dict() if firestore_doc.exists else None
    return in_memory_job, firestore_job

def update_job(job_id: str, status: str):
    # Update in-memory status for WebSocket
    if job_id in jobs:
        jobs[job_id]["status"] = status
    # Update Firestore
    db.collection("jobs").document(job_id).update({
        "status": status,
        "updatedAt": datetime.now()
    })

def send_transcription_ready_notification(user_id: str, content_title: Optional[str] = None):
    """Send Expo push notification when transcription completes."""
    try:
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists:
            print(f"Notification skipped: user {user_id} not found")
            return

        user_data = user_doc.to_dict()
        if not user_data.get("pushNotificationsEnabled"):
            print(f"Notification skipped: push notifications disabled for user {user_id}")
            return

        push_token = user_data.get("expoPushToken")
        if not push_token:
            print(f"Notification skipped: no expo push token for user {user_id}")
            return

        if content_title:
            body = f"Your transcription {content_title} is ready. Tap to open."
        else:
            body = "Your transcription is ready. Tap to open."

        response = requests.post(
            "https://exp.host/--/api/v2/push/send",
            json={
                "to": push_token,
                "title": "Transcryb",
                "body": body,
            },
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        response.raise_for_status()
        print(f"Sent transcription ready notification to user {user_id}")
    except Exception as e:
        print(f"Failed to send notification to user {user_id}: {e}")

def delete_job(job_id: str):
    del jobs[job_id]

def detect_platform(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")

    if "tiktok.com" in hostname:
        return "tiktok"
    if "instagram.com" in hostname:
        return "instagram"
    if "youtube.com" in hostname or hostname == "youtu.be":
        return "youtube"
    if "facebook.com" in hostname or hostname == "fb.watch":
        return "facebook"
    if "twitter.com" in hostname or hostname == "x.com":
        return "x"
    return "unknown"

def get_video_details(url: str) -> dict:
    api_key = os.environ.get("RAPIDAPI_KEY")
    if not api_key:
        raise fastapi.HTTPException(
            status_code=503,
            detail="RAPIDAPI_KEY is not configured (required for YouTube)",
        )

    api_url = f"https://{RAPIDAPI_HOST}/v1/social/autolink"
    response = requests.post(
        api_url,
        json={"url": url},
        headers={
            "x-rapidapi-host": RAPIDAPI_HOST,
            "Content-Type": "application/json",
            "x-rapidapi-key": api_key,
        },
        timeout=60,
    )

    try:
        result = response.json()
    except Exception:
        raise fastapi.HTTPException(
            status_code=502,
            detail=f"RapidAPI returned invalid response (HTTP {response.status_code})",
        )

    if not response.ok:
        raise fastapi.HTTPException(
            status_code=502,
            detail=f"RapidAPI request failed (HTTP {response.status_code})",
        )

    print(f"Video details response: {result}")
    return result

def extract_thumbnail_from_details(video_details: dict) -> Optional[str]:
    thumbnail_fields = ["thumbnail", "thumbnailUrl", "thumb", "picture", "cover", "image", "poster"]

    for field in thumbnail_fields:
        if field in video_details and video_details[field]:
            print(f"Found thumbnail in field '{field}': {video_details[field]}")
            return video_details[field]

    if video_details.get("medias"):
        for media in video_details["medias"]:
            if isinstance(media, dict):
                for field in thumbnail_fields:
                    if field in media and media[field]:
                        print(f"Found thumbnail in medias[].{field}: {media[field]}")
                        return media[field]

    if video_details.get("pictureUrl"):
        print(f"Found thumbnail in 'pictureUrl': {video_details['pictureUrl']}")
        return video_details["pictureUrl"]

    print(f"No thumbnail found in video details. Available keys: {list(video_details.keys())}")
    return None

def extract_title_from_details(video_details: dict, source: str = None) -> Optional[str]:
    useless_titles = [
        "- Facebook Reel",
        "Facebook Reel",
        "Instagram Reel",
        "- Instagram Reel",
        "Reel",
        "Video",
        "TikTok Video",
    ]

    if video_details.get("title"):
        title = video_details["title"].strip()
        if title in useless_titles or title.startswith("- ") and len(title) < 20:
            print(f"Ignoring useless title: {title}")
            return None
        print(f"Found title: {title[:50]}...")
        return title
    return None

def extract_video_url_from_details(video_details: dict, source: str) -> Optional[str]:
    if video_details.get("medias"):
        index = SOURCES_TO_INDEX.get(source, 0)
        if len(video_details["medias"]) > index:
            media = video_details["medias"][index]
            if isinstance(media, dict) and "url" in media:
                return media["url"]
            if isinstance(media, str):
                return media

    if "url" in video_details:
        return video_details["url"]
    if "videoUrl" in video_details:
        return video_details["videoUrl"]
    if "download" in video_details:
        return video_details["download"]

    print(f"Could not extract video URL. Available keys: {list(video_details.keys())}")
    return None

def _cobalt_headers() -> dict:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    api_key = os.environ.get("COBALT_API_KEY")
    if api_key:
        headers["Authorization"] = f"Api-Key {api_key}"
    bearer = os.environ.get("COBALT_BEARER_TOKEN")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers

def probe_media_duration(url: str) -> int:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        return max(1, int(duration))
    except Exception as e:
        print(f"ffprobe failed for {url[:80]}...: {e}")
        raise fastapi.HTTPException(
            status_code=400,
            detail="Could not determine video duration",
        )

def resolve_video_via_cobalt(url: str, probe_duration: bool = True) -> dict:
    response = requests.post(
        COBALT_API_URL,
        json={
            "url": url,
            "videoQuality": "1080",
            "downloadMode": "auto",
        },
        headers=_cobalt_headers(),
        timeout=60,
    )

    try:
        result = response.json()
    except Exception:
        raise fastapi.HTTPException(
            status_code=502,
            detail=f"Cobalt returned invalid response (HTTP {response.status_code})",
        )

    status = result.get("status")
    print(f"Cobalt response status={status}, filename={result.get('filename')}")

    source = detect_platform(url)
    video_url = None
    thumbnail_url = None
    filename = result.get("filename")

    if status in ("redirect", "tunnel"):
        video_url = result.get("url")
        if not video_url:
            raise fastapi.HTTPException(status_code=400, detail="Cobalt did not return a download URL")
    elif status == "picker":
        picker = result.get("picker") or []
        video_item = next((item for item in picker if item.get("type") == "video"), None)
        if not video_item:
            raise fastapi.HTTPException(status_code=400, detail="No video found in multi-media post")
        video_url = video_item.get("url")
        thumbnail_url = video_item.get("thumb")
        if not video_url:
            raise fastapi.HTTPException(status_code=400, detail="Cobalt picker did not return a video URL")
    elif status == "error":
        error = result.get("error") or {}
        code = error.get("code", "unknown_error")
        raise fastapi.HTTPException(status_code=400, detail=f"Could not download video: {code}")
    elif status == "local-processing":
        raise fastapi.HTTPException(
            status_code=400,
            detail="Unsupported media requires local processing",
        )
    else:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"Unexpected Cobalt status: {status}",
        )

    # The V3 (Deepgram) path skips this probe: Deepgram reports the true
    # duration, and ffprobe cannot read Cobalt's non-seekable tunnel URLs.
    duration = probe_media_duration(video_url) if probe_duration else None

    return {
        "video_url": video_url,
        "source": source,
        "duration": duration,
        "thumbnail_url": thumbnail_url,
        "title": None,
        "filename": filename,
    }

def resolve_video_via_rapidapi(url: str) -> dict:
    video_details = get_video_details(url)
    source = video_details.get("source") or detect_platform(url)
    video_url = extract_video_url_from_details(video_details, source)
    if not video_url:
        raise fastapi.HTTPException(status_code=400, detail="Could not extract video URL from source")

    thumbnail_url = extract_thumbnail_from_details(video_details)
    title = extract_title_from_details(video_details, source)
    duration = int(video_details.get("duration") or 0)

    if duration <= 0:
        try:
            duration = probe_media_duration(video_url)
        except fastapi.HTTPException:
            duration = None

    return {
        "video_url": video_url,
        "source": source,
        "duration": duration,
        "thumbnail_url": thumbnail_url,
        "title": title,
        "filename": None,
    }

def resolve_video(url: str, probe_duration: bool = True) -> dict:
    platform = detect_platform(url)
    if platform == "youtube":
        return resolve_video_via_rapidapi(url)
    return resolve_video_via_cobalt(url, probe_duration=probe_duration)

def generate_title_from_transcription(full_text: str, max_length: int = 100) -> str:
    """Generate a title from the beginning of transcription text"""
    if not full_text:
        return None
    # Clean up and truncate
    title = full_text.strip()[:max_length]
    # Try to cut at a word boundary
    if len(full_text) > max_length:
        last_space = title.rfind(' ')
        if last_space > 50:  # Only cut at space if we have reasonable length
            title = title[:last_space]
        title += "..."
    return title

def serialize_firestore_data(data: dict) -> dict:
    """
    Recursively convert Firestore DatetimeWithNanoseconds objects to ISO strings
    so the data can be JSON serialized.
    """
    if data is None:
        return None
    
    result = {}
    for key, value in data.items():
        if hasattr(value, 'isoformat'):  # datetime-like object
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = serialize_firestore_data(value)
        elif isinstance(value, list):
            result[key] = [
                serialize_firestore_data(item) if isinstance(item, dict) 
                else (item.isoformat() if hasattr(item, 'isoformat') else item)
                for item in value
            ]
        else:
            result[key] = value
    return result

def format_duration_minutes_seconds(duration_seconds: int) -> float:
    """
    Convert duration in seconds to minutes.seconds format.
    E.g., 58 seconds -> 0.58, 150 seconds (2m30s) -> 2.30, 345 seconds (5m45s) -> 5.45
    """
    minutes = duration_seconds // 60
    seconds = duration_seconds % 60
    return round(minutes + (seconds / 100), 2)

def fuzzy_search_item(item: dict, search_term: str, threshold: int = 85) -> tuple[bool, int]:
    """
    Perform fuzzy search on an item's searchable fields.
    Returns (is_match, best_score) tuple.
    
    Searches in:
    - title
    - fileName  
    - originalUrl
    - transcription.full_text
    
    Args:
        item: The item dict to search
        search_term: The search query
        threshold: Minimum score (0-100) to consider a match. Default 85 (strict).
    
    Returns:
        (is_match, best_score): Whether item matches and the highest score found
    """
    search_lower = search_term.lower().strip()
    
    if not search_lower:
        return (False, 0)
    
    # Fields to search
    searchable_fields = [
        item.get("title", ""),
        item.get("fileName", ""),
        item.get("originalUrl", ""),
    ]
    
    # Get transcription full_text if available
    transcription = item.get("transcription", {})
    if isinstance(transcription, dict):
        full_text = transcription.get("full_text", "")
        if full_text:
            searchable_fields.append(full_text)
    
    for field_value in searchable_fields:
        if not field_value:
            continue
        
        field_lower = str(field_value).lower()
        
        # Check for exact substring match (most relevant)
        if search_lower in field_lower:
            return (True, 100)
    
    # If no exact match, try fuzzy matching with high threshold
    best_score = 0
    for field_value in searchable_fields:
        if not field_value:
            continue
        
        field_lower = str(field_value).lower()
        
        # For short fields (title, fileName, url), use partial ratio
        if len(field_lower) < 500:
            score = fuzz.partial_ratio(search_lower, field_lower)
            best_score = max(best_score, score)
        else:
            # For long text (transcription), check if any word matches closely
            # Split into chunks and check each
            words = field_lower.split()
            for i in range(len(words)):
                # Check phrase of up to 10 words
                phrase = ' '.join(words[i:i+10])
                score = fuzz.partial_ratio(search_lower, phrase)
                if score >= threshold:
                    return (True, score)
    
    return (best_score >= threshold, best_score)

def get_raw_video(url: str, timeout: int = 600) -> bytes:
    """Download video bytes with a timeout. Streams in 1 MB chunks."""
    last_error = None
    for attempt in range(1, 4):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            chunks = []
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    chunks.append(chunk)
            return b"".join(chunks)
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            last_error = e
            print(f"Video download attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise last_error

def is_cobalt_tunnel_url(url: str) -> bool:
    """True for Cobalt /tunnel proxy URLs that Deepgram cannot fetch directly."""
    parsed = urlparse(url)
    cobalt_host = urlparse(COBALT_API_URL).netloc
    return parsed.netloc == cobalt_host and parsed.path.startswith("/tunnel")

def _deepgram_from_url_or_bytes(video_url: str, raw_video: bytes = None):
    """Transcribe via Deepgram URL ingestion when possible, else binary upload."""
    content_type = guess_content_type(video_url)

    try:
        return call_deepgram_transcribe_with_retry(video_url, is_url=True)
    except Exception as url_error:
        fallback_bytes = raw_video
        if fallback_bytes is None:
            print(f"Deepgram URL ingestion failed, downloading for binary fallback: {url_error}")
            fallback_bytes = get_raw_video(video_url)
        if not fallback_bytes:
            raise Exception(
                f"Deepgram URL ingestion failed and no video bytes available: {url_error}"
            ) from url_error
        print(f"Deepgram URL ingestion failed, falling back to binary upload: {url_error}")
        return call_deepgram_transcribe_with_retry(
            fallback_bytes,
            is_url=False,
            content_type=content_type,
        )

def upload_video(raw_video: bytes, job_id: str):
    bucket = storage.bucket("transcrybe-fe4cb.appspot.com")
    blob = bucket.blob("videos/" + job_id)
    # Increase timeout to 600s (10 minutes) for large uploads
    blob.upload_from_string(raw_video, content_type="video/mp4", timeout=600)
    blob.make_public()
    return blob.public_url

def download_thumbnail(url: str) -> bytes:
    """Download thumbnail from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading thumbnail: {e}")
        return None

def upload_thumbnail(thumbnail_data: bytes, job_id: str) -> str:
    """Upload thumbnail to Firebase Storage and return public URL"""
    try:
        bucket = storage.bucket("transcrybe-fe4cb.appspot.com")
        blob = bucket.blob(f"thumbnails/{job_id}.jpg")
        blob.upload_from_string(thumbnail_data, content_type="image/jpeg", timeout=60)
        blob.make_public()
        print(f"Thumbnail uploaded: {blob.public_url}")
        return blob.public_url
    except Exception as e:
        print(f"Error uploading thumbnail: {e}")
        return None

def process_and_upload_thumbnail(thumbnail_url: str, job_id: str) -> str:
    """Download thumbnail from source URL and upload to Firebase Storage"""
    if not thumbnail_url:
        return None
    
    thumbnail_data = download_thumbnail(thumbnail_url)
    if not thumbnail_data:
        return None
    
    return upload_thumbnail(thumbnail_data, job_id)

def _ffmpeg_extract_thumbnail(input_path: str, thumb_path: str) -> bool:
    for seek in ("1", "0"):
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", seek,
                "-i", input_path,
                "-vframes", "1",
                "-q:v", "2",
                "-vf", "scale=640:-1",
                thumb_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
            return True
        print(f"ffmpeg thumbnail seek={seek}s failed: {result.stderr[-500:]}")
    return False

def generate_thumbnail_from_video(video: bytes | str, job_id: str) -> Optional[str]:
    """Extract a poster frame with ffmpeg and upload to Firebase."""
    video_path = f"/tmp/{job_id}_thumb_src.mp4"
    thumb_path = f"/tmp/{job_id}_thumb.jpg"
    temp_video_path = None

    try:
        if isinstance(video, bytes):
            with open(video_path, "wb") as f:
                f.write(video)
            input_path = video_path
            temp_video_path = video_path
        else:
            input_path = video

        if not _ffmpeg_extract_thumbnail(input_path, thumb_path):
            return None

        with open(thumb_path, "rb") as f:
            thumbnail_data = f.read()
        uploaded = upload_thumbnail(thumbnail_data, job_id)
        if uploaded:
            print(f"Generated thumbnail from video for job {job_id}")
        return uploaded
    except Exception as e:
        print(f"Error generating thumbnail: {e}")
        return None
    finally:
        for path in (temp_video_path, thumb_path):
            if path and os.path.exists(path):
                os.remove(path)

def ensure_video_thumbnail(
    thumbnail_url: Optional[str],
    job_id: str,
    *,
    raw_video: Optional[bytes] = None,
    video_url: Optional[str] = None,
) -> Optional[str]:
    """Use Cobalt/picker thumbnail URL when available, else extract a frame from the video."""
    if thumbnail_url:
        uploaded = process_and_upload_thumbnail(thumbnail_url, job_id)
        if uploaded:
            return uploaded

    if raw_video:
        return generate_thumbnail_from_video(raw_video, job_id)
    if video_url:
        return generate_thumbnail_from_video(video_url, job_id)
    return None

def transcribe_video(original_url: str, video_url: str, job_id: str, credits_cost: int, source: str, duration: int, user_id: str, should_upload: bool = True, media_type: str = "video", thumbnail_url: str = None, title: str = None, file_name: str = None):
    try:
        # Track start time for processing duration
        start_time = time.time()
        
        _, job = get_job(job_id)
        update_job(job_id, "transcribing video")
        transcription = whisper.transcribe(video_url)
        
        update_job(job_id, "transcribed video")
        raw_video = get_raw_video(video_url)
        
        # Validate video was actually downloaded
        if len(raw_video) == 0:
            raise Exception("Failed to download video: file is empty. The video URL may be expired or invalid.")

        update_job(job_id, "processing video")

        if should_upload:
            upload_url = upload_video(raw_video, job_id)
        else:
            upload_url = video_url

        # Calculate processing duration
        end_time = time.time()
        processing_duration_seconds = round(end_time - start_time, 2)
        print(f"Total processing time: {processing_duration_seconds}s")

        # Determine mediaType: if internal, use the provided media_type, otherwise default to "video"
        stored_media_type = media_type if source == "internal" else "video"
        
        # Get full transcription text
        full_text = ''.join([s['text'] for s in transcription["segments"]])
        
        # For Facebook and Instagram, generate title from transcription if not provided
        stored_title = title
        if not stored_title and source in ["facebook", "instagram"]:
            stored_title = generate_title_from_transcription(full_text)
            print(f"Generated title from transcription: {stored_title}")
        
        video_data = {
            "userId": user_id,
            "durationMinutes": format_duration_minutes_seconds(duration),
            "fileName": file_name if file_name else job_id,
            "firebaseUrl": upload_url,
            "originalUrl": original_url,
            "platform": source,
            "mediaType": stored_media_type,
            "transcription": {
                "audio_duration": duration,
                "full_text": full_text,
                "speaker_segments": [{
                    "speaker": s['speaker'],
                    "text": s['text'],
                    "start": s['start'],
                    "end": s['end']
                } for s in transcription["segments"]]
            },
            "processingDurationSeconds": processing_duration_seconds,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now(),
            "jobId": job_id
        }
        
        uploaded_thumbnail_url = ensure_video_thumbnail(
            thumbnail_url,
            job_id,
            raw_video=raw_video,
            video_url=video_url,
        )
        if uploaded_thumbnail_url:
            video_data["thumbnailUrl"] = uploaded_thumbnail_url
        
        # Add title if available
        if stored_title:
            video_data["title"] = stored_title

        # Save video to Firestore
        db.collection("videos").add(video_data)

        update_job(job_id, "completed")
        send_transcription_ready_notification(user_id, stored_title)
    except Exception as e:
        update_job(job_id, "error")
        db.collection("jobs").document(job_id).update({
            "error": str(e),
            "updatedAt": datetime.now()
        })

def guess_content_type(url_or_path: str) -> str:
    """Best-effort MIME type for a media URL/path; defaults to video/mp4."""
    ctype, _ = mimetypes.guess_type(url_or_path)
    return ctype or "video/mp4"


def call_deepgram_transcribe(source, is_url: bool, content_type: str = None) -> dict:
    """Call Deepgram /listen with diarization + utterances.

    - URL mode (is_url=True): Deepgram fetches `source` (a URL) itself.
    - Binary mode (is_url=False): `source` is raw audio/video bytes.

    Raises on non-2xx responses (so callers can trigger a fallback).
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise Exception("DEEPGRAM_API_KEY is not configured")

    params = {
        "model": os.getenv("DEEPGRAM_MODEL", "nova-3"),
        "language": os.getenv("DEEPGRAM_LANGUAGE", "en"),
        # diarize=true is deprecated; diarize_model enables diarization + picks version.
        "diarize_model": os.getenv("DEEPGRAM_DIARIZE_MODEL", "latest"),
        "utterances": "true",
        "punctuate": "true",
        "smart_format": "true",
        "numerals": "true",
        "paragraphs": "true",
        "profanity_filter": "true",
    }
    max_speakers = os.getenv("DEEPGRAM_MAX_SPEAKERS", "6")
    if max_speakers:
        params["max_speakers"] = max_speakers

    headers = {"Authorization": f"Token {api_key}"}

    with httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0)) as client:
        if is_url:
            response = client.post(
                DEEPGRAM_URL,
                params=params,
                headers={**headers, "Content-Type": "application/json"},
                json={"url": source},
            )
        else:
            response = client.post(
                DEEPGRAM_URL,
                params=params,
                headers={**headers, "Content-Type": content_type or "video/mp4"},
                content=source,
            )

    response.raise_for_status()
    data = response.json()

    # Diagnostic logging for diarization quality.
    utterances = (data.get("results", {}) or {}).get("utterances", []) or []
    words = (
        (data.get("results", {}) or {}).get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("words", []) or []
    )
    utterance_speakers = sorted({u.get("speaker") for u in utterances if u.get("speaker") is not None})
    word_speakers = sorted({w.get("speaker") for w in words if w.get("speaker") is not None})
    request_id = (data.get("metadata", {}) or {}).get("request_id", "unknown")
    print(
        f"Deepgram diarization (request_id={request_id}): "
        f"utterance_speakers={utterance_speakers} word_speakers={word_speakers} "
        f"utterances={len(utterances)} words={len(words)}"
    )
    if len(utterance_speakers) <= 1 and len(utterances) > 1:
        print(
            "Warning: Deepgram returned multiple utterances but only one speaker label. "
            "Diarization may be weak for this audio (common with mono phone-camera interviews)."
        )

    return parse_deepgram_response(data)


def call_deepgram_transcribe_with_retry(
    source,
    is_url: bool,
    content_type: str = None,
    max_attempts: int = 3,
):
    """Retry Deepgram on transient network failures (e.g. disconnect mid-upload)."""
    retriable = (
        httpx.RemoteProtocolError,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ConnectError,
        httpx.NetworkError,
    )
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call_deepgram_transcribe(source, is_url=is_url, content_type=content_type)
        except retriable as e:
            last_error = e
            print(f"Deepgram attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    raise last_error


def format_speaker_label(speaker_idx: int) -> str:
    """Map Deepgram's 0-based speaker index to a human label (Speaker A, B, ...)."""
    idx = int(speaker_idx)
    if 0 <= idx < 26:
        return f"Speaker {chr(ord('A') + idx)}"
    return f"Speaker {idx + 1}"


def _append_speaker_segment(segments: list, speaker_label: str, text: str, start: float, end: float):
    """Append or extend a speaker segment, merging consecutive same-speaker runs."""
    text = (text or "").strip()
    if not text:
        return
    if segments and segments[-1]["speaker"] == speaker_label:
        previous = segments[-1]
        previous["text"] = (previous["text"] + " " + text).strip()
        previous["end"] = end
    else:
        segments.append({
            "speaker": speaker_label,
            "text": text,
            "start": start,
            "end": end,
        })


def _build_speaker_segments_from_utterances(utterances: list) -> list:
    segments = []
    for utterance in utterances:
        if utterance.get("speaker") is None:
            continue
        speaker_label = format_speaker_label(utterance["speaker"])
        _append_speaker_segment(
            segments,
            speaker_label,
            utterance.get("transcript") or "",
            utterance.get("start", 0),
            utterance.get("end", 0),
        )
    return segments


def _build_speaker_segments_from_words(words: list) -> list:
    """Build segments from per-word speaker labels (finer boundaries than utterances)."""
    segments = []
    for word in words:
        if word.get("speaker") is None:
            continue
        speaker_label = format_speaker_label(word["speaker"])
        piece = (word.get("punctuated_word") or word.get("word") or "").strip()
        if not piece:
            continue
        _append_speaker_segment(
            segments,
            speaker_label,
            piece,
            word.get("start", 0),
            word.get("end", 0),
        )
    return segments


def parse_deepgram_response(data: dict):
    """Parse a Deepgram /listen response.

    Returns (full_text, speaker_segments, audio_duration), where speaker_segments
    are built from per-word speaker labels when available (finer diarization
    boundaries), otherwise from utterances. Consecutive same-speaker runs are
    merged into a single segment shaped like {speaker, text, start, end}.
    """
    results = data.get("results", {}) or {}

    full_text = ""
    channels = results.get("channels", []) or []
    words = []
    if channels:
        alternatives = channels[0].get("alternatives", []) or []
        if alternatives:
            full_text = alternatives[0].get("transcript", "") or ""
            words = alternatives[0].get("words", []) or []

    utterances = results.get("utterances", []) or []
    utterance_segments = _build_speaker_segments_from_utterances(utterances)
    word_segments = _build_speaker_segments_from_words(words)

    utterance_speakers = {u.get("speaker") for u in utterances if u.get("speaker") is not None}
    word_speakers = {w.get("speaker") for w in words if w.get("speaker") is not None}

    # Prefer word-level segments when they detect more speakers than utterances.
    if word_segments and len(word_speakers) > len(utterance_speakers):
        speaker_segments = word_segments
    elif word_segments:
        speaker_segments = word_segments
    else:
        speaker_segments = utterance_segments

    # Fall back to the merged segments if the channel transcript is missing.
    if not full_text and speaker_segments:
        full_text = " ".join(segment["text"] for segment in speaker_segments).strip()

    audio_duration = (data.get("metadata", {}) or {}).get("duration")

    return full_text, speaker_segments, audio_duration


def transcribe_video_openai(original_url: str, video_url: str, job_id: str, credits_cost: int, source: str, duration: int, user_id: str, should_upload: bool = True, media_type: str = "video", thumbnail_url: str = None, title: str = None, file_name: str = None):
    """Transcribe video using Deepgram nova-3 with speaker diarization.

    Speed-optimized:
    - Internal uploads (should_upload=False): Deepgram URL ingestion with binary fallback.
    - Cobalt redirect URLs (Facebook, X, etc.): Deepgram fetches the CDN URL directly
      while the Firebase mirror downloads in parallel — avoids re-uploading large files.
    - Cobalt tunnel URLs (TikTok, etc.): download once, then Deepgram binary upload +
      Firebase upload in parallel (tunnel streams lack Content-Length for URL ingestion).
    """
    try:
        start_time = time.time()

        if not os.getenv("DEEPGRAM_API_KEY"):
            raise Exception("DEEPGRAM_API_KEY is not configured")

        _, job = get_job(job_id)
        raw_for_thumbnail = None
        upload_url = video_url

        if not should_upload:
            update_job(job_id, "transcribing video with Deepgram")
            full_text, all_segments, audio_duration = _deepgram_from_url_or_bytes(video_url)
        elif is_cobalt_tunnel_url(video_url):
            update_job(job_id, "downloading video")
            raw_video = get_raw_video(video_url)
            if not raw_video:
                raise Exception("Failed to download video: file is empty. The video URL may be expired or invalid.")
            raw_for_thumbnail = raw_video
            print(f"Video file size: {len(raw_video) / (1024 * 1024):.2f} MB")

            update_job(job_id, "uploading video")
            upload_url = upload_video(raw_video, job_id)
            update_job(job_id, "transcribing video with Deepgram")
            full_text, all_segments, audio_duration = _deepgram_from_url_or_bytes(
                upload_url,
                raw_video=raw_video,
            )
        else:
            # Direct CDN URL from Cobalt redirect — Deepgram can fetch it directly.
            def run_transcription_cdn():
                update_job(job_id, "transcribing video with Deepgram")
                return _deepgram_from_url_or_bytes(video_url)

            def run_mirror():
                update_job(job_id, "downloading video")
                raw = get_raw_video(video_url)
                if not raw:
                    raise Exception("Failed to download video: file is empty. The video URL may be expired or invalid.")
                print(f"Video file size: {len(raw) / (1024 * 1024):.2f} MB")
                print("Starting video upload to Firebase...")
                url = upload_video(raw, job_id)
                print("Video upload completed")
                return url, raw

            with ThreadPoolExecutor(max_workers=2) as executor:
                mirror_future = executor.submit(run_mirror)
                transcription_future = executor.submit(run_transcription_cdn)
                try:
                    full_text, all_segments, audio_duration = transcription_future.result()
                    upload_url, raw_for_thumbnail = mirror_future.result()
                except Exception as trans_error:
                    print(f"Deepgram CDN transcription failed, retrying via Firebase mirror: {trans_error}")
                    upload_url, raw_for_thumbnail = mirror_future.result()
                    update_job(job_id, "transcribing video with Deepgram")
                    full_text, all_segments, audio_duration = _deepgram_from_url_or_bytes(
                        upload_url,
                        raw_video=raw_for_thumbnail,
                    )

        update_job(job_id, "transcribed video")
        update_job(job_id, "processing video")

        # Prefer Deepgram's reported duration for the stored transcription.
        effective_duration = int(audio_duration) if audio_duration else duration

        # Deduct credits based on the actual transcribed duration (1 credit/min).
        final_credits_cost = int(effective_duration / 60) if effective_duration else 0
        if final_credits_cost > 0:
            try:
                db.collection("users").document(user_id).update({
                    "credits": firestore.Increment(-final_credits_cost)
                })
                print(f"Deducted {final_credits_cost} credits for {effective_duration}s")
            except Exception as credit_error:
                print(f"Failed to deduct credits: {credit_error}")

        # Calculate processing duration
        end_time = time.time()
        processing_duration_seconds = round(end_time - start_time, 2)
        print(f"Total processing time: {processing_duration_seconds}s")

        # Determine mediaType: if internal, use the provided media_type, otherwise default to "video"
        stored_media_type = media_type if source == "internal" else "video"

        # For Facebook and Instagram, generate title from transcription if not provided
        stored_title = title
        if not stored_title and source in ["facebook", "instagram"]:
            stored_title = generate_title_from_transcription(full_text)
            print(f"Generated title from transcription: {stored_title}")

        video_data = {
            "userId": user_id,
            "durationMinutes": format_duration_minutes_seconds(effective_duration),
            "fileName": file_name if file_name else job_id,
            "firebaseUrl": upload_url,
            "originalUrl": original_url,
            "platform": source,
            "mediaType": stored_media_type,
            "transcription": {
                "audio_duration": effective_duration,
                "full_text": full_text,
                "speaker_segments": all_segments
            },
            "processingDurationSeconds": processing_duration_seconds,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now(),
            "jobId": job_id
        }

        uploaded_thumbnail_url = ensure_video_thumbnail(
            thumbnail_url,
            job_id,
            raw_video=raw_for_thumbnail,
            video_url=video_url if raw_for_thumbnail is None else None,
        )
        if uploaded_thumbnail_url:
            video_data["thumbnailUrl"] = uploaded_thumbnail_url

        # Add title if available
        if stored_title:
            video_data["title"] = stored_title

        # Save video to Firestore
        db.collection("videos").add(video_data)

        update_job(job_id, "completed")
        send_transcription_ready_notification(user_id, stored_title)
    except Exception as e:
        print(f"Job failed: {str(e)}")
        update_job(job_id, "error")
        db.collection("jobs").document(job_id).update({
            "error": str(e),
            "updatedAt": datetime.now()
        })


# ---------------------------------------------------------------------------
# LEGACY: OpenAI GPT-4o transcription path (kept commented out for rollback).
# To roll back, restore this body into transcribe_video_openai above.
# ---------------------------------------------------------------------------
def _transcribe_video_openai_legacy(original_url: str, video_url: str, job_id: str, credits_cost: int, source: str, duration: int, user_id: str, should_upload: bool = True, media_type: str = "video", thumbnail_url: str = None, title: str = None, file_name: str = None):
    """Transcribe video using OpenAI's GPT-4o with speaker diarization"""
    try:
        # Track start time for processing duration
        start_time = time.time()
        
        # Use custom httpx client with long timeouts for large uploads
        # connect=60s, read=300s, write=600s (10 mins), pool=300s
        http_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0))
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=http_client
        )
        
        _, job = get_job(job_id)
        update_job(job_id, "downloading video")
        
        # Download the video file
        raw_video = get_raw_video(video_url)
        
        # Check file size (OpenAI has a 25 MB limit)
        file_size_mb = len(raw_video) / (1024 * 1024)
        print(f"Video file size: {file_size_mb:.2f} MB")
        
        # Validate video was actually downloaded
        if len(raw_video) == 0:
            raise Exception("Failed to download video: file is empty. The video URL may be expired or invalid.")
        
        if file_size_mb < 0.001:  # Less than 1KB is suspicious
            raise Exception(f"Downloaded file is too small ({len(raw_video)} bytes). The video URL may be invalid.")
        
        # Save video temporarily
        temp_file_path = f"/tmp/{job_id}.mp4"
        with open(temp_file_path, "wb") as f:
            f.write(raw_video)
        
        all_segments = []
        full_text = ""
        
        if file_size_mb > 25:
            # Use chunking for large files
            print(f"File exceeds 25 MB limit. Using chunking strategy...")
            update_job(job_id, "chunking large video")
            
            # Load audio with PyDub
            audio = AudioSegment.from_file(temp_file_path)
            
            # Get actual audio duration in seconds
            actual_duration_seconds = len(audio) / 1000
            print(f"Actual audio duration: {actual_duration_seconds:.1f}s")
            
            # Update duration and credits cost with actual values
            duration = int(actual_duration_seconds)
            credits_cost = int(duration / 60)
            print(f"Updated duration: {duration}s, credits: {credits_cost}")
            
            # Calculate chunk size (aim for ~20 MB chunks to stay under 25 MB limit)
            # Calculate MB per second based on actual audio duration
            mb_per_second = file_size_mb / actual_duration_seconds
            target_chunk_size_mb = 20  # Target 20 MB to have buffer
            chunk_duration_seconds = int(target_chunk_size_mb / mb_per_second)
            
            # Ensure minimum chunk size of 60 seconds and maximum of 1 minute (60s)
            # Reduced to 60s to maximize parallelization (more chunks = more speed)
            chunk_duration_seconds = max(60, min(chunk_duration_seconds, 60))
            
            # Use 15 second overlap
            overlap_seconds = 15
            chunk_duration_ms = chunk_duration_seconds * 1000
            overlap_ms = overlap_seconds * 1000
            
            
            print(f"Splitting into chunks of ~{chunk_duration_seconds}s with {overlap_seconds}s overlap")
            
            # Step 1: Prepare chunk metadata
            chunk_metadata = []
            chunk_num = 0
            position = 0
            
            while position < len(audio):
                chunk_num += 1
                end_position = min(position + chunk_duration_ms, len(audio))
                
                chunk_metadata.append({
                    'chunk_num': chunk_num,
                    'start_position': position,
                    'end_position': end_position,
                    'time_offset': position / 1000
                })
                
                # Move to next chunk (with overlap)
                position += chunk_duration_ms - overlap_ms
            
            total_chunks = len(chunk_metadata)
            print(f"Preparing {total_chunks} chunks...")
            
            # Step 2: Prepare chunks in parallel (extract and export to disk)
            def prepare_chunk(metadata):
                chunk_num = metadata['chunk_num']
                start_pos = metadata['start_position']
                end_pos = metadata['end_position']
                
                # Extract chunk
                chunk = audio[start_pos:end_pos]
                
                # Export to file
                chunk_file_path = f"/tmp/{job_id}_chunk_{chunk_num}.mp4"
                chunk.export(chunk_file_path, format="mp4")
                
                chunk_size_mb = os.path.getsize(chunk_file_path) / (1024 * 1024)
                print(f"Prepared chunk {chunk_num}/{total_chunks}: {chunk_size_mb:.2f} MB, duration: {len(chunk)/1000:.1f}s")
                
                return {
                    'chunk_num': chunk_num,
                    'file_path': chunk_file_path,
                    'time_offset': metadata['time_offset'],
                    'size_mb': chunk_size_mb
                }
            
            # Prepare chunks in parallel (use 3 workers for I/O operations)
            max_prep_workers = int(os.getenv("MAX_PREPARATION_WORKERS", "3"))
            print(f"Preparing chunks in parallel with {max_prep_workers} workers...")
            update_job(job_id, f"preparing {total_chunks} chunks")
            
            prepared_chunks = []
            with ThreadPoolExecutor(max_workers=max_prep_workers) as executor:
                futures = [executor.submit(prepare_chunk, metadata) for metadata in chunk_metadata]
                for future in as_completed(futures):
                    prepared_chunks.append(future.result())
            
            # Sort by chunk number
            prepared_chunks.sort(key=lambda x: x['chunk_num'])
            print(f"All {total_chunks} chunks prepared and ready for transcription")
            
            # Function to process a single chunk
            def process_chunk(chunk_info):
                chunk_num = chunk_info['chunk_num']
                chunk_file_path = chunk_info['file_path']
                time_offset = chunk_info['time_offset']
                chunk_size_mb = chunk_info['size_mb']
                
                print(f"Transcribing chunk {chunk_num}/{total_chunks}: {chunk_size_mb:.2f} MB")
                
                try:
                    # Transcribe chunk
                    with open(chunk_file_path, "rb") as audio_file:
                        transcription = client.audio.transcriptions.create(
                            model="gpt-4o-transcribe-diarize",
                            file=audio_file,
                            response_format="diarized_json",
                            chunking_strategy="auto"
                        )
                    
                    # Process segments with time offset
                    chunk_segments = []
                    chunk_text = ""
                    
                    if hasattr(transcription, 'segments'):
                        for segment in transcription.segments:
                            adjusted_start = segment.start + time_offset
                            adjusted_end = segment.end + time_offset
                            
                            # Skip overlapping segments from previous chunk
                            if chunk_num > 1 and adjusted_start < time_offset + overlap_seconds:
                                continue
                            
                            segment_data = {
                                "speaker": segment.speaker if hasattr(segment, 'speaker') else "SPEAKER_00",
                                "text": segment.text,
                                "start": adjusted_start,
                                "end": adjusted_end
                            }
                            chunk_segments.append(segment_data)
                            chunk_text += segment.text
                    
                    print(f"Completed chunk {chunk_num}/{total_chunks}")
                    
                    return {
                        'chunk_num': chunk_num,
                        'segments': chunk_segments,
                        'text': chunk_text,
                        'success': True
                    }
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_num}: {str(e)}")
                    return {
                        'chunk_num': chunk_num,
                        'segments': [],
                        'text': "",
                        'success': False,
                        'error': str(e)
                    }
                    
                finally:
                    # Clean up chunk file
                    if os.path.exists(chunk_file_path):
                        os.remove(chunk_file_path)
            
            # Process chunks in parallel using ThreadPoolExecutor
            # Use configurable max workers to avoid overwhelming the API
            max_chunk_workers = int(os.getenv("MAX_CHUNK_WORKERS", "5"))
            max_workers = min(max_chunk_workers, total_chunks)
            print(f"Starting parallel transcription with {max_workers} workers...")
            
            update_job(job_id, f"transcribing {total_chunks} chunks in parallel")
            
            chunk_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all prepared chunks for transcription
                future_to_chunk = {executor.submit(process_chunk, chunk_info): chunk_info for chunk_info in prepared_chunks}
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    result = future.result()
                    chunk_results.append(result)
                    
                    completed = len(chunk_results)
                    update_job(job_id, f"transcribed {completed}/{total_chunks} chunks")
            
            # Sort results by chunk number to maintain order
            chunk_results.sort(key=lambda x: x['chunk_num'])
            
            # Combine all segments and text
            for result in chunk_results:
                if result['success']:
                    all_segments.extend(result['segments'])
                    full_text += result['text']
                else:
                    print(f"Warning: Chunk {result['chunk_num']} failed: {result.get('error', 'Unknown error')}")
            
            print(f"Completed chunking: {total_chunks} chunks processed in parallel")
            
        else:
            # File is small enough, transcribe directly
            update_job(job_id, "transcribing video with OpenAI")
            print(f"Transcribing with OpenAI (duration: {duration}s, file size: {file_size_mb:.2f} MB)")
            
            with open(temp_file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe-diarize",
                    file=audio_file,
                    response_format="diarized_json",
                    chunking_strategy="auto"
                )
            
            # Parse segments
            if hasattr(transcription, 'segments'):
                for segment in transcription.segments:
                    all_segments.append({
                        "speaker": segment.speaker if hasattr(segment, 'speaker') else "SPEAKER_00",
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end
                    })
                    full_text += segment.text
        
        # Clean up main temp file
        os.remove(temp_file_path)
        
        update_job(job_id, "transcribed video")
        update_job(job_id, "processing video")

        if should_upload:
            try:
                print("Starting video upload to Firebase...")
                upload_url = upload_video(raw_video, job_id)
                print("Video upload completed")
            except Exception as e:
                print(f"Firebase upload error: {str(e)}")
                raise Exception(f"Firebase upload failed: {str(e)}")
        else:
            upload_url = video_url

        # Calculate processing duration
        end_time = time.time()
        processing_duration_seconds = round(end_time - start_time, 2)
        print(f"Total processing time: {processing_duration_seconds}s")

        # Determine mediaType: if internal, use the provided media_type, otherwise default to "video"
        stored_media_type = media_type if source == "internal" else "video"
        
        # For Facebook and Instagram, generate title from transcription if not provided
        stored_title = title
        if not stored_title and source in ["facebook", "instagram"]:
            stored_title = generate_title_from_transcription(full_text)
            print(f"Generated title from transcription: {stored_title}")

        video_data = {
            "userId": user_id,
            "durationMinutes": format_duration_minutes_seconds(duration),
            "fileName": file_name if file_name else job_id,
            "firebaseUrl": upload_url,
            "originalUrl": original_url,
            "platform": source,
            "mediaType": stored_media_type,
            "transcription": {
                "audio_duration": duration,
                "full_text": full_text,
                "speaker_segments": all_segments
            },
            "processingDurationSeconds": processing_duration_seconds,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now(),
            "jobId": job_id
        }
        
        uploaded_thumbnail_url = ensure_video_thumbnail(
            thumbnail_url,
            job_id,
            raw_video=raw_video,
            video_url=video_url,
        )
        if uploaded_thumbnail_url:
            video_data["thumbnailUrl"] = uploaded_thumbnail_url
        
        # Add title if available
        if stored_title:
            video_data["title"] = stored_title

        # Save video to Firestore
        db.collection("videos").add(video_data)

        update_job(job_id, "completed")
        send_transcription_ready_notification(user_id, stored_title)
    except Exception as e:
        print(f"Job failed: {str(e)}")
        update_job(job_id, "error")
        db.collection("jobs").document(job_id).update({
            "error": str(e),
            "updatedAt": datetime.now()
        })


def handle_transcribe(url: str, original_url: str, user_id: str, duration: int, source: str, thumbnail_url: str = None, media_type: str = "video", title: str = None, file_name: str = None):
    # Convert duration from milliseconds to seconds if it's too large
    # Assume if duration > 3600 (1 hour in seconds), it's in milliseconds
    if duration > 3600:
        print(f"Duration appears to be in milliseconds: {duration}ms, converting to seconds")
        duration = int(duration / 1000)
        print(f"Converted duration: {duration}s")
    
    credits_cost = int(duration / 60)
    
    user = db.collection("users").document(user_id)
    user_data = user.get().to_dict()
    
    print(f"Duration: {duration}s, Credits needed: {credits_cost}, User credits: {user_data.get('credits', 0)}")
    
    if user_data["credits"] < credits_cost:
        raise fastapi.HTTPException(
            status_code=400, 
            detail=f"Insufficient credits. Need {credits_cost}, have {user_data['credits']}"
        )
    
    user_data["credits"] -= credits_cost


    _, job = db.collection("jobs").add({
        "url": original_url,
        "error": None,
        "user_id": user_id,
        "status": "pending",
        "createdAt": datetime.now(),
        "updatedAt": datetime.now()
    })
    
    job_id = job.id
    print(job_id)

    transcription_thread = Thread(
        target=transcribe_video,
        args=(
            original_url,
            url,
            job_id,
            credits_cost,
            source,
            duration,
            user_id,
            source != "internal",
            media_type,
            thumbnail_url,
            title,
            file_name
        )
    )
    create_job(job_id, transcription_thread)
    transcription_thread.start()

    user.set(user_data)
    
    return {
        "message": "Transcribing video",
        "job_id": job_id,
        "url": url,
        "status": "transcribing"
    }

def handle_transcribe_openai(url: str, original_url: str, user_id: str, duration: int, source: str, media_type: str = "video", thumbnail_url: str = None, title: str = None, file_name: str = None):
    """Handle transcription using Deepgram nova-3 (V3 path).

    Duration may be unknown (None) for Cobalt tunnel URLs (e.g. YouTube) since
    ffprobe can't read the non-seekable tunnel stream. Deepgram reports the true
    duration during transcription, so credits are deducted *after* the job
    completes (see transcribe_video_openai). Upfront we only require >= 1 credit;
    when the duration is known (e.g. internal uploads) we also enforce it.
    """
    # Convert duration from milliseconds to seconds if it's too large
    if duration and duration > 3600:
        print(f"Duration appears to be in milliseconds: {duration}ms, converting to seconds")
        duration = int(duration / 1000)
        print(f"Converted duration: {duration}s")

    user = db.collection("users").document(user_id)
    user_data = user.get().to_dict()
    available_credits = user_data.get("credits", 0)

    # Upfront guard: must have at least 1 credit to start a job.
    if available_credits < 1:
        raise fastapi.HTTPException(status_code=400, detail="Insufficient credits")

    # If we already know the duration, enforce the exact cost upfront too.
    if duration:
        credits_cost = int(duration / 60)
        if available_credits < credits_cost:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"Insufficient credits. Need {credits_cost}, have {available_credits}"
            )

    print(f"Duration: {duration}s (unknown until Deepgram if None), User credits: {available_credits}")

    _, job = db.collection("jobs").add({
        "url": original_url,
        "error": None,
        "user_id": user_id,
        "status": "pending",
        "createdAt": datetime.now(),
        "updatedAt": datetime.now()
    })
    
    job_id = job.id
    print(job_id)

    transcription_thread = Thread(
        target=transcribe_video_openai,
        args=(
            original_url,
            url,
            job_id,
            0,  # credits deducted after transcription from Deepgram's duration
            source,
            duration or 0,
            user_id,
            source != "internal",
            media_type,
            thumbnail_url,
            title,
            file_name
        )
    )
    create_job(job_id, transcription_thread)
    transcription_thread.start()

    return {
        "message": "Transcribing video with Deepgram",
        "job_id": job_id,
        "url": url,
        "status": "transcribing"
    }


@api.post("/transcribe")
def transcribe(payload: TranscribePayload):
    print(payload.url, payload.user_id)
    user = db.collection("users").document(payload.user_id)

    print(user.get())
    
    if not user.get().exists:
        raise fastapi.HTTPException(status_code=404, detail="User not found")
    
    resolved = resolve_video(payload.url)

    return handle_transcribe(
        resolved["video_url"],
        payload.url,
        payload.user_id,
        resolved["duration"],
        resolved["source"],
        resolved.get("thumbnail_url"),
        "video",
        resolved.get("title"),
    )

@api.post("/transcribe/internal")
def transcribe_internal(payload: TranscribeInternalPayload):
    return handle_transcribe(payload.url, payload.url, payload.user_id, payload.duration, "internal", None, payload.media_type, None, payload.fileName)

# Add routes with trailing slashes to handle both cases
@api.post("/transcribe/")
def transcribe_with_slash(payload: TranscribePayload):
    return transcribe(payload)

@api.post("/transcribe/internal/")
def transcribe_internal_with_slash(payload: TranscribeInternalPayload):
    return transcribe_internal(payload)

@api.get("/sample-transcription")
def get_sample_transcription():
    """Get the sample transcription from sample_transcription collection"""
    try:
        sample_doc = db.collection("sample_transcription").document("sample").get()
        
        if not sample_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="Sample transcription not found")
        
        sample_data = sample_doc.to_dict()
        sample_data["id"] = sample_doc.id
        
        return serialize_firestore_data(sample_data)
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_sample_transcription: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api.get("/sample-transcription/")
def get_sample_transcription_with_slash():
    return get_sample_transcription()

@api.websocket("/transcribe/{job_id}")
async def transcribe_websocket(websocket: fastapi.WebSocket, job_id: str):
    await websocket.accept()

    while True:
        in_memory_job, firestore_job = get_job(job_id)
        
        # Determine status: prefer in-memory, fall back to Firestore
        if in_memory_job:
            status = in_memory_job.get("status", "pending")
        elif firestore_job:
            status = firestore_job.get("status", "unknown")
        else:
            # Job doesn't exist anywhere
            await websocket.send_json({
                "status": "not_found",
                "error": f"Job {job_id} not found"
            })
            break
        
        # If job is completed, find and return video details
        if status == "completed":
            # Find video by jobId first (new behavior)
            videos_query = db.collection("videos").where(filter=FieldFilter("jobId", "==", job_id)).limit(1).stream()
            video_doc = None
            for doc in videos_query:
                video_doc = doc
                break
            
            # Fallback for old videos: find by fileName == jobId
            if not video_doc:
                videos_query = db.collection("videos").where(filter=FieldFilter("fileName", "==", job_id)).limit(1).stream()
                for doc in videos_query:
                    video_doc = doc
                    break
            
            if video_doc:
                video_data = video_doc.to_dict()
                video_data["id"] = video_doc.id
                # Serialize to handle DatetimeWithNanoseconds
                await websocket.send_json({
                    "status": status,
                    "video": serialize_firestore_data(video_data)
                })
            else:
                await websocket.send_json({
                    "status": status,
                    "video": None,
                    "message": "Video not found"
                })
            break
        elif status == "error":
            # Return error details
            await websocket.send_json({
                "status": status,
                "error": firestore_job.get("error") if firestore_job else None
            })
            break
        else:
            # Send status update
            await websocket.send_json({"status": status})
            await asyncio.sleep(0.3)

# V3 API - Deepgram Transcription
api_v3 = fastapi.FastAPI()

@api_v3.post("/transcribe")
def transcribe_v3(payload: TranscribePayload):
    """Transcribe using Deepgram nova-3 with speaker diarization"""
    print(f"[V3] {payload.url}, {payload.user_id}")
    user = db.collection("users").document(payload.user_id)

    print(user.get())
    
    if not user.get().exists:
        raise fastapi.HTTPException(status_code=404, detail="User not found")
    
    # Skip ffprobe: Deepgram reports the true duration, and ffprobe can't read
    # Cobalt tunnel URLs. Credits are deducted after transcription completes.
    resolved = resolve_video(payload.url, probe_duration=False)

    return handle_transcribe_openai(
        resolved["video_url"],
        payload.url,
        payload.user_id,
        resolved["duration"],
        resolved["source"],
        "video",
        resolved.get("thumbnail_url"),
        resolved.get("title"),
    )

@api_v3.post("/transcribe/internal")
def transcribe_internal_v3(payload: TranscribeInternalPayload):
    """Transcribe internal videos using Deepgram nova-3"""
    return handle_transcribe_openai(payload.url, payload.url, payload.user_id, payload.duration, "internal", payload.media_type, None, None, payload.fileName)

# Add routes with trailing slashes to handle both cases
@api_v3.post("/transcribe/")
def transcribe_v3_with_slash(payload: TranscribePayload):
    return transcribe_v3(payload)

@api_v3.post("/transcribe/internal/")
def transcribe_internal_v3_with_slash(payload: TranscribeInternalPayload):
    return transcribe_internal_v3(payload)

@api_v3.get("/sample-transcription")
def get_sample_transcription_v3():
    """Get the sample transcription from sample_transcription collection"""
    return get_sample_transcription()

@api_v3.get("/sample-transcription/")
def get_sample_transcription_v3_with_slash():
    return get_sample_transcription()

@api_v3.post("/sample-transcription/share", response_model=ShareLinkResponse)
def create_sample_share_link_v3(
    current_user_id: str = Depends(get_current_user_id),
):
    """Create or reuse a public share link for the demo sample transcription."""
    try:
        sample_doc = db.collection("sample_transcription").document(SAMPLE_TRANSCRIPTION_ID).get()
        if not sample_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="Sample transcription not found")

        return _create_or_reuse_share(
            video_id=SAMPLE_TRANSCRIPTION_ID,
            user_id=current_user_id,
            source_type=SAMPLE_SOURCE_TYPE,
        )
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in create_sample_share_link_v3: {str(e)}")
        import traceback
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_v3.websocket("/transcribe/{job_id}")
async def transcribe_websocket_v3(websocket: fastapi.WebSocket, job_id: str):
    """WebSocket for job status updates with video details on completion"""
    await websocket.accept()

    async def safe_send(payload: dict) -> bool:
        """Send JSON payload and swallow disconnects so we can exit cleanly."""
        try:
            await websocket.send_json(payload)
            return True
        except (WebSocketDisconnect, ConnectionClosed, ConnectionClosedOK, ConnectionClosedError):
            return False
        except Exception as exc:
            # Log unexpected errors but stop looping to avoid noisy tracebacks
            print(f"WebSocket send error for job {job_id}: {exc}")
            return False

    while True:
        in_memory_job, firestore_job = get_job(job_id)
        
        # Determine status: prefer in-memory, fall back to Firestore
        if in_memory_job:
            status = in_memory_job.get("status", "pending")
        elif firestore_job:
            status = firestore_job.get("status", "unknown")
        else:
            # Job doesn't exist anywhere
            if not await safe_send({
                "status": "not_found",
                "error": f"Job {job_id} not found"
            }):
                break
            break
        
        # If job is completed, find and return video details
        if status == "completed":
            # Find video by jobId first (new behavior)
            videos_query = db.collection("videos").where(filter=FieldFilter("jobId", "==", job_id)).limit(1).stream()
            video_doc = None
            for doc in videos_query:
                video_doc = doc
                break
            
            # Fallback for old videos: find by fileName == jobId
            if not video_doc:
                videos_query = db.collection("videos").where(filter=FieldFilter("fileName", "==", job_id)).limit(1).stream()
                for doc in videos_query:
                    video_doc = doc
                    break
            
            if video_doc:
                video_data = video_doc.to_dict()
                video_data["id"] = video_doc.id
                # Serialize to handle DatetimeWithNanoseconds
                if not await safe_send({
                    "status": status,
                    "video": serialize_firestore_data(video_data)
                }):
                    break
            else:
                if not await safe_send({
                    "status": status,
                    "video": None,
                    "message": "Video not found"
                }):
                    break
            break
        elif status == "error":
            # Return error details
            if not await safe_send({
                "status": status,
                "error": firestore_job.get("error") if firestore_job else None
            }):
                break
            break
        else:
            # Send status update
            if not await safe_send({"status": status}):
                break
            await asyncio.sleep(0.3)

@api.get("/videos/{video_id}", response_model=TranscriptionHistoryItem)
def get_video(video_id: str):
    """Get a single video by ID"""
    try:
        video_doc = db.collection("videos").document(video_id).get()
        
        if not video_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="Video not found")
        
        video_data = video_doc.to_dict()
        video_data["id"] = video_doc.id
        
        return video_data
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_video: {str(e)}")
        import traceback
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_v3.get("/videos/{video_id}", response_model=TranscriptionHistoryItem)
def get_video_v3(
    video_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    """Get a single video by ID (owner only)."""
    try:
        video_doc = db.collection("videos").document(video_id).get()

        if not video_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="Video not found")

        video_data = video_doc.to_dict()
        if video_data.get("userId") != current_user_id:
            raise fastapi.HTTPException(status_code=403, detail="Not authorized")

        video_data["id"] = video_doc.id

        return video_data
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_video_v3: {str(e)}")
        import traceback
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_v3.post("/videos/{video_id}/share", response_model=ShareLinkResponse)
def create_share_link_v3(
    video_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    """Create or reuse a public share link for a transcription (opt-in)."""
    try:
        video_doc = db.collection("videos").document(video_id).get()
        if not video_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="Video not found")

        video_data = video_doc.to_dict()
        if video_data.get("userId") != current_user_id:
            raise fastapi.HTTPException(status_code=403, detail="Not authorized")

        return _create_or_reuse_share(
            video_id=video_id,
            user_id=current_user_id,
        )
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in create_share_link_v3: {str(e)}")
        import traceback
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_v3.delete("/videos/{video_id}/share")
def revoke_share_link_v3(
    video_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    """Revoke all active share links for a transcription."""
    try:
        video_doc = db.collection("videos").document(video_id).get()
        if not video_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="Video not found")

        video_data = video_doc.to_dict()
        if video_data.get("userId") != current_user_id:
            raise fastapi.HTTPException(status_code=403, detail="Not authorized")

        shares = (
            db.collection("shares")
            .where(filter=FieldFilter("videoId", "==", video_id))
            .where(filter=FieldFilter("isActive", "==", True))
            .get()
        )

        revoked = 0
        for share_doc in shares:
            share_doc.reference.update({"isActive": False})
            revoked += 1

        return {"message": "Share links revoked", "revokedCount": revoked}
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in revoke_share_link_v3: {str(e)}")
        import traceback
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_v3.get("/public/share/{token}", response_model=PublicShareResponse)
def get_public_share_v3(token: str):
    """Public read path — only works after owner has shared."""
    try:
        share_doc = db.collection("shares").document(token).get()
        if not share_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="Share link not found")

        share = share_doc.to_dict()
        if not share.get("isActive", True):
            raise fastapi.HTTPException(status_code=404, detail="Share link revoked")

        expires_at = share.get("expiresAt")
        if expires_at and expires_at < datetime.now():
            raise fastapi.HTTPException(status_code=404, detail="Share link expired")

        content_doc = _get_share_content_doc(share)
        if not content_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="Video not found")

        share_doc.reference.update({"viewCount": firestore.Increment(1)})

        return _build_public_share_response(content_doc)
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_public_share_v3: {str(e)}")
        import traceback
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_v3.get("/history", response_model=TranscriptionHistoryResponse)
def get_history(
    user_id: str,
    platform: str = None,
    media_type: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    search: str = None,
    limit: int = 20,
    offset: int = 0
):
    try:
        # Base query on videos collection (where transcriptions are stored)
        query = db.collection("videos").where(filter=FieldFilter("userId", "==", user_id))
        
        # Apply platform filter if provided
        if platform:
            query = query.where(filter=FieldFilter("platform", "==", platform))
            
        # Apply media type filter if provided
        if media_type:
            query = query.where(filter=FieldFilter("mediaType", "==", media_type))

        # Apply date filters if provided
        if start_date:
            query = query.where(filter=FieldFilter("createdAt", ">=", start_date))
        if end_date:
            query = query.where(filter=FieldFilter("createdAt", "<=", end_date))
        
        # Always order by createdAt descending (latest first)
        query = query.order_by("createdAt", direction=firestore.Query.DESCENDING)
        
        # Fetch all matching documents
        # We need to fetch all to perform in-memory search filtering and accurate pagination
        docs = query.stream()
        
        all_items = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            all_items.append(data)
            
        # Apply fuzzy search filter in memory if provided
        # Searches in: title, fileName, originalUrl, transcription.full_text
        if search:
            # Use fuzzy search to filter and score items
            scored_items = []
            for item in all_items:
                is_match, score = fuzzy_search_item(item, search)
                if is_match:
                    scored_items.append((item, score))
            
            # Helper to get timestamp for sorting
            def get_timestamp(item):
                created_at = item.get("createdAt")
                if created_at is None:
                    return 0
                # DatetimeWithNanoseconds already has timestamp() method
                if hasattr(created_at, 'timestamp'):
                    return created_at.timestamp()
                # If it's a regular datetime
                if isinstance(created_at, datetime):
                    return created_at.timestamp()
                return 0
            
            # Sort by score descending, then by date descending
            scored_items.sort(key=lambda x: (-x[1], -get_timestamp(x[0])))
            
            # Extract just the items (without scores)
            all_items = [item for item, score in scored_items]
        
        # Calculate total after filtering
        total = len(all_items)
        
        # Apply pagination manually
        paginated_items = all_items[offset : offset + limit]
        
        return {
            "items": paginated_items,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        print(f"Error in get_history: {str(e)}")
        import traceback
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api.get("/users/{user_id}/credits")
def get_user_credits(user_id: str):
    """Get user credits by User ID"""
    try:
        user_doc = db.collection("users").document(user_id).get()
        
        if not user_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="User not found")
        
        user_data = user_doc.to_dict()
        credits = user_data.get("credits", 0)
        
        return {"credits": credits}
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_user_credits: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api.get("/users/{user_id}", response_model=UserInfoResponse)
def get_user_info(user_id: str):
    """Get user info by User ID"""
    try:
        user_doc = db.collection("users").document(user_id).get()
        
        if not user_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="User not found")
        
        user_data = user_doc.to_dict()
        return user_data
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_user_info: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api.post("/users/{user_id}", response_model=UserInfoResponse)
def update_user_info(user_id: str, payload: UpdateUserInfoPayload):
    """Update user info by User ID"""
    try:
        user_ref = db.collection("users").document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="User not found")
        
        # Convert payload to dict, excluding None values
        update_data = {k: v for k, v in payload.dict(exclude_unset=True).items() if v is not None}
        
        if update_data:
            user_ref.update(update_data)
        
        # Return the updated user info
        updated_user_doc = user_ref.get()
        return updated_user_doc.to_dict()
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in update_user_info: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api.post("/users/{user_id}/credits")
def update_user_credits(user_id: str, payload: UserCreditsPayload):
    """Update user credits"""
    try:
        user_ref = db.collection("users").document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise fastapi.HTTPException(status_code=404, detail="User not found")
        
        user_data = user_doc.to_dict()
        
        # If freeTrial is true, set it to false when adding credits
        if user_data.get("freeTrial") is True:
            user_ref.update({"credits": payload.credits, "freeTrial": False})
            print(f"User {user_id} added credits. Setting freeTrial to False.")
        else:
            user_ref.update({"credits": payload.credits})
        
        return {
            "credits": payload.credits,
            "message": "Credits updated successfully"
        }
    except fastapi.HTTPException:
        raise
    except Exception as e:
        print(f"Error in update_user_credits: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

app.mount("/api/v1", api)
app.mount("/api/v3", api_v3)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
