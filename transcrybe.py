from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import fastapi
import firebase_admin
import requests
from firebase_admin import credentials, firestore, storage
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
from video_downloader import get_video_metadata, download_video, cleanup_download, VideoDownloadError

# Load environment variables from .env file
load_dotenv()

app = fastapi.FastAPI()
api = fastapi.FastAPI()

import base64
import json

service_account_base64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_BASE64")
if service_account_base64:
    service_account_data = json.loads(base64.b64decode(service_account_base64))
    cred = credentials.Certificate(service_account_data)
else:
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

SOURCES_TO_INDEX = {
    "tiktok": 1,
    "instagram": 0,
    "youtube": 0,
    "facebook": 0,
    "twitter": 0,
    "x": 0  # X (formerly Twitter)
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

def delete_job(job_id: str):
    del jobs[job_id]

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

def get_raw_video(url: str):    
    return requests.get(url).content

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


def transcribe_video(original_url: str, video_url: str, job_id: str, credits_cost: int, source: str, duration: int, user_id: str, should_upload: bool = True, media_type: str = "video", thumbnail_url: str = None, title: str = None, file_name: str = None):
    try:
        start_time = time.time()

        _, job = get_job(job_id)
        update_job(job_id, "downloading video")

        if source == "internal":
            raw_video = get_raw_video(video_url)
            local_file_path = None
        else:
            local_file_path, file_size = download_video(original_url, job_id)
            with open(local_file_path, "rb") as f:
                raw_video = f.read()

        if len(raw_video) == 0:
            raise Exception("Failed to download video: file is empty.")

        update_job(job_id, "processing video")

        if should_upload:
            upload_url = upload_video(raw_video, job_id)
        else:
            upload_url = video_url

        update_job(job_id, "transcribing video")
        transcription = whisper.transcribe(upload_url)

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
        
        # Upload thumbnail to Firebase Storage and store the uploaded URL
        if thumbnail_url:
            uploaded_thumbnail_url = process_and_upload_thumbnail(thumbnail_url, job_id)
            if uploaded_thumbnail_url:
                video_data["thumbnailUrl"] = uploaded_thumbnail_url
            else:
                # Fallback to original URL if upload fails
                video_data["thumbnailUrl"] = thumbnail_url
        
        # Add title if available
        if stored_title:
            video_data["title"] = stored_title

        # Save video to Firestore
        db.collection("videos").add(video_data)

        update_job(job_id, "completed")
    except Exception as e:
        update_job(job_id, "error")
        db.collection("jobs").document(job_id).update({
            "error": str(e),
            "updatedAt": datetime.now()
        })
    finally:
        if source != "internal":
            cleanup_download(job_id)

def transcribe_video_openai(original_url: str, video_url: str, job_id: str, credits_cost: int, source: str, duration: int, user_id: str, should_upload: bool = True, media_type: str = "video", thumbnail_url: str = None, title: str = None, file_name: str = None):
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

        if source == "internal":
            raw_video = get_raw_video(video_url)
            temp_file_path = f"/tmp/{job_id}.mp4"
            with open(temp_file_path, "wb") as f:
                f.write(raw_video)
        else:
            temp_file_path, file_size = download_video(original_url, job_id)
            with open(temp_file_path, "rb") as f:
                raw_video = f.read()

        file_size_mb = len(raw_video) / (1024 * 1024)
        print(f"Video file size: {file_size_mb:.2f} MB")

        if len(raw_video) == 0:
            raise Exception("Failed to download video: file is empty.")

        if file_size_mb < 0.001:
            raise Exception(f"Downloaded file is too small ({len(raw_video)} bytes).")
        
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
        
        # Upload thumbnail to Firebase Storage and store the uploaded URL
        if thumbnail_url:
            uploaded_thumbnail_url = process_and_upload_thumbnail(thumbnail_url, job_id)
            if uploaded_thumbnail_url:
                video_data["thumbnailUrl"] = uploaded_thumbnail_url
            else:
                # Fallback to original URL if upload fails
                video_data["thumbnailUrl"] = thumbnail_url
        
        # Add title if available
        if stored_title:
            video_data["title"] = stored_title

        # Save video to Firestore
        db.collection("videos").add(video_data)

        update_job(job_id, "completed")
    except Exception as e:
        print(f"Job failed: {str(e)}")
        update_job(job_id, "error")
        db.collection("jobs").document(job_id).update({
            "error": str(e),
            "updatedAt": datetime.now()
        })
    finally:
        if source != "internal":
            cleanup_download(job_id)


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
    """Handle transcription using OpenAI's GPT-4o model"""
    # Convert duration from milliseconds to seconds if it's too large
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
        target=transcribe_video_openai,
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
        "message": "Transcribing video with OpenAI",
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

    try:
        metadata = get_video_metadata(payload.url)
    except VideoDownloadError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))

    source = metadata.get('source', 'unknown')
    thumbnail_url = metadata.get('thumbnail')
    title = metadata.get('title')

    return handle_transcribe(
        payload.url,
        payload.url,
        payload.user_id,
        metadata.get('duration', 0),
        source,
        thumbnail_url,
        "video",
        title
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

# V3 API - OpenAI Transcription
api_v3 = fastapi.FastAPI()

@api_v3.post("/transcribe")
def transcribe_v3(payload: TranscribePayload):
    """Transcribe using OpenAI GPT-4o with speaker diarization"""
    print(f"[V3] {payload.url}, {payload.user_id}")
    user = db.collection("users").document(payload.user_id)

    print(user.get())

    if not user.get().exists:
        raise fastapi.HTTPException(status_code=404, detail="User not found")

    try:
        metadata = get_video_metadata(payload.url)
    except VideoDownloadError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))

    source = metadata.get('source', 'unknown')
    thumbnail_url = metadata.get('thumbnail')
    title = metadata.get('title')

    return handle_transcribe_openai(
        payload.url,
        payload.url,
        payload.user_id,
        metadata.get('duration', 0),
        source,
        "video",
        thumbnail_url,
        title
    )

@api_v3.post("/transcribe/internal")
def transcribe_internal_v3(payload: TranscribeInternalPayload):
    """Transcribe internal videos using OpenAI GPT-4o"""
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
def get_video_v3(video_id: str):
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
        print(f"Error in get_video_v3: {str(e)}")
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
