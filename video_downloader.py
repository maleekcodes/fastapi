import subprocess
import json
import os
import glob
import base64
from typing import Dict, Any, Tuple

DOWNLOAD_DIR = os.environ.get("YTDLP_DOWNLOAD_DIR", "/tmp/transcrybe_downloads")
METADATA_TIMEOUT = int(os.environ.get("YTDLP_METADATA_TIMEOUT", "60"))
DOWNLOAD_TIMEOUT = int(os.environ.get("YTDLP_DOWNLOAD_TIMEOUT", "300"))
DEFAULT_FORMAT = "worstvideo+worstaudio/worst"
COOKIES_FILE = os.path.join(DOWNLOAD_DIR, "cookies.txt")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _init_cookies():
    cookies_b64 = os.environ.get("YTDLP_COOKIES_BASE64", "")
    if cookies_b64:
        try:
            cookies_data = base64.b64decode(cookies_b64).decode("utf-8")
            with open(COOKIES_FILE, "w") as f:
                f.write(cookies_data)
            print("YouTube cookies loaded from env var")
        except Exception as e:
            print(f"Failed to decode cookies: {e}")


_init_cookies()


def get_ytdlp_base_args():
    args = [
        "yt-dlp",
        "--no-playlist",
        "--extractor-args", "youtube:player_client=android,web",
    ]
    if os.path.exists(COOKIES_FILE):
        args += ["--cookies", COOKIES_FILE]
    return args

YTDLP_EXTRACTOR_MAP = {
    "youtube": "youtube",
    "Youtube": "youtube",
    "TikTok": "tiktok",
    "tiktok": "tiktok",
    "Instagram": "instagram",
    "instagram": "instagram",
    "Facebook": "facebook",
    "facebook": "facebook",
    "Twitter": "twitter",
    "twitter": "twitter",
    "TwitterBroadcast": "twitter",
}

USELESS_TITLES = [
    "- Facebook Reel",
    "Facebook Reel",
    "Instagram Reel",
    "- Instagram Reel",
    "Reel",
    "Video",
    "TikTok Video",
]

ERROR_PATTERNS = {
    "Video unavailable": "This video is unavailable or has been removed.",
    "Private video": "This video is private and cannot be accessed.",
    "Sign in to confirm your age": "This video requires age verification.",
    "Sign in to confirm": "YouTube requires authentication. Try a different video or platform.",
    "Unsupported URL": "This URL is not supported.",
    "Unable to extract": "Could not extract video information.",
    "HTTP Error 403": "Access denied. The video may be geo-restricted.",
    "HTTP Error 404": "Video not found.",
}


class VideoDownloadError(Exception):
    pass


def get_user_friendly_error(stderr: str) -> str:
    for pattern, message in ERROR_PATTERNS.items():
        if pattern.lower() in stderr.lower():
            return message
    return stderr.strip().split("\n")[-1] if stderr.strip() else "Unknown error occurred."


def get_video_metadata(url: str) -> Dict[str, Any]:
    cmd = get_ytdlp_base_args() + ["-j", url]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=METADATA_TIMEOUT
        )

        if result.returncode != 0:
            error_msg = get_user_friendly_error(result.stderr)
            raise VideoDownloadError(error_msg)

        info = json.loads(result.stdout)

        extractor = info.get("extractor_key") or info.get("extractor", "unknown")
        source = YTDLP_EXTRACTOR_MAP.get(extractor, extractor.lower())

        title = info.get("title", "")
        if title in USELESS_TITLES or (title.startswith("- ") and len(title) < 20):
            title = None

        return {
            "title": title,
            "thumbnail": info.get("thumbnail", ""),
            "duration": info.get("duration", 0),
            "source": source,
            "uploader": info.get("uploader", ""),
        }

    except subprocess.TimeoutExpired:
        raise VideoDownloadError("Timed out fetching video information. Please try again.")
    except json.JSONDecodeError:
        raise VideoDownloadError("Failed to parse video information.")


def download_video(url: str, job_id: str) -> Tuple[str, int]:
    out_template = os.path.join(DOWNLOAD_DIR, f"{job_id}.%(ext)s")

    cmd = get_ytdlp_base_args() + [
        "-o", out_template,
        "-f", DEFAULT_FORMAT,
        "--merge-output-format", "mp4",
        url
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DOWNLOAD_TIMEOUT
        )

        if result.returncode != 0:
            error_msg = get_user_friendly_error(result.stderr)
            raise VideoDownloadError(error_msg)

        files = glob.glob(os.path.join(DOWNLOAD_DIR, f"{job_id}.*"))
        if not files:
            raise VideoDownloadError("Download completed but no file was found.")

        mp4_files = [f for f in files if f.endswith(".mp4")]
        chosen_file = mp4_files[0] if mp4_files else files[0]

        for f in files:
            if f != chosen_file:
                try:
                    os.remove(f)
                except OSError:
                    pass

        file_size = os.path.getsize(chosen_file)
        return chosen_file, file_size

    except subprocess.TimeoutExpired:
        cleanup_download(job_id)
        raise VideoDownloadError("Download timed out (5 minute limit).")


def cleanup_download(job_id: str) -> None:
    pattern = os.path.join(DOWNLOAD_DIR, f"{job_id}*")
    for f in glob.glob(pattern):
        try:
            os.remove(f)
        except OSError:
            pass
