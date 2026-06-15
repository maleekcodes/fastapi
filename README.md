# transcrybe-fastapi

FastAPI backend for Transcryb (Railway deployment).

## Share links

Transcriptions are **private by default**. They become public only when the owner calls:

- `POST /api/v3/videos/{video_id}/share` — requires Firebase Bearer token + ownership
- `POST /api/v3/sample-transcription/share` — requires auth; shares the demo sample (any logged-in user)
- `GET /api/v3/public/share/{token}` — public, no auth
- `DELETE /api/v3/videos/{video_id}/share` — revoke active share links

`GET /api/v3/videos/{video_id}` now requires auth and ownership.

### Environment

```bash
SHARE_BASE_URL=https://share.transcryb.app

# YouTube (RapidAPI Social Download All In One)
RAPIDAPI_KEY=...
# RAPIDAPI_HOST=social-download-all-in-one.p.rapidapi.com

# TikTok, Instagram, Facebook, X (Cobalt)
COBALT_API_URL=https://cobalt-production-7dd4.up.railway.app/
# COBALT_API_KEY=...   # only if your Cobalt instance requires auth

# Deepgram (V3 transcription)
DEEPGRAM_API_KEY=...
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=en
DEEPGRAM_DIARIZE_MODEL=latest
DEEPGRAM_MAX_SPEAKERS=6

# Replicate (V1 Whisper)
REPLICATE_API_TOKEN=...

# Railway Firebase credentials
FIREBASE_SERVICE_ACCOUNT_BASE64=...
```

### Video URL resolution (hybrid)

- **YouTube** — resolved via RapidAPI [Social Download All In One](https://rapidapi.com/nguyenmanhict-MuTUtGWD7K/api/social-download-all-in-one). Returns direct download URLs, title, thumbnail, and duration for upfront credit checks.
- **TikTok, Instagram, Facebook, X** — resolved via a [Cobalt](https://github.com/imputnet/cobalt) instance. Duration for V1 credit billing is probed with `ffprobe` after Cobalt returns a download URL — **ffmpeg/ffprobe must be on PATH** in production (same requirement as `pydub`).

### V3 transcription (Deepgram)

The V3 path (`POST /api/v3/transcribe`, `POST /api/v3/transcribe/internal`) transcribes with Deepgram `nova-3` and speaker diarization. It is optimized for speed:

- **Internal uploads** (`/transcribe/internal`) have a stable, directly-fetchable URL (with a `Content-Length`), so Deepgram fetches it itself via URL ingestion (`{"url": ...}`) — no download on our side. A binary upload is used as a fallback.
- **External videos** — YouTube uses RapidAPI direct URLs; other platforms use Cobalt tunnel URLs (no `Content-Length`, so Deepgram cannot URL-ingest them — HTTP 411). Those bytes are downloaded once for the Firebase mirror and sent to Deepgram as a binary upload.
- **Parallel Firebase mirror** — for external videos the download + Firebase Storage upload runs concurrently with transcription (sharing a single download), so total time is roughly `max(transcription, upload)` rather than the sum. A job is marked `completed` only once both finish.
- **No upfront ffprobe for Cobalt paths** — the V3 path skips ffprobe on Cobalt tunnels (non-seekable). YouTube gets duration from RapidAPI; other platforms rely on Deepgram for duration. **Credits are deducted after the job completes** for Cobalt sources (1 credit/min). Upfront we only require the user to have at least 1 credit; YouTube with known duration also enforces the exact cost before starting.

Query params sent to Deepgram: `model=nova-3`, `diarize_model=latest`, `max_speakers=6`, `utterances=true`, `punctuate=true`, `smart_format=true`, `numerals=true`, `paragraphs=true`, `profanity_filter=true`, `language=en`. Speaker segments are built from per-word speaker labels when available (finer boundaries than utterances). Labels are `Speaker A`, `Speaker B`, etc. Mono phone-camera interviews (e.g. Facebook street interviews) can still collapse to one speaker when Deepgram cannot separate voices on a single mixed track.

> The legacy OpenAI implementation is preserved for rollback as `_transcribe_video_openai_legacy`.

### Local development

```bash
conda activate transcryb
cd fastapi
pip install -r requirements.txt
hypercorn main:app --reload
# or: python transcrybe.py
```

Server runs at `http://localhost:8000`. Restart after pulling backend changes so new routes are registered.

### Railway deployment

Production starts via `hypercorn main:app --bind "[::]:$PORT"` (see `railway.json`). Set env vars in the Railway dashboard; use `FIREBASE_SERVICE_ACCOUNT_BASE64` instead of a local `service_account.json`.

### Firestore

New collection: `shares` (document ID = token). A composite index on `videoId` + `isActive` may be required — Firebase will prompt with a link on first query.
