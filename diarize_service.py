import os
import re
import json
import shutil
import time
import threading
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import av
import torch
import pandas as pd
import requests
import whisperx
from whisperx.audio import SAMPLE_RATE
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="MemoSplit Studio - Local & Cloud AI Diarization Engine")

# CORS middleware for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env if present
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 8 if DEVICE != "cpu" else 4
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
HF_TOKEN = os.getenv("HF_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
DEFAULT_ENGINE = os.getenv("MEMOSPLIT_ENGINE", "gemini")

# Priority default: mounted MyCloud EX2 Ultra share, otherwise local ./library
MYCLOUD_VOICE_CLIPS_DIR = "/Volumes/Public/Kyle/voice clips"
DEFAULT_LIBRARY_DIR = os.getenv(
    "MEMOSPLIT_LIBRARY_DIR",
    MYCLOUD_VOICE_CLIPS_DIR if os.path.isdir(MYCLOUD_VOICE_CLIPS_DIR) else os.path.abspath("./library")
)
os.makedirs(DEFAULT_LIBRARY_DIR, exist_ok=True)

whisper_model = None
diarize_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print(f"[*] Loading WhisperX model on {DEVICE} ({COMPUTE_TYPE})...")
        whisper_model = whisperx.load_model("small", DEVICE, compute_type=COMPUTE_TYPE)
    return whisper_model

def get_diarize_model(token=None):
    global diarize_model
    auth_token = token or HF_TOKEN
    if diarize_model is None and auth_token:
        print(f"[*] Loading Pyannote Diarization Pipeline on {DEVICE}...")
        try:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=auth_token, device=DEVICE)
        except Exception as e:
            print(f"[!] Diarization init error: {e}")
    return diarize_model

_AUDIO_DURATION_CACHE: Dict[str, tuple] = {}

def get_audio_duration(file_path: str) -> float:
    """
    Extract exact audio duration in seconds using PyAV container headers.
    Caches result by (file_path, mtime) to keep scans instantaneous.
    """
    try:
        if not os.path.isfile(file_path):
            return 0.0
        mtime = os.path.getmtime(file_path)
        if file_path in _AUDIO_DURATION_CACHE:
            cached_mtime, cached_dur = _AUDIO_DURATION_CACHE[file_path]
            if cached_mtime == mtime:
                return cached_dur
        with av.open(file_path) as container:
            if container.duration:
                dur = round(float(container.duration) / av.time_base, 2)
                _AUDIO_DURATION_CACHE[file_path] = (mtime, dur)
                return dur
    except Exception:
        pass
    return 0.0


# =====================================================================
# SRT & VTT Parsers / Serializers (Legacy Otter.ai & Standard Captions)
# =====================================================================

def parse_srt_time(time_str: str) -> float:
    """Convert '00:01:23,456' or '00:01:23.456' to seconds."""
    time_str = time_str.strip().replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0.0

def format_srt_time(seconds: float) -> str:
    """Format seconds into 00:00:00,000 SRT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_vtt_time(seconds: float) -> str:
    """Format seconds into 00:00:00.000 VTT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def parse_srt_content(content: str) -> List[Dict[str, Any]]:
    """
    Parse SubRip (.srt) text content into structured segments.
    Detects speaker names formatted like '[Speaker 1]: ...' (Otter.ai style)
    or 'Greg McKenzie: ...'.
    Preserves active speaker across continuation cues where Otter does not repeat the name.
    """
    segments = []
    blocks = re.split(r'\n\s*\n', content.strip())
    # Match leading speaker tag, e.g.:
    # "Greg McKenzie: text" or "[Greg McKenzie]: text" or "Speaker 1: text" or "[Speaker 1]: text"
    speaker_regex = re.compile(r"^\[?(?:Speaker\s*([0-9A-Za-z_-]+)|([A-Za-z0-9\s._\'-]{1,50}))\]?\s*:\s*(.*)", re.DOTALL)

    current_speaker = "Speaker 1"
    has_seen_first_speaker = False

    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) < 2:
            continue

        time_line_idx = 1 if lines[0].isdigit() else 0
        if time_line_idx >= len(lines):
            continue

        time_match = re.search(r'(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})', lines[time_line_idx])
        if not time_match:
            continue

        start_sec = parse_srt_time(time_match.group(1))
        end_sec = parse_srt_time(time_match.group(2))

        text_lines = lines[time_line_idx + 1:]
        raw_text = " ".join(text_lines).strip()

        spk_match = speaker_regex.match(raw_text)
        if spk_match:
            g1, g2, content_text = spk_match.groups()
            speaker_name = g1 or g2
            if speaker_name:
                cleaned = speaker_name.strip()
                if cleaned.isdigit():
                    current_speaker = f"Speaker {int(cleaned)}"
                else:
                    current_speaker = cleaned
                has_seen_first_speaker = True
            text = content_text.strip()
        else:
            # Continuation block: retains the currently active speaker
            text = raw_text

        segments.append({
            "id": len(segments) + 1,
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "speaker": current_speaker,
            "text": text
        })

    return segments

def parse_txt_to_segments(content: str, total_duration: float = 60.0) -> List[Dict[str, Any]]:
    """
    Parse plain .txt transcripts into structured segments.
    1. Detects timestamped lines (e.g. '00:01:23 Speaker: ...' or '[01:23] ...').
    2. If no timestamps exist (e.g. Otter note bullets, meeting text), strips bullet prefixes
       and distributes sentences/paragraphs proportionally across total_duration so that
       WhisperX Wav2Vec2 phonetic alignment can align words to precise timestamps.
    """
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        return []

    # Timestamp pattern: [00:01:23] or 00:01:23 or 01:23.45
    ts_pattern = re.compile(
        r'(?:(?:(\w[\w\s]{0,25}?)\s*[:\-])?\s*)?\[?(\d{1,2}:\d{2}(?::\d{2})?(?:[,\.]\d+)?)\]?(?:\s*-\s*\[?(\d{1,2}:\d{2}(?::\d{2})?(?:[,\.]\d+)?)\]?)?\s*:?\s*(.*)'
    )

    def _to_sec(s: str) -> float:
        s = s.replace(',', '.')
        parts = s.split(':')
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(s)

    timestamped_count = 0
    parsed_candidates = []
    for l in lines:
        m = ts_pattern.match(l)
        if m and m.group(2):
            timestamped_count += 1
            spk = m.group(1) or ""
            start_s = _to_sec(m.group(2))
            end_s = _to_sec(m.group(3)) if m.group(3) else None
            txt = m.group(4).strip()
            # If text has speaker prefix like "Greg McKenzie: ..."
            if ':' in txt and not txt.startswith('http'):
                pfx, rest = txt.split(':', 1)
                if len(pfx.strip().split()) <= 3 and len(pfx.strip()) <= 30:
                    spk = pfx.strip()
                    txt = rest.strip()
            parsed_candidates.append({
                'start': start_s,
                'end': end_s,
                'speaker': spk or 'SPEAKER_00',
                'text': txt
            })

    # If at least 35% of lines have timestamps, treat as timestamped transcript
    if timestamped_count >= len(lines) * 0.35 and timestamped_count >= 2:
        segments = []
        for idx, item in enumerate(parsed_candidates, 1):
            s_start = item['start']
            s_end = item['end']
            if s_end is None or s_end <= s_start:
                if idx < len(parsed_candidates):
                    s_end = parsed_candidates[idx]['start']
                else:
                    s_end = min(total_duration, s_start + 5.0) if total_duration > s_start else s_start + 5.0
            segments.append({
                'id': idx,
                'start': round(s_start, 2),
                'end': round(max(s_start + 0.1, s_end), 2),
                'speaker': item['speaker'] or 'SPEAKER_00',
                'text': item['text']
            })
        return segments

    # Otherwise: plain unsegmented text (e.g. bullets or paragraphs)
    cleaned_items = []
    # Drop potential single-line document title if present
    if len(lines) > 1 and not lines[0].startswith(('-', '*', '•')) and not any(lines[0].endswith(p) for p in ('.', '!', '?')) and len(lines[0]) < 80:
        lines = lines[1:]

    current_spk = "SPEAKER_00"
    for l in lines:
        c = re.sub(r'^[-*•]\s*', '', l).strip()
        # Check for explicit speaker prefix e.g. "Bob:" or "Alex Rivera:"
        spk_match = re.match(r'^([A-Z][a-zA-Z\s._\'-]{1,30}):\s*(.+)$', c)
        if spk_match:
            current_spk = spk_match.group(1).strip()
            c = spk_match.group(2).strip()

        if c:
            cleaned_items.append({'speaker': current_spk, 'text': c})

    if not cleaned_items:
        return []

    dur = max(1.0, total_duration)
    total_chars = sum(len(x['text']) for x in cleaned_items)
    segments = []
    curr_t = 0.0
    for idx, item in enumerate(cleaned_items, 1):
        seg_len = (len(item['text']) / total_chars) * dur
        seg_start = round(curr_t, 2)
        seg_end = round(curr_t + seg_len, 2)
        curr_t = seg_end
        segments.append({
            'id': idx,
            'start': seg_start,
            'end': seg_end,
            'speaker': item['speaker'],
            'text': item['text']
        })
    return segments

def find_existing_transcript_segments(audio_path: Path, total_duration: float = 0.0) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Check if a companion transcript (.json, .srt, .vtt, .txt) exists for the given audio.
    If found, parse and return (source_type, segments).
    This allows skipping Whisper ASR and jumping directly to phonetic alignment.
    """
    companion = find_companion_transcript(audio_path)
    if not companion or not companion.exists():
        return None, None

    ext = companion.suffix.lower()
    try:
        if ext == '.json':
            with open(companion, 'r', encoding='utf-8') as f:
                data = json.load(f)
                segs = data.get("segments", [])
                if segs and len(segs) > 0:
                    return "json", segs
        elif ext in ['.srt', '.vtt']:
            with open(companion, 'r', encoding='utf-8', errors='ignore') as f:
                segs = parse_srt_content(f.read())
                if segs and len(segs) > 0:
                    return ext.lstrip('.'), segs
        elif ext == '.txt':
            with open(companion, 'r', encoding='utf-8', errors='ignore') as f:
                segs = parse_txt_to_segments(f.read(), total_duration=total_duration)
                if segs and len(segs) > 0:
                    return "txt", segs
    except Exception as e:
        print(f"[!] Warning: Failed parsing companion transcript {companion}: {e}")
        return None, None

    return None, None

def serialize_to_srt(segments: List[Dict[str, Any]], speaker_meta: Dict[str, Any] = None) -> str:
    """Format segments list into standard .srt SubRip format with Otter-style speaker labels."""
    out = []
    meta = speaker_meta or {}
    for idx, seg in enumerate(segments, 1):
        spk_key = seg.get("speaker", "SPEAKER_00")
        label = meta.get(spk_key, {}).get("label", spk_key)
        start_str = format_srt_time(seg["start"])
        end_str = format_srt_time(seg["end"])
        out.append(f"{idx}\n{start_str} --> {end_str}\n[{label}]: {seg.get('text', '').strip()}\n")
    return "\n".join(out)

def serialize_to_vtt(segments: List[Dict[str, Any]], speaker_meta: Dict[str, Any] = None) -> str:
    """Format segments into standard WebVTT format with voice tags."""
    out = ["WEBVTT\n"]
    meta = speaker_meta or {}
    for seg in segments:
        spk_key = seg.get("speaker", "SPEAKER_00")
        label = meta.get(spk_key, {}).get("label", spk_key)
        start_str = format_vtt_time(seg["start"])
        end_str = format_vtt_time(seg["end"])
        out.append(f"{start_str} --> {end_str}\n<v {label}>{seg.get('text', '').strip()}\n")
    return "\n".join(out)

def serialize_to_txt(segments: List[Dict[str, Any]], speaker_meta: Dict[str, Any] = None) -> str:
    """Format segments into clean meeting transcript text."""
    out = []
    meta = speaker_meta or {}
    for seg in segments:
        spk_key = seg.get("speaker", "SPEAKER_00")
        label = meta.get(spk_key, {}).get("label", spk_key)
        mins = int(seg["start"] // 60)
        secs = int(seg["start"] % 60)
        time_str = f"[{mins:02d}:{secs:02d}]"
        out.append(f"{label} {time_str}:\n{seg.get('text', '').strip()}\n\n")
    return "".join(out)


# =====================================================================
# Library Scanner & File Helpers
# =====================================================================

AUDIO_EXTENSIONS = {'.m4a', '.mp3', '.wav', '.ogg', '.aac', '.flac'}

def normalize_smb_path(raw_path: str) -> str:
    """
    Translates smb:// URLs into local macOS /Volumes/ mount points.
    e.g. smb://MyCloudEX2Ultra._smb._tcp.local/Public/Kyle/voice clips
    -> /Volumes/Public/Kyle/voice clips
    """
    if not raw_path:
        return raw_path
    
    clean = raw_path.strip().strip("'").strip('"')
    if clean.startswith("smb://"):
        # Strip smb://
        sub = clean[6:]
        # Remove host part (everything before the first slash)
        if "/" in sub:
            share_and_path = sub.split("/", 1)[1]
            candidate = os.path.join("/Volumes", share_and_path)
            if os.path.isdir(candidate):
                return candidate
    return clean

def get_library_path(custom_path: Optional[str] = None) -> str:
    if custom_path:
        norm = normalize_smb_path(custom_path)
        if os.path.isdir(norm):
            return os.path.abspath(norm)
    return DEFAULT_LIBRARY_DIR

def find_companion_transcript(item_path: Path) -> Optional[Path]:
    """
    Find matching transcript file for an audio item.
    1. Check exact companion files (.json, .srt, .vtt, .txt) in the same folder.
    2. Check otter export directories like '{stem}_otter_ai' or matching folders in the parent dir.
    """
    parent = item_path.parent
    base = item_path.stem.strip().lower()

    # Priority 1: Exact match in same folder
    for ext in ['.json', '.srt', '.vtt', '.txt']:
        cf = item_path.with_suffix(ext)
        if cf.is_file():
            return cf

    # Priority 2: In same folder with fuzzy name match (e.g. slight date/space difference)
    for f in parent.iterdir():
        if f.is_file() and f.suffix.lower() in ['.json', '.srt', '.vtt', '.txt']:
            f_stem = f.stem.strip().lower()
            if base in f_stem or f_stem in base:
                return f

    # Priority 3: Companion otter folder in parent directory
    for d in parent.iterdir():
        if d.is_dir():
            d_clean = d.name.lower().replace('_otter_ai', '').strip()
            if d_clean == base or base in d_clean or d_clean in base:
                for ext in ['.json', '.srt', '.vtt', '.txt']:
                    for subf in d.iterdir():
                        if subf.is_file() and subf.suffix.lower() == ext:
                            return subf
    return None

def scan_library(library_dir: str) -> Dict[str, Any]:
    recordings = []
    folders_dict = {}
    p = Path(library_dir)
    if not p.exists():
        return {"recordings": [], "folders": []}

    # Root folder placeholder
    folders_dict[""] = {
        "path": "",
        "name": "All Folders / Root",
        "relative_path": "",
        "count": 0
    }

    # Recursively traverse directory tree
    for root, dirs, files in os.walk(library_dir):
        dirs.sort()
        files.sort()

        raw_rel = os.path.relpath(root, library_dir)
        if raw_rel == ".":
            raw_rel = ""

        # Normalize relative path: if inside an _otter_ai folder, logical folder is its parent
        parts = [p for p in raw_rel.split(os.sep) if p]
        is_otter_dir = False
        if parts and parts[-1].lower().endswith("_otter_ai"):
            is_otter_dir = True
            logical_rel = os.sep.join(parts[:-1])
        else:
            logical_rel = raw_rel

        # Register folder and all its ancestor directories
        if logical_rel:
            accum = ""
            for p in logical_rel.split(os.sep):
                accum = f"{accum}/{p}" if accum else p
                if accum not in folders_dict:
                    folders_dict[accum] = {
                        "path": accum,
                        "name": os.path.basename(accum),
                        "relative_path": accum,
                        "count": 0
                    }
        elif raw_rel and raw_rel not in folders_dict and not is_otter_dir:
            folders_dict[raw_rel] = {
                "path": raw_rel,
                "name": os.path.basename(raw_rel),
                "relative_path": raw_rel,
                "count": 0
            }

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                item = Path(root) / fname
                base_name = item.stem

                # Find companion transcript file (same directory or companion otter folder)
                transcript_file = find_companion_transcript(item)
                has_transcript = transcript_file is not None
                has_srt = transcript_file is not None and transcript_file.suffix.lower() == '.srt'
                duration = 0.0
                speakers = []
                preview_text = ""

                if transcript_file:
                    t_ext = transcript_file.suffix.lower()
                    if t_ext == '.json':
                        try:
                            with open(transcript_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                duration = data.get("duration", 0.0)
                                segments = data.get("segments", [])
                                speakers = list({s.get("speaker", "SPEAKER_00") for s in segments})
                                if segments:
                                    preview_text = " ".join(s.get("text", "") for s in segments[:3])[:200]
                        except Exception:
                            pass
                    elif t_ext == '.srt':
                        try:
                            with open(transcript_file, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                segs = parse_srt_content(content)
                                if segs:
                                    duration = segs[-1]["end"]
                                    speakers = list({s["speaker"] for s in segs})
                                    preview_text = " ".join(s["text"] for s in segs[:3])[:200]
                        except Exception:
                            pass
                    elif t_ext == '.txt':
                        try:
                            with open(transcript_file, 'r', encoding='utf-8', errors='ignore') as f:
                                preview_text = f.read(250).replace('\n', ' ')
                        except Exception:
                            pass

                if duration <= 0.0:
                    duration = get_audio_duration(str(item))

                stat = item.stat()
                rel_file_path = os.path.relpath(str(item), library_dir)
                has_speakers = bool(speakers and len(speakers) > 0)
                has_raw_transcript = bool(has_transcript and not has_speakers)

                rec_entry = {
                    "id": rel_file_path,
                    "filename": fname,
                    "relative_path": rel_file_path,
                    "relative_folder": logical_rel,
                    "folder_name": os.path.basename(logical_rel) if logical_rel else "Root",
                    "base_name": base_name,
                    "ext": ext,
                    "full_path": str(item),
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "has_transcript": has_transcript,
                    "has_speakers": has_speakers,
                    "has_raw_transcript": has_raw_transcript,
                    "has_srt": has_srt,
                    "duration": round(duration, 2),
                    "speakers": speakers,
                    "preview_text": preview_text
                }
                recordings.append(rec_entry)
                
                # Increment counts for root, logical folder, and all ancestor directories
                folders_dict[""]["count"] += 1
                if logical_rel:
                    accum = ""
                    for p in logical_rel.split(os.sep):
                        accum = f"{accum}/{p}" if accum else p
                        if accum in folders_dict:
                            folders_dict[accum]["count"] += 1

    # Sort recordings descending by modification time
    recordings.sort(key=lambda x: x["modified_at"], reverse=True)

    # Convert folders dict to list sorted by path
    folder_list = sorted(list(folders_dict.values()), key=lambda f: f["path"])
    return {
        "recordings": recordings,
        "folders": folder_list
    }


# =====================================================================
# API Endpoints
# =====================================================================

DASHBOARD_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memosplit_studio_local_diarization_dashboard.html")

@app.get("/")
async def root():
    if os.path.exists(DASHBOARD_HTML_PATH):
        return FileResponse(DASHBOARD_HTML_PATH, media_type="text/html")
    return {"message": "MemoSplit Studio Engine is running", "status": "online"}

@app.get("/health")
async def health():
    return {
        "status": "online",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "library_dir": DEFAULT_LIBRARY_DIR
    }

@app.get("/api/library")
async def get_library(path: Optional[str] = None):
    lib_path = get_library_path(path)
    scan_result = scan_library(lib_path)
    recordings = scan_result["recordings"]
    folders = scan_result["folders"]
    return {
        "library_path": lib_path,
        "total_recordings": len(recordings),
        "total_folders": len(folders),
        "folders": folders,
        "recordings": recordings
    }

@app.get("/api/recordings/{filename:path}/audio")
async def stream_audio(filename: str, path: Optional[str] = None):
    lib_path = get_library_path(path)
    file_path = os.path.join(lib_path, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Determine MIME type
    ext = os.path.splitext(filename)[1].lower()
    media_types = {
        '.m4a': 'audio/mp4',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.aac': 'audio/aac',
        '.flac': 'audio/flac'
    }
    media_type = media_types.get(ext, 'application/octet-stream')
    return FileResponse(file_path, media_type=media_type, filename=os.path.basename(filename))

@app.get("/api/recordings/{filename:path}/transcript")
async def get_transcript(filename: str, path: Optional[str] = None):
    lib_path = get_library_path(path)
    audio_full_path = Path(lib_path) / filename
    
    # Locate companion transcript
    transcript_file = find_companion_transcript(audio_full_path)
    if not transcript_file:
        file_base = os.path.splitext(filename)[0]
        json_path = os.path.join(lib_path, f"{file_base}.json")
        srt_path = os.path.join(lib_path, f"{file_base}.srt")
        if os.path.exists(json_path):
            transcript_file = Path(json_path)
        elif os.path.exists(srt_path):
            transcript_file = Path(srt_path)

    if transcript_file and transcript_file.exists():
        t_ext = transcript_file.suffix.lower()
        if t_ext == '.json':
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading JSON transcript: {e}")
        elif t_ext == '.srt':
            try:
                with open(transcript_file, 'r', encoding='utf-8', errors='ignore') as f:
                    segments = parse_srt_content(f.read())
                    duration = segments[-1]["end"] if segments else 0.0
                    speakers = list({s["speaker"] for s in segments})
                    return {
                        "source": "srt",
                        "duration": duration,
                        "speakers": speakers,
                        "segments": segments
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error parsing SRT transcript: {e}")
        elif t_ext == '.txt':
            try:
                with open(transcript_file, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_txt = f.read()
                    # Synthetic segment for viewing
                    return {
                        "source": "txt",
                        "duration": 0.0,
                        "speakers": ["Speaker"],
                        "segments": [{
                            "id": 1,
                            "start": 0.0,
                            "end": 0.0,
                            "speaker": "Speaker",
                            "text": raw_txt
                        }]
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading TXT transcript: {e}")

    return {
        "status": "not_transcribed",
        "duration": 0,
        "speakers": [],
        "segments": []
    }

class TranscriptUpdateRequest(BaseModel):
    segments: List[Dict[str, Any]]
    duration: Optional[float] = None
    speakers: Optional[List[str]] = None
    speaker_metadata: Optional[Dict[str, Any]] = None

def save_transcript_files(lib_path: str, filename: str, segments: List[Dict[str, Any]], duration: Optional[float] = None, speaker_meta: Optional[Dict[str, Any]] = None) -> List[str]:
    file_base = os.path.splitext(filename)[0]
    dur = duration or (segments[-1]["end"] if segments else 0.0)
    meta = speaker_meta or {}

    parent_dir = os.path.dirname(os.path.join(lib_path, filename))
    os.makedirs(parent_dir, exist_ok=True)

    print(f"[*] Saving transcripts for '{filename}' ({len(segments)} segments) to: {parent_dir}")

    json_path = os.path.join(lib_path, f"{file_base}.json")
    json_data = {
        "filename": filename,
        "duration": dur,
        "speaker_metadata": meta,
        "segments": segments
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"[✓] Wrote: {json_path}")

    srt_path = os.path.join(lib_path, f"{file_base}.srt")
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(serialize_to_srt(segments, meta))
    print(f"[✓] Wrote: {srt_path}")

    vtt_path = os.path.join(lib_path, f"{file_base}.vtt")
    with open(vtt_path, 'w', encoding='utf-8') as f:
        f.write(serialize_to_vtt(segments, meta))
    print(f"[✓] Wrote: {vtt_path}")

    txt_path = os.path.join(lib_path, f"{file_base}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(serialize_to_txt(segments, meta))
    print(f"[✓] Wrote: {txt_path}")

    print(f"[✓] All transcript files saved for '{filename}'")
    return [f"{file_base}.json", f"{file_base}.srt", f"{file_base}.vtt", f"{file_base}.txt"]

@app.post("/api/recordings/{filename:path}/transcript")
@app.put("/api/recordings/{filename:path}/transcript")
async def save_transcript(filename: str, payload: TranscriptUpdateRequest, path: Optional[str] = None):
    lib_path = get_library_path(path)
    saved = save_transcript_files(lib_path, filename, payload.segments, payload.duration, payload.speaker_metadata)
    return {"status": "success", "saved_files": saved}

from starlette.concurrency import run_in_threadpool

def _checkpoint_path(file_path: str, phase: str) -> str:
    """Return path for a hidden checkpoint file next to the audio."""
    base = os.path.splitext(file_path)[0]
    return f"{base}.{phase}_checkpoint.json"

def _load_checkpoint(file_path: str, phase: str) -> Optional[Any]:
    """Load checkpoint data if it exists and is newer than the audio file."""
    cp = _checkpoint_path(file_path, phase)
    if not os.path.isfile(cp):
        return None
    try:
        audio_mtime = os.path.getmtime(file_path)
        cp_mtime = os.path.getmtime(cp)
        if cp_mtime < audio_mtime:
            print(f"[!] Checkpoint '{cp}' is older than audio — ignoring.")
            return None
        with open(cp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[✓] Loaded {phase} checkpoint ({cp}) — skipping phase.")
        return data
    except Exception as e:
        print(f"[!] Failed to load checkpoint {cp}: {e}")
        return None

def _save_checkpoint(file_path: str, phase: str, data: Any):
    """Write checkpoint data to disk next to the audio file."""
    cp = _checkpoint_path(file_path, phase)
    try:
        with open(cp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[✓] Checkpoint saved: {cp}")
    except Exception as e:
        print(f"[!] Warning: Failed to save checkpoint {cp}: {e}")

def _delete_checkpoints(file_path: str):
    """Clean up checkpoint files after a successful final save."""
    for phase in ["align", "diarize"]:
        cp = _checkpoint_path(file_path, phase)
        if os.path.isfile(cp):
            try:
                os.remove(cp)
                print(f"[✓] Removed checkpoint: {cp}")
            except Exception as e:
                print(f"[!] Could not remove checkpoint {cp}: {e}")

def _sync_diarize(file_path: str, min_speakers: int, max_speakers: int, token: Optional[str] = None, progress_cb: Optional[Any] = None, skip_transcription_if_exists: bool = True):
    print(f"[*] Loading audio for {file_path}...")
    audio = whisperx.load_audio(file_path)
    duration = round(audio.shape[0] / 16000, 2)
    d_model = get_diarize_model(token or HF_TOKEN)

    # ── Phase 1: Transcription ────────────────────────────────────────────────
    # Check for pre-existing companion transcript to skip Whisper ASR
    existing_source = None
    existing_segments = None
    if skip_transcription_if_exists:
        existing_source, existing_segments = find_existing_transcript_segments(Path(file_path), duration)

    if existing_segments and len(existing_segments) > 0:
        print(f"[*] Pre-transcribed {existing_source.upper()} found ({len(existing_segments)} segments). Skipping Whisper transcription step ➔ Jumping directly to alignment!")
        if progress_cb:
            progress_cb("aligning", 40, f"Found existing {existing_source.upper()} transcript ({len(existing_segments)} segments). Skipping Whisper transcription ➔ Aligning phonetic word timestamps...")
        unaligned_segments = existing_segments
    else:
        w_model = get_whisper_model()
        if progress_cb:
            progress_cb("transcribing", 15, "Transcribing speech with WhisperX...")
        print(f"[*] Starting Whisper transcription for {file_path}...")
        asr_result = w_model.transcribe(audio, batch_size=BATCH_SIZE)
        unaligned_segments = asr_result.get("segments", [])

    # ── Phase 2: Phonetic Alignment (Wav2Vec2) ────────────────────────────────
    align_cp = _load_checkpoint(file_path, "align")
    if align_cp is not None:
        print(f"[*] Resuming from align checkpoint — skipping Wav2Vec2 alignment.")
        if progress_cb:
            progress_cb("clustering", 72, "Resuming from alignment checkpoint ➔ Running Pyannote speaker diarization...")
        aligned_result = align_cp  # already a dict with "segments" key
    else:
        if progress_cb:
            progress_cb("aligning", 55, "Aligning phonetic word timestamps with Wav2Vec2...")
        print(f"[*] Phonetic alignment...")
        align_model, metadata = whisperx.load_align_model(
            language_code="en",
            device=DEVICE
        )
        aligned_result = whisperx.align(unaligned_segments, align_model, metadata, audio, DEVICE, return_char_alignments=False)
        # Checkpoint: save aligned result so we can skip alignment on resume
        _save_checkpoint(file_path, "align", aligned_result)

    # ── Phase 3: Speaker Diarization (Pyannote) ───────────────────────────────
    diarize_cp = _load_checkpoint(file_path, "diarize")
    if diarize_cp is not None:
        print(f"[*] Resuming from diarize checkpoint — skipping Pyannote.")
        if progress_cb:
            progress_cb("done", 92, "Resuming from Pyannote checkpoint ➔ Finalizing segments...")
        diarized_result = diarize_cp
    else:
        if progress_cb:
            progress_cb("clustering", 72, "Clustering speaker voice profiles with Pyannote...")
        t0_dia = time.time()
        print(f"[*] Starting Pyannote neural speaker segmentation and clustering...")
        if d_model is not None:
            # Custom progress hook to stream progressive percent (72% -> 92%)
            def _pyannote_hook(step_name, step_artifact=None, file=None, total=None, completed=None):
                if total and completed is not None and total > 0:
                    frac = min(1.0, completed / total)
                    pct = int(72 + frac * 20)
                    step_lbl = str(step_name).split('.')[-1] if step_name else 'diarization'
                    if progress_cb:
                        progress_cb("clustering", pct, f"Clustering voice profiles ({step_lbl}: {int(frac*100)}%)...")

            try:
                if hasattr(d_model, 'model') and callable(getattr(d_model, 'model', None)):
                    audio_data = {
                        'waveform': torch.from_numpy(audio[None, :]),
                        'sample_rate': SAMPLE_RATE
                    }
                    raw_segments = d_model.model(
                        audio_data,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        hook=_pyannote_hook
                    )
                    diarize_segments = pd.DataFrame(raw_segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
                    diarize_segments['start'] = diarize_segments['segment'].apply(lambda x: x.start)
                    diarize_segments['end'] = diarize_segments['segment'].apply(lambda x: x.end)
                else:
                    diarize_segments = d_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            except Exception as e:
                print(f"[!] Hooked diarization fallback to standard call: {e}")
                diarize_segments = d_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

            dia_elapsed = time.time() - t0_dia
            print(f"[✓] Pyannote completed in {dia_elapsed:.1f}s. Assigning word speakers to segments...")
            diarized_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        else:
            diarized_result = aligned_result
        # Checkpoint: save diarized result so we can skip Pyannote on resume
        _save_checkpoint(file_path, "diarize", diarized_result)

    # ── Phase 4: Merge & finalize segments ───────────────────────────────────
    segments = []
    for idx, seg in enumerate(diarized_result.get("segments", []), 1):
        orig_spk = None
        if idx - 1 < len(unaligned_segments):
            orig_spk = unaligned_segments[idx - 1].get("speaker")

        spk = seg.get("speaker") or orig_spk or "SPEAKER_00"
        segments.append({
            "id": idx,
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "speaker": spk,
            "text": seg.get("text", "").strip()
        })

    if progress_cb:
        progress_cb("done", 100, f"Diarization complete! {len(segments)} segments processed.")

    return duration, segments

def _gemini_diarize(file_path: str, api_key: Optional[str] = None, model: Optional[str] = None, progress_cb: Optional[Any] = None) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Fast Cloud Diarization using Google Gemini Multimodal Audio API.
    Uploads audio via Google AI Studio Files API, prompts Gemini with structured JSON output,
    and returns timestamped segments labeled by speaker in ~10-20 seconds.
    Includes automatic failover across Flash-Lite and Flash models if high-demand (503) occurs.
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("Google Gemini API Key is required for Cloud Diarization.")

    chosen_model = model or GEMINI_MODEL or "gemini-flash-lite-latest"
    # Pool of candidate models to fallback through if 503 / 429 occurs
    candidate_pool = [chosen_model, "gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-latest"]
    models_to_try = []
    for m in candidate_pool:
        if m not in models_to_try:
            models_to_try.append(m)

    file_size = os.path.getsize(file_path)
    file_dur = get_audio_duration(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = {
        '.m4a': 'audio/mp4',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.aac': 'audio/aac',
        '.flac': 'audio/flac'
    }.get(ext, 'audio/mp4')

    print(f"[*] [Gemini Cloud] Starting cloud diarization for '{file_path}' ({file_size / (1024*1024):.1f} MB, {file_dur:.1f}s, Primary Model: {chosen_model})...")
    if progress_cb:
        progress_cb("uploading", 20, f"Uploading audio to Google AI Cloud ({file_size / (1024*1024):.1f} MB)...")

    # 1. Start Resumable Upload
    upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={key}"
    headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json"
    }
    metadata = {"file": {"display_name": os.path.basename(file_path), "mimeType": mime_type}}
    start_res = requests.post(upload_url, headers=headers, json=metadata, timeout=30)
    if start_res.status_code != 200:
        raise RuntimeError(f"Failed initiating Gemini file upload: {start_res.text}")

    upload_endpoint = start_res.headers.get("X-Goog-Upload-URL")
    if not upload_endpoint:
        raise RuntimeError("No upload URL returned by Gemini Files API")

    # 2. Upload file bytes
    if progress_cb:
        progress_cb("uploading", 45, "Streaming audio payload to Gemini...")

    upload_headers = {
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"
    }
    with open(file_path, "rb") as f:
        upload_res = requests.post(upload_endpoint, headers=upload_headers, data=f, timeout=300)
    
    if upload_res.status_code != 200:
        raise RuntimeError(f"Gemini file upload failed ({upload_res.status_code}): {upload_res.text}")

    file_info = upload_res.json().get("file", {})
    file_uri = file_info.get("uri")
    file_name = file_info.get("name")
    print(f"[✓] [Gemini Cloud] Uploaded file successfully. URI: {file_uri} (mimeType: {mime_type})")

    # 3. Wait until file state is ACTIVE
    if progress_cb:
        progress_cb("cloud_diarizing", 65, "Gemini Flash neural audio model analyzing speech & speakers...")

    max_wait = 45
    waited = 0
    while waited < max_wait:
        state_res = requests.get(f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={key}", timeout=15)
        if state_res.status_code == 200:
            st = state_res.json().get("state")
            if st == "ACTIVE":
                break
            elif st == "FAILED":
                raise RuntimeError("Gemini audio processing failed on server")
        time.sleep(2)
        waited += 2

    # 4. Generate Content (Transcription + Speaker Identification)
    # Check if a reference text transcript exists to assist Gemini
    ref_txt = ""
    companion = find_companion_transcript(Path(file_path))
    if companion and companion.suffix.lower() == '.txt':
        try:
            with open(companion, 'r', encoding='utf-8', errors='ignore') as f:
                ref_txt = f.read(8000)
        except Exception:
            pass

    prompt_lines = [
        "You are an expert audio transcriber and speaker identification engine.",
        "Transcribe the provided audio recording verbatim, identifying each distinct speaker and assigning start and end timestamps in seconds.",
        "",
        "Return ONLY a JSON array of segment objects conforming exactly to this schema:",
        "[",
        "  {",
        '    "id": 1,',
        '    "start": 0.0,',
        '    "end": 4.5,',
        '    "speaker": "Speaker 1",',
        '    "text": "Spoken words verbatim..."',
        "  }",
        "]",
        "",
        "Rules:",
        '- If a speaker\'s actual real name is introduced, addressed, or obvious from the conversation (e.g. "Bob", "LaShawn", "Greg", "Brian", "Troy"), use their actual human name for the \'speaker\' field. Otherwise use "Speaker 1", "Speaker 2", etc.',
        "- Timestamps must be numeric seconds with 1-2 decimal places.",
        "- Segments must be contiguous and capture every sentence or spoken thought.",
        "- Output pure JSON array without markdown formatting."
    ]

    if ref_txt:
        prompt_lines.append("")
        prompt_lines.append(f"Reference Transcript Text (for accuracy):\n{ref_txt}")

    prompt = "\n".join(prompt_lines)

    last_err_msg = ""
    parsed_segments = None
    successful_model = None

    for current_model in models_to_try:
        for attempt in range(2):
            if progress_cb:
                progress_cb("cloud_diarizing", 80, f"Identifying speakers via {current_model}...")

            gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "response_mime_type": "application/json"
                }
            }

            t0_gen = time.time()
            try:
                gen_res = requests.post(gen_url, json=payload, timeout=180)
                if gen_res.status_code == 200:
                    gen_data = gen_res.json()
                    candidates = gen_data.get("candidates", [])
                    if candidates:
                        raw_text = candidates[0]["content"]["parts"][0]["text"].strip()
                        raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text)
                        raw_text = re.sub(r'\s*```$', '', raw_text)
                        parsed = json.loads(raw_text)
                        if isinstance(parsed, dict) and "segments" in parsed:
                            parsed = parsed["segments"]
                        if isinstance(parsed, list):
                            parsed_segments = parsed
                            successful_model = current_model
                            print(f"[✓] [Gemini Cloud] Speaker identification succeeded with '{current_model}' in {time.time() - t0_gen:.1f}s.")
                            break
                else:
                    last_err_msg = f"Model '{current_model}' HTTP {gen_res.status_code}: {gen_res.text[:120]}"
                    print(f"[!] [Gemini Cloud] {last_err_msg}. (Attempt {attempt+1}/2)")
                    if progress_cb:
                        progress_cb("cloud_diarizing", 85, f"'{current_model}' busy ({gen_res.status_code}), retrying/switching...")
                    time.sleep(2)
            except Exception as e:
                last_err_msg = f"Model '{current_model}' exception: {e}"
                print(f"[!] [Gemini Cloud] {last_err_msg}. (Attempt {attempt+1}/2)")
                time.sleep(2)

        if parsed_segments is not None:
            break

    if parsed_segments is None:
        raise RuntimeError(f"All Gemini models failed. Last error: {last_err_msg}")

    segments = []
    for idx, s in enumerate(parsed_segments, 1):
        spk = s.get("speaker") or "Speaker 1"
        segments.append({
            "id": idx,
            "start": round(float(s.get("start", 0.0)), 2),
            "end": round(float(s.get("end", 0.0)), 2),
            "speaker": spk.strip(),
            "text": str(s.get("text", "")).strip()
        })

    calc_dur = max(file_dur, (segments[-1]["end"] if segments else 0.0))
    print(f"[✓] [Gemini Cloud] Finished successfully using {successful_model}. Generated {len(segments)} speaker segments.")

    # 5. Clean up remote file from Gemini Cloud to free quota
    try:
        requests.delete(f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={key}", timeout=10)
    except Exception:
        pass

    if progress_cb:
        progress_cb("done", 100, f"Gemini Diarization complete! {len(segments)} segments generated.")

    return calc_dur, segments

# Active single-job progress tracker
active_single_job: Dict[str, Any] = {
    "status": "idle",
    "filename": None,
    "audio_duration": 0.0,
    "stage": "",
    "stage_detail": "",
    "percent": 0,
    "start_time": 0.0,
    "elapsed": 0.0,
    "eta": 0.0
}

class BatchDiarizationManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle" # "idle", "running", "paused", "completed", "cancelled"
        self.queue: List[str] = []
        self.current_file: Optional[str] = None
        self.current_stage: str = ""
        self.current_stage_detail: str = ""
        self.current_percent: int = 0
        self.current_audio_duration: float = 0.0
        self.current_start_time: float = 0.0
        self.current_elapsed: float = 0.0
        self.current_eta: float = 0.0
        
        self.batch_total_count: int = 0
        self.batch_current_index: int = 0
        self.batch_completed_files: List[str] = []
        self.batch_failed_files: List[Dict[str, str]] = []
        self.batch_start_time: float = 0.0
        self.batch_elapsed_total: float = 0.0
        self.batch_total_audio_duration: float = 0.0
        self.batch_remaining_audio_duration: float = 0.0
        self.batch_eta_total: float = 0.0
        
        self.throttle_delay: int = 10
        self.library_path: str = DEFAULT_LIBRARY_DIR
        self.engine: str = DEFAULT_ENGINE
        self._thread: Optional[threading.Thread] = None
        self._stop_requested: bool = False
        self._pause_requested: bool = False

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            el_batch = round(time.time() - self.batch_start_time, 1) if (self.batch_start_time and self.status in ["running", "paused"]) else self.batch_elapsed_total
            el_curr = round(time.time() - self.current_start_time, 1) if (self.current_start_time and self.status == "running") else self.current_elapsed
            return {
                "status": self.status,
                "engine": self.engine,
                "current_file": self.current_file,
                "current_stage": self.current_stage,
                "current_stage_detail": self.current_stage_detail,
                "current_percent": self.current_percent,
                "current_audio_duration": round(self.current_audio_duration, 1),
                "current_elapsed": el_curr,
                "current_eta": self.current_eta,
                "batch_total_count": self.batch_total_count,
                "batch_current_index": self.batch_current_index,
                "batch_completed_count": len(self.batch_completed_files),
                "batch_failed_count": len(self.batch_failed_files),
                "batch_completed_files": self.batch_completed_files[-5:],
                "batch_elapsed_total": el_batch,
                "batch_total_audio_duration": round(self.batch_total_audio_duration, 1),
                "batch_remaining_audio_duration": round(self.batch_remaining_audio_duration, 1),
                "batch_eta_total": self.batch_eta_total,
                "throttle_delay": self.throttle_delay,
                "is_paused": self._pause_requested,
                "queue_length": len(self.queue)
            }

    def _recalculate_batch_eta(self):
        rem_queue_audio = 0.0
        rem_weighted_time = 0.0
        for f in self.queue:
            fp = os.path.join(self.library_path, f)
            d = get_audio_duration(fp)
            cf = find_companion_transcript(Path(fp))
            rem_queue_audio += d
            if self.engine == "gemini":
                rem_weighted_time += 18.0
            else:
                rem_weighted_time += (d * 0.28) if cf else (d * 0.7)
        self.batch_remaining_audio_duration = rem_queue_audio
        total_eta = self.current_eta + rem_weighted_time + (len(self.queue) * (self.throttle_delay if self.engine != "gemini" else 2))
        self.batch_eta_total = max(0.0, round(total_eta, 1))

    def start_batch(self, files: Optional[List[str]] = None, throttle_delay: int = 10, path: Optional[str] = None, engine: Optional[str] = None):
        with self.lock:
            if self.status in ["running", "paused"] and self._thread and self._thread.is_alive():
                return {"status": "error", "message": "A batch job is already running"}

            self.library_path = get_library_path(path)
            self.throttle_delay = max(0, throttle_delay)
            self.engine = engine or DEFAULT_ENGINE or ("gemini" if GEMINI_API_KEY else "local")
            self._stop_requested = False
            self._pause_requested = False

            if files and len(files) > 0:
                self.queue = list(files)
            else:
                scan = scan_library(self.library_path)
                self.queue = [
                    r["relative_path"] for r in scan["recordings"]
                    if not r.get("has_transcript")
                ]

            if not self.queue:
                return {"status": "empty", "message": "No voice memos found needing diarization"}

            self.batch_total_count = len(self.queue)
            self.batch_current_index = 0
            self.batch_completed_files = []
            self.batch_failed_files = []
            self.batch_start_time = time.time()
            self.batch_elapsed_total = 0.0

            total_dur = 0.0
            calc_eta = 0.0
            for f in self.queue:
                fp = os.path.join(self.library_path, f)
                d = get_audio_duration(fp)
                cf = find_companion_transcript(Path(fp))
                total_dur += d
                if self.engine == "gemini":
                    calc_eta += 18.0
                else:
                    calc_eta += (d * 0.28) if cf else (d * 0.7)
            self.batch_total_audio_duration = total_dur
            self.batch_remaining_audio_duration = total_dur
            self.batch_eta_total = round(calc_eta + (len(self.queue) * (self.throttle_delay if self.engine != "gemini" else 2)), 1)

            self.status = "running"
            self.current_file = None
            self.current_stage = "queued"
            self.current_stage_detail = f"Queued {self.batch_total_count} voice memos for {self.engine.upper()} processing..."

            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

            return {
                "status": "started",
                "engine": self.engine,
                "total_files": self.batch_total_count,
                "total_audio_duration": self.batch_total_audio_duration,
                "estimated_batch_time": self.batch_eta_total
            }

    def pause(self):
        with self.lock:
            if self.status == "running":
                self._pause_requested = True
                return {"status": "pausing", "message": "Batch will pause after current memo"}
        return {"status": self.status}

    def resume(self):
        with self.lock:
            if self._pause_requested or self.status == "paused":
                self._pause_requested = False
                self.status = "running"
                return {"status": "resumed", "message": "Batch resumed"}
        return {"status": self.status}

    def cancel(self):
        with self.lock:
            self._stop_requested = True
            self.status = "cancelled"
            return {"status": "cancelled", "message": "Batch cancelled"}

    def _worker_loop(self):
        print(f"[*] Batch Diarizer worker started with {self.batch_total_count} files (Engine: {self.engine})...")
        while self.queue and not self._stop_requested:
            while self._pause_requested and not self._stop_requested:
                with self.lock:
                    self.status = "paused"
                time.sleep(1)

            if self._stop_requested:
                break

            with self.lock:
                filename = self.queue.pop(0)
                self.current_file = filename
                self.batch_current_index += 1
                self.status = "running"
                self.current_percent = 5
                self.current_stage = "loading"
                self.current_stage_detail = f"Loading memo {self.batch_current_index} of {self.batch_total_count}..."
                self.current_start_time = time.time()
                self.current_elapsed = 0.0

            file_path = os.path.join(self.library_path, filename)
            dur = get_audio_duration(file_path)
            companion = find_companion_transcript(Path(file_path))
            has_pre = companion is not None
            c_ext = companion.suffix.lower().lstrip('.') if companion else ""
            with self.lock:
                self.current_audio_duration = dur
                if self.engine == "gemini":
                    self.current_eta = 18.0
                    self.current_stage = "uploading"
                    self.current_stage_detail = "⚡ Google Gemini Cloud: Streaming audio to Gemini..."
                    self.current_percent = 20
                else:
                    self.current_eta = max(1.0, round(dur * 0.28, 1)) if has_pre else max(1.0, round(dur * 0.7, 1))
                    if has_pre:
                        self.current_stage = "aligning"
                        self.current_stage_detail = f"Pre-transcribed {c_ext.upper()} detected. Skipping Whisper transcription ➔ Aligning..."
                        self.current_percent = 35
                self._recalculate_batch_eta()

            def progress_cb(stage: str, percent: int, detail: str):
                with self.lock:
                    self.current_stage = stage
                    self.current_percent = percent
                    self.current_stage_detail = detail
                    el = time.time() - self.current_start_time
                    self.current_elapsed = round(el, 1)
                    if self.engine == "gemini":
                        self.current_eta = max(0.0, round(20.0 - el, 1))
                    else:
                        mult = 0.28 if has_pre else 0.7
                        if 5 < percent < 95:
                            est_tot = el / (percent / 100.0)
                            self.current_eta = max(0.0, round(est_tot - el, 1))
                        elif self.current_audio_duration > 0:
                            self.current_eta = max(0.0, round(self.current_audio_duration * mult - el, 1))
                    self._recalculate_batch_eta()

            try:
                if self.engine == "gemini" and GEMINI_API_KEY:
                    try:
                        duration, segments = _gemini_diarize(file_path, GEMINI_API_KEY, model=GEMINI_MODEL, progress_cb=progress_cb)
                    except Exception as g_err:
                        print(f"[!] [Gemini Cloud] Failed on '{filename}': {g_err}. Gracefully falling back to Local Engine...")
                        progress_cb("aligning" if has_pre else "transcribing", 30, "Cloud busy (503). Auto-switching to Local CPU Engine...")
                        duration, segments = _sync_diarize(file_path, min_speakers=1, max_speakers=5, token=HF_TOKEN, progress_cb=progress_cb, skip_transcription_if_exists=True)
                        _delete_checkpoints(file_path)
                else:
                    duration, segments = _sync_diarize(file_path, min_speakers=1, max_speakers=5, token=HF_TOKEN, progress_cb=progress_cb, skip_transcription_if_exists=True)
                    _delete_checkpoints(file_path)

                save_transcript_files(self.library_path, filename, segments, duration)
                with self.lock:
                    self.batch_completed_files.append(filename)
                    self.current_percent = 100
                    self.current_stage = "done"
                    self.current_stage_detail = f"Completed ({len(segments)} segments)"
                    self._recalculate_batch_eta()
            except Exception as e:
                print(f"[!] Batch item error on {filename}: {e}")
                with self.lock:
                    self.batch_failed_files.append({"file": filename, "error": str(e)})

            if self.queue and self.throttle_delay > 0 and not self._stop_requested:
                with self.lock:
                    self.current_stage = "cooldown"
                    self.current_stage_detail = f"Cooling down ({self.throttle_delay}s) to keep CPU quiet..."
                for _ in range(self.throttle_delay):
                    if self._stop_requested or self._pause_requested:
                        break
                    time.sleep(1)

        with self.lock:
            if self._stop_requested:
                self.status = "cancelled"
            else:
                self.status = "completed"
                self.current_stage = "completed"
                self.current_stage_detail = f"Batch finished! {len(self.batch_completed_files)} memos diarized."
            self.current_file = None
            self.batch_elapsed_total = round(time.time() - self.batch_start_time, 1) if self.batch_start_time else 0.0
            self.batch_eta_total = 0.0
            self.current_eta = 0.0

batch_manager = BatchDiarizationManager()

class BatchStartRequest(BaseModel):
    files: Optional[List[str]] = None
    throttle_delay: Optional[int] = 10
    engine: Optional[str] = None
    path: Optional[str] = None

@app.get("/api/batch/status")
async def get_batch_status():
    return batch_manager.get_status()

@app.post("/api/batch/start")
async def start_batch(payload: Optional[BatchStartRequest] = Body(None)):
    files = payload.files if payload else None
    throttle = payload.throttle_delay if (payload and payload.throttle_delay is not None) else 10
    engine = payload.engine if payload else None
    path = payload.path if payload else None
    return batch_manager.start_batch(files=files, throttle_delay=throttle, path=path, engine=engine)

@app.post("/api/batch/pause")
async def pause_batch():
    return batch_manager.pause()

@app.post("/api/batch/resume")
async def resume_batch():
    return batch_manager.resume()

@app.post("/api/batch/cancel")
async def cancel_batch():
    return batch_manager.cancel()

@app.get("/api/job/status")
async def get_job_status():
    batch_status = batch_manager.get_status()
    if batch_status["status"] in ["running", "paused", "pausing"]:
        return {"active": True, "type": "batch", "data": batch_status}
    elif active_single_job.get("status") in ["running", "completed", "failed"]:
        el = round(time.time() - active_single_job["start_time"], 1) if active_single_job.get("start_time") else 0.0
        job_copy = dict(active_single_job)
        if active_single_job.get("status") == "running":
            job_copy["elapsed"] = el
        return {"active": active_single_job.get("status") == "running", "type": "single", "data": job_copy}
    return {"active": False, "type": "idle", "data": batch_status}

class SettingsUpdateRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    default_engine: Optional[str] = None

@app.get("/api/settings")
async def get_settings():
    global GEMINI_API_KEY, GEMINI_MODEL, DEFAULT_ENGINE
    return {
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_api_key_preview": f"{GEMINI_API_KEY[:8]}...{GEMINI_API_KEY[-6:]}" if GEMINI_API_KEY else "",
        "gemini_model": GEMINI_MODEL,
        "default_engine": DEFAULT_ENGINE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "library_dir": DEFAULT_LIBRARY_DIR
    }

@app.post("/api/settings")
async def update_settings(payload: SettingsUpdateRequest):
    global GEMINI_API_KEY, GEMINI_MODEL, DEFAULT_ENGINE
    if payload.gemini_api_key is not None:
        GEMINI_API_KEY = payload.gemini_api_key.strip()
    if payload.gemini_model is not None:
        GEMINI_MODEL = payload.gemini_model.strip()
    if payload.default_engine is not None:
        DEFAULT_ENGINE = payload.default_engine.strip()
    return {
        "status": "success",
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "default_engine": DEFAULT_ENGINE
    }

@app.post("/api/recordings/{filename:path}/diarize")
async def diarize_library_recording(
    filename: str,
    min_speakers: int = Query(1),
    max_speakers: int = Query(5),
    skip_transcription_if_exists: bool = Query(True),
    engine: Optional[str] = Query(None),
    path: Optional[str] = None
):
    lib_path = get_library_path(path)
    file_path = os.path.join(lib_path, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found in library")

    # Reject if a job is already running
    if active_single_job.get("status") == "running":
        raise HTTPException(status_code=409, detail="A diarization job is already running")

    selected_engine = engine or DEFAULT_ENGINE or ("gemini" if GEMINI_API_KEY else "local")
    dur = get_audio_duration(file_path)
    start_t = time.time()

    # Pre-check for existing companion transcript and phase checkpoints
    companion = find_companion_transcript(Path(file_path)) if skip_transcription_if_exists else None
    has_pre = companion is not None
    comp_ext = companion.suffix.lower().lstrip('.') if companion else ""

    align_cp_exists = _load_checkpoint(file_path, "align") is not None
    diarize_cp_exists = _load_checkpoint(file_path, "diarize") is not None

    if selected_engine == "gemini" and GEMINI_API_KEY:
        init_stage = "uploading"
        init_detail = "⚡ Google Gemini Cloud: Streaming audio to Gemini..."
        init_pct = 20
        init_eta = 18.0
    elif diarize_cp_exists:
        init_stage = "done"
        init_detail = "Found Pyannote checkpoint! Resuming to final saving phase..."
        init_pct = 90
        init_eta = 3.0
    elif align_cp_exists:
        init_stage = "clustering"
        init_detail = "Found alignment checkpoint! Skipping Wav2Vec2 ➔ Running Pyannote..."
        init_pct = 70
        init_eta = max(1.0, round(dur * 0.15, 1))
    elif has_pre:
        init_stage = "aligning"
        init_detail = f"Pre-transcribed {comp_ext.upper()} detected. Skipping Whisper transcription ➔ Aligning..."
        init_pct = 35
        init_eta = max(1.0, round(dur * 0.28, 1))
    else:
        init_stage = "transcribing"
        init_detail = "Transcribing speech with WhisperX (Local CPU)..."
        init_pct = 15
        init_eta = max(1.0, round(dur * 0.7, 1))

    active_single_job.update({
        "status": "running",
        "filename": filename,
        "engine": selected_engine,
        "audio_duration": dur,
        "skipped_transcription": has_pre if selected_engine == "local" else False,
        "transcript_source": comp_ext if selected_engine == "local" else "gemini_cloud",
        "stage": init_stage,
        "stage_detail": init_detail,
        "percent": init_pct,
        "start_time": start_t,
        "elapsed": 0.0,
        "eta": init_eta,
        "result": None,
        "error": None,
    })

    def single_progress_cb(stage: str, percent: int, detail: str):
        el = time.time() - start_t
        if selected_engine == "gemini":
            eta = max(0.0, round(20.0 - el, 1))
        else:
            mult = 0.28 if has_pre else 0.7
            eta = max(0.0, round((el / (percent / 100.0)) - el, 1)) if (5 < percent < 95) else max(0.0, round(dur * mult - el, 1))
        active_single_job.update({
            "stage": stage,
            "percent": percent,
            "stage_detail": detail,
            "elapsed": round(el, 1),
            "eta": eta
        })

    async def _run_job():
        try:
            used_engine = selected_engine
            if selected_engine == "gemini" and GEMINI_API_KEY:
                try:
                    duration, segments = await run_in_threadpool(
                        _gemini_diarize, file_path, GEMINI_API_KEY, GEMINI_MODEL, single_progress_cb
                    )
                except Exception as g_err:
                    print(f"[!] [Gemini Cloud] Failed on '{filename}': {g_err}. Falling back to Local Engine...")
                    single_progress_cb("aligning" if has_pre else "transcribing", 30, "Cloud busy (503). Auto-switching to Local CPU Engine...")
                    used_engine = "local (cloud fallback)"
                    duration, segments = await run_in_threadpool(
                        _sync_diarize, file_path, min_speakers, max_speakers, HF_TOKEN, single_progress_cb, skip_transcription_if_exists
                    )
                    _delete_checkpoints(file_path)
            else:
                duration, segments = await run_in_threadpool(
                    _sync_diarize, file_path, min_speakers, max_speakers, HF_TOKEN, single_progress_cb, skip_transcription_if_exists
                )
                _delete_checkpoints(file_path)

            save_transcript_files(lib_path, filename, segments, duration)
            active_single_job.update({
                "status": "completed",
                "percent": 100,
                "stage": "done",
                "stage_detail": f"Saved {len(segments)} segments to MyCloud.",
                "result": {
                    "status": "success",
                    "duration": duration,
                    "segments": segments,
                    "engine": selected_engine,
                    "skipped_transcription": has_pre if selected_engine == "local" else False,
                    "transcript_source": comp_ext if selected_engine == "local" else "gemini_cloud"
                }
            })
            print(f"[✓] Single job ({selected_engine}) complete for '{filename}' ({len(segments)} segments)")
        except Exception as e:
            print(f"[!] Single job FAILED for '{filename}': {e}")
            active_single_job.update({
                "status": "failed",
                "error": str(e),
                "stage_detail": f"Error: {e}"
            })

    # Fire off the job — return immediately so the HTTP connection closes
    asyncio.create_task(_run_job())
    return {
        "status": "accepted",
        "filename": filename,
        "engine": selected_engine,
        "skipped_transcription": has_pre if selected_engine == "local" else False,
        "transcript_source": comp_ext if selected_engine == "local" else "gemini_cloud"
    }

@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    min_speakers: int = Form(1),
    max_speakers: int = Form(5),
    hf_token: str = Form(None)
):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        duration, segments = await run_in_threadpool(_sync_diarize, temp_path, min_speakers, max_speakers, hf_token)

        # Also auto-persist into default library folder
        try:
            lib_dest = os.path.join(DEFAULT_LIBRARY_DIR, file.filename)
            shutil.copyfile(temp_path, lib_dest)
            update_req = TranscriptUpdateRequest(segments=segments, duration=duration)
            await save_transcript(file.filename, update_req, path=DEFAULT_LIBRARY_DIR)
        except Exception as e:
            print(f"[!] Note: Could not auto-archive into library: {e}")

        return {"status": "success", "duration": duration, "segments": segments}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    print(f"[*] MemoSplit Engine running on http://localhost:8080")
    print(f"[*] Default Library Directory: {DEFAULT_LIBRARY_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
