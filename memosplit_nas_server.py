#!/usr/bin/env python3
"""
MemoSplit Studio - Lightweight NAS Server (WD My Cloud EX2 Ultra Edition)
========================================================================
Zero-dependency, standalone HTTP server running directly on Linux / My Cloud OS.
- Serves the MemoSplit Studio Web Dashboard.
- Scans local NAS shares (/shares/Public/Kyle/voice clips, /shares/Public, etc.).
- Streams audio with HTTP Range (206 Partial Content) support for iOS Safari.
- Reads & writes companion transcripts (.srt, .json, .vtt, .txt).
- Proxies or coordinates Google Gemini Multimodal Cloud AI diarization.
"""

import os
import sys
import re
import time
import json
import mimetypes
import urllib.request
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8088"))
HOST = os.getenv("HOST", "0.0.0.0")

# Register Apple Voice Memos QuickTime Audio (.qta) and MPEG-4 Audio (.m4a)
mimetypes.add_type("audio/mp4", ".qta")
mimetypes.add_type("audio/mp4", ".m4a")

AUDIO_EXTENSIONS = {".m4a", ".qta", ".mp3", ".wav", ".aac", ".ogg", ".flac"}

# Candidate root paths for voice memos on My Cloud NAS
DEFAULT_SEARCH_PATHS = [
    "/shares/Public/Kyle/voice clips",
    "/shares/Public/voice clips",
    "/shares/Public",
    "/Volumes/Public/Kyle/voice clips",
    "/Volumes/Public/voice clips",
    "/Volumes/Public",
    os.path.abspath("./library"),
    os.path.abspath(".")
]

def get_default_media_root():
    for p in DEFAULT_SEARCH_PATHS:
        if os.path.isdir(p):
            return p
    return os.path.abspath(".")

MEDIA_ROOT = os.getenv("MEMOSPLIT_MEDIA_ROOT", get_default_media_root())
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Transcript & SRT Helpers
# -----------------------------------------------------------------------------
def parse_srt_time(time_str: str) -> float:
    time_str = time_str.strip().replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return 0.0

def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def parse_srt_content(content: str):
    clean = re.sub(r'^\ufeff', '', content).replace('\r\n', '\n').replace('\r', '\n').strip()
    blocks = re.split(r'\n\s*\n', clean)
    segs = []
    speaker_regex = re.compile(r"^\[?(?:Speaker\s*([0-9A-Za-z_-]+)|([A-Za-z0-9\s._\'-]{1,50}))\]?\s*:\s*(.*)", re.DOTALL)
    current_speaker = "Speaker 1"

    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
        time_line_idx = -1
        for j, line in enumerate(lines):
            if '-->' in line:
                time_line_idx = j
                break
        if time_line_idx == -1:
            continue

        time_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?(?:[,\.]\d{1,3})?)\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[,\.]\d{1,3})?)', lines[time_line_idx])
        if not time_match:
            continue

        start_sec = parse_srt_time(time_match.group(1))
        end_sec = parse_srt_time(time_match.group(2))
        raw_text = " ".join(lines[time_line_idx + 1:]).strip()
        if not raw_text:
            continue

        spk_match = speaker_regex.match(raw_text)
        if spk_match:
            g1, g2, content_text = spk_match.groups()
            speaker_name = (g1 or g2 or "").strip()
            if speaker_name:
                if speaker_name.isdigit():
                    current_speaker = f"Speaker {int(speaker_name)}"
                else:
                    current_speaker = speaker_name
            text = content_text.strip()
        else:
            text = raw_text

        segs.append({
            "id": len(segs) + 1,
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "speaker": current_speaker,
            "text": text
        })
    return segs

def serialize_to_srt(segments, speaker_meta=None) -> str:
    lines = []
    meta = speaker_meta or {}
    for i, seg in enumerate(segments, 1):
        spk_key = seg.get("speaker", "Speaker 1")
        display_label = spk_key
        if spk_key in meta and meta[spk_key].get("label"):
            display_label = meta[spk_key]["label"]
        start_str = format_srt_time(seg.get("start", 0))
        end_str = format_srt_time(seg.get("end", 0))
        text = seg.get("text", "")
        lines.append(f"{i}\n{start_str} --> {end_str}\n[{display_label}]: {text}\n")
    return "\n".join(lines)

def format_vtt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def serialize_to_vtt(segments, speaker_meta=None) -> str:
    out = ["WEBVTT\n"]
    meta = speaker_meta or {}
    for seg in segments:
        spk_key = seg.get("speaker", "Speaker 1")
        label = meta.get(spk_key, {}).get("label", spk_key)
        start_str = format_vtt_time(seg.get("start", 0))
        end_str = format_vtt_time(seg.get("end", 0))
        out.append(f"{start_str} --> {end_str}\n<v {label}>{seg.get('text', '').strip()}\n")
    return "\n".join(out)

def serialize_to_txt(segments, speaker_meta=None) -> str:
    out = []
    meta = speaker_meta or {}
    for seg in segments:
        spk_key = seg.get("speaker", "Speaker 1")
        label = meta.get(spk_key, {}).get("label", spk_key)
        start_sec = seg.get("start", 0)
        mins = int(start_sec // 60)
        secs = int(start_sec % 60)
        time_str = f"[{mins:02d}:{secs:02d}]"
        out.append(f"{label} {time_str}:\n{seg.get('text', '').strip()}\n\n")
    return "".join(out)

def safe_write_file(file_path: str, content, is_json: bool = False):
    """Atomic write to disk to prevent partial corruption or zero-byte file writes."""
    tmp_path = f"{file_path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            if is_json:
                json.dump(content, f, indent=2, ensure_ascii=False)
            else:
                f.write(content)
        os.replace(tmp_path, file_path)
        return True
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise e

# -----------------------------------------------------------------------------
# HTTP Request Handler
# -----------------------------------------------------------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MemoSplitHandler(BaseHTTPRequestHandler):
    server_version = "MemoSplitStudioNAS/1.0"

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Accept-Ranges, Content-Length")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/health":
            self.send_json({"status": "ok", "service": "MemoSplit Studio NAS", "root": MEDIA_ROOT})
            return

        if path == "/api/settings":
            self.send_json({
                "library_dir": MEDIA_ROOT,
                "default_engine": "gemini",
                "platform": "WD My Cloud EX2 Ultra"
            })
            return

        if path in ("/api/recordings", "/api/library"):
            raw_path = query.get("path", [""])[0].strip()
            scan_path = raw_path if raw_path else MEDIA_ROOT
            if not os.path.isabs(scan_path):
                scan_path = os.path.join(MEDIA_ROOT, scan_path)
            self.handle_api_recordings(scan_path)
            return

        # /api/recordings/<filename>/audio
        audio_match = re.match(r"^/api/recordings/(.+)/audio$", path)
        if audio_match:
            rel_file = audio_match.group(1)
            raw_root = query.get("path", [""])[0].strip()
            custom_root = raw_root if raw_root else MEDIA_ROOT
            full_path = os.path.join(custom_root, rel_file)
            self.handle_stream_audio(full_path)
            return

        # /api/recordings/<filename>/transcript
        transcript_match = re.match(r"^/api/recordings/(.+)/transcript$", path)
        if transcript_match:
            rel_file = transcript_match.group(1)
            raw_root = query.get("path", [""])[0].strip()
            custom_root = raw_root if raw_root else MEDIA_ROOT
            full_path = os.path.join(custom_root, rel_file)
            self.handle_get_transcript(full_path)
            return

        # /api/profiles
        if path == "/api/profiles":
            raw_root = query.get("path", [""])[0].strip()
            custom_root = raw_root if raw_root else MEDIA_ROOT
            self.handle_get_profiles(custom_root)
            return

        # /api/profiles/<filename>/audio
        profile_audio_match = re.match(r"^/api/profiles/(.+)/audio$", path)
        if profile_audio_match:
            rel_file = profile_audio_match.group(1)
            raw_root = query.get("path", [""])[0].strip()
            custom_root = raw_root if raw_root else MEDIA_ROOT
            p_dir = self.get_profiles_dir(custom_root)
            full_path = os.path.join(p_dir, rel_file)
            self.handle_stream_audio(full_path)
            return

        # Batch status endpoint
        if path == "/api/batch/status":
            self.send_json({
                "status": "idle",
                "active": False,
                "batch_total_count": 0,
                "batch_current_index": 0,
                "batch_completed_count": 0,
                "batch_failed_count": 0,
                "batch_elapsed_total": 0.0,
                "batch_eta_total": 0.0,
                "current_file": "",
                "current_percent": 0,
                "current_stage_detail": "Idle"
            })
            return

        # Serve static files (index.html, js, css, icons)
        self.handle_static_file(path)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

        # Voice profiles save endpoint
        if path in ("/api/profiles", "/api/profiles/save"):
            raw_root = query.get("path", [""])[0].strip()
            custom_root = raw_root if raw_root else MEDIA_ROOT
            self.handle_save_profile(custom_root, query)
            return

        # Batch control endpoints
        if path.startswith("/api/batch/"):
            action = path.replace("/api/batch/", "")
            self.send_json({
                "status": "delegate_client" if action == "start" else "ok",
                "action": action,
                "message": "Batch coordination handled via client-side Gemini pipeline"
            })
            return

        # Upload audio recording endpoint
        if path in ("/api/upload", "/api/recordings/upload"):
            folder = query.get("folder", [""])[0].strip()
            filename = query.get("filename", [""])[0].strip()
            raw_root = query.get("path", [""])[0].strip()
            custom_root = raw_root if raw_root else MEDIA_ROOT
            self.handle_upload_recording(custom_root, folder, filename)
            return

        # Save transcript endpoint
        transcript_match = re.match(r"^/api/recordings/(.+)/transcript$", path)
        if transcript_match:
            rel_file = transcript_match.group(1)
            raw_root = query.get("path", [""])[0].strip()
            custom_root = raw_root if raw_root else MEDIA_ROOT
            full_path = os.path.join(custom_root, rel_file)
            self.handle_save_transcript(full_path)
            return

        self.send_error(404, "Not Found")

    do_PUT = do_POST

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

        profile_match = re.match(r"^/api/profiles/(.+)$", path)
        if profile_match:
            filename = profile_match.group(1)
            raw_root = query.get("path", [""])[0].strip()
            custom_root = raw_root if raw_root else MEDIA_ROOT
            self.handle_delete_profile(custom_root, filename)
            return

        self.send_error(404, "Not Found")

    def handle_static_file(self, req_path):
        if req_path in ("/", "", "/index.html"):
            file_name = "index.html"
        else:
            file_name = req_path.lstrip("/")

        target = os.path.join(STATIC_DIR, file_name)
        if not os.path.isfile(target):
            # Fallback to index.html for client-side routing
            target = os.path.join(STATIC_DIR, "index.html")

        if not os.path.isfile(target):
            self.send_error(404, "File Not Found")
            return

        mime_type, _ = mimetypes.guess_type(target)
        mime_type = mime_type or "application/octet-stream"

        try:
            with open(target, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def handle_api_recordings(self, root_dir):
        if not os.path.isdir(root_dir):
            self.send_json({"recordings": [], "folders": [], "total": 0, "path": root_dir})
            return

        audio_exts = AUDIO_EXTENSIONS
        recordings_map = {}
        folders_map = {"": {"name": os.path.basename(root_dir) or "Root", "path": "", "count": 0}}

        for dirpath, dirnames, filenames in os.walk(root_dir):
            rel_folder = os.path.relpath(dirpath, root_dir)
            if rel_folder == ".":
                rel_folder = ""
            elif rel_folder.startswith("_profiles") or rel_folder.startswith("."):
                continue
            elif rel_folder not in folders_map:
                folders_map[rel_folder] = {"name": os.path.basename(dirpath), "path": rel_folder, "count": 0}

            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                base = os.path.splitext(f)[0]
                norm_base = base.strip().lower()
                rel_key = f"{rel_folder.lower()}/{norm_base}" if rel_folder else norm_base
                rel_file_path = f"{rel_folder}/{f}" if rel_folder else f
                abs_path = os.path.join(dirpath, f)

                if rel_key not in recordings_map:
                    recordings_map[rel_key] = {
                        "filename": f,
                        "base_name": base,
                        "relative_path": rel_file_path,
                        "relative_folder": rel_folder,
                        "folder_name": os.path.basename(dirpath) if rel_folder else "Root",
                        "size_bytes": 0,
                        "modified_at": os.path.getmtime(abs_path) if os.path.exists(abs_path) else 0,
                        "has_audio": False,
                        "has_transcript": False,
                        "has_speakers": False,
                        "has_raw_transcript": False,
                        "speakers": [],
                        "preview_text": "",
                        "duration": 0
                    }

                item = recordings_map[rel_key]

                if ext in audio_exts:
                    item["has_audio"] = True
                    item["filename"] = f
                    item["relative_path"] = rel_file_path
                    item["size_bytes"] = os.path.getsize(abs_path)
                elif ext == ".json":
                    item["has_transcript"] = True
                    item["has_speakers"] = True
                    try:
                        with open(abs_path, "r", encoding="utf-8") as jf:
                            data = json.load(jf)
                            if "duration" in data:
                                item["duration"] = data["duration"]
                            if "segments" in data and data["segments"]:
                                item["speakers"] = list({s.get("speaker") for s in data["segments"] if s.get("speaker")})
                                item["preview_text"] = " ".join([s.get("text", "") for s in data["segments"][:3]])
                    except Exception:
                        pass
                elif ext in (".srt", ".vtt"):
                    item["has_transcript"] = True
                    item["has_speakers"] = True
                    try:
                        with open(abs_path, "r", encoding="utf-8") as sf:
                            segs = parse_srt_content(sf.read())
                            if segs:
                                item["duration"] = segs[-1]["end"]
                                item["speakers"] = list({s["speaker"] for s in segs})
                                item["preview_text"] = " ".join([s["text"] for s in segs[:3]])
                    except Exception:
                        pass
                elif ext == ".txt":
                    item["has_transcript"] = True
                    if not item["has_speakers"]:
                        item["has_raw_transcript"] = True
                        try:
                            with open(abs_path, "r", encoding="utf-8") as tf:
                                item["preview_text"] = tf.read()[:150]
                        except Exception:
                            pass

        recordings_list = [r for r in recordings_map.values() if r["has_audio"] or r["has_transcript"]]
        for r in recordings_list:
            fkey = r["relative_folder"]
            if fkey in folders_map:
                folders_map[fkey]["count"] += 1

        self.send_json({
            "recordings": recordings_list,
            "folders": list(folders_map.values()),
            "total": len(recordings_list),
            "path": root_dir
        })

    def handle_stream_audio(self, full_path):
        if not os.path.isfile(full_path):
            self.send_error(404, "Audio file not found")
            return

        file_size = os.path.getsize(full_path)
        range_header = self.headers.get("Range")
        mime_type, _ = mimetypes.guess_type(full_path)
        mime_type = mime_type or "audio/mpeg"

        if range_header:
            range_match = re.search(r"bytes=(\d+)-(\d*)", range_header)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_cors_headers()
                self.end_headers()

                with open(full_path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return

        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_cors_headers()
        self.end_headers()

        with open(full_path, "rb") as f:
            self.wfile.write(f.read())

    def handle_get_transcript(self, full_path):
        base_path = os.path.splitext(full_path)[0]
        json_file = base_path + ".json"
        srt_file = base_path + ".srt"
        vtt_file = base_path + ".vtt"
        txt_file = base_path + ".txt"

        raw_text_content = ""
        # Check if companion txt file exists
        if os.path.isfile(txt_file):
            try:
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as tf:
                    raw_text_content = tf.read().strip()
            except Exception:
                pass

        # 1. Try loading structured JSON if it contains valid segments
        if os.path.isfile(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    if data.get("segments") and len(data["segments"]) > 0:
                        if raw_text_content and not data.get("raw_text"):
                            data["raw_text"] = raw_text_content
                        self.send_json(data)
                        return
            except Exception as e:
                pass

        # 2. Try loading SRT or VTT if it contains valid segments
        if os.path.isfile(srt_file) or os.path.isfile(vtt_file):
            target = srt_file if os.path.isfile(srt_file) else vtt_file
            try:
                with open(target, "r", encoding="utf-8", errors="ignore") as sf:
                    content = sf.read().strip()
                    if content:
                        segs = parse_srt_content(content)
                        if segs and len(segs) > 0:
                            self.send_json({
                                "segments": segs,
                                "duration": segs[-1]["end"] if segs else 0,
                                "raw_text": raw_text_content,
                                "source": "srt"
                            })
                            return
            except Exception as e:
                pass

        # 3. If raw text exists, return it as draft transcript
        if raw_text_content:
            self.send_json({
                "segments": [],
                "raw_text": raw_text_content,
                "duration": 0,
                "source": "txt",
                "has_raw_transcript": True
            })
            return

        self.send_json({"segments": [], "speaker_metadata": {}, "duration": 0})

    def handle_save_transcript(self, full_path):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)

            base_path = os.path.splitext(full_path)[0]
            segments = data.get("segments", [])
            speaker_meta = data.get("speaker_metadata", {})
            duration = data.get("duration", 0)
            raw_text = data.get("raw_text", "")

            # Guard against accidental empty overwrite of valid files
            if not segments and not raw_text:
                self.send_json({"status": "ignored_empty"})
                return

            saved_map = {}

            # 1. Write JSON atomically if segments provided
            if segments:
                json_path = base_path + ".json"
                safe_write_file(json_path, {
                    "duration": duration,
                    "segments": segments,
                    "speaker_metadata": speaker_meta,
                    "updated_at": os.path.getmtime(full_path) if os.path.exists(full_path) else time.time()
                }, is_json=True)
                saved_map["json"] = json_path

            # 2. Write Otter-compatible SRT atomically if segments provided
            if segments:
                srt_path = base_path + ".srt"
                safe_write_file(srt_path, serialize_to_srt(segments, speaker_meta))
                saved_map["srt"] = srt_path

            # 3. Write WebVTT atomically if segments provided
            if segments:
                vtt_path = base_path + ".vtt"
                safe_write_file(vtt_path, serialize_to_vtt(segments, speaker_meta))
                saved_map["vtt"] = vtt_path

            # 4. Write TXT atomically if raw_text or segments provided
            if raw_text or segments:
                txt_path = base_path + ".txt"
                if raw_text:
                    safe_write_file(txt_path, raw_text)
                else:
                    safe_write_file(txt_path, serialize_to_txt(segments, speaker_meta))
                saved_map["txt"] = txt_path

            self.send_json({"status": "saved", **saved_map})
        except Exception as e:
            self.send_error(500, f"Failed to save transcript: {e}")

    def handle_upload_recording(self, custom_root: str, folder: str, filename: str):
        try:
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))

            file_payload = None

            if "multipart/form-data" in content_type:
                import email
                # Use standard email parser to extract multipart file and form fields
                body_bytes = self.rfile.read(length)
                msg_bytes = b"Content-Type: " + content_type.encode("latin1") + b"\r\n\r\n" + body_bytes
                msg = email.message_from_bytes(msg_bytes)
                for part in msg.walk():
                    fn = part.get_filename()
                    if fn:
                        filename = filename or fn
                        file_payload = part.get_payload(decode=True)
                    elif part.get_param("name", header="content-disposition") == "folder":
                        folder = folder or part.get_payload(decode=True).decode("utf-8", errors="ignore").strip()

            if not filename:
                filename = self.headers.get("X-Filename", "").strip()

            if not filename:
                filename = f"Recording_{int(time.time())}.m4a"

            filename = os.path.basename(filename)
            clean_folder = folder.strip("/\\")

            dest_dir = os.path.join(custom_root, clean_folder) if clean_folder else custom_root
            abs_custom = os.path.abspath(custom_root)
            abs_dest = os.path.abspath(dest_dir)

            if not abs_dest.startswith(abs_custom):
                self.send_error(403, "Forbidden destination folder")
                return

            os.makedirs(abs_dest, exist_ok=True)
            final_path = os.path.join(abs_dest, filename)
            tmp_path = f"{final_path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"

            if file_payload is not None:
                with open(tmp_path, "wb") as f:
                    f.write(file_payload)
            else:
                remaining = length
                with open(tmp_path, "wb") as f:
                    while remaining > 0:
                        chunk_size = min(remaining, 65536)
                        chunk = self.rfile.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)

            os.replace(tmp_path, final_path)

            rel_path = os.path.join(clean_folder, filename) if clean_folder else filename
            file_size = os.path.getsize(final_path)

            self.send_json({
                "status": "ok",
                "filename": filename,
                "folder": clean_folder,
                "relative_path": rel_path,
                "saved_path": final_path,
                "size_bytes": file_size
            })
        except Exception as e:
            self.send_error(500, f"Failed to upload recording: {e}")

    def get_profiles_dir(self, custom_root: str) -> str:
        p_dir = os.path.join(custom_root or MEDIA_ROOT, "_profiles")
        os.makedirs(p_dir, exist_ok=True)
        return p_dir

    def handle_get_profiles(self, custom_root: str):
        try:
            p_dir = self.get_profiles_dir(custom_root)
            meta_file = os.path.join(p_dir, "profiles.json")
            profiles = []
            if os.path.isfile(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        profiles = json.load(f)
                except Exception:
                    profiles = []

            audio_exts = AUDIO_EXTENSIONS
            existing_files = {p.get("filename") for p in profiles if p.get("filename")}
            changed = False
            for f in os.listdir(p_dir):
                ext = os.path.splitext(f)[1].lower()
                if ext in audio_exts and f not in existing_files:
                    base = os.path.splitext(f)[0]
                    clean_name = base.replace("_", " ").strip()
                    file_path = os.path.join(p_dir, f)
                    profiles.append({
                        "id": f"prof_{int(time.time())}_{len(profiles)}",
                        "name": clean_name,
                        "role": "Speaker",
                        "filename": f,
                        "size_bytes": os.path.getsize(file_path),
                        "created_at": os.path.getmtime(file_path)
                    })
                    changed = True

            valid_profiles = []
            for p in profiles:
                fn = p.get("filename")
                if fn and os.path.isfile(os.path.join(p_dir, fn)):
                    valid_profiles.append(p)
                else:
                    changed = True

            if changed:
                safe_write_file(meta_file, valid_profiles, is_json=True)

            self.send_json({"profiles": valid_profiles, "count": len(valid_profiles)})
        except Exception as e:
            self.send_error(500, f"Error getting profiles: {e}")

    def handle_save_profile(self, custom_root: str, query):
        try:
            p_dir = self.get_profiles_dir(custom_root)
            meta_file = os.path.join(p_dir, "profiles.json")
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))

            name = query.get("name", [""])[0].strip()
            role = query.get("role", [""])[0].strip()
            filename = query.get("filename", [""])[0].strip()

            file_payload = None

            if "multipart/form-data" in content_type:
                import email
                body_bytes = self.rfile.read(length)
                msg_bytes = b"Content-Type: " + content_type.encode("latin1") + b"\r\n\r\n" + body_bytes
                msg = email.message_from_bytes(msg_bytes)
                for part in msg.walk():
                    fn = part.get_filename()
                    if fn:
                        filename = filename or fn
                        file_payload = part.get_payload(decode=True)
                    elif part.get_param("name", header="content-disposition") == "name":
                        name = part.get_payload(decode=True).decode("utf-8", errors="ignore").strip() or name
                    elif part.get_param("name", header="content-disposition") == "role":
                        role = part.get_payload(decode=True).decode("utf-8", errors="ignore").strip() or role

            if not name:
                name = self.headers.get("X-Profile-Name", "").strip() or "Unknown Speaker"
            if not role:
                role = self.headers.get("X-Profile-Role", "").strip() or "Speaker"

            if not filename:
                safe_base = re.sub(r'[^a-zA-Z0-9_\- ]', '', name).replace(' ', '_')
                filename = f"{safe_base}.wav"

            filename = os.path.basename(filename)
            final_path = os.path.join(p_dir, filename)
            tmp_path = f"{final_path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"

            if file_payload is not None:
                with open(tmp_path, "wb") as f:
                    f.write(file_payload)
            else:
                remaining = length
                with open(tmp_path, "wb") as f:
                    while remaining > 0:
                        chunk_size = min(remaining, 65536)
                        chunk = self.rfile.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)

            os.replace(tmp_path, final_path)

            profiles = []
            if os.path.isfile(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        profiles = json.load(f)
                except Exception:
                    profiles = []

            existing = next((p for p in profiles if p.get("filename") == filename or p.get("name", "").lower() == name.lower()), None)
            if existing:
                existing["name"] = name
                existing["role"] = role
                existing["filename"] = filename
                existing["size_bytes"] = os.path.getsize(final_path)
                existing["modified_at"] = time.time()
                profile_obj = existing
            else:
                profile_obj = {
                    "id": f"prof_{int(time.time())}",
                    "name": name,
                    "role": role,
                    "filename": filename,
                    "size_bytes": os.path.getsize(final_path),
                    "created_at": time.time()
                }
                profiles.append(profile_obj)

            safe_write_file(meta_file, profiles, is_json=True)

            self.send_json({
                "status": "ok",
                "profile": profile_obj,
                "message": f"Successfully enrolled voice profile for {name}"
            })
        except Exception as e:
            self.send_error(500, f"Error saving profile: {e}")

    def handle_delete_profile(self, custom_root: str, filename: str):
        try:
            p_dir = self.get_profiles_dir(custom_root)
            filename = os.path.basename(filename)
            target = os.path.join(p_dir, filename)
            if os.path.isfile(target):
                os.remove(target)

            meta_file = os.path.join(p_dir, "profiles.json")
            if os.path.isfile(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        profiles = json.load(f)
                    profiles = [p for p in profiles if p.get("filename") != filename]
                    safe_write_file(meta_file, profiles, is_json=True)
                except Exception:
                    pass

            self.send_json({"status": "deleted", "filename": filename})
        except Exception as e:
            self.send_error(500, f"Error deleting profile: {e}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------
def run_server():
    server_address = (HOST, PORT)
    httpd = ThreadedHTTPServer(server_address, MemoSplitHandler)
    print("=" * 70)
    print("   🚀 MemoSplit Studio - WD My Cloud EX2 Ultra Server Running!")
    print(f"   📂 Media Storage Root : {MEDIA_ROOT}")
    print(f"   🌐 Local Web Access   : http://0.0.0.0:{PORT}")
    print(f"   📱 iPhone Safari Access: http://<mycloud-ip>:{PORT}  or  http://mycloudex2ultra.local:{PORT}")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
