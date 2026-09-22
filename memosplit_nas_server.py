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

        # Serve static files (index.html, js, css, icons)
        self.handle_static_file(path)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

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

        audio_exts = {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac"}
        recordings_map = {}
        folders_map = {"": {"name": os.path.basename(root_dir) or "Root", "path": "", "count": 0}}

        for dirpath, dirnames, filenames in os.walk(root_dir):
            rel_folder = os.path.relpath(dirpath, root_dir)
            if rel_folder == ".":
                rel_folder = ""
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

        if os.path.isfile(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    self.send_json(data)
                    return
            except Exception as e:
                pass

        if os.path.isfile(srt_file) or os.path.isfile(vtt_file):
            target = srt_file if os.path.isfile(srt_file) else vtt_file
            try:
                with open(target, "r", encoding="utf-8") as sf:
                    segs = parse_srt_content(sf.read())
                    self.send_json({
                        "segments": segs,
                        "duration": segs[-1]["end"] if segs else 0,
                        "source": "srt"
                    })
                    return
            except Exception as e:
                pass

        if os.path.isfile(txt_file):
            try:
                with open(txt_file, "r", encoding="utf-8") as tf:
                    raw = tf.read()
                    self.send_json({
                        "segments": [],
                        "raw_text": raw,
                        "source": "txt"
                    })
                    return
            except Exception as e:
                pass

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

            # 1. Write JSON
            json_path = base_path + ".json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({
                    "duration": duration,
                    "segments": segments,
                    "speaker_metadata": speaker_meta,
                    "updated_at": os.path.getmtime(full_path) if os.path.exists(full_path) else None
                }, jf, indent=2, ensure_ascii=False)

            # 2. Write Otter-compatible SRT
            srt_path = base_path + ".srt"
            with open(srt_path, "w", encoding="utf-8") as sf:
                sf.write(serialize_to_srt(segments, speaker_meta))

            self.send_json({"status": "saved", "json": json_path, "srt": srt_path})
        except Exception as e:
            self.send_error(500, f"Failed to save transcript: {e}")

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
