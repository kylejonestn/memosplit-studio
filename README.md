# 🎙️ MemoSplit Studio

> **Private Voice Memo Speaker Diarization, Interactive Audio Player & AI Transcription**  
> Run directly in your browser with zero installation via **Google Gemini Cloud AI**, or 100% offline with **Local WhisperX + Pyannote**.

[![GitHub Pages Deployment](https://github.com/kylejonestn/memosplit-studio/actions/workflows/deploy.yml/badge.svg)](https://kylejonestn.github.io/memosplit-studio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-indigo.svg)](LICENSE)

---

## ✨ Features

- ⚡ **Zero-Install Web App (GitHub Pages)**: Runs directly in your browser using the **Native File System Access API** (`window.showDirectoryPicker()`). Open your local Voice Memos folder, process audio with Gemini Cloud AI, and write `.json`, `.srt`, `.vtt`, and `.txt` files directly back to disk.
- 🤖 **Google Gemini Cloud Diarization (Default)**: Multimodal audio diarization completing in **~15 seconds** with intelligent auto-failover (`gemini-flash-lite-latest` ➔ `gemini-3.5-flash-lite` ➔ `gemini-3.6-flash`).
- 💻 **100% Offline Local Engine (Optional)**: Faster-Whisper + Wav2Vec2 + Pyannote INT8 pipeline running entirely offline on your Mac CPU/GPU.
- 📝 **Diarization Studio with Draft Transcript Alignment**: Paste raw notes (from Apple Notes, Otter, Zoom, ChatGPT, Whisper) and align words to speaker turns and audio timestamps with 1 click.
- 🎧 **Interactive Playback & Speaker Reassignment**: Click any speaker badge to instantly reassign a single line to another speaker with instant local saving.
- 📁 **Plex-Style Multi-Folder Library**: Automatically indexes nested subfolders, displays status badges (🟢 *Speakers Identified*, 🔵 *Text Only*, 🟡 *Needs Identification*), and searches across spoken transcripts.
- 🌙 **Overnight Batch Queue**: Automated batch diarization queue with configurable thermal pacing.
- ✂️ **Local Audio Slicer**: Auto-generates `ffmpeg` scripts to export clean single-speaker tracks.

---

## 🚀 Quick Start (Zero Install)

1. Open the **[MemoSplit Studio Web App](https://kylejonestn.github.io/memosplit-studio/)** in a Chromium browser (Chrome, Edge, Brave, Opera).
2. Click **"📁 Open Folder"** and select your local voice memo directory.
3. Click the **"⚡ Google Gemini Cloud AI"** icon and enter your [free Google AI Studio API key](https://aistudio.google.com/app/apikey).
4. Click on any recording and select **"Run Diarization"** or paste an existing draft transcript to align speakers.
5. All generated `.json`, `.srt`, `.vtt`, and `.txt` transcripts are automatically written back to your local folder.

---

## 💻 Optional: Running Local Python Engine (Offline)

If you prefer 100% offline transcription with local WhisperX + Pyannote:

### 1. Prerequisites
- Python 3.9 - 3.11
- `ffmpeg` installed (`brew install ffmpeg`)

### 2. Setup Virtual Environment
```bash
git clone https://github.com/kylejonestn/memosplit-studio.git
cd memosplit-studio

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables (Optional)
Copy the example environment file and add your keys:
```bash
cp .env.example .env
```

### 4. Start Local Engine
```bash
python diarize_service.py
```
Open `http://localhost:8080` in your browser.

---

## 🔒 Privacy & Security

- **Client-Side Storage**: Your Google Gemini API key and Hugging Face tokens are stored exclusively in your browser's `localStorage` or your local gitignored `.env` file.
- **No Analytics or Telemetry**: MemoSplit Studio does not track users, log data, or maintain remote databases.

---

## 📄 License

MIT License. Free and open source for personal and commercial use.
