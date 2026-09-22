#!/bin/sh
# MemoSplit Studio - Startup Script for My Cloud EX2 Ultra
# Runs standalone in the background (nohup) on port 8088

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Kill any existing instance
pkill -f "memosplit_nas_server.py" 2>/dev/null

echo "[*] Starting MemoSplit Studio on My Cloud EX2 Ultra..."
nohup python3 "$DIR/memosplit_nas_server.py" > "$DIR/server.log" 2>&1 &

sleep 1
if pgrep -f "memosplit_nas_server.py" > /dev/null; then
    echo "[✓] MemoSplit Studio is running on http://$(hostname -I | awk '{print $1}'):8088"
    echo "    (Also accessible via http://mycloudex2ultra.local:8088)"
else
    echo "[!] Server failed to start. Check $DIR/server.log"
fi
