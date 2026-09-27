#!/usr/bin/env bash
# Sets up the venv, builds the frontend, and runs the server as a launchd agent (survives reboots).
set -euo pipefail
cd "$(dirname "$0")/.."
APP_DIR="$(pwd)"
PORT="$(cat PORT)"
LABEL="com.merolaagi.microopus"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

rm -f weights/model.json
PY="$(command -v python3.12 || command -v python3)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python tools/build.py
mkdir -p logs "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$APP_DIR/.venv/bin/uvicorn</string><string>server:app</string>
    <string>--host</string><string>127.0.0.1</string><string>--port</string><string>$PORT</string>
  </array>
  <key>WorkingDirectory</key><string>$APP_DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$APP_DIR/logs/server.log</string>
  <key>StandardErrorPath</key><string>$APP_DIR/logs/server.log</string>
</dict></plist>
PLIST

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    echo "Micro Opus $(cat VERSION) is running at http://localhost:$PORT"
    exit 0
  fi
  sleep 1
done
echo "Server did not answer on port $PORT. Last log lines:"
tail -20 logs/server.log
exit 1
