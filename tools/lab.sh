#!/usr/bin/env bash
# Run the lab server in the background.
#   tools/lab.sh start | stop | restart | status | open | logs | install-autostart | remove-autostart
#
# install-autostart registers the lab as a macOS login service (a LaunchAgent): it starts when you log in,
# and launchd restarts it within seconds whenever it exits or crashes. It always uses the same port, so the
# Cloudflare tunnel keeps working. Once installed, start/stop/restart go through launchd.
set -euo pipefail

PROJECT="${MALECNS_PROJECT:-$HOME/Sites/malecns-lab}"
PIDFILE="$PROJECT/.server.pid"
LOGFILE="$PROJECT/.server.log"
LABEL="com.merolaagi.malecns-lab"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
cd "$PROJECT"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
PYABS="$PROJECT/.venv/bin/python"; [ -x "$PYABS" ] || PYABS=$(command -v python3)

port() { "$PY" -c "import json;print(json.load(open('.runtime.json'))['port'])" 2>/dev/null || echo 53680; }
url() { echo "http://127.0.0.1:$(port)"; }

# --- login service -----------------------------------------------------------------------------------------
installed() { [ -f "$PLIST" ]; }
loaded() { launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; }
service_pid() { launchctl print "$DOMAIN/$LABEL" 2>/dev/null | awk '$1 == "pid" {print $3; exit}'; }

wait_ready() {  # the server loads its circuits before listening
  for _ in $(seq 1 90); do
    if curl -s -o /dev/null --max-time 2 "$(url)/"; then echo "Lab running as login service (pid $(service_pid)): $(url)"; return 0; fi
    sleep 0.5
  done
  echo "Lab service started but isn't answering yet. See: tools/lab.sh logs" >&2; return 1
}

install_autostart() {
  local p; p=$(port)
  legacy_stop
  mkdir -p "$(dirname "$PLIST")"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$PYABS</string><string>$PROJECT/server.py</string></array>
  <key>WorkingDirectory</key><string>$PROJECT</string>
  <key>EnvironmentVariables</key><dict><key>MALECNS_PORT</key><string>$p</string><key>PYTHONUNBUFFERED</key><string>1</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$LOGFILE</string>
  <key>StandardErrorPath</key><string>$LOGFILE</string>
</dict></plist>
EOF
  if loaded; then launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true; sleep 1; fi
  launchctl bootstrap "$DOMAIN" "$PLIST"
  echo "Installed login service $LABEL on port $p: starts at login, restarts on failure."
  wait_ready
}

remove_autostart() {
  if loaded; then launchctl bootout "$DOMAIN/$LABEL" || true; fi
  rm -f "$PLIST"; echo "Login service removed. Start the lab by hand with: tools/lab.sh start"
}

# --- plain background process (no login service) -------------------------------------------------------------
our_pid() {  # the PID in .server.pid, only if it is still this project's server.py
  [ -f "$PIDFILE" ] || return 1
  local pid; pid=$(cat "$PIDFILE")
  ps -p "$pid" -o command= 2>/dev/null | grep -q "server.py" || return 1
  echo "$pid"
}

legacy_stop() {
  local pid
  if pid=$(our_pid); then
    kill "$pid"
    for _ in $(seq 1 20); do ps -p "$pid" >/dev/null 2>&1 || break; sleep 0.25; done
    if ps -p "$pid" >/dev/null 2>&1; then kill -9 "$pid"; fi
    echo "Lab stopped."
  fi
  rm -f "$PIDFILE"
}

start() {
  if installed; then
    if loaded; then launchctl kickstart "$DOMAIN/$LABEL"; else launchctl bootstrap "$DOMAIN" "$PLIST"; fi
    wait_ready; return
  fi
  if pid=$(our_pid); then echo "Lab already running (pid $pid): $(url)"; return 0; fi
  nohup "$PY" server.py >> "$LOGFILE" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 40); do
    sleep 0.5
    if ! our_pid >/dev/null; then echo "Server exited during startup. Last log lines:" >&2; tail -n 20 "$LOGFILE" >&2; return 1; fi
    if grep -q "MaleCNS Lab: http" <(tail -n 5 "$LOGFILE"); then echo "Lab running (pid $(cat "$PIDFILE")): $(url)"; return 0; fi
  done
  echo "Server started (pid $(cat "$PIDFILE")) but hasn't printed its URL yet. See: tools/lab.sh logs"
}

stop() {
  if installed && loaded; then
    launchctl bootout "$DOMAIN/$LABEL" && echo "Lab stopped (login service unloaded until the next start or login)."
  elif our_pid >/dev/null; then legacy_stop
  else echo "Lab not running."; fi
}

restart() {
  if installed && loaded; then launchctl kickstart -k "$DOMAIN/$LABEL"; wait_ready
  else stop; start; fi
}

status() {
  if installed && loaded && [ -n "$(service_pid)" ]; then echo "Lab running as login service (pid $(service_pid)): $(url)"
  elif pid=$(our_pid); then echo "Lab running (pid $pid): $(url)"
  else echo "Lab not running. Start it with: tools/lab.sh start"; fi
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  open) start >/dev/null; u=$(url); echo "$u"; command -v open >/dev/null && open "$u" ;;
  logs) tail -n 40 "$LOGFILE" 2>/dev/null || echo "No log yet." ;;
  install-autostart) install_autostart ;;
  remove-autostart) remove_autostart ;;
  *) sed -n '2,3p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
