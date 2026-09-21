#!/usr/bin/env bash
# Run the lab server in the background.
#   tools/lab.sh start | stop | restart | status | open | logs
set -euo pipefail

PROJECT="${MALECNS_PROJECT:-$HOME/Sites/malecns-lab}"
PIDFILE="$PROJECT/.server.pid"
LOGFILE="$PROJECT/.server.log"
cd "$PROJECT"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3

our_pid() {  # the PID in .server.pid, only if it is still this project's server.py
  [ -f "$PIDFILE" ] || return 1
  local pid; pid=$(cat "$PIDFILE")
  ps -p "$pid" -o command= 2>/dev/null | grep -q "server.py" || return 1
  echo "$pid"
}

url() {
  "$PY" -c "import json;print('http://127.0.0.1:%d' % json.load(open('.runtime.json'))['port'])" 2>/dev/null || true
}

start() {
  if pid=$(our_pid); then echo "Lab already running (pid $pid): $(url)"; return 0; fi
  nohup "$PY" server.py >> "$LOGFILE" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 40); do   # the server loads both circuits before listening; wait for its URL
    sleep 0.5
    if ! our_pid >/dev/null; then echo "Server exited during startup. Last log lines:" >&2; tail -n 20 "$LOGFILE" >&2; return 1; fi
    if grep -q "MaleCNS Lab: http" <(tail -n 5 "$LOGFILE"); then echo "Lab running (pid $(cat "$PIDFILE")): $(url)"; return 0; fi
  done
  echo "Server started (pid $(cat "$PIDFILE")) but hasn't printed its URL yet. See: tools/lab.sh logs"
}

stop() {
  if pid=$(our_pid); then
    kill "$pid"
    for _ in $(seq 1 20); do ps -p "$pid" >/dev/null 2>&1 || break; sleep 0.25; done
    ps -p "$pid" >/dev/null 2>&1 && kill -9 "$pid"
    echo "Lab stopped."
  else echo "Lab not running."; fi
  rm -f "$PIDFILE"
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) if pid=$(our_pid); then echo "Lab running (pid $pid): $(url)"; else echo "Lab not running. Start it with: tools/lab.sh start"; fi ;;
  open) start >/dev/null; u=$(url); echo "$u"; command -v open >/dev/null && open "$u" ;;
  logs) tail -n 40 "$LOGFILE" 2>/dev/null || echo "No log yet." ;;
  *) sed -n '2,3p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
