#!/usr/bin/env bash
# Automatic sync: apply, test, commit and push each new malecns-lab*.zip that lands in Downloads.
#
#   tools/autosync.sh install     start a background agent (launchd) that watches Downloads
#   tools/autosync.sh uninstall   stop and remove the agent
#   tools/autosync.sh status      show whether the agent is loaded and the last log lines
#   tools/autosync.sh watch       same job in the foreground (Ctrl-C to stop); use if macOS blocks the agent
#   tools/autosync.sh run         one pass: sync the newest zip if there is one (what the agent calls)
set -euo pipefail

PROJECT="${MALECNS_PROJECT:-$HOME/Sites/malecns-lab}"
DOWNLOADS="${MALECNS_DOWNLOADS:-$HOME/Downloads}"
LABEL="com.merolaagi.malecns-lab.autosync"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/malecns-lab-autosync.log"
LOCK="${TMPDIR:-/tmp}/malecns-lab-autosync.lock"

notify() {  # macOS banner; silently skipped elsewhere
  command -v osascript >/dev/null 2>&1 && osascript -e "display notification \"$2\" with title \"MaleCNS lab\" subtitle \"$1\"" >/dev/null 2>&1 || true
}

run_once() {
  mkdir "$LOCK" 2>/dev/null || exit 0          # another pass is already running
  trap 'rmdir "$LOCK"' EXIT
  ZIP=$(ls -t "$DOWNLOADS"/malecns-lab.zip "$DOWNLOADS"/malecns-lab\ \(*\).zip 2>/dev/null | head -n 1 || true)
  [ -n "$ZIP" ] || exit 0
  # Browsers rename to .zip only when the download finishes; still confirm the archive is complete.
  sleep 2
  unzip -tq "$ZIP" >/dev/null 2>&1 || { echo "$(date '+%F %T') $(basename "$ZIP") not readable yet; will retry"; exit 0; }
  echo "$(date '+%F %T') Syncing $(basename "$ZIP")"
  set +e; OUT=$("$PROJECT/tools/sync.sh" "$ZIP" 2>&1); CODE=$?; set -e
  echo "$OUT"
  case $CODE in
    0) MSG=$(echo "$OUT" | grep -m1 '^Committed:' | sed 's/^Committed: //' || true)
       URL=$(echo "$OUT" | grep -o 'http://127.0.0.1:[0-9]*' | tail -n 1 || true)
       if echo "$OUT" | grep -q '^Pushed'; then notify "Pushed to GitHub" "${MSG:-Iteration applied}.${URL:+ Lab running at $URL}"
       else notify "Up to date" "That download matched the last commit."; fi ;;
    4) notify "Refused a download" "$(basename "$ZIP") isn't a normal lab iteration. See $LOG." ;;
    3) notify "Skipped an older download" "$(basename "$ZIP") predates the committed version." ;;
    *) if echo "$OUT" | grep -q 'uncommitted changes'; then notify "Sync paused" "Commit or stash your local edits, then run tools/sync.sh."
       elif echo "$OUT" | grep -q 'Tests failed'; then notify "Tests failed" "Nothing was committed. See $LOG."
       else notify "Sync failed" "See $LOG."; fi ;;
  esac
}

case "${1:-}" in
  run) run_once ;;
  watch)
    echo "Watching $DOWNLOADS for malecns-lab.zip downloads (Ctrl-C to stop)…"
    while true; do "$0" run || true; sleep 5; done ;;
  install)
    [ "$(uname)" = Darwin ] || { echo "install uses launchd (macOS). Use: tools/autosync.sh watch" >&2; exit 1; }
    mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"
    cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$PROJECT/tools/autosync.sh</string><string>run</string></array>
  <key>WatchPaths</key><array><string>$DOWNLOADS</string></array>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>AbandonProcessGroup</key><true/>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>MALECNS_PROJECT</key><string>$PROJECT</string>
    <key>MALECNS_DOWNLOADS</key><string>$DOWNLOADS</string>
  </dict>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict></plist>
PL
    plutil -lint "$PLIST" >/dev/null || { echo "Generated agent file is invalid: $PLIST" >&2; exit 1; }
    # Unloading is asynchronous; loading again too soon fails with "Bootstrap failed: 5".
    launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || break; sleep 0.5; done
    ok=0
    for attempt in 1 2 3; do
      if launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null; then ok=1; break; fi
      sleep 2
    done
    [ $ok = 1 ] || { echo "launchctl couldn't load the agent. Try: tools/autosync.sh uninstall && tools/autosync.sh install" >&2; echo "Or run the same job in a Terminal tab: tools/autosync.sh watch" >&2; exit 1; }
    echo "Autosync is on. New malecns-lab*.zip downloads are applied, tested, committed and pushed."
    echo "Log: $LOG"
    echo "If the log shows 'Operation not permitted', macOS is blocking background access to Downloads:"
    echo "  System Settings > Privacy & Security > Full Disk Access, add /bin/bash, then run: tools/autosync.sh install"
    echo "  or skip the agent and keep a Terminal tab running: tools/autosync.sh watch" ;;
  uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
    rm -f "$PLIST"; echo "Autosync removed." ;;
  status)
    if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then echo "Autosync agent: loaded"; else echo "Autosync agent: not loaded"; fi
    [ -f "$LOG" ] && { echo "Last log lines:"; tail -n 15 "$LOG"; } || echo "No log yet." ;;
  *) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
