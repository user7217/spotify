#!/usr/bin/env bash
# On-demand status dashboard for the overnight build. Run it over SSH anytime:
#   ./scripts/status.sh            (or: bash scripts/status.sh)
#
# Shows: container up/down, download/analyze counts + %, a classification of
# what's going wrong right now (throttled? age-gated? errored?), and the tail
# of the live log — so one command tells you the whole picture.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."          # -> repo root (audiolens/)
COMPOSE="docker compose -f docker-compose.overnight.yml"
DB="work/catalog.db"
LINES="${1:-10}"                                 # log lines to show (default 10)
SCAN=250                                          # log lines to scan for issues

c() { sqlite3 "$DB" "$1" 2>/dev/null || echo 0; }

# ── container state ──────────────────────────────────────────────────────────
CID=$($COMPOSE ps -q overnight 2>/dev/null)
if [ -n "$CID" ] && [ -n "$(docker ps -q --no-trunc | grep -F "$CID" 2>/dev/null)" ]; then
  UPTIME=$(docker ps --filter "id=$CID" --format '{{.Status}}')
  STATE="RUNNING — $UPTIME"
else
  STATE="STOPPED  (re-run: $COMPOSE up -d)"
fi

# ── counts ───────────────────────────────────────────────────────────────────
TOTAL=$(c "SELECT COUNT(*) FROM catalog_tracks")
GOT=$(c "SELECT COUNT(*) FROM catalog_tracks WHERE audio_track_id IS NOT NULL")
FAILED=$(c "SELECT COUNT(*) FROM download_failures")
ANALYZED=$(c "SELECT COUNT(*) FROM processing_state WHERE stage='document' AND status='done'")
FEAT=$(c "SELECT COUNT(*) FROM reccobeats_features WHERE status='ok'")
PCT=$(awk -v g="${GOT:-0}" -v t="${TOTAL:-0}" 'BEGIN{if(t>0)printf "%.1f",100*g/t; else print "0"}')

echo "════════════════════ AudioLens status ════════════════════"
echo " Container : $STATE"
echo " Features  : $FEAT (ReccoBeats)"
echo " Downloads : $GOT / $TOTAL  (${PCT}%)"
echo " Failed    : $FAILED        Analyzed : $ANALYZED"

# ── classify what's happening in the recent log ──────────────────────────────
LOG=$($COMPOSE logs --tail=$SCAN 2>/dev/null)
throttle=$(grep -c -iE "rate-limited|try again later" <<<"$LOG")
agegate=$(grep -c -iE "confirm your age" <<<"$LOG")
forbid=$(grep -c -iE "403|forbidden" <<<"$LOG")
unavail=$(grep -c -iE "Video unavailable" <<<"$LOG")
downloaded_now=$(grep -c -iE "\[download\] Destination|has already been downloaded" <<<"$LOG")

echo "───────────── recent activity (last $SCAN log lines) ─────────────"
printf " downloads   : %s\n" "$downloaded_now"
printf " throttled   : %s\n" "$throttle"
printf " age-gated   : %s\n" "$agegate"
printf " 403/forbid  : %s\n" "$forbid"
printf " unavailable : %s\n" "$unavail"

# verdict
if   [ "$throttle" -gt 5 ]; then echo " => ⚠️  THROTTLED by YouTube — will recover; sleeps are active"
elif [ "$forbid"  -gt 5 ]; then echo " => ⚠️  403s — check yt-dlp/Deno (see logs)"
elif [ "$downloaded_now" -gt 0 ]; then echo " => ✅ downloading normally"
elif [ -z "$CID" ]; then echo " => ⚠️  container not running"
else echo " => (quiet — check the log tail below)"
fi

echo "──────────────────── last $LINES log lines ────────────────────"
$COMPOSE logs --tail="$LINES" 2>/dev/null | sed 's/^overnight-1  | //'
echo "═══════════════════════════════════════════════════════════"
