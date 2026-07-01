"""Push a one-line progress update to your phone via ntfy.sh.

Runs on the HOST (not in the container) from cron, so it still reports even if
the job has stopped. Reads the SQLite catalog directly, computes a rate + ETA
by diffing against the previous run (state file), checks whether the Docker
container is still running, and POSTs a compact message to an ntfy topic.

No account/token needed — ntfy delivers to whatever topic you subscribe to in
the app. Pick an unguessable topic name (it's effectively your password).

Usage (cron):
    NTFY_TOPIC=audiolens-anand-7h3k \
    python3 scripts/notify_progress.py \
        --sqlite /home/anand/spotify/audiolens/work/catalog.db \
        --total 14304
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import subprocess
import time
import urllib.request

NTFY_BASE = os.environ.get("NTFY_BASE", "https://ntfy.sh")


def _count(db, sql: str) -> int:
    try:
        return db.execute(sql).fetchone()[0]
    except Exception:
        return 0


def _container_running(name_filter: str):
    """True/False if known, None if we can't tell (e.g. no docker perms) —
    so a permission error never gets misreported as a stopped job."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", f"name={name_filter}",
             "--filter", "status=running", "-q"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except Exception:
        return None


def _fmt_eta(hours: float) -> str:
    if hours <= 0 or hours != hours:  # 0 or NaN
        return "—"
    if hours < 1:
        return f"~{int(hours * 60)} min"
    if hours < 48:
        return f"~{hours:.1f} h"
    return f"~{hours / 24:.1f} days"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--topic", default=os.environ.get("NTFY_TOPIC"))
    ap.add_argument("--total", type=int, help="total tracks (default: from catalog)")
    ap.add_argument("--container", default="audiolens-overnight",
                    help="name filter to check if the job is running")
    ap.add_argument("--state", help="rate state file (default: alongside the db)")
    ap.add_argument("--dry-run", action="store_true", help="print the message, don't send")
    a = ap.parse_args()
    if not a.topic:
        raise SystemExit("set --topic or NTFY_TOPIC")

    db = sqlite3.connect(a.sqlite)
    total = a.total or _count(db, "SELECT COUNT(*) FROM catalog_tracks")
    got = _count(db, "SELECT COUNT(*) FROM catalog_tracks WHERE audio_track_id IS NOT NULL")
    failed = _count(db, "SELECT COUNT(*) FROM download_failures")
    analyzed = _count(
        db,
        "SELECT COUNT(*) FROM processing_state WHERE stage='document' AND status='done'",
    )

    # rate + ETA from the delta since last run
    state_path = pathlib.Path(a.state or (str(a.sqlite) + ".notify_state.json"))
    now = time.time()
    prev = {}
    if state_path.exists():
        try:
            prev = json.loads(state_path.read_text())
        except Exception:
            prev = {}
    dt_h = (now - prev.get("ts", now)) / 3600 if prev else 0

    dl_rate = (got - prev.get("got", got)) / dt_h if dt_h > 0 else 0
    an_rate = (analyzed - prev.get("analyzed", analyzed)) / dt_h if dt_h > 0 else 0
    state_path.write_text(json.dumps({"ts": now, "got": got, "analyzed": analyzed}))

    running = _container_running(a.container) if not a.dry_run else True

    # pick the phase to show an ETA for
    if got < total and dl_rate > 0.5:
        phase, eta = "downloading", _fmt_eta((total - got) / dl_rate)
        rate_line = f"{dl_rate:.0f} dl/hr · ETA {eta}"
    elif analyzed < got and an_rate > 0.5:
        phase, eta = "analyzing", _fmt_eta((got - analyzed) / an_rate)
        rate_line = f"{an_rate:.0f} tracks/hr · ETA {eta}"
    elif got >= total and analyzed >= total:
        phase, rate_line = "complete", "all done"
    else:
        phase, rate_line = "idle/slow", "rate ~0 (check the job)"

    pct = 100 * got / total if total else 0
    # NB: HTTP headers are latin-1 only — no emoji in Title. ntfy renders the
    # emoji from the Tags header instead.
    stopped = running is False
    title = f"AudioLens — {'STOPPED' if stopped else 'running'} ({phase})"
    body = (
        f"Downloads: {got:,}/{total:,} ({pct:.1f}%)\n"
        f"{rate_line}\n"
        f"Failed: {failed:,}   Analyzed: {analyzed:,}"
    )
    if stopped and phase != "complete":
        body += "\n\n⚠️ container not running — re-run `up -d`"

    # ntfy: POST body to the topic; headers set title / priority / tags
    priority = "default"
    tags = "musical_note"
    if stopped and phase != "complete":
        priority, tags = "high", "warning"
    elif phase == "complete":
        priority, tags = "high", "white_check_mark"

    if a.dry_run:
        print(f"--- would send to {NTFY_BASE}/{a.topic} ---")
        print(f"Title: {title}\nPriority: {priority}  Tags: {tags}\n{body}")
        return 0

    req = urllib.request.Request(
        f"{NTFY_BASE}/{a.topic}",
        data=body.encode("utf-8"),
        headers={
            "Title": title,
            "Priority": priority,
            "Tags": tags,
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=20)
        print("sent:", title, "|", rate_line)
    except Exception as e:
        print("ntfy send failed:", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
