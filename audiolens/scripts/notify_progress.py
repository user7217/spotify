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


def _classify_recent(name_filter: str, n: int = 250) -> dict:
    """Scan the container's recent log lines and classify what's happening now:
    throttling / age-gates / 403s / actual downloads. Returns counts + a verdict
    so the phone push says *why* it's slow, not just that it is."""
    empty = {"downloads": 0, "throttled": 0, "agegate": 0, "forbid": 0,
             "unavail": 0, "verdict": ""}
    try:
        cid = subprocess.run(["docker", "ps", "--filter", f"name={name_filter}", "-q"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        cid = cid.split("\n")[0]
        if not cid:
            return empty
        r = subprocess.run(["docker", "logs", "--tail", str(n), cid],
                           capture_output=True, text=True, timeout=20)
        log = (r.stdout or "") + (r.stderr or "")
    except Exception:
        return empty
    low = log.lower()
    d = {
        "downloads": low.count("[download] destination") + low.count("has already been downloaded"),
        "throttled": low.count("rate-limited") + low.count("try again later"),
        "agegate": low.count("confirm your age"),
        "forbid": low.count("403") + low.count("forbidden"),
        "unavail": low.count("video unavailable"),
    }
    if d["throttled"] > 5:
        d["verdict"] = "throttled by YouTube (will recover; sleeps active)"
    elif d["forbid"] > 5:
        d["verdict"] = "403s — check yt-dlp/Deno"
    elif d["downloads"] > 0:
        d["verdict"] = "downloading normally"
    else:
        d["verdict"] = "quiet"
    return d


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
    feat = _count(db, "SELECT COUNT(*) FROM reccobeats_features WHERE status='ok'")

    # rate + ETA from the delta since last run. Default the state file to the
    # user's home, NOT next to the db — work/ is owned by root (the container
    # runs as root), so a sibling file would be unwritable by the cron user.
    default_state = os.path.join(os.path.expanduser("~"), ".audiolens_notify_state.json")
    state_path = pathlib.Path(a.state or default_state)
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
    cls = _classify_recent(a.container) if not a.dry_run else {
        "downloads": 5, "throttled": 118, "agegate": 12, "forbid": 0,
        "unavail": 121, "verdict": "throttled by YouTube (will recover; sleeps active)",
    }
    # Title becomes an HTTP header (latin-1 only) — keep it plain ASCII.
    title = f"AudioLens - {'STOPPED' if stopped else 'running'} ({phase})"
    body = (
        f"Downloads: {got:,}/{total:,} ({pct:.1f}%)\n"
        f"{rate_line}\n"
        f"Failed: {failed:,}   Analyzed: {analyzed:,}\n"
        f"Features: {feat:,}\n"
        f"— recent —\n"
        f"state: {cls['verdict'] or '—'}\n"
        f"dl {cls['downloads']} · throttled {cls['throttled']} · "
        f"agegate {cls['agegate']} · 403 {cls['forbid']}"
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
            "Title": title.encode("ascii", "ignore").decode(),  # header = latin-1 safe
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
