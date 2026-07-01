"""Tiny live status page for the overnight build — reachable from your phone
over Tailscale, anywhere, always current.

Serves one auto-refreshing HTML page (no deps beyond the stdlib). Reads the
SQLite catalog for counts and classifies the container's recent logs
(throttled / 403 / downloading) exactly like status.sh, plus a log tail.

Run on the HOST (needs docker + read access to work/catalog.db):

    python3 scripts/status_server.py \
        --sqlite ~/spotify/audiolens/work/catalog.db --port 8899

Then, with Tailscale up on both the server and your phone, open:
    http://<server-tailscale-name>:8899     (or http://100.x.y.z:8899)

Bind is 0.0.0.0 so it's reachable on your tailnet (a private network — only
your devices). Keep the port off the public internet.
"""

from __future__ import annotations

import argparse
import html
import sqlite3
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None


def _count(db, sql):
    try:
        return db.execute(sql).fetchone()[0]
    except Exception:
        return 0


def _container(name):
    try:
        cid = subprocess.run(["docker", "ps", "--filter", f"name={name}", "-q"],
                             capture_output=True, text=True, timeout=15).stdout.strip().split("\n")[0]
        if not cid:
            return None, ""
        st = subprocess.run(["docker", "ps", "--filter", f"id={cid}", "--format", "{{.Status}}"],
                            capture_output=True, text=True, timeout=15).stdout.strip()
        logs = subprocess.run(["docker", "logs", "--tail", "250", cid],
                              capture_output=True, text=True, timeout=20)
        return st, (logs.stdout or "") + (logs.stderr or "")
    except Exception:
        return None, ""


def _snapshot():
    db = sqlite3.connect(ARGS.sqlite)
    total = _count(db, "SELECT COUNT(*) FROM catalog_tracks")
    got = _count(db, "SELECT COUNT(*) FROM catalog_tracks WHERE audio_track_id IS NOT NULL")
    failed = _count(db, "SELECT COUNT(*) FROM download_failures")
    analyzed = _count(db, "SELECT COUNT(*) FROM processing_state WHERE stage='document' AND status='done'")
    feat = _count(db, "SELECT COUNT(*) FROM reccobeats_features WHERE status='ok'")
    st, log = _container(ARGS.container)
    low = log.lower()
    cls = {
        "downloads": low.count("[download] destination") + low.count("has already been downloaded"),
        "throttled": low.count("rate-limited") + low.count("try again later"),
        "agegate": low.count("confirm your age"),
        "forbid": low.count("403") + low.count("forbidden"),
        "unavail": low.count("video unavailable"),
    }
    if st is None:
        verdict, vclass = "container STOPPED — re-run `up -d`", "bad"
    elif cls["throttled"] > 5:
        verdict, vclass = "throttled by YouTube — recovering, sleeps active", "warn"
    elif cls["forbid"] > 5:
        verdict, vclass = "403s — check yt-dlp / Deno", "bad"
    elif cls["downloads"] > 0:
        verdict, vclass = "downloading normally", "ok"
    else:
        verdict, vclass = "quiet / idle", "warn"
    tail = "\n".join(log.strip().splitlines()[-14:])
    pct = (100 * got / total) if total else 0
    return dict(total=total, got=got, failed=failed, analyzed=analyzed, feat=feat,
                status=st or "stopped", verdict=verdict, vclass=vclass, cls=cls,
                pct=pct, tail=tail)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh}">
<title>AudioLens status</title>
<style>
 body{{font-family:-apple-system,system-ui,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:18px}}
 h1{{font-size:17px;margin:0 0 4px}} .sub{{color:#8a8f98;font-size:12px;margin-bottom:14px}}
 .bar{{height:12px;background:#22262e;border-radius:6px;overflow:hidden;margin:10px 0}}
 .bar>i{{display:block;height:100%;width:{pct}%;background:#3b82f6}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:14px 0}}
 .card{{background:#171a21;border:1px solid #262b34;border-radius:10px;padding:12px}}
 .card b{{font-size:22px;display:block}} .card span{{color:#8a8f98;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
 .state{{padding:12px;border-radius:10px;font-weight:600;margin:6px 0 14px}}
 .ok{{background:#0f2a17;color:#5ee08a;border:1px solid #1c4a2b}}
 .warn{{background:#2a230f;color:#e0b95e;border:1px solid #4a3c1c}}
 .bad{{background:#2a0f13;color:#e05e6e;border:1px solid #4a1c24}}
 .mini{{display:flex;gap:14px;flex-wrap:wrap;color:#8a8f98;font-size:12px;margin-bottom:12px}}
 pre{{background:#0b0d11;border:1px solid #262b34;border-radius:10px;padding:10px;font-size:11px;
      overflow:auto;white-space:pre-wrap;word-break:break-word;max-height:40vh}}
</style></head><body>
<h1>🎵 AudioLens — {status}</h1>
<div class="sub">auto-refresh {refresh}s · {ts}</div>
<div class="state {vclass}">{verdict}</div>
<div class="bar"><i></i></div>
<div class="sub">{got} / {total} downloaded ({pct:.1f}%)</div>
<div class="grid">
 <div class="card"><span>Downloaded</span><b>{got}</b></div>
 <div class="card"><span>Failed</span><b>{failed}</b></div>
 <div class="card"><span>Analyzed</span><b>{analyzed}</b></div>
 <div class="card"><span>Features</span><b>{feat}</b></div>
</div>
<div class="mini">
 <span>recent: dl {c_dl}</span><span>throttled {c_thr}</span>
 <span>age-gated {c_age}</span><span>403 {c_403}</span><span>unavail {c_un}</span>
</div>
<div class="sub">last log lines</div>
<pre>{tail}</pre>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_response(404); self.end_headers(); return
        import datetime
        s = _snapshot()
        page = PAGE.format(
            refresh=ARGS.refresh, pct=s["pct"], status=html.escape(s["status"]),
            verdict=html.escape(s["verdict"]), vclass=s["vclass"],
            got=f"{s['got']:,}", total=f"{s['total']:,}", failed=f"{s['failed']:,}",
            analyzed=f"{s['analyzed']:,}", feat=f"{s['feat']:,}",
            c_dl=s["cls"]["downloads"], c_thr=s["cls"]["throttled"], c_age=s["cls"]["agegate"],
            c_403=s["cls"]["forbid"], c_un=s["cls"]["unavail"],
            tail=html.escape(s["tail"]) or "(no logs)",
            ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--container", default="audiolens-overnight")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--refresh", type=int, default=30, help="page auto-refresh seconds")
    ARGS = ap.parse_args()
    print(f"status page on http://0.0.0.0:{ARGS.port}  (open via your Tailscale name/IP)")
    ThreadingHTTPServer(("0.0.0.0", ARGS.port), H).serve_forever()


if __name__ == "__main__":
    main()
