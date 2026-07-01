"""Bulk catalog ingestion: walk a directory, upload everything to the API.

Usage: python scripts/bulk_ingest.py /path/to/music --api http://localhost:8000
"""
import argparse
import pathlib
import httpx

EXTS = {".mp3", ".flac", ".wav", ".aac", ".m4a", ".ogg"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--api", default="http://localhost:8000")
    args = p.parse_args()

    files = [f for f in pathlib.Path(args.root).rglob("*") if f.suffix.lower() in EXTS]
    print(f"found {len(files)} audio files")

    with httpx.Client(timeout=120) as client:
        for i, f in enumerate(files, 1):
            with open(f, "rb") as fh:
                r = client.post(f"{args.api}/v1/tracks",
                                files={"file": (f.name, fh)})
            status = r.json()
            dedup = " (dedup)" if status.get("deduplicated") else ""
            print(f"[{i}/{len(files)}] {f.name} -> {r.status_code}{dedup}")


if __name__ == "__main__":
    main()
