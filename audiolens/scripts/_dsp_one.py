"""Analyze ONE audio file and print its DSP document as JSON to stdout.

Run as an isolated subprocess by run_pipeline so a native crash (madmom /
libsndfile segfault) kills only this process — the parent sees a non-zero exit
code and marks just that track failed, instead of the whole run dying.

    python scripts/_dsp_one.py /path/to/audio.opus   ->  JSON on stdout
Logs go to stderr (inherited by the parent, so they still show in docker logs).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> int:
    from services.extractor.app.pipeline import CorruptedAudioError, analyze_track
    path = sys.argv[1]
    try:
        doc = analyze_track(path)
    except CorruptedAudioError:
        sys.stderr.write("corrupted_audio\n")
        return 3
    sys.stdout.write(json.dumps(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
