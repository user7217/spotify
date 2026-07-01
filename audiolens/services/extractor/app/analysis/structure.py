"""Structural segmentation + functional section labeling (spec).

Pipeline:
  1. Boundary detection — spectral clustering on a recurrence + path-enhanced
     affinity over beat-synchronous MFCC+chroma (McFee & Ellis 2014 style).
  2. Cluster segments by timbre+chroma centroid -> repeated section groups.
  3. Heuristic functional labels (intro/verse/chorus/bridge/breakdown/solo/
     outro) from position, repetition count, relative energy and vocal band.

Output matches the spec: sections[{start,end,label,confidence}] plus
section_repetition, structural_similarity, boundaries.
"""

import logging

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.ndimage import median_filter
from scipy.sparse.csgraph import laplacian

import librosa

log = logging.getLogger("audiolens.extractor.structure")

EXTRACTOR_VERSION = "structure-v2"


def _segment_features(y: np.ndarray, sr: int, hop: int = 512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    if len(beats) < 8:  # fallback to fixed 0.5 s grid
        beats = np.arange(0, mfcc.shape[1], int(sr / hop / 2))
    Msync = librosa.util.sync(mfcc, beats, aggregate=np.median)
    Csync = librosa.util.sync(chroma, beats, aggregate=np.median)
    Rsync = librosa.util.sync(rms[None, :], beats, aggregate=np.median)[0]
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    return Msync, Csync, Rsync, beat_times


def _boundaries(Msync: np.ndarray, Csync: np.ndarray, k: int | None = None) -> np.ndarray:
    X = np.vstack([librosa.util.normalize(Msync, axis=0),
                   librosa.util.normalize(Csync, axis=0)])
    R = librosa.segment.recurrence_matrix(X, width=3, mode="affinity", sym=True)
    df = librosa.segment.timelag_filter(median_filter)
    Rf = df(R, size=(1, 7))
    path = np.exp(-np.diff(X, axis=1) ** 2).mean(axis=0)
    A = Rf * 0.5
    n = A.shape[0]
    A[np.arange(n - 1), np.arange(1, n)] += path * 0.5
    A[np.arange(1, n), np.arange(n - 1)] += path * 0.5
    L = laplacian(A, normed=True)
    evals, evecs = np.linalg.eigh(L)
    k = k or int(np.clip(np.searchsorted(np.cumsum(evals[:12]) / (evals[:12].sum() + 1e-12), 0.5) + 3, 4, 10))
    E = evecs[:, :k]
    E = librosa.util.normalize(E, axis=1)
    labels = sch.fcluster(sch.linkage(E, method="ward"), t=k, criterion="maxclust")
    return np.flatnonzero(np.diff(labels)) + 1, labels


def _merge_short(segs: list[dict], min_len: float = 3.0) -> list[dict]:
    """Absorb segments shorter than min_len into their more similar neighbor.

    Contiguity is preserved; cluster/energy of the absorbing segment are kept
    (duration-weighted energy update)."""
    out: list[dict] = []
    for s in segs:
        dur = s["end"] - s["start"]
        if out and dur < min_len:
            prev = out[-1]
            w_prev = prev["end"] - prev["start"]
            prev["_energy"] = (prev["_energy"] * w_prev + s["_energy"] * dur) / (w_prev + dur + 1e-9)
            prev["end"] = s["end"]
        elif not out and dur < min_len and len(segs) > 1:
            # head fragment: push into the next one by deferring
            s["_defer"] = True
            out.append(s)
        else:
            if out and out[-1].get("_defer"):
                frag = out.pop()
                s["start"] = frag["start"]
            out.append(s)
    for s in out:
        s.pop("_defer", None)
    return out


def _label_sections(segs: list[dict], duration: float) -> None:
    """Heuristic functional labels in place."""
    if not segs:
        return
    energies = np.array([s["_energy"] for s in segs])
    counts = {}
    for s in segs:
        counts[s["_cluster"]] = counts.get(s["_cluster"], 0) + 1
    e_norm = (energies - energies.min()) / (np.ptp(energies) + 1e-12)
    # chorus cluster: most repeated among high-energy segments
    rep_clusters = sorted(counts, key=lambda c: (-counts[c], -np.mean(e_norm[[i for i, s in enumerate(segs) if s["_cluster"] == c]])))
    chorus = rep_clusters[0] if counts[rep_clusters[0]] >= 2 else None
    verse = rep_clusters[1] if len(rep_clusters) > 1 and counts[rep_clusters[1]] >= 2 else None

    for i, s in enumerate(segs):
        c, e = s["_cluster"], e_norm[i]
        if i == 0 and s["end"] < duration * 0.2:
            label = "intro"
        elif i == len(segs) - 1 and s["start"] > duration * 0.75:
            label = "outro"
        elif c == chorus:
            label = "chorus"
        elif c == verse:
            label = "verse"
        elif e < 0.25:
            label = "breakdown"
        elif counts[c] == 1 and e > 0.6 and duration * 0.4 < s["start"] < duration * 0.85:
            label = "solo" if e > 0.8 else "bridge"
        elif counts[c] == 1:
            label = "bridge"
        else:
            label = "verse"
        s["label"] = label


def extract_structure(y: np.ndarray, sr: int) -> dict:
    duration = len(y) / sr
    Msync, Csync, Rsync, beat_times = _segment_features(y, sr)
    try:
        bounds, labels = _boundaries(Msync, Csync)
    except Exception as e:  # noqa: BLE001 — degenerate audio
        log.warning("segmentation failed (%s); single-section fallback", e)
        return {
            "sections": [{"start": 0.0, "end": round(duration, 2), "label": "verse",
                          "confidence": 0.1}],
            "section_repetition": 0.0,
            "structural_similarity": 0.0,
            "boundaries": [0.0, round(duration, 2)],
            "extractor_version": EXTRACTOR_VERSION,
        }

    idx = np.concatenate([[0], bounds, [len(labels)]])
    segs = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b <= a:
            continue
        start = float(beat_times[a]) if a < len(beat_times) else duration
        end = float(beat_times[b - 1]) if b - 1 < len(beat_times) else duration
        cluster = int(np.bincount(labels[a:b]).argmax())
        segs.append({
            "start": round(start, 2),
            "end": round(max(end, start + 0.1), 2),
            "_cluster": cluster,
            "_energy": float(np.mean(Rsync[a:b])),
            "confidence": round(float(np.mean(labels[a:b] == cluster)), 3),
        })
    if segs:
        segs[0]["start"] = 0.0
        segs[-1]["end"] = round(duration, 2)
    segs = _merge_short(segs, min_len=3.0)
    _label_sections(segs, duration)

    # repetition: fraction of time in clusters that appear 2+ times
    counts = {}
    for s in segs:
        counts[s["_cluster"]] = counts.get(s["_cluster"], 0) + 1
    rep_time = sum(s["end"] - s["start"] for s in segs if counts[s["_cluster"]] > 1)
    total = sum(s["end"] - s["start"] for s in segs) or 1.0

    # structural similarity: mean off-diagonal recurrence within same-cluster pairs
    X = np.vstack([librosa.util.normalize(Msync, axis=0), librosa.util.normalize(Csync, axis=0)])
    R = librosa.segment.recurrence_matrix(X, width=3, mode="affinity", sym=True)
    struct_sim = float(np.mean(R[R > 0])) if (R > 0).any() else 0.0

    for s in segs:
        s.pop("_cluster"), s.pop("_energy")

    return {
        "sections": segs,
        "section_repetition": round(rep_time / total, 3),
        "structural_similarity": round(struct_sim, 3),
        "boundaries": [s["start"] for s in segs] + [round(duration, 2)],
        "extractor_version": EXTRACTOR_VERSION,
    }
