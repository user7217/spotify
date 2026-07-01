"""Full low-level feature extraction (spec: spectral / energy / frequency / pitch).

All librosa-based; LUFS via pyloudnorm when available (falls back to gated
RMS approximation). Returns scalars + a `detail` dict matching the
low_level_features table.
"""

import logging

import numpy as np

import librosa

log = logging.getLogger("audiolens.extractor.lowlevel")

try:
    import pyloudnorm
except ImportError:  # pragma: no cover
    pyloudnorm = None

EXTRACTOR_VERSION = "lowlevel-v2"
N_MFCC = 24  # spec: 20-40


def _stats(M: np.ndarray, axis: int = 1) -> dict:
    return {
        "mean": np.mean(M, axis=axis).tolist(),
        "std": np.std(M, axis=axis).tolist(),
    }


def _spectral_entropy(S: np.ndarray) -> float:
    """Mean frame-wise Shannon entropy of the normalized power spectrum (0-1)."""
    P = S / (S.sum(axis=0, keepdims=True) + 1e-12)
    H = -(P * np.log2(P + 1e-12)).sum(axis=0)
    return float(np.mean(H) / np.log2(S.shape[0]))


def _spectral_flux(S: np.ndarray) -> float:
    diff = np.diff(S, axis=1)
    return float(np.mean(np.sqrt((np.maximum(diff, 0) ** 2).sum(axis=0))))


def _lufs(y: np.ndarray, sr: int) -> float:
    if pyloudnorm is not None and len(y) > sr * 0.5:
        try:
            return float(pyloudnorm.Meter(sr).integrated_loudness(y.astype(np.float64)))
        except Exception as e:  # noqa: BLE001
            log.debug("pyloudnorm failed: %s", e)
    # gated-RMS approximation (≈ LUFS for typical program material)
    rms = librosa.feature.rms(y=y)[0]
    gate = rms > (np.max(rms) * 0.03)
    val = np.sqrt(np.mean(rms[gate] ** 2)) if gate.any() else np.sqrt(np.mean(rms**2))
    return float(20 * np.log10(val + 1e-12) - 0.691)


def extract_low_level(y: np.ndarray, sr: int) -> dict:
    S = np.abs(librosa.stft(y)) ** 2
    S_mag = np.sqrt(S)

    # --- spectral -----------------------------------------------------------
    centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(S=S_mag, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S_mag, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S_mag)[0]

    # --- energy --------------------------------------------------------------
    rms = librosa.feature.rms(y=y)[0]
    peak = float(np.max(np.abs(y)))
    rms_global = float(np.sqrt(np.mean(y**2)))
    crest = float(peak / (rms_global + 1e-12))
    # dynamic range: loudest vs quietest active frames (dB)
    active = rms[rms > np.max(rms) * 0.01]
    dyn_range = float(
        20 * np.log10((np.percentile(active, 99) + 1e-12) / (np.percentile(active, 5) + 1e-12))
    ) if len(active) else 0.0

    # --- frequency -----------------------------------------------------------
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    d_mfcc = librosa.feature.delta(mfcc)
    dd_mfcc = librosa.feature.delta(mfcc, order=2)

    # --- pitch -----------------------------------------------------------------
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # HPCP approximation: energy-weighted CENS chroma (use Essentia HPCP if present)
    hpcp = librosa.feature.chroma_cens(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    voiced = f0[np.isfinite(f0)]
    pitch_hist, _ = np.histogram(
        librosa.hz_to_midi(voiced[voiced > 0]) if len(voiced) else np.array([]),
        bins=np.arange(24, 109),
    )

    return {
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "spectral_flatness": float(np.mean(flatness)),
        "spectral_flux": _spectral_flux(S_mag),
        "spectral_entropy": _spectral_entropy(S),
        "rms_energy": rms_global,
        "dynamic_range_db": dyn_range,
        "loudness_lufs": _lufs(y, sr),
        "peak_amplitude": peak,
        "crest_factor": crest,
        "detail": {
            "spectral_centroid": {"std": float(np.std(centroid))},
            "spectral_contrast": _stats(contrast),
            "mfcc": _stats(mfcc),
            "delta_mfcc": _stats(d_mfcc),
            "delta2_mfcc": _stats(dd_mfcc),
            "mel": {
                "mean_db": float(np.mean(librosa.power_to_db(mel))),
                "band_means": np.mean(librosa.power_to_db(mel), axis=1).tolist(),
            },
            "chroma": _stats(chroma),
            "hpcp": _stats(hpcp),
            "tonnetz": _stats(tonnetz),
            "pitch_histogram": pitch_hist.tolist(),
            "n_mfcc": N_MFCC,
        },
        "extractor_version": EXTRACTOR_VERSION,
    }
