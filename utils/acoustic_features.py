#!/usr/bin/env python3
"""
Handcrafted acoustic feature extractor for emotion-subtype analysis.

Twelve interpretable descriptors per utterance:

    f0_mean_voiced         mean pitch across voiced frames (Hz)
    f0_std_voiced          pitch variation across voiced frames (Hz)
    jitter                 short-term F0 stability: mean |dF0| / mean F0
                           across voiced frames. Larger = shakier pitch.
    voiced_frac            fraction of frames flagged voiced by an
                           RMS-threshold VAD (top 10% of max RMS).
    rms_mean               mean short-time energy
    rms_std                energy variation
    zcr_mean               mean zero crossing rate
    hnr_proxy              harmonic-to-noise proxy: RMS ratio of harmonic
                           component to residual, from HPSS. Higher =
                           more voiced / cleaner voice quality.
    spectral_centroid_mean brightness
    spectral_bandwidth_mean spectral spread
    spectral_rolloff_mean  high-frequency content
    mfcc1_mean             first MFCC coefficient (voice quality proxy)

Feature-set id bumped to acoustic_v2 to invalidate acoustic_v1 caches.
The v1 extractor had a broken voiced_frac (YIN never marks unvoiced).

The output is deterministic given the waveform, so it is safe to cache.
librosa is used for the FFT-based features and the yin pitch tracker.
"""

from typing import List

import numpy as np


FEATURE_SET_ID = "acoustic_v2"

FEATURE_NAMES: List[str] = [
    "f0_mean_voiced",
    "f0_std_voiced",
    "jitter",
    "voiced_frac",
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "hnr_proxy",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "mfcc1_mean",
]

FEATURE_DIM = len(FEATURE_NAMES)


def extract_features(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Extract a fixed-length acoustic feature vector from a waveform.

    Args:
        waveform: 1D float32 waveform, mono, arbitrary length.
        sr: sampling rate in Hz.

    Returns:
        [FEATURE_DIM] float32 vector matching FEATURE_NAMES order. Frames
        with numerical issues are guarded so the output is always finite.

    Raises:
        ImportError: if librosa is not installed.
    """
    import librosa  # local import so callers without librosa see a clear error

    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    if y.size == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # Normalize amplitude to reduce cross-corpus recording-level bias.
    peak = float(np.max(np.abs(y))) + 1e-8
    y = y / peak

    frame_length = 1024
    hop_length = 256

    # RMS energy (used both as a feature and as the voicing gate)
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length,
    )[0]
    rms_mean = float(np.mean(rms)) if rms.size > 0 else 0.0
    rms_std = float(np.std(rms)) if rms.size > 0 else 0.0

    # Energy-based voicing detection. YIN gives no unvoiced signal (it
    # always returns a pitch estimate clipped to [fmin, fmax]), so use an
    # RMS threshold at 10% of per-utterance max RMS as the VAD gate.
    if rms.size > 0 and rms.max() > 1e-6:
        voiced_thresh = float(rms.max()) * 0.1
        voiced_mask = rms > voiced_thresh
    else:
        voiced_mask = np.zeros_like(rms, dtype=bool)
    voiced_frac = float(voiced_mask.mean()) if voiced_mask.size > 0 else 0.0

    # F0 via YIN over the whole signal, then mask to voiced frames.
    try:
        f0 = librosa.yin(
            y, fmin=50, fmax=500, sr=sr,
            frame_length=frame_length, hop_length=hop_length,
        )
    except Exception:
        f0 = np.zeros_like(rms)

    n_frames = min(len(f0), len(voiced_mask))
    f0 = f0[:n_frames]
    voiced_here = voiced_mask[:n_frames]

    f0_voiced = f0[voiced_here] if voiced_here.any() else np.zeros(0)
    if f0_voiced.size >= 2:
        f0_mean = float(np.mean(f0_voiced))
        f0_std = float(np.std(f0_voiced))
        # Jitter proxy: normalized short-term F0 change
        d_f0 = np.abs(np.diff(f0_voiced))
        jitter = float(d_f0.mean() / (f0_mean + 1e-8))
    else:
        f0_mean = f0_std = jitter = 0.0

    # HNR proxy via HPSS: RMS ratio of harmonic to residual.
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        h_rms = float(np.sqrt(np.mean(y_harm ** 2)))
        p_rms = float(np.sqrt(np.mean(y_perc ** 2))) + 1e-8
        hnr_proxy = h_rms / p_rms
    except Exception:
        hnr_proxy = 0.0

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=frame_length, hop_length=hop_length,
    )[0]
    zcr_mean = float(np.mean(zcr)) if zcr.size > 0 else 0.0

    # Spectral shape
    sc = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length,
    )[0]
    sb = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length,
    )[0]
    sr_ = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length,
    )[0]
    sc_mean = float(np.mean(sc)) if sc.size > 0 else 0.0
    sb_mean = float(np.mean(sb)) if sb.size > 0 else 0.0
    sr_mean = float(np.mean(sr_)) if sr_.size > 0 else 0.0

    # First MFCC (voice-quality proxy)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=2, n_fft=frame_length, hop_length=hop_length,
    )
    mfcc1_mean = float(np.mean(mfcc[0])) if mfcc.shape[1] > 0 else 0.0

    vec = np.array([
        f0_mean, f0_std, jitter, voiced_frac,
        rms_mean, rms_std, zcr_mean, hnr_proxy,
        sc_mean, sb_mean, sr_mean,
        mfcc1_mean,
    ], dtype=np.float32)

    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
