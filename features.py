"""
features.py
------------
Signal feature extraction:
- FFT (single-sided frequency spectrum)
- Time-domain: RMS, Peak, Peak-to-Peak, Mean, Variance, Crest Factor,
  Kurtosis, Skewness, Shape/Impulse/Clearance Factor
  (the "factor" family and kurtosis/skewness are the classical indicators
  used in industrial bearing diagnostics because they react strongly to
  short, sharp impacts - the vibration signature of a localized defect -
  long before RMS energy noticeably changes)
- Dominant frequency / spectral energy (derived from FFT)
- Windowing utilities to turn one long signal into many feature samples,
  which is what makes per-window statistics (and Isolation Forest /
  classifiers trained on them) meaningful for a single uploaded file.
"""
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import skew as _scipy_skew

FEATURE_COLUMNS = [
    "rms",
    "peak",
    "peak_to_peak",
    "mean",
    "variance",
    "crest_factor",
    "kurtosis",
    "skewness",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "dominant_freq",
    "spectral_energy",
]


def compute_fft(signal: np.ndarray, fs: float):
    """
    Compute the single-sided amplitude spectrum of a real-valued signal.

    Parameters
    ----------
    signal : np.ndarray
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freqs : np.ndarray
    magnitude : np.ndarray
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])

    window = np.hanning(n)
    spectrum = rfft(signal * window)
    freqs = rfftfreq(n, d=1.0 / fs)
    magnitude = np.abs(spectrum) / n
    return freqs, magnitude


def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_peak(signal: np.ndarray) -> float:
    return float(np.max(np.abs(signal)))


def compute_peak_to_peak(signal: np.ndarray) -> float:
    """Max - min. Captures total swing of the signal, robust to DC bias."""
    return float(np.max(signal) - np.min(signal))


def compute_mean(signal: np.ndarray) -> float:
    return float(np.mean(signal))


def compute_variance(signal: np.ndarray) -> float:
    return float(np.var(signal))


def compute_crest_factor(signal: np.ndarray) -> float:
    """
    Peak / RMS. WHY: a healthy, roughly sinusoidal vibration signal has a
    crest factor near sqrt(2)~=1.41. Localized impacts (e.g. a spalled
    bearing race) create sharp peaks that raise the peak value much more
    than the RMS, so crest factor rises - an early, simple fault indicator.
    """
    rms = compute_rms(signal)
    if rms == 0:
        return 0.0
    return float(compute_peak(signal) / rms)


def compute_kurtosis(signal: np.ndarray) -> float:
    """
    Standard (non-excess) kurtosis, fisher=False so a Gaussian/healthy
    signal scores ~3. WHY: kurtosis is extremely sensitive to the sharp,
    intermittent impulses produced by bearing defects (much more than RMS,
    which averages energy over the whole window) - it is one of the most
    widely used indicators in bearing diagnostics.
    """
    if len(signal) < 4 or np.std(signal) == 0:
        return 3.0
    return float(_scipy_kurtosis(signal, fisher=False, bias=True))


def compute_skewness(signal: np.ndarray) -> float:
    """Distribution asymmetry. A healthy signal is typically near-symmetric (~0)."""
    if len(signal) < 3 or np.std(signal) == 0:
        return 0.0
    return float(_scipy_skew(signal, bias=True))


def compute_shape_factor(signal: np.ndarray) -> float:
    """RMS / mean(|x|). Sensitive to the overall waveform shape."""
    mean_abs = np.mean(np.abs(signal))
    if mean_abs == 0:
        return 0.0
    return float(compute_rms(signal) / mean_abs)


def compute_impulse_factor(signal: np.ndarray) -> float:
    """Peak / mean(|x|). Like crest factor, but normalized by mean absolute value."""
    mean_abs = np.mean(np.abs(signal))
    if mean_abs == 0:
        return 0.0
    return float(compute_peak(signal) / mean_abs)


def compute_clearance_factor(signal: np.ndarray) -> float:
    """
    Peak / (mean(sqrt(|x|)))^2. The most sensitive of the "factor" family
    to early-stage impulsive faults, at the cost of being noisier.
    """
    sqrt_mean = np.mean(np.sqrt(np.abs(signal)))
    denom = sqrt_mean ** 2
    if denom == 0:
        return 0.0
    return float(compute_peak(signal) / denom)


def compute_dominant_frequency(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    """Frequency bin with the highest magnitude, excluding the DC bin."""
    if len(freqs) <= 1:
        return 0.0
    idx = int(np.argmax(magnitude[1:])) + 1
    return float(freqs[idx])


def compute_spectral_energy(magnitude: np.ndarray) -> float:
    return float(np.sum(np.square(magnitude)))


def extract_features_window(window: np.ndarray, fs: float) -> dict:
    """Extract the full feature set for a single window of signal."""
    freqs, mag = compute_fft(window, fs)
    return {
        "rms": compute_rms(window),
        "peak": compute_peak(window),
        "peak_to_peak": compute_peak_to_peak(window),
        "mean": compute_mean(window),
        "variance": compute_variance(window),
        "crest_factor": compute_crest_factor(window),
        "kurtosis": compute_kurtosis(window),
        "skewness": compute_skewness(window),
        "shape_factor": compute_shape_factor(window),
        "impulse_factor": compute_impulse_factor(window),
        "clearance_factor": compute_clearance_factor(window),
        "dominant_freq": compute_dominant_frequency(freqs, mag),
        "spectral_energy": compute_spectral_energy(mag),
    }


def segment_signal(signal: np.ndarray, window_size: int, overlap: float = 0.0):
    """
    Split a signal into (possibly overlapping) fixed-length windows.

    overlap : float in [0, 1)
        Fraction of overlap between consecutive windows.

    Returns
    -------
    windows : list[np.ndarray]
    starts  : list[int]   (start sample index of each window)
    """
    signal = np.asarray(signal, dtype=float)
    window_size = max(1, min(int(window_size), len(signal)))
    step = max(1, int(window_size * (1 - overlap)))

    windows, starts = [], []
    for start in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[start:start + window_size])
        starts.append(start)

    if not windows:  # signal shorter than window_size
        windows.append(signal)
        starts.append(0)

    return windows, starts


def extract_features_dataframe(
    signal: np.ndarray, fs: float, window_size: int, overlap: float = 0.0
) -> pd.DataFrame:
    """
    Segment the signal into windows and extract features for each window.

    Returns a DataFrame with one row per window, including the feature
    columns plus the window's start sample/time.
    """
    windows, starts = segment_signal(signal, window_size, overlap)

    rows = []
    for w, s in zip(windows, starts):
        feats = extract_features_window(w, fs)
        feats["window_start_sample"] = s
        feats["window_start_time"] = s / fs
        rows.append(feats)

    cols = ["window_start_sample", "window_start_time"] + FEATURE_COLUMNS
    return pd.DataFrame(rows)[cols]
