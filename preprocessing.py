"""
preprocessing.py
-----------------
Signal preprocessing utilities:
- Cleaning (handling missing / infinite values)
- Normalization
- Digital filtering (Butterworth low/high/band-pass) - WHY: bearing-fault
  energy and structural resonances often live in specific frequency bands;
  isolating them before further analysis (e.g. envelope analysis) improves
  the signal-to-noise ratio of the fault signature.
- Synthetic Gaussian noise injection + SNR estimation - WHY: lets us
  quantify how robust a detection method is as measurement conditions
  degrade, and to demonstrate the benefit of filtering.
- Downsampling - WHY: lets us study the Nyquist trade-off between
  sampling rate and the highest fault frequency we can still resolve.
"""
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, decimate


def clean_signal(series: pd.Series) -> np.ndarray:
    """
    Clean a raw sensor column:
    - Replace +/- inf with NaN
    - Linearly interpolate interior NaNs
    - Back/forward-fill any remaining NaNs at the edges

    Returns a 1-D numpy array of floats.
    """
    s = series.replace([np.inf, -np.inf], np.nan)
    s = s.interpolate(limit_direction="both")
    s = s.bfill().ffill()

    if s.isna().any():
        # Entire column was empty/NaN -> fall back to zeros
        s = s.fillna(0.0)

    return s.to_numpy(dtype=float)


def normalize_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize a signal.

    method:
        'zscore' -> (x - mean) / std
        'minmax' -> (x - min) / (max - min)
    """
    signal = np.asarray(signal, dtype=float)

    if method == "zscore":
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal - mean
        return (signal - mean) / std

    if method == "minmax":
        min_v, max_v = np.min(signal), np.max(signal)
        rng = max_v - min_v
        if rng == 0:
            return np.zeros_like(signal)
        return (signal - min_v) / rng

    raise ValueError("method must be 'zscore' or 'minmax'")


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """Subtract the mean (DC component) from the signal, useful before FFT."""
    signal = np.asarray(signal, dtype=float)
    return signal - np.mean(signal)


# --------------------------------------------------------------------------
# Digital filtering (Butterworth)
# --------------------------------------------------------------------------
FilterType = str  # "lowpass" | "highpass" | "bandpass"


def design_butterworth(
    filter_type: FilterType,
    cutoff: Union[float, Tuple[float, float]],
    fs: float,
    order: int = 4,
):
    """
    Design a Butterworth filter.

    Parameters
    ----------
    filter_type : 'lowpass' | 'highpass' | 'bandpass'
    cutoff : float for lowpass/highpass, or (low_hz, high_hz) for bandpass
    fs : sampling frequency in Hz
    order : filter order (higher = steeper roll-off, but more ringing/delay;
            order 4 is a common, well-behaved default for vibration analysis)

    Returns
    -------
    (b, a) : filter coefficients for use with scipy.signal.filtfilt
    """
    nyquist = fs / 2.0

    if filter_type in ("lowpass", "highpass"):
        normal_cutoff = float(cutoff) / nyquist
        normal_cutoff = float(np.clip(normal_cutoff, 1e-6, 0.999999))
        b, a = butter(order, normal_cutoff, btype=filter_type)
    elif filter_type == "bandpass":
        low, high = cutoff
        low_n = float(np.clip(low / nyquist, 1e-6, 0.999998))
        high_n = float(np.clip(high / nyquist, low_n + 1e-6, 0.999999))
        b, a = butter(order, [low_n, high_n], btype="band")
    else:
        raise ValueError("filter_type must be 'lowpass', 'highpass' or 'bandpass'")

    return b, a


def apply_filter(
    signal: np.ndarray,
    filter_type: FilterType,
    cutoff: Union[float, Tuple[float, float]],
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth filter to a signal.

    Uses filtfilt (forward-backward filtering) instead of lfilter so the
    filter introduces no phase shift / time delay - important for vibration
    analysis where the exact timing of impulsive fault events matters.
    """
    signal = np.asarray(signal, dtype=float)
    b, a = design_butterworth(filter_type, cutoff, fs, order)

    # filtfilt needs the signal to be longer than ~3x the filter order
    min_len = 3 * (max(len(a), len(b)))
    if len(signal) <= min_len:
        return signal.copy()

    return filtfilt(b, a, signal)


# --------------------------------------------------------------------------
# Noise injection & SNR
# --------------------------------------------------------------------------
def compute_signal_power(signal: np.ndarray) -> float:
    """Average power of a signal: mean(x^2)."""
    signal = np.asarray(signal, dtype=float)
    return float(np.mean(np.square(signal)))


def compute_snr_db(clean_signal_: np.ndarray, noisy_signal: np.ndarray) -> float:
    """
    Estimate SNR (in dB) given a clean reference and its noisy version.
    SNR = 10*log10(P_signal / P_noise), where the noise is the residual
    (noisy - clean).
    """
    clean_signal_ = np.asarray(clean_signal_, dtype=float)
    noisy_signal = np.asarray(noisy_signal, dtype=float)
    noise = noisy_signal - clean_signal_

    p_signal = compute_signal_power(clean_signal_)
    p_noise = compute_signal_power(noise)

    if p_noise == 0:
        return float("inf")
    return float(10 * np.log10(p_signal / p_noise))


def add_gaussian_noise(signal: np.ndarray, snr_db: float, random_state: int = 42) -> np.ndarray:
    """
    Add zero-mean white Gaussian noise to a signal so that the resulting
    SNR matches the requested value (in dB).

    WHY: this lets us empirically test, e.g., "at what SNR do BPFO/BPFI
    peaks disappear from the raw spectrum but remain visible in the
    envelope spectrum?" - a standard robustness study in bearing diagnostics.
    """
    signal = np.asarray(signal, dtype=float)
    rng = np.random.default_rng(random_state)

    p_signal = compute_signal_power(signal)
    snr_linear = 10 ** (snr_db / 10.0)
    p_noise = p_signal / snr_linear if snr_linear > 0 else p_signal

    noise = rng.normal(loc=0.0, scale=np.sqrt(max(p_noise, 1e-30)), size=signal.shape)
    return signal + noise


# --------------------------------------------------------------------------
# Downsampling
# --------------------------------------------------------------------------
def downsample_signal(signal: np.ndarray, fs: float, target_fs: float) -> Tuple[np.ndarray, float]:
    """
    Downsample a signal to (approximately) target_fs, applying an
    anti-aliasing filter first (via scipy.signal.decimate) to respect the
    Nyquist criterion and avoid folding high-frequency fault harmonics
    back into the band of interest.

    Returns
    -------
    new_signal : np.ndarray
    new_fs : float (actual resulting sampling frequency)
    """
    signal = np.asarray(signal, dtype=float)
    if target_fs >= fs:
        return signal.copy(), fs

    factor = int(round(fs / target_fs))
    factor = max(1, factor)
    if factor == 1:
        return signal.copy(), fs

    # decimate applies a built-in anti-aliasing low-pass filter (Chebyshev)
    new_signal = decimate(signal, factor, zero_phase=True)
    new_fs = fs / factor
    return new_signal, new_fs
