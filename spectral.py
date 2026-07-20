"""
spectral.py
-----------
Frequency-domain and time-frequency analysis beyond the basic FFT in
features.py:

- Power Spectral Density (Welch's method)   -> compute_psd_welch
- Periodogram (single-segment PSD estimate) -> compute_periodogram
- Spectrogram / Short-Time Fourier Transform -> compute_spectrogram

WHY add PSD/Welch on top of the plain FFT:
A single long FFT gives high frequency resolution but a very noisy
(high-variance) amplitude estimate. Welch's method splits the signal into
overlapping segments, computes a periodogram for each, and averages them -
trading some frequency resolution for a much more statistically reliable
power estimate. This matters when comparing healthy vs. faulty spectra,
since we want differences in peak height to reflect the underlying physics,
not estimation noise.

WHY add a spectrogram:
Bearing faults are often non-stationary (load/speed varies, or fault
severity changes over the recording). A single FFT only shows the
*average* frequency content over the whole signal and can hide a fault
signature that's only present part of the time. The spectrogram (STFT)
shows how the spectrum evolves over time.
"""
from typing import Optional

import numpy as np
from scipy.signal import welch, periodogram, spectrogram


def compute_psd_welch(signal: np.ndarray, fs: float, nperseg: Optional[int] = None):
    """
    Estimate the Power Spectral Density using Welch's method.

    Returns
    -------
    freqs : np.ndarray
    psd : np.ndarray (power per Hz)
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 8:
        return np.array([]), np.array([])

    if nperseg is None:
        nperseg = min(1024, n)
    nperseg = max(8, min(int(nperseg), n))

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd


def compute_periodogram(signal: np.ndarray, fs: float):
    """
    Single-segment periodogram (no averaging - higher variance than Welch,
    but full frequency resolution; useful as a quick comparison).
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 4:
        return np.array([]), np.array([])
    freqs, pxx = periodogram(signal, fs=fs)
    return freqs, pxx


def compute_spectrogram(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
):
    """
    Compute the spectrogram (STFT magnitude squared per time/frequency bin).

    Returns
    -------
    freqs : np.ndarray
    times : np.ndarray
    Sxx : np.ndarray, shape (len(freqs), len(times))
        Power spectral density per segment, suitable for plotting with
        pcolormesh / imshow on a log scale.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    nperseg = max(8, min(int(nperseg), n))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = max(0, min(int(noverlap), nperseg - 1))

    freqs, times, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, times, Sxx
