"""
noise_study.py
--------------
All noise-analysis computations for Tab 5 of Motor Insight.

This module is deliberately kept Streamlit-free so every function is
independently testable and reusable outside the web app.

Functions
---------
run_snr_sweep()        : sweep multiple SNR targets and collect metrics
plot_zoomed_triplet()  : short zoomed time-domain view (Original/Noisy/Filtered)
plot_psd_triplet()     : PSD overlay showing noise-floor rise and filter effect
plot_snr_curves()      : multi-panel metric curves from the SNR sweep

Signal & Systems rationale
--------------------------
Showing three full 20-second time-domain plots of a signal with added
Gaussian noise teaches almost nothing: the noise is not visible at that
scale and the signal looks identical to the original. The meaningful
demonstrations are:

1. A SHORT zoomed segment (0.1–0.3 s): the individual noise samples
   become visible and the smoothing effect of the filter is clear.

2. The FREQUENCY DOMAIN: added white noise raises the PSD floor uniformly
   across all frequencies. A band-pass / low-pass filter then cuts the
   floor in the stop-band, recovering the signal SNR in the pass-band.
   This is the textbook illustration of "filtering = SNR improvement in
   the frequency domain".

3. The SNR SWEEP (the core experiment): as target SNR decreases from 20 dB
   to -5 dB we track how observable quantities change:
   • Kurtosis drops from fault-like (impulsive) toward ~3 (Gaussian)
     because noise washes out the impulsive structure that carries the
     bearing-fault signature.
   • The spectral noise floor rises, eventually burying the fault-frequency
     peaks that envelope analysis relies on.
   • Filtering partially reverses both degradations — this is the practical
     result that justifies using a band-pass filter before envelope analysis.
   This is a real DSP experiment, not just a visual demonstration.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.signal import welch


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _apply_filter_safe(signal: np.ndarray, fs: float, filter_cfg: Optional[dict]) -> np.ndarray:
    """
    Apply a filter defined by a config dict, with a safe fallback.
    filter_cfg = {"type": "bandpass"|"lowpass"|"highpass",
                  "cutoff": float or (float, float), "order": int}
    Returns the unmodified signal if filter_cfg is None or filtering fails.
    """
    if filter_cfg is None:
        return signal.copy()
    try:
        from preprocessing import apply_filter
        return apply_filter(
            signal,
            filter_cfg["type"],
            filter_cfg["cutoff"],
            fs,
            order=filter_cfg.get("order", 4),
        )
    except Exception:
        return signal.copy()


def _estimate_spectral_noise_floor(magnitude: np.ndarray) -> float:
    """
    Estimate the background noise floor of a spectrum.

    Uses the median of the lower 40% of magnitude values, which excludes
    the signal peaks while being robust to a few outliers.  Median is
    preferred over mean because a single strong peak would otherwise
    inflate the estimate.
    """
    if len(magnitude) == 0:
        return 0.0
    threshold = np.percentile(magnitude, 40)
    floor_vals = magnitude[magnitude <= threshold]
    return float(np.median(floor_vals)) if len(floor_vals) > 0 else float(np.median(magnitude))


def _compute_psd(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD — compact wrapper used throughout this module."""
    n = len(signal)
    if n < 8:
        return np.array([0.0]), np.array([1e-30])
    nperseg = min(1024, n)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd


# --------------------------------------------------------------------------
# SNR sweep
# --------------------------------------------------------------------------

DEFAULT_SNR_LEVELS: List[float] = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]


def run_snr_sweep(
    signal: np.ndarray,
    fs: float,
    snr_levels: Optional[List[float]] = None,
    filter_cfg: Optional[dict] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    For each target SNR level, inject Gaussian noise and measure how
    key diagnostic indicators degrade and how much filtering recovers them.

    Parameters
    ----------
    signal : np.ndarray
        The clean reference signal (working_signal in app.py).
    fs : float
    snr_levels : list of target SNR values in dB.
        Defaults to [-5, 0, 5, 10, 15, 20, 25] dB.
    filter_cfg : dict or None
        Filter configuration {"type", "cutoff", "order"}.
        If None, a default low-pass at 0.25×fs is used.
    random_state : int
        Base seed; each SNR level uses (random_state + i) so the noise
        realisations are independent but reproducible.

    Returns
    -------
    pd.DataFrame with one row per SNR level and columns:
        target_snr_db, achieved_snr_noisy_db, achieved_snr_filtered_db,
        kurtosis_clean, kurtosis_noisy, kurtosis_filtered,
        noise_floor_noisy, noise_floor_filtered, noise_floor_clean,
        rms_error_noisy, rms_error_filtered, snr_improvement_db
    """
    if snr_levels is None:
        snr_levels = DEFAULT_SNR_LEVELS

    from preprocessing import add_gaussian_noise, compute_snr_db

    default_filter_cfg = (
        filter_cfg
        if filter_cfg is not None
        else {"type": "lowpass", "cutoff": fs * 0.25, "order": 4}
    )

    clean = np.asarray(signal, dtype=float)
    kurt_clean = float(_scipy_kurtosis(clean, fisher=False, bias=True))
    _, psd_clean = _compute_psd(clean, fs)
    nf_clean = _estimate_spectral_noise_floor(psd_clean)

    rows = []
    for i, target_snr in enumerate(snr_levels):
        noisy = add_gaussian_noise(clean, snr_db=target_snr, random_state=random_state + i)
        filtered = _apply_filter_safe(noisy, fs, default_filter_cfg)

        achieved_snr_noisy    = compute_snr_db(clean, noisy)
        achieved_snr_filtered = compute_snr_db(clean, filtered)

        kurt_noisy    = float(_scipy_kurtosis(noisy,    fisher=False, bias=True))
        kurt_filtered = float(_scipy_kurtosis(filtered, fisher=False, bias=True))

        _, psd_noisy    = _compute_psd(noisy,    fs)
        _, psd_filtered = _compute_psd(filtered, fs)
        nf_noisy    = _estimate_spectral_noise_floor(psd_noisy)
        nf_filtered = _estimate_spectral_noise_floor(psd_filtered)

        noise_component = noisy - clean
        rms_error_noisy    = float(np.sqrt(np.mean(noise_component ** 2)))
        rms_error_filtered = float(np.sqrt(np.mean((filtered - clean) ** 2)))

        rows.append({
            "target_snr_db":           target_snr,
            "achieved_snr_noisy_db":   achieved_snr_noisy,
            "achieved_snr_filtered_db": achieved_snr_filtered,
            "kurtosis_clean":          kurt_clean,
            "kurtosis_noisy":          kurt_noisy,
            "kurtosis_filtered":       kurt_filtered,
            "noise_floor_clean":       nf_clean,
            "noise_floor_noisy":       nf_noisy,
            "noise_floor_filtered":    nf_filtered,
            "rms_error_noisy":         rms_error_noisy,
            "rms_error_filtered":      rms_error_filtered,
            "snr_improvement_db":      achieved_snr_filtered - achieved_snr_noisy,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Zoomed time-domain comparison
# --------------------------------------------------------------------------

def plot_zoomed_triplet(
    t: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray,
    filtered: np.ndarray,
    zoom_start: float = 0.05,
    zoom_end: float = 0.25,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Three-panel zoomed time-domain view: Original / Noisy / Filtered.

    WHY zoom instead of showing the full signal:
    Gaussian noise on a 20-second signal at SNR ≥ 5 dB is completely
    invisible when the x-axis spans thousands of samples per pixel.  A
    short window (0.1–0.3 s, ~1000–3500 samples at 12 kHz) shows
    individual noise samples, the distortion they cause on signal peaks,
    and the smoothing effect of the filter — the essential visual lesson.
    """
    mask = (t >= zoom_start) & (t <= zoom_end)
    tz, xc, xn, xf = t[mask], clean[mask], noisy[mask], filtered[mask]

    fig, axes = plt.subplots(3, 1, figsize=(9, 5), sharex=True)
    styles = [
        (xc, "#2ca02c", "Original (clean)"),
        (xn, "#d62728", "Noisy"),
        (xf, "#1f77b4", "Filtered"),
    ]
    for ax, (sig, col, lbl) in zip(axes, styles):
        ax.plot(tz * 1000, sig, color=col, linewidth=0.8)  # time in ms
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_title(lbl, fontsize=9, loc="left", pad=2)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time (ms)", fontsize=9)
    suf = f" — {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"Zoomed segment [{zoom_start*1000:.0f}–{zoom_end*1000:.0f} ms]{suf}",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# PSD triple-overlay
# --------------------------------------------------------------------------

def plot_psd_triplet(
    clean: np.ndarray,
    noisy: np.ndarray,
    filtered: np.ndarray,
    fs: float,
    highlight_band: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Overlay PSD of Original / Noisy / Filtered on a single log-scale plot.

    WHY this is the key noise illustration:
    • White Gaussian noise has a FLAT PSD — adding it raises the entire
      spectral floor by a constant amount determined by the noise power.
    • A band-pass filter cuts the noise power outside the pass-band,
      lowering the floor in the stop-band and recovering the signal in
      the pass-band — exactly what envelope analysis relies on.
    • This plot makes both effects immediately visible: the floor lifts
      uniformly for the noisy signal, and the filtered signal sits below
      the noisy one except in its pass-band.

    Parameters
    ----------
    highlight_band : (low_hz, high_hz) or None
        If given, shade the filter pass-band to connect this plot to the
        envelope analysis tab.
    """
    fc, psd_c = _compute_psd(clean,    fs)
    fn, psd_n = _compute_psd(noisy,    fs)
    ff, psd_f = _compute_psd(filtered, fs)

    nf_clean    = _estimate_spectral_noise_floor(psd_c)
    nf_noisy    = _estimate_spectral_noise_floor(psd_n)
    nf_filtered = _estimate_spectral_noise_floor(psd_f)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(fc, psd_c + 1e-30, color="#2ca02c", linewidth=0.9,
                label=f"Original  (floor {nf_clean:.2e})")
    ax.semilogy(fn, psd_n + 1e-30, color="#d62728", linewidth=0.9,
                label=f"Noisy     (floor {nf_noisy:.2e})")
    ax.semilogy(ff, psd_f + 1e-30, color="#1f77b4", linewidth=0.9,
                label=f"Filtered  (floor {nf_filtered:.2e})")

    # horizontal reference lines for each floor
    for nf, col in [(nf_clean, "#2ca02c"), (nf_noisy, "#d62728"), (nf_filtered, "#1f77b4")]:
        ax.axhline(nf, color=col, linewidth=0.6, linestyle="--", alpha=0.5)

    if highlight_band is not None:
        ax.axvspan(highlight_band[0], highlight_band[1],
                   alpha=0.08, color="#9467bd",
                   label=f"Filter band ({highlight_band[0]:.0f}–{highlight_band[1]:.0f} Hz)")

    ax.set_title("Power Spectral Density (Welch) — Original vs Noisy vs Filtered")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (power/Hz, log scale)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# SNR sweep curves
# --------------------------------------------------------------------------

def plot_snr_curves(sweep_df: pd.DataFrame) -> plt.Figure:
    """
    Three-panel figure from the SNR sweep results:

    Panel 1 — Kurtosis vs target SNR
        Shows how the impulsive fault signature (high kurtosis) degrades
        with noise and is partially recovered by filtering.  The horizontal
        dashed line at kurtosis = 3 is the Gaussian baseline; values above
        it indicate non-Gaussian (impulsive) content.

    Panel 2 — Spectral noise floor vs target SNR
        Shows the noise floor rising as SNR decreases, and how the filter
        lowers it back — the direct frequency-domain effect of filtering.

    Panel 3 — RMS reconstruction error vs target SNR
        Quantifies how accurately the original signal is recovered after
        filtering.  Lower is better.  The gap between the noisy and
        filtered curves shows the practical benefit of the filter at each
        noise level.
    """
    snr_x = sweep_df["target_snr_db"].values

    fig = plt.figure(figsize=(10, 7))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    # ── Panel 1: Kurtosis ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.axhline(3.0, color="gray", linewidth=0.8, linestyle="--",
                label="Gaussian baseline (κ = 3)")
    ax1.plot(snr_x, sweep_df["kurtosis_clean"],    "o--", color="#2ca02c",
             linewidth=1.0, markersize=4, label="Original")
    ax1.plot(snr_x, sweep_df["kurtosis_noisy"],    "s-",  color="#d62728",
             linewidth=1.2, markersize=4, label="Noisy")
    ax1.plot(snr_x, sweep_df["kurtosis_filtered"], "^-",  color="#1f77b4",
             linewidth=1.2, markersize=4, label="Filtered")
    ax1.set_title(
        "Kurtosis vs SNR  —  fault impulses are masked by noise (κ → 3) "
        "and partially recovered by filtering",
        fontsize=9,
    )
    ax1.set_ylabel("Kurtosis")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.25)

    # ── Panel 2: Spectral noise floor ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.semilogy(snr_x, sweep_df["noise_floor_noisy"],    "s-",  color="#d62728",
                 linewidth=1.2, markersize=4, label="Noisy")
    ax2.semilogy(snr_x, sweep_df["noise_floor_filtered"], "^-",  color="#1f77b4",
                 linewidth=1.2, markersize=4, label="Filtered")
    ax2.axhline(sweep_df["noise_floor_clean"].iloc[0], color="#2ca02c",
                linewidth=0.8, linestyle="--", label="Original floor")
    ax2.set_title(
        "Spectral noise floor vs SNR  —  noise raises the floor; "
        "filter brings it back down",
        fontsize=9,
    )
    ax2.set_ylabel("Noise floor\n(power/Hz)", fontsize=8)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(alpha=0.25, which="both")

    # ── Panel 3: RMS reconstruction error ────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(snr_x, sweep_df["rms_error_noisy"],    "s-",  color="#d62728",
             linewidth=1.2, markersize=4, label="Noisy vs clean")
    ax3.plot(snr_x, sweep_df["rms_error_filtered"], "^-",  color="#1f77b4",
             linewidth=1.2, markersize=4, label="Filtered vs clean")
    ax3.set_title(
        "RMS reconstruction error vs SNR  —  filtering reduces the error "
        "most at low SNR",
        fontsize=9,
    )
    ax3.set_ylabel("RMS error")
    ax3.set_xlabel("Target SNR (dB)")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(alpha=0.25)
    ax3.invert_xaxis()

    return fig
