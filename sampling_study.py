"""
sampling_study.py
-----------------
Sampling-rate / Nyquist robustness study for Motor Insight.

This module is designed to answer the question:

    "How does reducing the sampling rate affect fault observability,
     aliasing risk, and the final diagnosis?"

What it studies
---------------
For a clean reference signal, it evaluates a list of target sampling rates
and compares:
1) Time-domain waveform (zoomed)
2) FFT spectrum before/after downsampling
3) PSD / Welch spectrum before/after downsampling
4) Envelope spectrum before/after downsampling
5) Rule-based diagnosis before/after downsampling
6) Nyquist / aliasing indicators
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from features import extract_features_dataframe, compute_fft
from spectral import compute_psd_welch
from preprocessing import downsample_signal, remove_dc_offset
from envelope import envelope_analysis_pipeline
from diagnosis import rule_based_diagnosis, DiagnosisThresholds
from utils import plot_fft, plot_psd, plot_envelope_spectrum


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class SamplingTrial:
    target_fs: float
    actual_fs: float
    nyquist_hz: float
    aliasing_risk: bool
    aliased_band_limit_hz: float


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_envelope_band(band: Tuple[float, float], current_fs: float) -> Tuple[float, float]:
    """
    Adjusts envelope bandpass boundaries to strictly lie below the Nyquist frequency
    of current_fs to prevent IIR filter instability (numerical blowup).
    """
    nyq = current_fs / 2.0
    low, high = band
    safe_low = min(low, nyq * 0.25)
    safe_high = min(high, nyq * 0.90)
    if safe_high <= safe_low:
        safe_low = max(1.0, nyq * 0.10)
        safe_high = nyq * 0.85
    return (safe_low, safe_high)


def _dominant_freq_above(freqs: np.ndarray, magnitude: np.ndarray, threshold_hz: float) -> Optional[float]:
    """Return the dominant frequency above a threshold, or None if absent."""
    if len(freqs) == 0 or len(magnitude) == 0:
        return None
    mask = freqs > threshold_hz
    if not np.any(mask):
        return None
    idx = np.argmax(magnitude[mask])
    return float(freqs[mask][idx])


def _max_peak_freq(freqs: np.ndarray, magnitude: np.ndarray, fmin: float = 0.0, fmax: Optional[float] = None) -> Optional[float]:
    """Return the frequency of the maximum magnitude within [fmin, fmax]."""
    if len(freqs) == 0 or len(magnitude) == 0:
        return None
    if fmax is None:
        fmax = float(freqs[-1])
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None
    idx = np.argmax(magnitude[mask])
    return float(freqs[mask][idx])


def _aliasing_warning_string(actual_fs: float, max_signal_freq_est: Optional[float]) -> str:
    nyquist = actual_fs / 2.0
    if max_signal_freq_est is None or np.isnan(max_signal_freq_est):
        return "Unknown"
    return "Yes" if max_signal_freq_est > nyquist else "No"


# ---------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------

def run_sampling_study(
    signal: np.ndarray,
    fs: float,
    target_fs_levels: Sequence[float],
    fault_freqs: Optional[Dict[str, float]] = None,
    window_size: int = 2048,
    overlap: float = 0.5,
    envelope_band: Optional[Tuple[float, float]] = None,
    filter_order: int = 4,
    diagnosis_thresholds: Optional[DiagnosisThresholds] = None,
    compute_without_anti_aliasing: bool = False,
) -> dict:
    """
    Evaluate the effect of sampling-rate reduction on signal observability
    and diagnosis quality.
    """
    clean = remove_dc_offset(np.asarray(signal, dtype=float))
    diagnosis_thresholds = diagnosis_thresholds or DiagnosisThresholds()

    # Reference analysis at the original rate
    ref_features = extract_features_dataframe(clean, fs, int(window_size), overlap)
    ref_fft_freqs, ref_fft_mag = compute_fft(clean, fs)
    ref_psd_freqs, ref_psd = compute_psd_welch(clean, fs)

    if envelope_band is None:
        nyq = fs / 2.0
        low = min(1000.0, nyq * 0.25)
        high = min(5000.0, nyq * 0.9)
        if high <= low:
            high = min(nyq * 0.9, low + max(200.0, nyq * 0.1))
        envelope_band = (low, high)

    ref_env_band = _safe_envelope_band(envelope_band, fs)
    ref_env, ref_env_freqs, ref_env_mag = envelope_analysis_pipeline(
        clean, fs, band=ref_env_band, filter_order=filter_order
    )
    ref_diag = rule_based_diagnosis(ref_features, diagnosis_thresholds)

    reference = {
        "signal": clean,
        "fs": fs,
        "nyquist_hz": fs / 2.0,
        "features": ref_features,
        "fft": (ref_fft_freqs, ref_fft_mag),
        "psd": (ref_psd_freqs, ref_psd),
        "envelope": (ref_env, ref_env_freqs, ref_env_mag),
        "diagnosis": ref_diag,
        "kurtosis": float(ref_features["kurtosis"].mean()),
        "rms": float(ref_features["rms"].mean()),
    }

    trials: Dict[float, dict] = {}
    summary_rows: List[dict] = []

    ref_dom_freq = _max_peak_freq(ref_fft_freqs, ref_fft_mag, fmin=0.0)

    for idx, target_fs in enumerate(target_fs_levels):
        target_fs = float(target_fs)
        if target_fs <= 0:
            continue

        if target_fs >= fs:
            down_sig = clean.copy()
            actual_fs = fs
            anti_alias_used = False
        else:
            down_sig, actual_fs = downsample_signal(clean, fs, target_fs)
            anti_alias_used = True

        if compute_without_anti_aliasing and target_fs < fs:
            factor = max(1, int(round(fs / target_fs)))
            naive_sig = clean[::factor].copy()
            naive_fs = fs / factor
        else:
            naive_sig = None
            naive_fs = None

        nyquist = actual_fs / 2.0
        aliasing_risk = bool(ref_dom_freq is not None and ref_dom_freq > nyquist)

        # Signal-level analyses on the downsampled signal
        ds_features = extract_features_dataframe(
            down_sig, actual_fs, min(int(window_size), len(down_sig)), overlap
        )
        ds_fft_freqs, ds_fft_mag = compute_fft(down_sig, actual_fs)
        ds_psd_freqs, ds_psd = compute_psd_welch(down_sig, actual_fs)

        # Dynamic safe envelope band for actual downsampled rate
        ds_env_band = _safe_envelope_band(envelope_band, actual_fs)
        ds_env, ds_env_freqs, ds_env_mag = envelope_analysis_pipeline(
            down_sig, actual_fs, band=ds_env_band, filter_order=filter_order
        )
        ds_diag = rule_based_diagnosis(ds_features, diagnosis_thresholds)

        # Naive branch (optional)
        if naive_sig is not None:
            naive_features = extract_features_dataframe(
                naive_sig, naive_fs, min(int(window_size), len(naive_sig)), overlap
            )
            naive_fft_freqs, naive_fft_mag = compute_fft(naive_sig, naive_fs)
            naive_psd_freqs, naive_psd = compute_psd_welch(naive_sig, naive_fs)

            naive_env_band = _safe_envelope_band(envelope_band, naive_fs)
            naive_env, naive_env_freqs, naive_env_mag = envelope_analysis_pipeline(
                naive_sig, naive_fs, band=naive_env_band, filter_order=filter_order
            )
            naive_diag = rule_based_diagnosis(naive_features, diagnosis_thresholds)
        else:
            naive_features = None
            naive_fft_freqs = naive_fft_mag = None
            naive_psd_freqs = naive_psd = None
            naive_env = naive_env_freqs = naive_env_mag = None
            naive_diag = None

        # Time-domain metrics
        rms_ref = float(reference["rms"])
        rms_ds = float(ds_features["rms"].mean())
        kurt_ref = float(reference["kurtosis"])
        kurt_ds = float(ds_features["kurtosis"].mean())
        verdict_ref = reference["diagnosis"]["level"]
        verdict_ds = ds_diag["level"]

        max_sig_freq_est = _max_peak_freq(ref_fft_freqs, ref_fft_mag, fmin=0.0)
        aliasing_label = _aliasing_warning_string(actual_fs, max_sig_freq_est)

        trials[target_fs] = {
            "target_fs": target_fs,
            "actual_fs": actual_fs,
            "nyquist_hz": nyquist,
            "aliasing_risk": aliasing_risk,
            "aliasing_label": aliasing_label,
            "anti_alias_used": anti_alias_used,
            "signal": down_sig,
            "features": ds_features,
            "fft": (ds_fft_freqs, ds_fft_mag),
            "psd": (ds_psd_freqs, ds_psd),
            "envelope": (ds_env, ds_env_freqs, ds_env_mag),
            "diagnosis": ds_diag,
            "kurtosis": kurt_ds,
            "rms": rms_ds,
            "kurtosis_delta": kurt_ds - kurt_ref,
            "rms_ratio": rms_ds / rms_ref if rms_ref > 0 else np.nan,
            "verdict": verdict_ds,
            "verdict_ref": verdict_ref,
            "naive": {
                "signal": naive_sig,
                "fs": naive_fs,
                "features": naive_features,
                "fft": (naive_fft_freqs, naive_fft_mag),
                "psd": (naive_psd_freqs, naive_psd),
                "envelope": (naive_env, naive_env_freqs, naive_env_mag),
                "diagnosis": naive_diag,
            } if naive_sig is not None else None,
        }

        summary_rows.append({
            "target_fs_hz": target_fs,
            "actual_fs_hz": actual_fs,
            "nyquist_hz": nyquist,
            "anti_aliasing": "Yes" if anti_alias_used else "No",
            "aliasing_risk": "Yes" if aliasing_risk else "No",
            "ref_verdict": verdict_ref,
            "downsampled_verdict": verdict_ds,
            "ref_kurtosis": kurt_ref,
            "downsampled_kurtosis": kurt_ds,
            "kurtosis_delta": kurt_ds - kurt_ref,
            "ref_rms": rms_ref,
            "downsampled_rms": rms_ds,
            "rms_ratio": rms_ds / rms_ref if rms_ref > 0 else np.nan,
            "ref_dom_freq_hz": ref_dom_freq,
        })

    return {
        "reference": reference,
        "trials": trials,
        "summary": pd.DataFrame(summary_rows),
        "envelope_band": envelope_band,
        "fault_freqs": fault_freqs,
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_sampling_time_domain(
    study_results: dict,
    zoom_seconds: Tuple[float, float] = (0.05, 0.15),
) -> plt.Figure:
    """Plot zoomed time-domain comparison across trials."""
    reference = study_results["reference"]
    trials = study_results["trials"]

    ref_sig = reference["signal"]
    fs = float(reference["fs"])
    t = np.arange(len(ref_sig)) / fs
    mask = (t >= zoom_seconds[0]) & (t <= zoom_seconds[1])

    n_rows = 1 + len(trials)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.2 * n_rows), sharex=True, constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    axes[0].plot(t[mask] * 1000.0, ref_sig[mask], linewidth=0.85, color="#2ca02c")
    axes[0].set_title("Reference signal (original fs)", loc="left", fontsize=9)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)

    for ax, (target_fs, trial) in zip(axes[1:], trials.items()):
        ds_sig = trial["signal"]
        ds_fs = float(trial["actual_fs"])
        t_ds = np.arange(len(ds_sig)) / ds_fs
        mask_ds = (t_ds >= zoom_seconds[0]) & (t_ds <= zoom_seconds[1])
        ax.plot(t_ds[mask_ds] * 1000.0, ds_sig[mask_ds], linewidth=0.85, color="#1f77b4")
        ax.set_title(
            f"Downsampled to {target_fs:.0f} Hz  |  Nyquist={trial['nyquist_hz']:.0f} Hz  |  Alias risk: {trial['aliasing_label']}",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Zoomed time-domain comparison under different sampling rates", fontsize=11, fontweight="bold")
    return fig


def plot_sampling_fft(study_results: dict) -> plt.Figure:
    """Plot FFT comparison across all sampling-rate trials."""
    reference = study_results["reference"]
    trials = study_results["trials"]

    fig = plt.figure(figsize=(11, 2.2 * (len(trials) + 1)))
    gs = gridspec.GridSpec(len(trials) + 1, 1, hspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    f_ref, m_ref = reference["fft"]
    ax0.plot(f_ref, m_ref, linewidth=0.9, color="#2ca02c", label=f"Reference (fs={reference['fs']:.0f} Hz)")
    ax0.set_title("FFT reference", loc="left", fontsize=9)
    ax0.set_ylabel("Magnitude")
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=8, loc="upper right")

    ax = ax0
    for i, (target_fs, trial) in enumerate(trials.items(), start=1):
        ax = fig.add_subplot(gs[i])
        f_ds, m_ds = trial["fft"]
        ax.plot(f_ds, m_ds, linewidth=0.9, color="#1f77b4", label=f"Downsampled (fs={trial['actual_fs']:.0f} Hz)")
        ax.set_title(
            f"Target fs = {target_fs:.0f} Hz | Nyquist = {trial['nyquist_hz']:.0f} Hz | Alias risk: {trial['aliasing_label']}",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("Magnitude")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    ax.set_xlabel("Frequency (Hz)")
    fig.suptitle("FFT under different sampling rates", fontsize=11, fontweight="bold")
    return fig


def plot_sampling_psd(study_results: dict) -> plt.Figure:
    """Plot PSD comparison across all sampling-rate trials."""
    reference = study_results["reference"]
    trials = study_results["trials"]

    fig, axes = plt.subplots(1, len(trials) + 1, figsize=(4.2 * (len(trials) + 1), 3.8), sharey=True, constrained_layout=True)
    if len(trials) == 0:
        axes = [axes]

    f_ref, psd_ref = reference["psd"]
    axes[0].semilogy(f_ref, psd_ref + 1e-30, linewidth=0.9, color="#2ca02c")
    axes[0].set_title(f"Reference\nfs={reference['fs']:.0f} Hz", fontsize=9)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].set_ylabel("PSD (power/Hz)")

    for ax, (target_fs, trial) in zip(axes[1:], trials.items()):
        f_ds, psd_ds = trial["psd"]
        ax.semilogy(f_ds, psd_ds + 1e-30, linewidth=0.9, color="#1f77b4")
        ax.set_title(f"fs={trial['actual_fs']:.0f} Hz\nNyquist={trial['nyquist_hz']:.0f} Hz", fontsize=9)
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(alpha=0.25, which="both")

    fig.suptitle("PSD (Welch) under different sampling rates", fontsize=11, fontweight="bold")
    return fig


def plot_sampling_envelope(study_results: dict) -> plt.Figure:
    """Plot envelope spectra across sampling-rate trials."""
    reference = study_results["reference"]
    trials = study_results["trials"]
    fault_freqs = study_results.get("fault_freqs") or {}

    fig, axes = plt.subplots(len(trials) + 1, 1, figsize=(10, 2.5 * (len(trials) + 1)), sharex=True, constrained_layout=True)
    if len(trials) == 0:
        axes = [axes]

    ref_env, ref_env_freqs, ref_env_mag = reference["envelope"]
    axes[0].plot(ref_env_freqs, ref_env_mag, linewidth=0.85, color="#2ca02c", label="Reference")
    if fault_freqs:
        for name, freq in fault_freqs.items():
            if freq is None:
                continue
            axes[0].axvline(freq, linestyle="--", linewidth=0.9, alpha=0.8, label=f"{name} ({freq:.1f} Hz)")
    axes[0].set_title("Envelope spectrum reference", loc="left", fontsize=9)
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper right")

    for ax, (target_fs, trial) in zip(axes[1:], trials.items()):
        _, env_freqs, env_mag = trial["envelope"]
        ax.plot(env_freqs, env_mag, linewidth=0.85, color="#1f77b4", label=f"fs={trial['actual_fs']:.0f} Hz")
        if fault_freqs:
            for name, freq in fault_freqs.items():
                if freq is None:
                    continue
                if freq <= trial["nyquist_hz"]:
                    ax.axvline(freq, linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_title(
            f"Target fs = {target_fs:.0f} Hz | Nyquist = {trial['nyquist_hz']:.0f} Hz | Verdict: {trial['diagnosis']['level']}",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("Magnitude")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Envelope spectrum under different sampling rates", fontsize=11, fontweight="bold")
    return fig


def plot_sampling_summary(study_results: dict) -> plt.Figure:
    """Plot a compact summary of key metrics versus sampling rate."""
    df = study_results["summary"].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    x = df["target_fs_hz"].to_numpy()

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 1, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, df["downsampled_kurtosis"], marker="o", linewidth=1.1, label="Downsampled kurtosis")
    ax1.axhline(df["ref_kurtosis"].iloc[0], linestyle="--", linewidth=0.8, color="gray", label="Reference kurtosis")
    ax1.set_ylabel("Kurtosis")
    ax1.set_title("Kurtosis vs sampling rate", loc="left", fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="best")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x, df["rms_ratio"], marker="s", linewidth=1.1, color="#d62728")
    ax2.set_ylabel("RMS ratio")
    ax2.set_title("RMS ratio (downsampled / reference)", loc="left", fontsize=9)
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[2])
    verdict_numeric = [1 if v == "Faulty" else 0.5 if v == "Warning" else 0 for v in df["downsampled_verdict"]]
    ax3.step(x, verdict_numeric, where="mid", linewidth=1.2, color="#1f77b4")
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_yticklabels(["Healthy", "Warning", "Faulty"])
    ax3.set_xlabel("Sampling rate (Hz)")
    ax3.set_ylabel("Verdict")
    ax3.set_title("Diagnosis outcome vs sampling rate", loc="left", fontsize=9)
    ax3.grid(alpha=0.25)
    ax3.invert_xaxis()

    fig.suptitle("Sampling-rate sensitivity summary", fontsize=11, fontweight="bold")
    return fig


# ---------------------------------------------------------------------
# Streamlit UI helper (Interactive & Reactive)
# ---------------------------------------------------------------------

def render_sampling_study_tab(
    signal: np.ndarray,
    fs: float,
    fault_freqs: Optional[Dict[str, float]] = None,
    window_size: int = 2048,
    overlap: float = 0.5,
    diagnosis_thresholds: Optional[DiagnosisThresholds] = None,
):
    """
    Render an interactive and dynamic Streamlit tab for sampling-rate / Nyquist study.
    """
    st.markdown("### 📉 بررسی نرخ نمونه‌برداری، قضیه نایکویست و Aliasing")
    st.info(
        "در این بخش اثر کاهش نرخ نمونه‌برداری بر FFT، PSD، طیف پاکت و تصمیم نهایی سیستم بررسی می‌شود. "
        "تغییر پارامترها به‌صورت زنده و آنی در نمودارها اعمال می‌شود."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        presets = {
            "حفظ کامل (fs اصلی)": [fs],
            "پله‌ای": [fs, fs * 2 / 3, fs / 2, fs / 3, fs / 4],
            "آموزشی (شدید)": [fs, 6000, 2000, 800],
        }
        preset_name = st.selectbox("سناریوی نمونه‌برداری", list(presets.keys()))
        levels = [float(v) for v in presets[preset_name] if v > 0]

    with col2:
        use_custom = st.checkbox("تعیین دستی نرخ‌ها", value=False)
        if use_custom:
            raw = st.text_input("نرخ‌ها (Hz) جداشده با کاما", ", ".join(str(int(v)) for v in levels))
            try:
                levels = [float(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                st.warning("ورودی نامعتبر بود؛ نرخ‌های پیش‌فرض استفاده شد.")

    with col3:
        zoom_start = st.number_input("شروع زوم زمانی (s)", min_value=0.0, value=0.05, step=0.01)
        zoom_end = st.number_input("پایان زوم زمانی (s)", min_value=0.01, value=0.15, step=0.01)

    nyq = fs / 2.0
    default_low = min(1000.0, nyq * 0.25)
    default_high = min(5000.0, nyq * 0.9)
    if default_high <= default_low:
        default_high = min(nyq * 0.9, default_low + max(200.0, nyq * 0.1))
    envelope_band = (default_low, default_high)

    filter_order = st.slider("مرتبه فیلتر envelope", 1, 8, 4)

    # اجرای آنی محاسبات بدون گیر افتادن در state دکمه
    with st.spinner("در حال محاسبه آنی..."):
        study = run_sampling_study(
            signal=signal,
            fs=fs,
            target_fs_levels=levels,
            fault_freqs=fault_freqs,
            window_size=window_size,
            overlap=overlap,
            envelope_band=envelope_band,
            filter_order=filter_order,
            diagnosis_thresholds=diagnosis_thresholds,
            compute_without_anti_aliasing=False,
        )

    # Summary table
    st.markdown("#### 📋 جدول خلاصه")
    summary_df = study["summary"].copy()
    display_cols = [
        "target_fs_hz",
        "actual_fs_hz",
        "nyquist_hz",
        "aliasing_risk",
        "anti_aliasing",
        "ref_kurtosis",
        "downsampled_kurtosis",
        "kurtosis_delta",
        "ref_verdict",
        "downsampled_verdict",
    ]
    st.dataframe(summary_df[display_cols].round(3), use_container_width=True)

    # Charts
    tab_time, tab_fft, tab_psd, tab_env, tab_sum = st.tabs([
        "📈 حوزه زمان",
        "FFT",
        "PSD",
        "Envelope",
        "خلاصه",
    ])

    with tab_time:
        fig = plot_sampling_time_domain(study, zoom_seconds=(zoom_start, zoom_end))
        st.pyplot(fig)
        plt.close(fig)

    with tab_fft:
        fig = plot_sampling_fft(study)
        st.pyplot(fig)
        plt.close(fig)

    with tab_psd:
        fig = plot_sampling_psd(study)
        st.pyplot(fig)
        plt.close(fig)

    with tab_env:
        fig = plot_sampling_envelope(study)
        st.pyplot(fig)
        plt.close(fig)

    with tab_sum:
        fig = plot_sampling_summary(study)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### تفسیر خودکار")
        ref_verdict = study["reference"]["diagnosis"]["level"]
        rows = []
        for _, row in summary_df.iterrows():
            rows.append(
                f"- fs={int(row['target_fs_hz'])} Hz → Nyquist={row['nyquist_hz']:.0f} Hz، "
                f"aliasing={'Yes' if row['aliasing_risk'] == 'Yes' else 'No'}، "
                f"Verdict: {row['downsampled_verdict']}"
            )
        st.write(
            f"در سیگنال مرجع، حکم سیستم {ref_verdict} است. با کاهش fs، اگر نرخ نمونه‌برداری به اندازه کافی بالا نماند، "
            f"پیک‌های طیفی و ویژگی‌های ضربه‌ای ضعیف‌تر می‌شوند و ممکن است تشخیص پایدار نباشد."
        )
        for r in rows:
            st.write(r)


# ---------------------------------------------------------------------
# Optional: PDF-friendly exports
# ---------------------------------------------------------------------

def study_to_table_dataframe(study_results: dict) -> pd.DataFrame:
    """Return the summary table as a dataframe ready for PDF/report export."""
    return study_results["summary"].copy()


def figures_as_png_bytes(
    study_results: dict,
    zoom_seconds: Tuple[float, float] = (0.05, 0.15),
) -> dict:
    """Convenience function for embedding into PDF reports."""
    fig_time = plot_sampling_time_domain(study_results, zoom_seconds=zoom_seconds)
    fig_fft = plot_sampling_fft(study_results)
    fig_psd = plot_sampling_psd(study_results)
    fig_env = plot_sampling_envelope(study_results)
    fig_sum = plot_sampling_summary(study_results)

    return {
        "time": fig_time,
        "fft": fig_fft,
        "psd": fig_psd,
        "envelope": fig_env,
        "summary": fig_sum,
    }
