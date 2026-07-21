"""
noise_study.py
-----------------
ماژول تحلیل جامع اثر نویز گوسی (AWGN) و فیلترسازی بر عیب‌یابی بلبرینگ.

ویژگی‌های کلیدی:
1. استفاده کامل از توابع ماژول‌های پروژه (DRY): preprocessing, features, envelope, diagnosis, spectral, utils
2. عدم وابستگی به فایل‌های محلی سخت‌افزار (دریافت ورودی از Streamlit)
3. بررسی اثر نویز روی سیگنال زمانی، طیف PSD و طیف پاکت (Envelope Spectrum)
4. بررسی تغییرات وضعیت تشخیص نهایی (Healthy / Warning / Faulty) قبل و بعد از فیلتر
5. مدیریت حافظه جهت جلوگیری از Memory Leak در Matplotlib و Streamlit
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st

# وارد کردن توابع موجود در پروژه جهت رعایت اصل DRY
from preprocessing import (
    add_gaussian_noise,
    compute_snr_db,
    apply_filter,
    remove_dc_offset,
)
from envelope import compute_envelope_spectrum
from spectral import compute_psd_welch
from features import extract_features_dataframe
from diagnosis import rule_based_diagnosis
from utils import fig_to_png_bytes


# ──────────────────────────────────────────────────────
# ۱. هسته محاسباتی مطالعه نویز (Core Pipeline)
# ──────────────────────────────────────────────────────
def run_noise_study(
    clean_signal: np.ndarray,
    fs: float,
    snr_levels: List[int] = [20, 10, 0],
    filter_cfg: Optional[dict] = None,
    fault_freqs: Optional[dict] = None,
    window_size: int = 2048,
    overlap: float = 0.5,
) -> dict:
    """
    تحلیل جامع اثر نویز و فیلتر روی سیگنال، ویژگی‌ها، طیف پاکت و تشخیص نهایی.
    """
    if filter_cfg is None:
        filter_cfg = {"type": "bandpass", "cutoff": (1000.0, 5000.0), "order": 4}

    clean_signal = remove_dc_offset(clean_signal)

    # ۱. استخراج ویژگی و تشخیص برای سیگنال تمیز (مرجع)
    clean_feats = extract_features_dataframe(clean_signal, fs, window_size, overlap)
    c_env_f, c_env_m = compute_envelope_spectrum(clean_signal, fs)
    c_psd_f, c_psd = compute_psd_welch(clean_signal, fs)
    clean_diag = rule_based_diagnosis(clean_feats)

    study_results = {
        "clean": {
            "signal": clean_signal,
            "features": clean_feats,
            "psd_f": c_psd_f,
            "psd": c_psd,
            "env_f": c_env_f,
            "env_m": c_env_m,
            "diagnosis": clean_diag,
            "kurtosis": float(clean_feats["kurtosis"].mean()) if "kurtosis" in clean_feats else 0.0,
        },
        "snr_trials": {},
        "filter_cfg": filter_cfg,
    }

    # ۲. شبیه‌سازی سطوح مختلف نویز و ارزیابی
    for snr_target in snr_levels:
        # تزریق نویز (استفاده از preprocessing)
        noisy = add_gaussian_noise(clean_signal, snr_target)
        
        # اعمال فیلتر (استفاده از preprocessing)
        filtered = apply_filter(
            noisy,
            filter_cfg["type"],
            filter_cfg["cutoff"],
            fs,
            filter_cfg.get("order", 4),
        )

        # محاسبات SNR واقعی
        snr_noisy_actual = compute_snr_db(clean_signal, noisy)
        snr_filtered_actual = compute_snr_db(clean_signal, filtered)

        # استخراج ویژگی‌ها (استفاده از features)
        noisy_feats = extract_features_dataframe(noisy, fs, window_size, overlap)
        filt_feats = extract_features_dataframe(filtered, fs, window_size, overlap)

        # محاسبه طیف PSD و پاکت (استفاده از spectral و envelope)
        n_psd_f, n_psd = compute_psd_welch(noisy, fs)
        f_psd_f, f_psd = compute_psd_welch(filtered, fs)

        n_env_f, n_env_m = compute_envelope_spectrum(noisy, fs)
        f_env_f, f_env_m = compute_envelope_spectrum(filtered, fs)

        # ارزیابی تشخیص نهایی (استفاده از diagnosis)
        noisy_diag = rule_based_diagnosis(noisy_feats)
        filt_diag = rule_based_diagnosis(filt_feats)

        k_noisy = float(noisy_feats["kurtosis"].mean()) if "kurtosis" in noisy_feats else 0.0
        k_filt = float(filt_feats["kurtosis"].mean()) if "kurtosis" in filt_feats else 0.0

        study_results["snr_trials"][snr_target] = {
            "noisy_signal": noisy,
            "filtered_signal": filtered,
            "snr_noisy": snr_noisy_actual,
            "snr_filtered": snr_filtered_actual,
            "snr_gain": snr_filtered_actual - snr_noisy_actual,
            "noisy_kurtosis": k_noisy,
            "filtered_kurtosis": k_filt,
            "noisy_psd": (n_psd_f, n_psd),
            "filtered_psd": (f_psd_f, f_psd),
            "noisy_env": (n_env_f, n_env_m),
            "filtered_env": (f_env_f, f_env_m),
            "noisy_diagnosis": noisy_diag,
            "filtered_diagnosis": filt_diag,
        }

    return study_results


# ──────────────────────────────────────────────────────
# ۲. توابع رسم نمودارها (Returning Matplotlib Figures)
# ──────────────────────────────────────────────────────
def plot_noise_envelope_comparison(
    study_results: dict,
    fault_freqs: Optional[dict] = None,
) -> plt.Figure:
    """
    رسم مقایسه‌ای طیف پاکت  برای سیگنال تمیز، نویزی و فیلترشده.
    """
    snr_trials = study_results["snr_trials"]
    snr_levels = list(snr_trials.keys())
    n_snr = len(snr_levels)

    fig, axes = plt.subplots(
        n_snr, 1, figsize=(9, 2.8 * n_snr), sharex=True, constrained_layout=True
    )
    if n_snr == 1:
        axes = [axes]

    c_env_f, c_env_m = study_results["clean"]["env_f"], study_results["clean"]["env_m"]
    max_freq_view = 500.0  # محدوده فرکانسی تحلیل عیب بلبرینگ

    for i, snr in enumerate(snr_levels):
        ax = axes[i]
        trial = snr_trials[snr]
        n_env_f, n_env_m = trial["noisy_env"]
        f_env_f, f_env_m = trial["filtered_env"]

        # رسم طیف نویزی، فیلترشده و مرجع
        ax.plot(n_env_f, n_env_m, color="#e74c3c", lw=0.7, alpha=0.6, label=f"Noisy ({snr} dB)")
        ax.plot(f_env_f, f_env_m, color="#1f77b4", lw=1.1, label="Filtered Envelope")
        ax.plot(c_env_f, c_env_m, color="#2ca02c", lw=0.8, ls=":", label="Clean Reference")

        # نشانه‌گذاری فرکانس‌های عیب بلبرینگ (BPFO, BPFI, BSF, FTF)
        if fault_freqs:
            colors_cycle = ["#9467bd", "#8c564b", "#e377c2", "#17becf"]
            for j, (name, f_val) in enumerate(fault_freqs.items()):
                if f_val and isinstance(f_val, (int, float)) and f_val <= max_freq_view:
                    ax.axvline(
                        f_val,
                        color=colors_cycle[j % len(colors_cycle)],
                        ls="--",
                        lw=1.0,
                        alpha=0.8,
                        label=f"{name} ({f_val:.1f} Hz)" if i == 0 else "",
                    )

        ax.set_xlim(0, max_freq_view)
        n_verdict = trial['noisy_diagnosis'].get('level', 'Unknown')
        f_verdict = trial['filtered_diagnosis'].get('level', 'Unknown')
        ax.set_title(
            f"Envelope Spectrum (Target SNR = {snr} dB) | Verdict: {n_verdict} ➔ {f_verdict}",
            fontsize=9.5,
            loc="left",
        )
        ax.set_ylabel("Magnitude")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.8)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Envelope Spectrum Recoverability under AWGN Noise", fontsize=11, fontweight="bold")
    return fig


def plot_noise_psd_comparison(study_results: dict) -> plt.Figure:
    """رسم مقایسه‌ای چگالی طیفی توان (PSD)."""
    snr_trials = study_results["snr_trials"]
    snr_levels = list(snr_trials.keys())
    filter_cfg = study_results["filter_cfg"]
    
    cutoff = filter_cfg.get("cutoff")
    if isinstance(cutoff, (tuple, list)) and len(cutoff) == 2:
        low_f, high_f = cutoff
    else:
        low_f, high_f = 0.0, float(cutoff) if cutoff else 1000.0

    fig, axes = plt.subplots(
        1, len(snr_levels), figsize=(3.8 * len(snr_levels), 3.5), sharey=True, constrained_layout=True
    )
    if len(snr_levels) == 1:
        axes = [axes]

    c_psd_f, c_psd = study_results["clean"]["psd_f"], study_results["clean"]["psd"]

    for i, (ax, snr) in enumerate(zip(axes, snr_levels)):
        trial = snr_trials[snr]
        n_psd_f, n_psd = trial["noisy_psd"]
        f_psd_f, f_psd = trial["filtered_psd"]

        eps = 1e-20
        ax.semilogy(c_psd_f, c_psd + eps, color="#2ca02c", lw=1.0, label="Clean Ref")
        ax.semilogy(n_psd_f, n_psd + eps, color="#e74c3c", lw=0.7, alpha=0.7, label=f"Noisy {snr} dB")
        ax.semilogy(f_psd_f, f_psd + eps, color="#1f77b4", lw=1.1, label="Filtered")

        # نمایش محدوده گذر فیلتر
        if low_f > 0 or high_f > 0:
            ax.axvspan(low_f, high_f, alpha=0.08, color="purple")
            ax.axvline(low_f, color="purple", ls="--", lw=0.8)
            ax.axvline(high_f, color="purple", ls="--", lw=0.8)

        ax.set_title(f"Target SNR = {snr} dB", fontsize=9.5)
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(alpha=0.3, which="both", ls="--")
        ax.legend(fontsize=7.5, loc="upper right")

    axes[0].set_ylabel("PSD (Power/Hz, log scale)")
    fig.suptitle("Power Spectral Density (PSD) & Filter Attenuation", fontsize=11, fontweight="bold")
    return fig


def plot_noise_time_domain(study_results: dict, fs: float, zoom_ms: Tuple[float, float] = (50, 150)) -> plt.Figure:
    """رسم مقایسه‌ای موج زمانی زوم‌شده."""
    snr_trials = study_results["snr_trials"]
    snr_levels = list(snr_trials.keys())
    clean_sig = study_results["clean"]["signal"]

    t = np.arange(len(clean_sig)) / fs * 1000.0  # میلی‌ثانیه
    mask = (t >= zoom_ms[0]) & (t <= zoom_ms[1])
    t_z = t[mask]

    fig, axes = plt.subplots(
        len(snr_levels) + 1, 1, figsize=(9, 2.0 * (len(snr_levels) + 1)), sharex=True, constrained_layout=True
    )

    # سیگنال مرجع تمیز
    axes[0].plot(t_z, clean_sig[mask], color="#2ca02c", lw=0.9)
    axes[0].set_title("Clean Reference Signal", loc="left", fontsize=9, color="#2ca02c")
    axes[0].set_ylabel("Amp")
    axes[0].grid(alpha=0.3, ls="--")

    for i, snr in enumerate(snr_levels, start=1):
        ax = axes[i]
        trial = snr_trials[snr]
        ax.plot(t_z, trial["noisy_signal"][mask], color="#e74c3c", lw=0.6, alpha=0.75, label=f"Noisy ({snr} dB)")
        ax.plot(t_z, trial["filtered_signal"][mask], color="#1f77b4", lw=1.0, label="Filtered")
        ax.set_title(f"Target SNR = {snr} dB", loc="left", fontsize=9)
        ax.set_ylabel("Amp")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(loc="upper right", fontsize=7.5)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Time Domain Waveform (Zoomed)", fontsize=11, fontweight="bold")
    return fig


# ──────────────────────────────────────────────────────
# ۳. رندر کامل تب اختصاصی در Streamlit UI
# ──────────────────────────────────────────────────────
def render_noise_study_tab(
    clean_signal: np.ndarray,
    fs: float,
    fault_freqs: Optional[dict] = None,
    window_size: int = 2048,
    overlap: float = 0.5,
):
    """
    رندر رابط کاربری تعاملی مطالعه نویز در Streamlit.
    """
    st.markdown("###  مطالعه تعاملی اثر نویز و فیلتر بر عیب‌یابی ")
    st.info(
        "در این بخش می‌توانید اثر نویز سفید گوسی  را بر ویژگی‌های سیگنال، طیف پاکت بلبرینگ "
        "و **تشخیص نهایی سیستم ** ارزیابی کنید."
    )

    # کنترل‌های ورودی در استریم‌لیت
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**تنظیمات فیلتر باندگذر**")
        bp_low = st.number_input("قطع پایین (Hz):", min_value=10, max_value=int(fs / 2) - 100, value=1000, step=100)
        bp_high = st.number_input("قطع بالا (Hz):", min_value=bp_low + 100, max_value=int(fs / 2) - 10, value=5000, step=100)
        order = st.slider("مرتبه فیلتر (Order):", 1, 8, 4)

    with col2:
        st.markdown("**سطوح SNR آزمایشی (dB)**")
        snr_str = st.text_input("مقادیر SNR (جدا شده با کاما):", "20, 10, 0")
        try:
            snr_levels = [int(s.strip()) for s in snr_str.split(",") if s.strip()]
        except ValueError:
            snr_levels = [20, 10, 0]

    with col3:
        st.markdown("**تنظیمات زوم زمانی**")
        max_t = (len(clean_signal) / fs) * 1000.0
        zoom_ms = st.slider("بازه زوم (ms):", 0.0, float(max_t), (50.0, min(200.0, max_t)))

    filter_cfg = {"type": "bandpass", "cutoff": (float(bp_low), float(bp_high)), "order": order}

    # اجرای محاسبات اصلی
    results = run_noise_study(
        clean_signal=clean_signal,
        fs=fs,
        snr_levels=snr_levels,
        filter_cfg=filter_cfg,
        fault_freqs=fault_freqs,
        window_size=window_size,
        overlap=overlap,
    )

    # ۱. جدول مقایسه کمی و وضعیت تشخیص نهایی
    st.markdown("#### 📋 جدول مقایسه کمی و ارزیابی تغییرات تشخیص ")

    summary_data = []
    # سطر مرجع
    clean_diag = results["clean"]["diagnosis"]
    summary_data.append({
        "SNR Target": "Reference (Clean)",
        "Measured Noisy SNR": "∞",
        "Filtered SNR": "—",
        "SNR Gain": "—",
        "Noisy Kurtosis": f"{results['clean']['kurtosis']:.2f}",
        "Filtered Kurtosis": f"{results['clean']['kurtosis']:.2f}",
        "Noisy Verdict": clean_diag.get("level", "Unknown"),
        "Filtered Verdict": clean_diag.get("level", "Unknown"),
    })

    for snr, trial in results["snr_trials"].items():
        summary_data.append({
            "SNR Target": f"{snr} dB",
            "Measured Noisy SNR": f"{trial['snr_noisy']:.1f} dB",
            "Filtered SNR": f"{trial['snr_filtered']:.1f} dB",
            "SNR Gain": f"+{trial['snr_gain']:.1f} dB",
            "Noisy Kurtosis": f"{trial['noisy_kurtosis']:.2f}",
            "Filtered Kurtosis": f"{trial['filtered_kurtosis']:.2f}",
            "Noisy Verdict": trial["noisy_diagnosis"].get("level", "Unknown"),
            "Filtered Verdict": trial["filtered_diagnosis"].get("level", "Unknown"),
        })

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)

    # ۲. نمایش نمودارها در تب‌های مجزا
    tab_env, tab_psd, tab_time = st.tabs([
        "طیف پاکت و فرکانس‌های عیب (Envelope)",
        " چگالی طیفی توان (PSD)",
        " سیگنال زمانی (Time Domain)",
    ])

    with tab_env:
        fig_env = plot_noise_envelope_comparison(results, fault_freqs)
        st.pyplot(fig_env)
        plt.close(fig_env)

    with tab_psd:
        fig_psd = plot_noise_psd_comparison(results)
        st.pyplot(fig_psd)
        plt.close(fig_psd)

    with tab_time:
        fig_time = plot_noise_time_domain(results, fs, zoom_ms)
        st.pyplot(fig_time)
        plt.close(fig_time)
