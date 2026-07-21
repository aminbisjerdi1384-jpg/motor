"""
app.py
------
Motor Insight - main Streamlit application (v2).

v2 adds, on top of the original CSV/Excel + FFT + features + PDF report:
    - Native .mat (CWRU) support, alongside CSV/Excel
    - Butterworth low/high/band-pass filtering (user-selectable, toggle)
    - Extended time-domain features (crest factor, kurtosis, skewness,
      shape/impulse/clearance factor, peak-to-peak)
    - PSD (Welch) and Periodogram
    - Spectrogram (STFT)
    - Envelope analysis (band-pass -> Hilbert -> envelope -> envelope FFT)
      with theoretical bearing fault frequency overlay (BPFO/BPFI/BSF/FTF)
    - Noise injection / SNR study and downsampling / Nyquist study
    - TWO diagnosis modes:
        Mode A - engineering rule-based thresholds (diagnosis.py)
        Mode B - supervised ML classifier trained on labeled files
                 (ml_classifier.py), fixing the original bug where
                 Isolation Forest, fit only on the uploaded file itself,
                 could never flag a uniformly faulty recording.
    - An extended PDF report covering all of the above.

Run with:
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

import preprocessing as pp
import features as ft
import spectral as sp
import envelope as env
import bearing as brg
import diagnosis as diag
import anomaly as an
from noise_study import render_noise_study_tab
from sampling_study import run_sampling_study
import ml_classifier as mlc
import cwru_loader as cw
from report import generate_pdf_report
from utils import (
    load_data, get_numeric_columns, detect_time_column,
    plot_raw_signal, plot_fft, plot_anomaly_scores, fig_to_png_bytes,
    plot_filtered_overlay, plot_psd, plot_spectrogram, plot_envelope,
    plot_envelope_spectrum, plot_noise_comparison, plot_confusion_matrix,
    plot_roc_curves,
)

st.set_page_config(page_title="Motor Insight", page_icon="⚙️", layout="wide")

st.title(" Motor Insight")
st.caption("سامانه پایش وضعیت و تشخیص عیب موتور/بلبرینگ بر پایه پردازش سیگنال")


# ==========================================================================
# Helpers
# ==========================================================================
def load_main_uploaded_file(uploaded_file, fs_default: float):
    """
    Load the main analysis file (CSV/Excel/.mat) and return a dict of
    available {channel_name: np.ndarray}, plus rpm if found in the file.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".mat"):
        channels = cw.load_mat_file(uploaded_file)
        rpm = channels.pop("rpm", None)
        return channels, rpm, None
    else:
        df = load_data(uploaded_file)
        numeric_cols = get_numeric_columns(df)
        time_col = detect_time_column(df)
        candidate_cols = [c for c in numeric_cols if c != time_col]
        channels = {c: pp.clean_signal(df[c]) for c in candidate_cols}
        return channels, None, (df, time_col)


def load_training_file(uploaded_file):
    """Load one labeled training file (CSV or .mat) for Mode B."""
    name = uploaded_file.name
    label = cw.infer_label_from_filename(name)
    if name.lower().endswith(".mat"):
        channels = cw.load_mat_file(uploaded_file)
        rpm = channels.get("rpm")
        signal = channels.get("DE") if "DE" in channels else next(iter(channels.values()))
        return signal, rpm, label
    else:
        df = pd.read_csv(uploaded_file)
        rpm = None
        rpm_cols = [c for c in df.columns if c.lower() == "rpm"]
        if rpm_cols:
            rpm = float(df[rpm_cols[0]].iloc[0])
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() != "rpm"]
        signal = df[numeric_cols[0]].to_numpy(dtype=float)
        return signal, rpm, label


# ==========================================================================
# Sidebar
# ==========================================================================
with st.sidebar:
    st.header("۱) فایل ورودی")
    uploaded_file = st.file_uploader(
        "بارگذاری فایل (CSV, Excel یا CWRU .mat)", type=["csv", "xlsx", "xls", "mat"]
    )
    fs = st.number_input("فرکانس نمونه‌برداری (Hz)", min_value=1.0, value=12000.0, step=100.0)

    st.header("۲) اطلاعات بلبرینگ/دور موتور (اختیاری)")
    st.caption("برای محاسبه فرکانس‌های مشخصه عیب (BPFO/BPFI/BSF/FTF) و تحلیل پاکت")
    show_bearing_freqs = st.checkbox("نمایش فرکانس‌های مشخصه بلبرینگ", value=True)
    rpm_input = st.number_input("دور موتور (RPM)", min_value=1.0, value=1797.0, step=1.0)
    bearing_position = st.selectbox(
        "محل بلبرینگ (مشخصات استاندارد CWRU)", ["drive_end", "fan_end"], index=0
    )

    st.header("۳) فیلتر دیجیتال (Butterworth)")
    filter_enabled = st.checkbox("فعال‌سازی فیلتر", value=False)
    st.caption(
        "⚠️ توجه: عیوب بلبرینگ معمولاً انرژی‌شان را در فرکانس‌های تشدید (اغلب بالا) آزاد می‌کنند. "
        "یک باند بسیار محدود یا پایین ممکن است خودِ سیگنال عیب را هم حذف کند. "
        "در صورت شک، فیلتر را غیرفعال بگذارید یا باند گسترده انتخاب کنید."
    )
    filter_type = st.selectbox("نوع فیلتر", ["lowpass", "highpass", "bandpass"], index=2, disabled=not filter_enabled)
    filter_order = st.slider("مرتبه فیلتر", min_value=1, max_value=10, value=4, disabled=not filter_enabled)
    if filter_type == "bandpass":
        f_low = st.number_input("فرکانس پایین (Hz)", min_value=0.1, value=500.0, disabled=not filter_enabled)
        f_high = st.number_input("فرکانس بالا (Hz)", min_value=1.0, value=4000.0, disabled=not filter_enabled)
        filter_cutoff = (f_low, f_high)
    else:
        filter_cutoff = st.number_input("فرکانس قطع (Hz)", min_value=0.1, value=1000.0, disabled=not filter_enabled)

    st.header("۴) تنظیمات پنجره‌بندی")
    window_size = st.number_input("اندازه پنجره (تعداد نمونه)", min_value=64, value=1024, step=64)
    overlap = st.slider("همپوشانی پنجره‌ها", min_value=0.0, max_value=0.9, value=0.25, step=0.05)

    st.header("۵) حالت تشخیص عیب")
    diagnosis_mode = st.radio(
        "روش تشخیص نهایی",
        ["Mode A: قانون‌محور (مهندسی)", "Mode B: یادگیری ماشین (RF/SVM)"],
    )

    mode_a_thresholds = diag.DiagnosisThresholds()
    if diagnosis_mode.startswith("Mode A"):
        with st.expander("تنظیم آستانه‌های Mode A (پیشرفته)"):
            mode_a_thresholds.kurtosis_warning = st.number_input("آستانه هشدار Kurtosis", value=4.0)
            mode_a_thresholds.kurtosis_fault = st.number_input("آستانه خرابی Kurtosis", value=6.0)
            mode_a_thresholds.crest_factor_warning = st.number_input("آستانه هشدار Crest Factor", value=4.0)
            mode_a_thresholds.crest_factor_fault = st.number_input("آستانه خرابی Crest Factor", value=6.0)

    if diagnosis_mode.startswith("Mode B"):
        st.subheader("آموزش مدل (Mode B)")
        st.caption(
            "چند فایل برچسب‌دار (سالم/IR/OR/Ball) بارگذاری کنید - مثلاً خروجی "
            "download_cwru_data.py یا فایل‌های .mat اصلی CWRU."
        )
        training_files = st.file_uploader(
            "فایل‌های آموزشی (CSV یا .mat)", type=["csv", "mat"], accept_multiple_files=True
        )
        confirmed_labels = []
        if training_files:
            st.caption("برچسب هر فایل را تایید/اصلاح کنید:")
            for i, tf in enumerate(training_files):
                inferred = cw.infer_label_from_filename(tf.name)
                default_idx = cw.FAULT_CLASSES.index(inferred) if inferred in cw.FAULT_CLASSES else 0
                lbl = st.selectbox(
                    f"{tf.name}", cw.FAULT_CLASSES, index=default_idx, key=f"train_label_{i}_{tf.name}"
                )
                confirmed_labels.append(lbl)

        model_type_label = st.radio("نوع طبقه‌بند", ["Random Forest", "SVM"])
        model_type = "random_forest" if model_type_label == "Random Forest" else "svm"

        train_clicked = st.button("🎯 آموزش مدل", disabled=not training_files)
        if train_clicked and training_files:
            with st.spinner("در حال استخراج ویژگی و آموزش مدل..."):
                filter_cfg = (
                    {"type": filter_type, "cutoff": filter_cutoff, "order": int(filter_order)}
                    if filter_enabled else None
                )
                tf_objects = []
                for tf, lbl in zip(training_files, confirmed_labels):
                    sig, _rpm, _ = load_training_file(tf)
                    sig = pp.clean_signal(pd.Series(sig))
                    tf_objects.append(mlc.TrainingFile(signal=sig, fs=fs, label=lbl))

                # IMPORTANT: must apply the exact same filter setting used on the
                # signal at prediction time (see ml_classifier.py docstring) -
                # otherwise the classifier learns a systematically different
                # feature distribution than what it will see at inference time.
                train_feat_df = mlc.extract_training_features(
                    tf_objects, int(window_size), overlap, filter_cfg=filter_cfg
                )
                if train_feat_df["label"].nunique() < 2:
                    st.error("حداقل ۲ کلاس متفاوت برای آموزش مدل لازم است.")
                else:
                    result = mlc.train_classifier(train_feat_df, model_type=model_type)
                    st.session_state["mlc_result"] = result
                    st.session_state["mlc_model_type"] = model_type_label
                    st.session_state["mlc_filter_cfg"] = filter_cfg
                    st.success(
                        f"مدل آموزش دید. دقت روی داده آزمون: "
                        f"{result['classification_report']['accuracy']*100:.1f}٪"
                    )

if uploaded_file is None:
    
    st.markdown(
        """
        <div style="
            background-color: #f3e8ff;
            border-right: 5px solid #9333ea;
            padding: 12px 16px;
            border-radius: 6px;
            direction: rtl;
            text-align: right;
            color: #581c87;
            margin-bottom: 20px;
            font-size: 0.95rem;
        ">
            ℹ️ لطفاً یک فایل با فرمت <b>CSV</b>، <b>Excel</b> یا <b>mat.</b> (دیتابیس CWRU) بارگذاری کنید.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="direction: rtl; text-align: right;">

        **راهنمای فرمت فایل ورودی:**
        * **CSV / Excel:** هر ستون عددی یک کانال سنسور است (مثلاً `vibration_x` یا `current`).
        * **فایل اصلی دیتابیس mat.:** فایل‌های CWRU (کانال‌های DE/FE/BA و RPM) به صورت خودکار خوانده می‌شوند.
        * برای دریافت داده واقعی CWRU، اسکریپت `download_cwru_data.py` را اجرا کنید (این محیط دسترسی اینترنت ندارد).

        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# --------------------------------------------------------------------------
# Load file + channel selection
# --------------------------------------------------------------------------
try:
    channels, detected_rpm, extra = load_main_uploaded_file(uploaded_file, fs)
except Exception as e:
    st.error(f"خطا در خواندن فایل: {e}")
    st.stop()

if not channels:
    st.error("هیچ کانال/ستون عددی قابل تحلیل در فایل یافت نشد.")
    st.stop()

if extra is not None:
    df, time_col = extra
    st.subheader("پیش‌نمایش داده")
    st.dataframe(df.head(20), use_container_width=True)
else:
    time_col = None

selected_col = st.selectbox("کانال/ستون سنسور برای تحلیل", list(channels.keys()))
raw_signal = np.asarray(channels[selected_col], dtype=float)

if detected_rpm:
    st.markdown(
    f"""
    <div style="
        direction: rtl;
        text-align: right;
        color: #9333ea;
        font-size: 0.85rem;
        margin-top: -8px;
        margin-bottom: 12px;
    ">
        دور موتور تشخیص‌داده‌شده از فایل: <b>{detected_rpm:.0f} RPM</b> (در صورت تمایل از سایدبار اصلاح کنید)
    </div>
    """,
    unsafe_allow_html=True
)
    rpm = detected_rpm
else:
    rpm = rpm_input

if extra is not None and time_col is not None:
    t = df[time_col].to_numpy(dtype=float)
    if len(t) != len(raw_signal):
        t = np.arange(len(raw_signal)) / fs
else:
    t = np.arange(len(raw_signal)) / fs

detrended = pp.remove_dc_offset(raw_signal)

# --------------------------------------------------------------------------
# Global filter
# --------------------------------------------------------------------------
if filter_enabled:
    filtered_signal = pp.apply_filter(detrended, filter_type, filter_cutoff, fs, order=int(filter_order))
    working_signal = filtered_signal
else:
    filtered_signal = detrended
    working_signal = detrended

current_filter_cfg = (
    {"type": filter_type, "cutoff": filter_cutoff, "order": int(filter_order)} if filter_enabled else None
)

fault_freqs = brg.cwru_fault_frequencies(rpm, bearing_position) if show_bearing_freqs else None

# ==========================================================================
# Tabs
# ==========================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "خام و فیلتر",
        "تحلیل فرکانسی",
        "زمان-فرکانس",
        "پاکت (Envelope)",
        "نویز",
        "نمونه‌برداری",
        "ویژگی‌ها",
        "تشخیص عیب",
        "گزارش PDF",
    ]
)
# --- Tab 1: Raw & Filter ---------------------------------------------------
with tab1:
    st.subheader("سیگنال خام")
    raw_fig = plot_raw_signal(t, raw_signal, selected_col)
    st.pyplot(raw_fig)

    if filter_enabled:
        st.subheader("سیگنال خام در مقابل فیلترشده")
        overlay_fig = plot_filtered_overlay(t, detrended, filtered_signal, selected_col)
        st.pyplot(overlay_fig)
    else:
        st.info("فیلتر غیرفعال است. برای فعال‌سازی به سایدبار (بخش ۳) مراجعه کنید.")

    st.subheader("آمار کلی سیگنال (کل فایل)")
    quick_feats = ft.extract_features_window(working_signal, fs)
    st.dataframe(pd.DataFrame([quick_feats]), use_container_width=True)

# --- Tab 2: Frequency domain -------------------------------------------------
with tab2:
    st.subheader("طیف فرکانسی (FFT)")
    fft_freqs, fft_mag = ft.compute_fft(working_signal, fs)
    fft_fig = plot_fft(fft_freqs, fft_mag, selected_col)
    st.pyplot(fft_fig)

    col_psd, col_pg = st.columns(2)
    with col_psd:
        st.subheader("PSD (روش Welch)")
        psd_freqs, psd_vals = sp.compute_psd_welch(working_signal, fs)
        psd_fig = plot_psd(psd_freqs, psd_vals, "PSD (Welch)")
        st.pyplot(psd_fig)
    with col_pg:
        st.subheader("Periodogram")
        pg_freqs, pg_vals = sp.compute_periodogram(working_signal, fs)
        pg_fig = plot_psd(pg_freqs, pg_vals, "Periodogram")
        st.pyplot(pg_fig)

    if show_bearing_freqs:
        st.subheader("فرکانس‌های مشخصه بلبرینگ (نظری)")
        st.caption(
            "بر اساس مشخصات رسمی بلبرینگ CWRU (6205-2RS / 6203-2RS) و RPM وارد شده. "
            "برای موتور/بلبرینگ دیگر، این مقادیر صرفاً جهت آشنایی با روش هستند."
        )
        freq_table = pd.DataFrame([fault_freqs])
        st.dataframe(freq_table, use_container_width=True)

# --- Tab 3: Time-Frequency ---------------------------------------------------
with tab3:
    st.subheader("اسپکتروگرام (STFT)")
    nperseg = st.slider("طول هر بازه STFT (نمونه)", min_value=64, max_value=2048, value=256, step=64)
    sg_freqs, sg_times, sg_Sxx = sp.compute_spectrogram(working_signal, fs, nperseg=int(nperseg))
    sg_fig = plot_spectrogram(sg_freqs, sg_times, sg_Sxx)
    st.pyplot(sg_fig)
    st.caption(
        "اسپکتروگرام نشان می‌دهد محتوای فرکانسی سیگنال در طول زمان چگونه تغییر می‌کند - "
        "مفید برای عیوبی که شدت‌شان با زمان یا سرعت موتور تغییر می‌کند."
    )


# In your app.py, replace the `with tab4:` block with the code below.
# (Keep all imports at the top of your file as-is.)
# Additional import needed at top of app.py:
#   from envelope import auto_envelope_analysis

# ──────────────────────────────────────────────────────────────────────────
# TAB 4  —  Envelope Analysis
# ──────────────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("تحلیل پاکت (Envelope Analysis)")
    st.caption(
        "روش: فیلتر میان‌گذر حول فرکانس تشدید سازه‌ای ← تبدیل هیلبرت ← پاکت دامنه ← FFT پاکت. "
        "امضای عیب بلبرینگ (BPFO/BPFI/BSF) معمولاً در طیف خام گم می‌شود "
        "ولی در طیف پاکت آشکار می‌گردد."
    )

    # ── Band selection ────────────────────────────────────────────────────
    st.markdown("**انتخاب باند تشدید (Resonance Band)**")
    auto_band_toggle = st.checkbox(
        "🔍 تشخیص خودکار باند (Kurtogram-lite)",
        value=True,
        help=(
            "الگوریتم Kurtogram-lite چندین باند فرکانسی را آزمایش می‌کند و باندی را "
            "انتخاب می‌کند که کurtosis پاکت آن بیشینه است. "
            "این باند معمولاً با فرکانس تشدید سازه‌ای بلبرینگ مطابقت دارد و "
            "برای داده‌های واقعی CWRU بسیار مؤثرتر از انتخاب دستی است."
        ),
    )

    nyquist = fs / 2.0

    if auto_band_toggle:
        kurtogram_centers = st.slider(
            "تعداد باندهای آزمایشی", min_value=4, max_value=20, value=12, step=2,
        )
        kurtogram_bw = st.number_input(
            "عرض هر باند (Hz)", min_value=200.0,
            max_value=nyquist * 0.6, value=1000.0, step=100.0,
        )

        with st.spinner("در حال جستجوی باند بهینه (Kurtogram-lite)…"):
            auto_result = env.auto_envelope_analysis(
                detrended, fs,
                auto_band=True,
                n_centers=int(kurtogram_centers),
                bandwidth_hz=float(kurtogram_bw),
            )
        band_low, band_high = auto_result["band_used"]
        band_kurt           = auto_result["band_kurtosis"]
        envelope_signal     = auto_result["envelope"]
        env_freqs           = auto_result["freqs"]
        env_mag             = auto_result["magnitude"]

        st.success(
            f"باند انتخاب‌شده: **{band_low:.0f} – {band_high:.0f} Hz**  "
            f"(کurtosis پاکت: **{band_kurt:.2f}**)"
        )
        if auto_result.get("band_scan"):
            scan_df = pd.DataFrame(
                [{"باند (Hz)": f"{lo:.0f}–{hi:.0f}", "Kurtosis": round(k, 2)}
                 for (lo, hi), k in sorted(auto_result["band_scan"].items(),
                                           key=lambda x: -x[1])]
            )
            with st.expander("نتایج جستجوی تمام باندها"):
                st.dataframe(scan_df, use_container_width=True)

    else:
        default_low  = min(1000.0, nyquist * 0.3)
        default_high = min(4000.0, nyquist * 0.9)
        band_low  = st.number_input(
            "فرکانس پایین باند تشدید (Hz)", min_value=1.0,
            value=float(default_low),
        )
        band_high = st.number_input(
            "فرکانس بالای باند تشدید (Hz)", min_value=2.0,
            value=float(default_high),
        )
        # Validate
        if band_high <= band_low:
            st.error("فرکانس بالای باند باید از فرکانس پایین بزرگ‌تر باشد.")
            st.stop()

        envelope_signal, env_freqs, env_mag = env.envelope_analysis_pipeline(
            detrended, fs, band=(band_low, band_high), filter_order=4,
        )

    # ── Time-domain envelope ──────────────────────────────────────────────
    env_time_fig = plot_envelope(t, envelope_signal, selected_col)
    st.pyplot(env_time_fig)

    # ── Envelope spectrum with fault-freq markers ─────────────────────────
    fault_markers = None
    if show_bearing_freqs and fault_freqs:
        fault_markers = {
            "BPFO": fault_freqs.get("BPFO"),
            "BPFI": fault_freqs.get("BPFI"),
            "BSF":  fault_freqs.get("BSF"),
        }
    env_spec_fig = plot_envelope_spectrum(
        env_freqs, env_mag, fault_freq_markers=fault_markers
    )
    st.pyplot(env_spec_fig)

    # ── Fault-frequency peak matching table ──────────────────────────────
    if show_bearing_freqs and fault_freqs:
        st.subheader("تطبیق پیک با فرکانس‌های مشخصه بلبرینگ")
        st.caption(
            "الگوریتم fundamental و ۳ هارمونیک بعدی (h=1..4) هر فرکانس مشخصه را جستجو "
            "می‌کند و بهترین SNR را در همه هارمونیک‌ها گزارش می‌دهد. "
            "عیوب اولیه اغلب در هارمونیک‌های بالاتر ظاهر می‌شوند، نه در فرکانس پایه."
        )

        # ── اسکن fundamental + هارمونیک‌ها با scan_fault_frequencies ────────
        _scan_input = {
            "BPFO": fault_freqs.get("BPFO"),
            "BPFI": fault_freqs.get("BPFI"),
            "BSF":  fault_freqs.get("BSF"),
        }
        _scan_results = env.scan_fault_frequencies(
            env_freqs, env_mag, _scan_input,
            tolerance_hz=2.0,
            include_harmonics=3,           # بررسی h=1,2,3,4
            rpm_uncertainty_pct=2.0,       # پنجره تطبیقی با عدم‌قطعیت RPM
            use_adaptive_tolerance=True,
        )
        _summary = env.fault_scan_summary(_scan_results)

        _LABEL_MAP = {
            "BPFO": "Outer Race Fault",
            "BPFI": "Inner Race Fault",
            "BSF":  "Ball Fault",
        }
        rows = []
        for fault_key, label_name in _LABEL_MAP.items():
            info = _summary.get(fault_key)
            if info is None:
                continue
            best_h      = info["best_harmonic"]
            best_snr    = info["best_snr_db"]
            n_found     = info["n_harmonics_found"]
            cls         = info["best_classification"]
            best_detail = info["details"].get(best_h, {}) if best_h else {}
            target      = fault_freqs.get(fault_key)

            snr_str  = f"{best_snr:.1f}"                       if best_snr is not None          else "—"
            freq_str = f"{best_detail['freq']:.2f}"            if best_detail.get("freq")       else "—"
            mag_str  = f"{best_detail['magnitude']:.4f}"       if best_detail.get("magnitude")  else "—"
            nf_str   = f"{best_detail['noise_floor']:.4f}"     if best_detail.get("noise_floor") else "—"

            if best_detail.get("freq") and best_h and target:
                ferr_str = f"{abs(best_detail['freq'] - target * best_h):.2f}"
            else:
                ferr_str = "—"

            cls_display = (f"🟢 {cls}" if cls == "Strong"
                           else f"🟡 {cls}" if cls == "Weak"
                           else f"🔴 {cls}")

            rows.append({
                "نوع عیب":            label_name,
                "فرکانس پایه (Hz)":   round(target, 2) if target else "—",
                "بهترین هارمونیک":    f"h={best_h}"    if best_h  else "—",
                "فرکانس تشخیص (Hz)": freq_str,
                "خطا (Hz)":           ferr_str,
                "Magnitude":          mag_str,
                "Noise Floor":        nf_str,
                "SNR (dB)":           snr_str,
                "هارمونیک تأیید":     n_found,
                "تشخیص":              cls_display,
            })

        df_peaks = pd.DataFrame(rows)
        st.dataframe(df_peaks, use_container_width=True)

        # جزئیات هارمونیک‌ها در اکسپندر
        with st.expander("جزئیات هر هارمونیک"):
            for fault_key, label_name in _LABEL_MAP.items():
                info = _summary.get(fault_key)
                if not info:
                    continue
                st.markdown(f"**{label_name} ({fault_key})**")
                h_rows = []
                for h, hr in info["details"].items():
                    target = fault_freqs.get(fault_key)
                    h_rows.append({
                        "h":           h,
                        "فرکانس هدف": f"{target * h:.1f}" if target else "—",
                        "فرکانس یافت": f"{hr['freq']:.2f}" if hr.get("freq") else "—",
                        "SNR (dB)":   f"{hr['snr_db']:.1f}" if hr.get("snr_db") else "—",
                        "تشخیص":      hr["classification"],
                    })
                st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)

        # پیام خلاصه
        detected = [r["نوع عیب"] for r in rows if "Not Detected" not in r["تشخیص"]]
        if detected:
            st.warning(
                f"⚠️ پیک مشخصه در طیف پاکت تشخیص داده شد: "
                f"**{', '.join(detected)}**. "
                "این یک شاخص حمایتی است — تشخیص نهایی در تب «تشخیص عیب» انجام می‌شود."
            )
        else:
            st.info(
                "هیچ پیک مشخصه‌ای با SNR کافی در طیف پاکت یافت نشد. "
                "اگر Mode A/B عیب تشخیص داده، "
                "باند تشدید را تغییر دهید یا از 'تشخیص خودکار باند' استفاده کنید."
            )

# --- Tab 5: Noise Study (مطالعه اثر نویز و فیلتر) -----------------------------
with tab5:
    render_noise_study_tab(
        clean_signal=detrended,
        fs=fs,
        fault_freqs=fault_freqs if show_bearing_freqs else None,
        window_size=int(window_size),
        overlap=overlap,
    )
with tab6:

    st.header("Sampling Rate & Nyquist Study")

    st.write(
        """
        This experiment evaluates the influence of sampling frequency on
        vibration-based fault diagnosis.

        The original signal is resampled to several lower sampling rates,
        then the diagnostic features are recalculated.

        The experiment illustrates the Nyquist criterion and aliasing.
        """
    )

    sampling_rates = st.multiselect(
        "Sampling frequencies (Hz)",
        [12000, 8000, 6000, 4000, 2000, 1000],
        default=[12000, 6000, 2000]
    )

    if st.button("Run Sampling Study"):

        results = run_sampling_study(
            signal=working_signal,
            fs=fs,
            sampling_rates=sampling_rates
        )

        st.dataframe(results["table"])

        st.pyplot(results["time_fig"])

        st.pyplot(results["fft_fig"])

        st.pyplot(results["feature_fig"])
        
# --- Tab 7: Features ---------------------------------------------------------
with tab7:
    st.subheader("ویژگی‌های استخراج‌شده در هر پنجره")
    feature_df = ft.extract_features_dataframe(working_signal, fs, int(window_size), overlap)
    st.dataframe(feature_df, use_container_width=True)

    st.subheader("آمار خلاصه")
    st.dataframe(feature_df[ft.FEATURE_COLUMNS].describe().T, use_container_width=True)

    with st.expander("بررسی نوسانات درون همین فایل (Isolation Forest - تکمیلی، نه تشخیص نهایی)"):
        st.caption(
            "این روش فقط پنجره‌هایی را که نسبت به بقیه همین فایل غیرعادی به‌نظر می‌رسند مشخص می‌کند. "
            "اگر کل فایل به‌طور یکنواخت معیوب باشد، این روش به‌تنهایی آن را تشخیص نمی‌دهد - برای "
            "تشخیص نهایی به تب «تشخیص عیب» مراجعه کنید."
        )
        contamination = st.slider("حساسیت (contamination)", 0.01, 0.4, 0.1, 0.01)
        an_result = an.detect_anomalies(feature_df, contamination=contamination)
        an_fig = plot_anomaly_scores(an_result["anomaly_score"].to_numpy(), an_result["anomaly_label"].tolist())
        st.pyplot(an_fig)

# --- Tab 8: Diagnosis (the primary verdict) ----------------------------------
with tab8:
    st.subheader("تشخیص نهایی")
    feature_df = ft.extract_features_dataframe(working_signal, fs, int(window_size), overlap)

    if diagnosis_mode.startswith("Mode A"):
        result = diag.rule_based_diagnosis(feature_df, mode_a_thresholds)
        level = result["level"]
        if level == "Faulty":
            st.error("🔴 وضعیت: **معیوب (Faulty)**")
        elif level == "Warning":
            st.warning("🟡 وضعیت: **هشدار (Warning)**")
        else:
            st.success("🟢 وضعیت: **سالم (Healthy)**")

        st.markdown("**دلایل (Mode A - قانون‌محور):**")
        for line in result["explanation"]:
            st.write("- " + line)

    else:  # Mode B
        mlc_result = st.session_state.get("mlc_result")
        if mlc_result is None:
            st.warning("هنوز مدلی آموزش ندیده است. ابتدا در سایدبار (بخش ۵) فایل‌های آموزشی را بارگذاری و مدل را آموزش دهید.")
        else:
            if st.session_state.get("mlc_filter_cfg") != current_filter_cfg:
                st.warning(
                    "⚠️ تنظیمات فیلتر سایدبار از زمان آموزش مدل تغییر کرده است. "
                    "نتیجه پیش‌بینی ممکن است نامعتبر باشد - بهتر است مدل را دوباره آموزش دهید."
                )
            pred_df = mlc.predict_file(mlc_result["pipeline"], feature_df)
            summary = mlc.summarize_prediction(pred_df)

            label = summary["predicted_label"]
            if label == "Healthy":
                st.success(f"🟢 کلاس پیش‌بینی‌شده: **{label}**")
            else:
                st.error(f"🔴 کلاس پیش‌بینی‌شده: **{label}**")

            c1, c2 = st.columns(2)
            c1.metric("سهم رأی پنجره‌ها", f"{summary['vote_share']*100:.1f}٪")
            c2.metric("میانگین اطمینان مدل", f"{summary['mean_confidence']*100:.1f}٪")

            st.markdown("**توزیع رأی پنجره‌ها بین کلاس‌ها:**")
            st.bar_chart(pd.Series(summary["class_vote_shares"]))

            with st.expander("ارزیابی مدل روی داده آزمون آموزش (Confusion Matrix / ROC)"):
                st.caption(f"نوع مدل: {st.session_state.get('mlc_model_type', '')}")
                cm_fig = plot_confusion_matrix(mlc_result["confusion_matrix"], mlc_result["classes"])
                st.pyplot(cm_fig)
                roc_fig = plot_roc_curves(mlc_result["roc_curves"])
                st.pyplot(roc_fig)
                report_df = pd.DataFrame(mlc_result["classification_report"]).T
                st.dataframe(report_df, use_container_width=True)

# --- Tab 9: PDF report --------------------------------------------------------
with tab9:
    st.subheader("دانلود گزارش PDF")
    motor_name = st.text_input("نام/شناسه موتور (برای درج در گزارش)", value="Motor-01")

    if st.button("ساخت گزارش PDF"):
        with st.spinner("در حال تولید گزارش..."):
            feature_df_report = ft.extract_features_dataframe(working_signal, fs, int(window_size), overlap)

            if diagnosis_mode.startswith("Mode A"):
                diag_result = diag.rule_based_diagnosis(feature_df_report, mode_a_thresholds)
                diagnosis_summary = {
                    "mode": "Mode A (Rule-based)",
                    "level": diag_result["level"],
                    "explanation": diag_result["explanation"],
                }
            else:
                mlc_result = st.session_state.get("mlc_result")
                if mlc_result is None:
                    st.error("ابتدا مدل Mode B را آموزش دهید یا حالت Mode A را انتخاب کنید.")
                    st.stop()
                if st.session_state.get("mlc_filter_cfg") != current_filter_cfg:
                    st.warning(
                        "⚠️ تنظیمات فیلتر سایدبار از زمان آموزش مدل تغییر کرده است؛ "
                        "نتیجه این گزارش ممکن است نامعتبر باشد."
                    )
                pred_df_report = mlc.predict_file(mlc_result["pipeline"], feature_df_report)
                pred_summary = mlc.summarize_prediction(pred_df_report)
                diagnosis_summary = {
                    "mode": f"Mode B ({st.session_state.get('mlc_model_type', 'ML')})",
                    "level": pred_summary["predicted_label"],
                    "explanation": [
                        f"Vote share: {pred_summary['vote_share']*100:.1f}%",
                        f"Mean confidence: {pred_summary['mean_confidence']*100:.1f}%",
                    ],
                }

            fft_freqs_r, fft_mag_r = ft.compute_fft(working_signal, fs)
            psd_freqs_r, psd_vals_r = sp.compute_psd_welch(working_signal, fs)
            sg_freqs_r, sg_times_r, sg_Sxx_r = sp.compute_spectrogram(working_signal, fs, nperseg=256)
            env_sig_r, env_freqs_r, env_mag_r = env.envelope_analysis_pipeline(
                detrended, fs, band=(band_low, band_high)
            )

            pdf_buffer = generate_pdf_report(
                motor_name=motor_name,
                column_name=selected_col,
                diagnosis_summary=diagnosis_summary,
                feature_df=feature_df_report,
                raw_fig_bytes=fig_to_png_bytes(plot_raw_signal(t, raw_signal, selected_col)),
                fft_fig_bytes=fig_to_png_bytes(plot_fft(fft_freqs_r, fft_mag_r, selected_col)),
                psd_fig_bytes=fig_to_png_bytes(plot_psd(psd_freqs_r, psd_vals_r)),
                spectrogram_fig_bytes=fig_to_png_bytes(plot_spectrogram(sg_freqs_r, sg_times_r, sg_Sxx_r)),
                envelope_fig_bytes=fig_to_png_bytes(plot_envelope_spectrum(env_freqs_r, env_mag_r)),
            )
        st.download_button(
            label="⬇️ دانلود گزارش PDF",
            data=pdf_buffer,
            file_name=f"motor_insight_report_{motor_name}.pdf",
            mime="application/pdf",
        )
