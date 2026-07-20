"""
utils.py
--------
Utility functions for Motor Insight:
- Loading CSV / Excel data
- Detecting numeric / time columns
- Plotting raw signal, FFT spectrum and anomaly scores
- Converting matplotlib figures to PNG bytes (for embedding in PDF reports)

NOTE on plot text language:
Matplotlib does not perform Arabic/Persian text shaping (letter joining) or
bidi reordering out of the box, so Persian text rendered inside a plot would
look broken. Plot titles/labels are therefore kept in English, while the
Streamlit UI (rendered by the browser, which *does* support Persian shaping)
uses Persian.
"""
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    uploaded_file : file-like object with a `.name` attribute
        Typically the object returned by st.file_uploader().

    Returns
    -------
    pd.DataFrame
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(
            "Unsupported file format. Please upload a CSV or Excel file."
        )

    if df.empty:
        raise ValueError("The uploaded file does not contain any data.")

    return df


def get_numeric_columns(df: pd.DataFrame):
    """Return the list of numeric column names in the DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def detect_time_column(df: pd.DataFrame):
    """
    Try to find a column that represents time, based on common naming
    conventions. Returns the column name, or None if not found.
    """
    candidates = ("time", "timestamp", "t", "sec", "seconds")
    for col in df.columns:
        if str(col).strip().lower() in candidates:
            return col
    return None


def plot_raw_signal(t, signal, column_name: str):
    """Plot the raw (cleaned) time-domain signal."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, signal, linewidth=0.8, color="#1f77b4")
    ax.set_title(f"Raw Signal - {column_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_fft(freqs, magnitude, column_name: str):
    """Plot the (single-sided) frequency spectrum of the signal."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs, magnitude, linewidth=0.8, color="#d62728")
    ax.set_title(f"Frequency Spectrum (FFT) - {column_name}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_anomaly_scores(scores, labels):
    """
    Scatter-plot the Isolation Forest anomaly score for every analysis
    window, color-coded by Normal / Abnormal label.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#d62728" if lbl == "Abnormal" else "#2ca02c" for lbl in labels]
    ax.scatter(range(len(scores)), scores, c=colors, s=18)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Anomaly Score per Window")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Anomaly score (lower = more abnormal)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> io.BytesIO:
    """Render a matplotlib figure to an in-memory PNG (BytesIO)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


def plot_filtered_overlay(t, raw_signal, filtered_signal, column_name: str):
    """Overlay raw vs filtered signal so the effect of filtering is visible."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, raw_signal, linewidth=0.7, color="#9aa5b1", label="Raw")
    ax.plot(t, filtered_signal, linewidth=0.9, color="#1f77b4", label="Filtered")
    ax.set_title(f"Raw vs Filtered Signal - {column_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_psd(freqs, psd, title="Power Spectral Density (Welch)"):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(freqs, psd + 1e-20, linewidth=0.8, color="#9467bd")
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (power/Hz, log scale)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_spectrogram(freqs, times, Sxx, title="Spectrogram (STFT)"):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    pcm = ax.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-20), shading="gouraud", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    fig.tight_layout()
    return fig


def plot_envelope(t, envelope, column_name: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, envelope, linewidth=0.8, color="#ff7f0e")
    ax.set_title(f"Envelope (Hilbert transform magnitude) - {column_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope amplitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_envelope_spectrum(freqs, magnitude, fault_freq_markers: dict = None):
    """
    Plot the envelope spectrum, optionally marking theoretical fault
    frequencies (e.g. {'BPFO': 87.3, 'BPFI': 130.9}) as vertical lines so
    measured peaks can be visually compared against theory.
    """
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(freqs, magnitude, linewidth=0.8, color="#d62728")
    ax.set_title("Envelope Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    if fault_freq_markers:
        colors_cycle = ["#2ca02c", "#9467bd", "#17becf", "#8c564b"]
        for i, (label, freq) in enumerate(fault_freq_markers.items()):
            if freq is None:
                continue
            ax.axvline(freq, color=colors_cycle[i % len(colors_cycle)], linestyle="--", linewidth=1.0, label=f"{label} ({freq:.1f} Hz)")
        ax.legend(loc="upper right", fontsize=8)

    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_noise_comparison(t, original, noisy, filtered, column_name: str = ""):
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, original, linewidth=0.7, color="#2ca02c")
    axes[0].set_title(f"Original {column_name}")
    axes[1].plot(t, noisy, linewidth=0.7, color="#d62728")
    axes[1].set_title("Noisy (Gaussian noise added)")
    axes[2].plot(t, filtered, linewidth=0.7, color="#1f77b4")
    axes[2].set_title("Filtered (noise reduced)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_roc_curves(roc_curves: dict, title="ROC Curves (One-vs-Rest)"):
    """roc_curves: dict[class_label] -> (fpr, tpr, auc_value)"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, (fpr, tpr, auc_value) in roc_curves.items():
        ax.plot(fpr, tpr, linewidth=1.2, label=f"{label} (AUC={auc_value:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
