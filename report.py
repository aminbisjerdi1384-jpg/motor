"""
report.py
---------
Generates a downloadable PDF report summarizing the analysis: signal
statistics, FFT, PSD, spectrogram, envelope analysis, extracted features,
and the final diagnosis (from either Mode A or Mode B) with explanation.

Uses reportlab (the standard choice for composing a multi-section PDF with
text, tables and embedded images in pure Python; not in the original
core library list, but required for a real PDF deliverable).

NOTE on language: kept in English - reportlab does not perform Arabic/
Persian glyph shaping or bidi reordering by default, so Persian text would
render as disconnected, incorrectly-ordered letters without extra
dependencies (a Persian TTF font + arabic-reshaper + python-bidi). The
Streamlit UI (app.py) uses Persian since the browser renders that correctly.
"""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)

from features import FEATURE_COLUMNS

FEATURE_DISPLAY_NAMES = {
    "rms": "RMS",
    "peak": "Peak value",
    "peak_to_peak": "Peak-to-Peak",
    "mean": "Mean",
    "variance": "Variance",
    "crest_factor": "Crest Factor",
    "kurtosis": "Kurtosis",
    "skewness": "Skewness",
    "shape_factor": "Shape Factor",
    "impulse_factor": "Impulse Factor",
    "clearance_factor": "Clearance Factor",
    "dominant_freq": "Dominant frequency (Hz)",
    "spectral_energy": "Spectral energy",
}

_LEVEL_COLORS = {
    "Faulty": colors.red,
    "Warning": colors.HexColor("#d4a017"),
    "Healthy": colors.HexColor("#2ca02c"),
}


def _image_flowable(png_bytes_io: io.BytesIO, width=16 * cm, aspect=0.4):
    png_bytes_io.seek(0)
    return Image(png_bytes_io, width=width, height=width * aspect)


def generate_pdf_report(
    motor_name: str,
    column_name: str,
    diagnosis_summary: dict,
    feature_df,
    raw_fig_bytes: io.BytesIO,
    fft_fig_bytes: io.BytesIO,
    psd_fig_bytes: io.BytesIO,
    spectrogram_fig_bytes: io.BytesIO,
    envelope_fig_bytes: io.BytesIO,
) -> io.BytesIO:
    """
    Build the PDF report and return it as an in-memory BytesIO buffer,
    ready to be passed to st.download_button.

    diagnosis_summary : dict with keys
        mode        : str, e.g. "Mode A (Rule-based)" or "Mode B (Random Forest)"
        level       : str, e.g. "Healthy" / "Warning" / "Faulty" or a fault class name
        explanation : list[str]
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Motor Insight Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=20)
    h2_style = ParagraphStyle("H2X", parent=styles["Heading2"], spaceBefore=12)
    normal_style = styles["Normal"]
    bullet_style = ParagraphStyle("BulletX", parent=styles["Normal"], leftIndent=12)

    level = diagnosis_summary.get("level", "Unknown")
    diag_color = _LEVEL_COLORS.get(level, colors.black)
    diag_style = ParagraphStyle("DiagX", parent=styles["Heading1"], textColor=diag_color, fontSize=18)

    elements = [
        Paragraph("Motor Insight - Condition Monitoring & Fault Diagnosis Report", title_style),
        Spacer(1, 0.3 * cm),
        Paragraph(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style),
        Paragraph(f"Motor / Asset ID: {motor_name}", normal_style),
        Paragraph(f"Analyzed sensor channel: {column_name}", normal_style),
        Paragraph(f"Diagnosis method: {diagnosis_summary.get('mode', 'N/A')}", normal_style),
        Spacer(1, 0.5 * cm),
        Paragraph(f"Final diagnosis: {level}", diag_style),
    ]

    for line in diagnosis_summary.get("explanation", []):
        elements.append(Paragraph(f"&bull; {line}", bullet_style))

    elements += [
        Spacer(1, 0.5 * cm),
        Paragraph("Raw Signal", h2_style),
        _image_flowable(raw_fig_bytes),
        Spacer(1, 0.3 * cm),
        Paragraph("Frequency Spectrum (FFT)", h2_style),
        _image_flowable(fft_fig_bytes),
        Spacer(1, 0.3 * cm),
        Paragraph("Power Spectral Density (Welch)", h2_style),
        _image_flowable(psd_fig_bytes),
        PageBreak(),
        Paragraph("Spectrogram (Time-Frequency, STFT)", h2_style),
        _image_flowable(spectrogram_fig_bytes),
        Spacer(1, 0.3 * cm),
        Paragraph("Envelope Spectrum (Bearing Fault Demodulation)", h2_style),
        _image_flowable(envelope_fig_bytes),
        PageBreak(),
        Paragraph("Average Extracted Features", h2_style),
    ]

    avg_feats = feature_df[FEATURE_COLUMNS].mean()
    table_data = [["Feature", "Average value"]]
    for key in FEATURE_COLUMNS:
        label = FEATURE_DISPLAY_NAMES.get(key, key)
        table_data.append([label, f"{avg_feats[key]:.4f}"])

    table = Table(table_data, colWidths=[8 * cm, 8 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(table)

    elements.append(Spacer(1, 0.6 * cm))
    elements.append(Paragraph("Recommendations", h2_style))
    if level == "Faulty":
        rec = (
            "The analyzed signal shows strong indicators of an active mechanical fault "
            "(elevated kurtosis/crest factor and/or a clear bearing characteristic "
            "frequency in the envelope spectrum). Schedule an inspection of this asset "
            "and consider increasing monitoring frequency until resolved."
        )
    elif level == "Warning":
        rec = (
            "Some indicators are elevated above the normal baseline but not "
            "conclusively faulty. Recommend closer monitoring and a follow-up "
            "measurement to track the trend over time."
        )
    elif level == "Healthy":
        rec = "No significant fault indicators detected. Continue routine monitoring."
    else:
        rec = f"Predicted condition: {level}. Review the per-class probabilities and confusion matrix for context."
    elements.append(Paragraph(rec, normal_style))

    elements.append(Spacer(1, 0.4 * cm))
    elements.append(
        Paragraph(
            "This report was generated automatically by the Motor Insight "
            "system and is intended to support, not replace, expert "
            "technical judgement.",
            normal_style,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer
