"""
diagnosis.py
------------
Mode A: Engineering rule-based diagnosis.

WHY this replaces the old "fit Isolation Forest on the uploaded file and
see which windows look like outliers *within that same file*" approach:
that method can only ever detect windows that look unusual *relative to
the rest of the same recording*. If you upload a file that is faulty
*from start to end* (e.g. a CWRU bearing-fault recording), every window
looks equally "faulty" to the model, so nothing stands out and the file
gets reported as Normal - which is exactly the bug you ran into.

Rule-based diagnosis instead compares each feature to fixed, physically
meaningful thresholds (independent of what else is in the file), so a
uniformly faulty file is correctly flagged.

These default thresholds are illustrative engineering heuristics drawn
from common bearing-diagnostics practice (e.g. kurtosis ~3 for a Gaussian/
healthy signal, rising sharply with impulsive faults; crest factor ~1.4
for a clean sinusoid). They are NOT a certified standard (unlike, e.g.,
ISO 10816 velocity limits, which require calibrated velocity units this
app does not assume). For real deployments, thresholds should be
calibrated against that specific machine's own healthy baseline.
"""
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class DiagnosisThresholds:
    kurtosis_warning: float = 4.0
    kurtosis_fault: float = 6.0
    crest_factor_warning: float = 4.0
    crest_factor_fault: float = 6.0
    rms_warning_ratio: float = 2.0   # x times the window-median RMS
    rms_fault_ratio: float = 4.0
    spectral_energy_warning_ratio: float = 2.0
    spectral_energy_fault_ratio: float = 4.0


@dataclass
class IndicatorResult:
    name: str
    value: float
    level: str          # "ok" | "warning" | "fault"
    explanation: str


def _classify_absolute(value: float, warn_th: float, fault_th: float, name: str, unit_hint: str = "") -> IndicatorResult:
    if value >= fault_th:
        level = "fault"
        explanation = f"{name} = {value:.2f}{unit_hint} >= fault threshold {fault_th:.2f}"
    elif value >= warn_th:
        level = "warning"
        explanation = f"{name} = {value:.2f}{unit_hint} >= warning threshold {warn_th:.2f}"
    else:
        level = "ok"
        explanation = f"{name} = {value:.2f}{unit_hint} is within the normal range (< {warn_th:.2f})"
    return IndicatorResult(name=name, value=value, level=level, explanation=explanation)


def _classify_relative(value: float, baseline: float, warn_ratio: float, fault_ratio: float, name: str) -> IndicatorResult:
    if baseline <= 0:
        baseline = 1e-9
    ratio = value / baseline
    if ratio >= fault_ratio:
        level = "fault"
        explanation = f"{name} is {ratio:.1f}x the file's own median - far above normal ({fault_ratio:.0f}x threshold)"
    elif ratio >= warn_ratio:
        level = "warning"
        explanation = f"{name} is {ratio:.1f}x the file's own median - elevated ({warn_ratio:.0f}x threshold)"
    else:
        level = "ok"
        explanation = f"{name} is close to the file's own median ({ratio:.1f}x)"
    return IndicatorResult(name=name, value=value, level=level, explanation=explanation)


def rule_based_diagnosis(feature_df: pd.DataFrame, thresholds: DiagnosisThresholds = None) -> dict:
    """
    Apply threshold rules to the per-window feature table and produce an
    overall verdict.

    Indicators are split into two roles:
    - PRIMARY (kurtosis, crest factor): absolute, dimensionless, and
      individually diagnostic for impulsive bearing faults per standard
      practice - either one alone reaching the "fault" threshold is
      sufficient to call the file Faulty, even if it is uniformly bad
      across the whole recording (this is exactly the case the old
      Isolation-Forest-only approach missed).
    - SECONDARY (RMS ratio, spectral energy ratio, both relative to the
      file's own median): useful supporting/contextual evidence that
      something changed *within* this particular recording, but - because
      they are relative to the file's own median - they cannot by
      themselves catch a file that is uniformly faulty from start to end.
      They can only raise the verdict to Warning on their own.

    Returns
    -------
    dict with keys:
        level: 'Healthy' | 'Warning' | 'Faulty'
        indicators: list[IndicatorResult] (one aggregated result per metric)
        explanation: list[str] (human-readable reasons)
    """
    thresholds = thresholds or DiagnosisThresholds()

    median_rms = float(feature_df["rms"].median())
    median_energy = float(feature_df["spectral_energy"].median())

    mean_kurtosis = float(feature_df["kurtosis"].mean())
    mean_crest = float(feature_df["crest_factor"].mean())
    max_rms = float(feature_df["rms"].max())
    max_energy = float(feature_df["spectral_energy"].max())

    kurt_ind = _classify_absolute(mean_kurtosis, thresholds.kurtosis_warning, thresholds.kurtosis_fault, "Kurtosis")
    crest_ind = _classify_absolute(mean_crest, thresholds.crest_factor_warning, thresholds.crest_factor_fault, "Crest factor")
    rms_ind = _classify_relative(max_rms, median_rms, thresholds.rms_warning_ratio, thresholds.rms_fault_ratio, "Peak-window RMS")
    energy_ind = _classify_relative(max_energy, median_energy, thresholds.spectral_energy_warning_ratio, thresholds.spectral_energy_fault_ratio, "Peak-window spectral energy")

    indicators: List[IndicatorResult] = [kurt_ind, crest_ind, rms_ind, energy_ind]
    primary = [kurt_ind, crest_ind]
    secondary = [rms_ind, energy_ind]

    if any(ind.level == "fault" for ind in primary):
        level = "Faulty"
    elif any(ind.level == "warning" for ind in primary) or any(ind.level == "fault" for ind in secondary):
        level = "Warning"
    elif any(ind.level == "warning" for ind in secondary):
        level = "Warning"
    else:
        level = "Healthy"

    return {
        "level": level,
        "indicators": indicators,
        "explanation": [ind.explanation for ind in indicators],
    }
