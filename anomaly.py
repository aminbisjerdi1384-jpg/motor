"""
anomaly.py
----------
Unsupervised novelty screening over the per-window features extracted by
features.py, using scikit-learn's Isolation Forest.

IMPORTANT - role of this module in the current architecture:
Isolation Forest is fit *only on the windows of the single uploaded file*,
so it can only ever flag windows that look unusual *relative to the rest
of that same file*. If an entire file is uniformly faulty (e.g. a CWRU
bearing-fault recording from start to finish), every window looks equally
"faulty" to the model and nothing stands out - so this method alone WILL
miss a uniformly faulty file. This was the original bug in this project.

This module is therefore kept as a *supplementary, within-file* novelty
detector (e.g. "did something change partway through this recording?",
useful for spotting a developing fault or a one-off transient event), and
is no longer used to produce the application's primary Healthy/Warning/
Faulty verdict. That verdict now comes from diagnosis.py (Mode A,
rule-based thresholds) and/or ml_classifier.py (Mode B, a classifier
trained on multiple labeled files) - see app.py.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS


def detect_anomalies(
    features_df: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit an Isolation Forest on the extracted window features and label
    each window as 'Normal' or 'Abnormal'.

    Returns a copy of features_df with two extra columns:
        anomaly_label : 'Normal' | 'Abnormal' | 'Unknown'
        anomaly_score : Isolation Forest decision function value
                        (lower / more negative => more abnormal)
    """
    df = features_df.copy()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    if len(df) < 2:
        # Not enough windows to fit a meaningful model.
        df["anomaly_label"] = "Unknown"
        df["anomaly_score"] = 0.0
        return df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination must lie in (0, 0.5]
    contamination = float(np.clip(contamination, 0.01, 0.5))

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_scaled)

    raw_pred = model.predict(X_scaled)          # 1 = normal, -1 = anomaly
    scores = model.decision_function(X_scaled)  # lower => more abnormal

    df["anomaly_label"] = np.where(raw_pred == -1, "Abnormal", "Normal")
    df["anomaly_score"] = scores
    return df


def summarize_diagnosis(df: pd.DataFrame, abnormal_ratio_threshold: float = 0.15) -> dict:
    """
    Turn per-window labels into a single overall diagnosis for the file.

    If the fraction of windows labeled 'Abnormal' is at or above
    `abnormal_ratio_threshold`, the overall diagnosis is 'Abnormal'.
    """
    total = len(df)
    abnormal_count = int((df["anomaly_label"] == "Abnormal").sum())
    ratio = abnormal_count / total if total > 0 else 0.0
    diagnosis = "Abnormal" if ratio >= abnormal_ratio_threshold else "Normal"

    return {
        "total_windows": total,
        "abnormal_windows": abnormal_count,
        "abnormal_ratio": ratio,
        "diagnosis": diagnosis,
    }
