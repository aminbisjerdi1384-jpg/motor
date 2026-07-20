"""
ml_classifier.py
-----------------
Mode B: Machine-learning fault classification.

Unlike Isolation Forest (unsupervised, "what looks different from the rest
of *this* file"), this module trains a *supervised* classifier on labeled
examples from several files (ideally from the CWRU dataset, which has
known fault types), so it learns what each class actually looks like and
can correctly label a file that is uniformly healthy or uniformly faulty.

Pipeline:
1. extract_training_features(): for each labeled training file, clean ->
   (optionally filter) -> window -> extract the same feature set used
   everywhere else in this app (features.FEATURE_COLUMNS), tagging every
   window with the file's label.
2. train_classifier(): StandardScaler + RandomForestClassifier or SVC,
   with a held-out test split for honest evaluation (confusion matrix,
   classification report, ROC curves).
3. predict(): apply the trained pipeline to a new file's window features.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline

from preprocessing import apply_filter
from features import extract_features_dataframe, FEATURE_COLUMNS


@dataclass
class TrainingFile:
    signal: np.ndarray
    fs: float
    label: str


def extract_training_features(
    training_files: List[TrainingFile],
    window_size: int,
    overlap: float = 0.0,
    filter_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a labeled feature table from several training signals.

    filter_cfg, if given, is a dict like
        {"type": "bandpass", "cutoff": (500, 4000), "order": 4}
    applied to every training signal before windowing/feature extraction -
    it MUST match whatever filter (if any) will be applied to new files at
    prediction time, otherwise the classifier sees a systematic mismatch.

    Returns
    -------
    pd.DataFrame with the usual FEATURE_COLUMNS plus a 'label' column.
    """
    all_rows = []
    for tf in training_files:
        signal = tf.signal
        if filter_cfg:
            signal = apply_filter(signal, filter_cfg["type"], filter_cfg["cutoff"], tf.fs, filter_cfg.get("order", 4))

        feat_df = extract_features_dataframe(signal, tf.fs, window_size, overlap)
        feat_df["label"] = tf.label
        all_rows.append(feat_df)

    return pd.concat(all_rows, ignore_index=True)


def train_classifier(
    feature_df: pd.DataFrame,
    model_type: str = "random_forest",
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """
    Train a supervised classifier on labeled window features.

    Returns a dict with:
        pipeline      : fitted sklearn Pipeline (StandardScaler + classifier)
        classes       : sorted list of class labels
        X_test, y_test, y_pred, y_proba
        confusion_matrix : np.ndarray
        classification_report : dict
        roc_curves : dict[class_label] -> (fpr, tpr, auc) for one-vs-rest ROC
    """
    X = feature_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = feature_df["label"].to_numpy()

    classes = sorted(pd.unique(y).tolist())
    if len(classes) < 2:
        raise ValueError("Need at least 2 distinct classes to train a classifier.")

    stratify = y if min(pd.Series(y).value_counts()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight="balanced")
    elif model_type == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=random_state, class_weight="balanced")
    else:
        raise ValueError("model_type must be 'random_forest' or 'svm'")

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

    # One-vs-rest ROC curves (meaningful for binary or multiclass)
    y_test_bin = label_binarize(y_test, classes=classes)
    roc_curves = {}
    for i, cls in enumerate(classes):
        if y_test_bin.shape[1] == 1:  # binary edge case from label_binarize
            fpr, tpr, _ = roc_curve(y_test_bin[:, 0], y_proba[:, 1] if i == 1 else y_proba[:, 0])
        else:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_curves[cls] = (fpr, tpr, float(auc(fpr, tpr)))

    return {
        "pipeline": pipeline,
        "classes": classes,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_curves": roc_curves,
    }


def predict_file(pipeline: Pipeline, feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a trained pipeline to a new (unlabeled) file's window features.
    Returns the feature_df with 'predicted_label' and per-class probability
    columns appended.
    """
    X = feature_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    classes = pipeline.classes_

    out = feature_df.copy()
    out["predicted_label"] = pred
    for i, cls in enumerate(classes):
        out[f"proba_{cls}"] = proba[:, i]
    out["confidence"] = proba.max(axis=1)
    return out


def summarize_prediction(pred_df: pd.DataFrame) -> dict:
    """
    Aggregate per-window predictions into a single file-level verdict:
    the majority predicted class, its vote share, and mean confidence.
    """
    counts = pred_df["predicted_label"].value_counts(normalize=True)
    top_label = counts.index[0]
    return {
        "predicted_label": top_label,
        "vote_share": float(counts.iloc[0]),
        "mean_confidence": float(pred_df.loc[pred_df["predicted_label"] == top_label, "confidence"].mean()),
        "class_vote_shares": counts.to_dict(),
    }
