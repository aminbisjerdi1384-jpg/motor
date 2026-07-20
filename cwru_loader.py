"""
cwru_loader.py
--------------
Native support for the CWRU Bearing Data Center dataset:
- Load the original .mat files (scipy.io.loadmat) and robustly extract the
  DE_time / FE_time / BA_time channels and the RPM metadata, regardless of
  the exact variable-name prefix used in a given file (this varies slightly
  across the dataset, e.g. X097_DE_time vs X105_DE_time).
- Load CSV files that were already converted from .mat (one signal column
  per file, e.g. via the bundled download_cwru_data.py script).
- Infer the fault-class label from the filename, following the official
  CWRU naming convention (IR = inner race, OR = outer race, B = ball,
  Normal = healthy), since the raw .mat files carry no label field.
"""
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio

FAULT_CLASSES = ["Healthy", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]


def _find_key(keys, suffix: str) -> Optional[str]:
    for k in keys:
        if k.endswith(suffix):
            return k
    return None


def load_mat_file(path: str) -> dict:
    """
    Load a CWRU .mat file and extract whichever channels are present.

    Returns
    -------
    dict with keys among {'DE', 'FE', 'BA', 'rpm'}; 'DE'/'FE'/'BA' are
    1-D numpy arrays, 'rpm' is a float or None if not stored in the file.
    """
    mat = sio.loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    result = {}
    de_key = _find_key(keys, "_DE_time") or _find_key(keys, "DE_time")
    fe_key = _find_key(keys, "_FE_time") or _find_key(keys, "FE_time")
    ba_key = _find_key(keys, "_BA_time") or _find_key(keys, "BA_time")
    rpm_key = _find_key(keys, "RPM")

    if de_key:
        result["DE"] = np.asarray(mat[de_key]).squeeze().astype(float)
    if fe_key:
        result["FE"] = np.asarray(mat[fe_key]).squeeze().astype(float)
    if ba_key:
        result["BA"] = np.asarray(mat[ba_key]).squeeze().astype(float)
    if rpm_key:
        try:
            result["rpm"] = float(np.asarray(mat[rpm_key]).squeeze())
        except (TypeError, ValueError):
            result["rpm"] = None

    if not result:
        raise ValueError(
            f"No recognizable DE/FE/BA channel found in {path}. "
            "Expected a CWRU-format .mat file."
        )

    return result


def infer_label_from_filename(filename: str) -> str:
    """
    Heuristically infer the fault class from a CWRU-style filename, e.g.:
        'IR007_0.mat'  -> 'Inner Race Fault'
        'OR007@6_1.mat'-> 'Outer Race Fault'
        'B021_2.mat'   -> 'Ball Fault'
        'Normal_0.mat' -> 'Healthy'

    Returns 'Unknown' if no pattern matches - the caller (UI) should then
    ask the user to confirm/select the correct label.
    """
    name = Path(filename).stem

    if re.search(r"(?i)normal", name) or re.search(r"(?i)healthy", name):
        return "Healthy"
    if re.search(r"(?i)^IR", name) or re.search(r"(?i)inner", name):
        return "Inner Race Fault"
    if re.search(r"(?i)^OR", name) or re.search(r"(?i)outer", name):
        return "Outer Race Fault"
    if re.search(r"(?i)^B\d", name) or re.search(r"(?i)ball", name):
        return "Ball Fault"
    return "Unknown"


def load_training_signal(path: str, channel: str = "DE") -> Tuple[np.ndarray, Optional[float], str]:
    """
    Load a single training file (.mat or .csv) and return
    (signal, rpm, inferred_label).

    For .csv files, the first numeric column is used as the signal and RPM
    is unknown (None) unless a 'rpm' column is present.
    """
    path_obj = Path(path)
    label = infer_label_from_filename(path_obj.name)

    if path_obj.suffix.lower() == ".mat":
        channels = load_mat_file(path)
        signal = channels.get(channel) or next(iter(channels.values()))
        rpm = channels.get("rpm")
        return signal, rpm, label

    if path_obj.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        rpm = None
        if "rpm" in [c.lower() for c in df.columns]:
            rpm_col = [c for c in df.columns if c.lower() == "rpm"][0]
            rpm = float(df[rpm_col].iloc[0])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c.lower() != "rpm"]
        if not numeric_cols:
            raise ValueError(f"No numeric signal column found in {path}")
        signal = df[numeric_cols[0]].to_numpy(dtype=float)
        return signal, rpm, label

    raise ValueError(f"Unsupported file type for CWRU loader: {path_obj.suffix}")
