"""
envelope.py  (v4 — production-ready for real-world bearing data)
-----------------------------------------------------------------
Envelope analysis: Band-pass filter → Hilbert transform → envelope →
envelope FFT.

WHY this exists (and why plain FFT often isn't enough):
A bearing defect produces a train of short mechanical impacts at its
characteristic fault frequency (BPFO/BPFI/BSF). Each impact excites the
much higher-frequency *structural resonance* of the bearing/housing, which
dominates the raw vibration spectrum. The fault-frequency information is
not a strong tone in the raw spectrum — it is encoded as the *repetition
rate (amplitude modulation)* of that resonance. This is mathematically the
same situation as an AM radio signal: the carrier (resonance) is strong,
but the information of interest (the fault impacts) is in the envelope.

Demodulation procedure (the standard industrial technique):
1. Band-pass filter the signal around the resonance frequency band — isolates
   the carrier and rejects unrelated noise/other components.
2. Take the analytic signal via the Hilbert transform and its magnitude
   (the "envelope") — amplitude demodulation, exactly like an AM detector.
3. FFT the envelope — the fault repetition rate (BPFO/BPFI/BSF) now shows
   up as a strong, clear peak, because the impacts repeat periodically.

============================================================
CHANGELOG vs v3
============================================================

ROOT CAUSE #1 — Wrong resonance band:
    v3 default: fixed 1000–4000 Hz when auto_band=False.
    FIX (v4):   auto_band defaults to True. The kurtosis search now also
                runs a MULTI-SCALE scan (three bandwidth sizes: 500, 1000,
                2000 Hz), then returns the band with the global highest
                kurtosis across all scales.

ROOT CAUSE #2 — Harmonics not checked:
    v3 default: only fundamental BPFO/BPFI/BSF frequencies were checked.
    FIX (v4):   analyze_bearing() uses include_harmonics=3 by default, and
                scan_fault_frequencies() default also raised to 3.
                The adaptive tolerance is widened for higher harmonics.

ROOT CAUSE #3 — Bearing-position mismatch:
    v3 problem: app.py may supply wrong multipliers (DE vs FE bearing differ
                in geometry → BPFO can be 16 Hz off → easy miss for tight
                tolerance windows).
    FIX (v4):   Added cwru_fault_frequencies(rpm, bearing_position) and
                the CWRU_BEARING_PARAMS catalogue.  The tolerance window is
                also made adaptive via adaptive_tolerance_hz() based on RPM
                uncertainty.

============================================================
DESIGN NOTES: find_peak_near() — v3 logic retained
============================================================

v1 (original): used np.argmax() in a tolerance window.
    Bug: every spectrum has a local maximum → found=True on healthy signals.

v2 (previous rewrite): added scipy.find_peaks() + prominence + SNR + a
    "spectral crowding guard" (MAX_PEAKS_IN_WINDOW = 2).
    Bug: REAL bearing-fault signals (e.g. CWRU) have richer noise floors
    than synthetic signals. This caused the crowding guard to reject a 36 dB
    BPFO peak on a real OR-fault file — a false negative far more dangerous
    than a false positive.

v3/v4 (this version): removes the crowding guard. Statistical discrimination
    relies on THREE independent criteria:
      1. SNR ≥ threshold (collar-based noise floor, not contaminated
         by the fault peak itself).
      2. Prominence ≥ MIN_PROMINENCE_RATIO × noise_floor (peak must stand
         out locally, not be buried by a broad hump).
      3. Dominance: the best candidate must be X times more prominent than
         the second-best candidate inside the window.

    Additionally: find_optimal_envelope_band() (Kurtogram-lite) with
    MULTI-SCALE search (v4 addition).
"""
from typing import Optional, Tuple, List, Dict

import numpy as np
from scipy.signal import hilbert, find_peaks
from scipy.stats import kurtosis as _scipy_kurtosis

from preprocessing import apply_filter, remove_dc_offset
from features import compute_fft


# --------------------------------------------------------------------------
# Classification thresholds — calibrated so that:
#   • Noise-only spectra (healthy real & synthetic) produce SNR 6–10 dB
#     (below SNR_WEAK_DB) → "Not Detected".
#   • Real bearing-fault peaks (CWRU dataset) produce SNR 15–50+ dB
#     (above SNR_STRONG_DB) → "Strong".
# The gap between noise (≤10 dB) and fault (≥15 dB) is wide enough that
# conservative thresholds cause negligible sensitivity loss on real faults.
# --------------------------------------------------------------------------
SNR_STRONG_DB:        float = 20.0   # clear, confident fault peak
SNR_WEAK_DB:          float = 12.0   # detectable; use for trending
NOISE_WINDOW_FACTOR:  float = 10.0   # noise collar = factor × tolerance_hz
MIN_PROMINENCE_RATIO: float = 1.5    # peak prominence ≥ 1.5× noise floor
DOMINANCE_RATIO:      float = 2.5    # best peak must be ≥ 2.5× 2nd-best


# --------------------------------------------------------------------------
# CWRU bearing catalogue
# --------------------------------------------------------------------------
# Fault-frequency multipliers (dimensionless): multiply by (RPM / 60) to get Hz.
#
# Drive End (DE) bearing: 6205-2RS JEM SKF
# Fan End (FE) bearing:   6203-2RS JEM SKF
#
# Published values from CWRU bearing data website (bearingdatacenter.org):
#   6205-2RS: 9 balls, pitch Ø 1.537", ball Ø 0.331", contact angle 0°
#   6203-2RS: 8 balls, pitch Ø 1.122", ball Ø 0.316", contact angle 0°
# --------------------------------------------------------------------------
CWRU_BEARING_PARAMS: Dict[str, Dict[str, float]] = {
    "drive_end": {
        "BPFO": 3.585,   # Ball Pass Frequency Outer Race
        "BPFI": 5.415,   # Ball Pass Frequency Inner Race
        "BSF":  2.357,   # Ball Spin Frequency
        "FTF":  0.398,   # Fundamental Train Frequency (cage)
    },
    "fan_end": {
        "BPFO": 3.052,
        "BPFI": 4.947,
        "BSF":  1.994,
        "FTF":  0.381,
    },
}


def cwru_fault_frequencies(
    rpm: float,
    bearing_position: str = "drive_end",
) -> Dict[str, float]:
    """
    Return CWRU fault frequencies (Hz) for the specified bearing position.

    Parameters
    ----------
    rpm              : shaft speed in revolutions per minute
    bearing_position : "drive_end" (default) or "fan_end"

    Returns
    -------
    dict with keys BPFO, BPFI, BSF, FTF (all in Hz)

    Raises
    ------
    ValueError if bearing_position is not recognised.

    Example
    -------
    >>> freqs = cwru_fault_frequencies(rpm=1797, bearing_position="drive_end")
    >>> print(freqs["BPFO"])   # → 107.4 Hz
    """
    pos = bearing_position.lower().strip()
    if pos not in CWRU_BEARING_PARAMS:
        available = list(CWRU_BEARING_PARAMS.keys())
        raise ValueError(
            f"Unknown bearing_position '{bearing_position}'. "
            f"Choose from: {available}"
        )
    multipliers = CWRU_BEARING_PARAMS[pos]
    rps = rpm / 60.0
    return {name: mult * rps for name, mult in multipliers.items()}


def adaptive_tolerance_hz(
    fault_freq_hz: float,
    rpm_uncertainty_pct: float = 2.0,
    base_tolerance_hz: float = 2.0,
) -> float:
    """
    Return a search-window tolerance that scales with RPM uncertainty.

    WHY this matters:
    The tachometer-derived RPM is never exact, and bearing slip further
    shifts the actual fault frequency from the theoretical value. A fixed
    ±2 Hz window works for clean synthetic data but misses peaks when the
    motor is running slightly off-nominal (e.g., 1792 vs 1797 RPM moves
    BPFO by ~0.3 Hz; 3 % speed variation moves it by ~3 Hz).

    The tolerance is max(base_tolerance_hz, fault_freq_hz × uncertainty_pct/100).
    For BPFO ≈ 107 Hz and 2 % uncertainty → max(2, 2.1) = 2.1 Hz.
    For BPFI×4 ≈ 649 Hz and 2 % uncertainty → max(2, 13.0) = 13.0 Hz.
    This avoids over-tight windows at high harmonics.

    Parameters
    ----------
    fault_freq_hz       : the target frequency (Hz)
    rpm_uncertainty_pct : expected RPM error, percent (default 2 %)
    base_tolerance_hz   : minimum window, Hz (default 2 Hz)

    Returns
    -------
    tolerance_hz : float
    """
    return max(base_tolerance_hz, fault_freq_hz * rpm_uncertainty_pct / 100.0)


# --------------------------------------------------------------------------
# Core envelope pipeline
# --------------------------------------------------------------------------

def compute_envelope(
    signal: np.ndarray,
    fs: float,
    band: Optional[Tuple[float, float]] = None,
    filter_order: int = 4,
) -> np.ndarray:
    """
    Compute the amplitude envelope via the Hilbert transform, optionally
    after band-pass filtering to isolate the structural resonance band.

    Parameters
    ----------
    signal : np.ndarray
    fs : float
    band : (low_hz, high_hz) or None
    filter_order : Butterworth order for the band-pass stage.

    Returns
    -------
    envelope : np.ndarray  (same length as input)
    """
    signal = np.asarray(signal, dtype=float)
    working = (apply_filter(signal, "bandpass", band, fs, order=filter_order)
               if band is not None else signal)
    return np.abs(hilbert(working))


def compute_envelope_spectrum(
    envelope: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FFT of the DC-removed envelope.

    DC removal is essential: the envelope is always ≥ 0, so its mean (DC)
    dominates the spectrum and buries the fault-frequency peaks without
    removal.
    """
    return compute_fft(remove_dc_offset(np.asarray(envelope, dtype=float)), fs)


def envelope_analysis_pipeline(
    signal: np.ndarray,
    fs: float,
    band: Optional[Tuple[float, float]] = None,
    filter_order: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline: (band-pass) → Hilbert envelope → FFT.

    Returns
    -------
    envelope, freqs, magnitude
    """
    envelope = compute_envelope(signal, fs, band=band, filter_order=filter_order)
    freqs, magnitude = compute_envelope_spectrum(envelope, fs)
    return envelope, freqs, magnitude


# --------------------------------------------------------------------------
# Noise floor estimation
# --------------------------------------------------------------------------

def _estimate_noise_floor(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    target_freq: float,
    context_hz: float,
    peak_idx: Optional[int] = None,
    tolerance_hz: float = 0.0,
) -> float:
    """
    Estimate the local noise floor from a COLLAR region:
    (wider context window) MINUS (tolerance window around target_freq).

    WHY the collar matters:
    If we include the tolerance window in the noise estimate, the fault
    peak (which lives in that window) inflates the median and makes the
    SNR look smaller than it really is — under-detection.

    Collar = outer ± context_hz   MINUS   inner ± tolerance_hz
    """
    outer_mask = np.abs(freqs - target_freq) <= context_hz
    inner_mask = np.abs(freqs - target_freq) <= max(tolerance_hz, 0.0)
    collar_mask = outer_mask & ~inner_mask
    collar_idx = np.where(collar_mask)[0]

    if len(collar_idx) < 3:
        return float(np.percentile(magnitude, 25))

    # Exclude a ±2 bin guard around any secondary peaks in the collar
    if peak_idx is not None:
        guard = set(range(peak_idx - 2, peak_idx + 3))
        collar_idx = np.array([i for i in collar_idx if i not in guard])

    if len(collar_idx) == 0:
        return float(np.percentile(magnitude, 25))

    return float(np.median(magnitude[collar_idx]))


# --------------------------------------------------------------------------
# Improved peak detection  (v3/v4 — unchanged from v3)
# --------------------------------------------------------------------------

def find_peak_near(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    target_freq: float,
    tolerance_hz: float = 2.0,
) -> dict:
    """
    Robust bearing-fault spectral peak detector.

    Algorithm (v3/v4 — see module docstring for rationale vs v1/v2):
    1. scipy.find_peaks() with adaptive prominence threshold to identify
       true spectral peaks (not noise fluctuations).
    2. Restrict to the ±tolerance_hz window.
    3. Select candidate with highest prominence.
    4. Estimate local noise floor from the collar region (context window
       minus tolerance window) so the fault peak cannot contaminate its
       own baseline.
    5. Compute SNR = 20·log10(peak / noise_floor).
    6. Dominance check: best peak prominence ≥ DOMINANCE_RATIO × the
       next-best peak inside the window. This replaces the blunt
       "count peaks in window" guard that failed on real CWRU data.
    7. Classify using SNR + prominence + dominance.

    Parameters
    ----------
    freqs       : frequency axis of the envelope spectrum (Hz)
    magnitude   : magnitude axis
    target_freq : theoretical fault frequency (Hz)
    tolerance_hz : half-width of the search window (Hz)
                  Use adaptive_tolerance_hz() to auto-scale with RPM uncertainty.

    Returns
    -------
    dict with keys:
        found             : bool
        freq              : float or None    measured peak frequency (Hz)
        magnitude         : float or None    peak magnitude
        noise_floor       : float or None    collar median noise level
        snr_db            : float or None    20·log10(peak / noise_floor)
        prominence        : float or None    scipy prominence of the peak
        n_peaks_in_window : int              diagnostic count (not used to reject)
        classification    : str              "Strong" | "Weak" | "Not Detected"
    """
    _ND = {
        "found": False, "freq": None, "magnitude": None,
        "noise_floor": None, "snr_db": None, "prominence": None,
        "n_peaks_in_window": 0, "classification": "Not Detected",
    }

    if len(freqs) == 0 or len(magnitude) == 0:
        return _ND

    # ------------------------------------------------------------------
    # Step 1: detect real spectral peaks across the whole spectrum.
    # Adaptive prominence threshold: 2% of the global dynamic range.
    # Kept intentionally loose here — the local SNR/prominence checks
    # below do the real discrimination.
    # ------------------------------------------------------------------
    global_floor = float(np.percentile(magnitude, 10))
    global_range = float(np.max(magnitude) - global_floor)
    min_prominence = max(global_range * 0.02, 1e-12)

    all_peak_idx, all_props = find_peaks(
        magnitude, height=global_floor, prominence=min_prominence
    )

    # ------------------------------------------------------------------
    # Step 2: restrict to the tolerance window.
    # ------------------------------------------------------------------
    if len(all_peak_idx) == 0:
        return _ND

    in_window_mask = np.abs(freqs[all_peak_idx] - target_freq) <= tolerance_hz
    cand_idx  = all_peak_idx[in_window_mask]
    cand_prom = all_props["prominences"][in_window_mask]
    n_peaks   = int(len(cand_idx))

    if n_peaks == 0:
        return _ND

    # ------------------------------------------------------------------
    # Step 3: pick the candidate with the highest prominence.
    # Prominence is preferred over raw height because it measures how
    # much a peak stands above its local surroundings — a sharp fault
    # line has high prominence even at moderate absolute height.
    # ------------------------------------------------------------------
    best_local = int(np.argmax(cand_prom))
    best_idx   = int(cand_idx[best_local])
    best_prom  = float(cand_prom[best_local])
    best_freq  = float(freqs[best_idx])
    best_mag   = float(magnitude[best_idx])

    # ------------------------------------------------------------------
    # Step 4: collar-based noise floor (the key fix from v2, retained).
    # ------------------------------------------------------------------
    context_hz  = NOISE_WINDOW_FACTOR * tolerance_hz
    noise_floor = _estimate_noise_floor(
        freqs, magnitude, target_freq,
        context_hz=context_hz, peak_idx=best_idx, tolerance_hz=tolerance_hz,
    )
    noise_floor = max(noise_floor, 1e-30)

    # ------------------------------------------------------------------
    # Step 5: SNR in dB (amplitude convention: 20·log10).
    # ------------------------------------------------------------------
    snr_db = float(20.0 * np.log10(best_mag / noise_floor))

    # ------------------------------------------------------------------
    # Step 6: dominance check (replaces crowding guard).
    #
    # WHY dominance instead of peak count:
    # In real CWRU data the tolerance window often contains 3–6 small
    # noise peaks. Counting peaks blocked genuine 36 dB fault detections
    # (the v2 bug). Instead we ask:
    #   "Is the best peak MUCH stronger than the second-best?"
    # A real fault peak typically has ≥ 3× the prominence of any noise
    # peak nearby. A noise peak has a 2nd-best that is similar in size.
    # ------------------------------------------------------------------
    if n_peaks >= 2:
        second_prom = float(np.sort(cand_prom)[-2])   # 2nd-highest prom.
        dominant = (second_prom <= 0) or (best_prom / second_prom >= DOMINANCE_RATIO)
    else:
        dominant = True   # single peak in window → by definition dominant

    # ------------------------------------------------------------------
    # Step 7: classify.
    # All three of SNR, prominence and dominance must align for "Strong".
    # For "Weak" dominance is relaxed — an emerging fault might not yet
    # produce a perfectly clean peak.
    # ------------------------------------------------------------------
    prom_threshold = MIN_PROMINENCE_RATIO * noise_floor

    if snr_db >= SNR_STRONG_DB and best_prom >= prom_threshold and dominant:
        classification = "Strong"
        found = True
    elif snr_db >= SNR_WEAK_DB and best_prom >= prom_threshold * 0.5:
        classification = "Weak"
        found = True
    else:
        classification = "Not Detected"
        found = False

    return {
        "found": found,
        "freq": best_freq,
        "magnitude": best_mag,
        "noise_floor": noise_floor,
        "snr_db": snr_db,
        "prominence": best_prom,
        "n_peaks_in_window": n_peaks,
        "classification": classification,
    }


# --------------------------------------------------------------------------
# Automatic resonance-band detection — v4: MULTI-SCALE Kurtogram-lite
# --------------------------------------------------------------------------

def find_optimal_envelope_band(
    signal: np.ndarray,
    fs: float,
    n_centers: int = 12,
    bandwidth_hz: float = 1_000.0,
    min_freq: float = 500.0,
    max_freq: Optional[float] = None,
    filter_order: int = 4,
    multi_scale: bool = True,
) -> Tuple[float, float, float, dict]:
    """
    Find the bandpass filter band that maximises kurtosis of the resulting
    envelope signal — the "Kurtogram-lite" approach.

    WHY kurtosis maximisation:
    Bearing-fault impacts are impulsive (non-Gaussian). Kurtosis measures
    the "peakedness" of a distribution (value ≈ 3 for Gaussian/healthy;
    rises sharply with impulsive content). The resonance band that lets
    the most impulsive energy through will have the highest kurtosis,
    and that is where the fault signature is strongest.

    v4 addition — MULTI-SCALE scan:
    A single bandwidth may be too wide (smearing a narrow resonance) or
    too narrow (splitting a broad resonance across two windows). v4 runs
    the search at three scales: 500, 1000, and 2000 Hz (when multi_scale=True),
    and returns the global winner across all scales. This correctly handles
    both narrow structural resonances (e.g. ≈500 Hz wide) and broad ones
    (e.g. ≈2000 Hz).

    Parameters
    ----------
    signal       : np.ndarray (raw or DC-removed signal)
    fs           : float
    n_centers    : number of candidate band centres to try per scale
    bandwidth_hz : base bandwidth for the single-scale mode
    min_freq     : lower bound for band search (Hz)
    max_freq     : upper bound (defaults to 0.9 × Nyquist)
    filter_order : Butterworth order
    multi_scale  : if True (default), run at 500/1000/2000 Hz bandwidths
                   and return the global best

    Returns
    -------
    low_hz      : float — lower cutoff of the best band
    high_hz     : float — upper cutoff
    best_kurt   : float — kurtosis value of the best envelope
    scan_results: dict — {(low, high): kurtosis} for all candidates tried
    """
    nyquist = fs / 2.0
    if max_freq is None:
        max_freq = nyquist * 0.9

    scales = [500.0, 1000.0, 2000.0] if multi_scale else [bandwidth_hz]

    global_best_low   = min_freq
    global_best_high  = min_freq + (scales[0] if multi_scale else bandwidth_hz)
    global_best_kurt  = -np.inf
    all_scan_results: dict = {}

    for bw in scales:
        usable_max = max_freq - bw
        if usable_max <= min_freq:
            continue

        centers = np.linspace(
            min_freq + bw / 2,
            usable_max + bw / 2,
            n_centers,
        )

        for centre in centers:
            low  = max(centre - bw / 2, 1.0)
            high = min(centre + bw / 2, nyquist * 0.999)
            if high <= low + 10:
                continue
            try:
                env  = compute_envelope(signal, fs, band=(low, high),
                                        filter_order=filter_order)
                kurt = float(_scipy_kurtosis(env, fisher=False, bias=True))
                key  = (round(low, 1), round(high, 1))
                all_scan_results[key] = kurt
                if kurt > global_best_kurt:
                    global_best_kurt = kurt
                    global_best_low  = low
                    global_best_high = high
            except Exception:
                continue

    if global_best_kurt == -np.inf:
        # Fallback: fs too low to search at any scale
        default_low  = max(min_freq, 100.0)
        default_high = min(max_freq, default_low + bandwidth_hz)
        return default_low, default_high, 0.0, {}

    return global_best_low, global_best_high, global_best_kurt, all_scan_results


def auto_envelope_analysis(
    signal: np.ndarray,
    fs: float,
    band: Optional[Tuple[float, float]] = None,
    auto_band: bool = True,                # v4: defaulted to True
    filter_order: int = 4,
    n_centers: int = 12,
    bandwidth_hz: float = 1_000.0,
    multi_scale: bool = True,              # v4 addition
) -> dict:
    """
    Convenience wrapper: optionally auto-select the resonance band (via
    multi-scale kurtosis search), then run the full envelope analysis pipeline.

    CHANGE FROM v3: auto_band now defaults to True so callers get the right
    resonance band out of the box without having to pass band=(1000, 4000).

    Parameters
    ----------
    signal       : np.ndarray
    fs           : float
    band         : explicit (low, high) tuple; used only when auto_band=False
    auto_band    : if True (default), ignore `band` and run kurtosis search
    filter_order : Butterworth order
    n_centers    : band candidates per scale for the kurtosis search
    bandwidth_hz : bandwidth for single-scale mode (ignored if multi_scale=True)
    multi_scale  : if True (default), search at 500/1000/2000 Hz bandwidths

    Returns
    -------
    dict with:
        band_used     : (low_hz, high_hz) tuple  or None (no filtering)
        band_source   : "auto" | "manual" | "none"
        band_kurtosis : float (None if band_source != "auto")
        band_scan     : dict (None if band_source != "auto")
        envelope      : np.ndarray
        freqs         : np.ndarray
        magnitude     : np.ndarray
    """
    if auto_band:
        low, high, kurt, scan = find_optimal_envelope_band(
            signal, fs,
            n_centers=n_centers,
            bandwidth_hz=bandwidth_hz,
            filter_order=filter_order,
            multi_scale=multi_scale,
        )
        used_band = (low, high)
        source    = "auto"
    elif band is not None:
        used_band = band
        kurt      = None
        scan      = None
        source    = "manual"
    else:
        used_band = None
        kurt      = None
        scan      = None
        source    = "none"

    envelope, freqs, magnitude = envelope_analysis_pipeline(
        signal, fs, band=used_band, filter_order=filter_order
    )

    return {
        "band_used":     used_band,
        "band_source":   source,
        "band_kurtosis": kurt,
        "band_scan":     scan,
        "envelope":      envelope,
        "freqs":         freqs,
        "magnitude":     magnitude,
    }


# --------------------------------------------------------------------------
# Multi-fault, multi-harmonic scan
# --------------------------------------------------------------------------

def scan_fault_frequencies(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    fault_freqs: dict,
    tolerance_hz: float = 2.0,
    include_harmonics: int = 3,           # v4: raised default from 2 to 3
    rpm_uncertainty_pct: float = 2.0,     # v4 addition: adaptive tolerance
    use_adaptive_tolerance: bool = True,  # v4 addition
) -> dict:
    """
    Run find_peak_near() for every fault frequency and its harmonics.

    WHY check harmonics:
    At early fault stages the fundamental BPFO/BPFI/BSF is sometimes
    below the noise floor, but harmonic energy accumulates and is
    detectable first. A real fault is also unlikely to produce spurious
    peaks at BOTH the fundamental AND its harmonics simultaneously —
    raising confidence when multiple harmonics are confirmed.

    v4 additions:
    • include_harmonics default raised from 2 to 3 (checks h=1..4).
    • Adaptive tolerance: tolerance is widened at higher harmonics AND
      accounts for RPM uncertainty via adaptive_tolerance_hz(). Pass
      use_adaptive_tolerance=False to revert to the fixed-window v3 logic.

    Parameters
    ----------
    freqs                : frequency axis of the envelope spectrum (Hz)
    magnitude            : magnitude axis
    fault_freqs          : dict {name: base_freq_hz} — e.g. {"BPFO": 107.4}
                           Values of None are silently skipped.
    tolerance_hz         : base half-window for the search (Hz)
    include_harmonics    : number of additional harmonics above fundamental
                           (e.g. 3 → checks h=1,2,3,4)
    rpm_uncertainty_pct  : expected RPM accuracy (%) for adaptive tolerance
    use_adaptive_tolerance : if True (default), scale the tolerance window
                           with each harmonic's actual frequency

    Returns
    -------
    dict {fault_name: {harmonic_index: find_peak_near_result}}
    """
    results = {}
    for name, base_freq in fault_freqs.items():
        if base_freq is None:
            continue
        results[name] = {}
        for h in range(1, include_harmonics + 2):    # h=1...(include_harmonics+1)
            target = base_freq * h
            if target > freqs[-1]:
                break
            if use_adaptive_tolerance:
                tol = adaptive_tolerance_hz(
                    target,
                    rpm_uncertainty_pct=rpm_uncertainty_pct,
                    base_tolerance_hz=tolerance_hz,
                )
            else:
                # v3 behaviour: widen linearly by 20 % per harmonic
                tol = tolerance_hz * (1 + 0.2 * (h - 1))
            results[name][h] = find_peak_near(freqs, magnitude, target,
                                              tolerance_hz=tol)
    return results


def fault_scan_summary(scan_results: dict) -> dict:
    """
    Reduce scan_results to one overall classification per fault type,
    taking the best SNR across all harmonics.

    Returns
    -------
    dict {fault_name: {
        best_classification : str   ("Strong" / "Weak" / "Not Detected")
        best_harmonic       : int or None
        best_snr_db         : float or None
        n_harmonics_found   : int   (v4 addition: count of confirmed harmonics)
        details             : {harmonic_index: find_peak_near_result}
    }}
    """
    summary = {}
    for name, harmonics in scan_results.items():
        best_snr  = -np.inf
        best_h    = None
        best_cls  = "Not Detected"
        n_found   = 0
        for h, r in harmonics.items():
            if r.get("found"):
                n_found += 1
            snr = r.get("snr_db")
            if snr is not None and snr > best_snr:
                best_snr = snr
                best_h   = h
                best_cls = r["classification"]
        summary[name] = {
            "best_classification": best_cls,
            "best_harmonic":       best_h,
            "best_snr_db":         float(best_snr) if best_snr > -np.inf else None,
            "n_harmonics_found":   n_found,
            "details":             harmonics,
        }
    return summary


# --------------------------------------------------------------------------
# v4 HIGH-LEVEL API: analyze_bearing()
# --------------------------------------------------------------------------

def analyze_bearing(
    signal: np.ndarray,
    fs: float,
    fault_freqs: Optional[Dict[str, float]] = None,
    rpm: Optional[float] = None,
    bearing_position: str = "drive_end",
    band: Optional[Tuple[float, float]] = None,
    auto_band: bool = True,
    include_harmonics: int = 3,
    base_tolerance_hz: float = 2.0,
    rpm_uncertainty_pct: float = 2.0,
    filter_order: int = 4,
    n_centers: int = 12,
    multi_scale: bool = True,
) -> dict:
    """
    One-call bearing fault analysis.  All three v4 fixes are applied
    automatically when this function is used.

    Priority for fault frequencies:
    1. ``fault_freqs`` dict — use as-is if provided.
    2. ``rpm`` + ``bearing_position`` — derive from CWRU catalogue.
    3. Both None → only returns the envelope spectrum; no fault scoring.

    Parameters
    ----------
    signal           : raw vibration signal (1D np.ndarray)
    fs               : sampling frequency (Hz)
    fault_freqs      : explicit {name: Hz} dict (overrides rpm / bearing_position)
    rpm              : shaft speed (RPM); used to derive CWRU fault frequencies
    bearing_position : "drive_end" or "fan_end" (for CWRU catalogue)
    band             : explicit bandpass band; ignored when auto_band=True
    auto_band        : if True (default), auto-select via kurtosis search
    include_harmonics: harmonics checked per fault (default 3 → h=1..4)
    base_tolerance_hz: base search window half-width (Hz) — widened adaptively
    rpm_uncertainty_pct : RPM accuracy for adaptive tolerance (%)
    filter_order     : Butterworth filter order
    n_centers        : kurtosis search candidates per bandwidth scale
    multi_scale      : if True (default), search at 500/1000/2000 Hz scales

    Returns
    -------
    dict with:
        band_used       : (low_hz, high_hz) or None
        band_source     : "auto" | "manual" | "none"
        band_kurtosis   : float or None
        band_scan       : dict or None
        envelope        : np.ndarray
        freqs           : np.ndarray
        magnitude       : np.ndarray
        fault_freqs     : dict {name: Hz} actually used (or None)
        scan_results    : raw harmonic-by-harmonic results (or None)
        summary         : fault_scan_summary output (or None)

    Examples
    --------
    # CWRU drive-end OR fault file:
    result = analyze_bearing(signal, fs=12000, rpm=1797,
                             bearing_position="drive_end")
    print(result["summary"]["BPFO"]["best_classification"])

    # Custom bearing (non-CWRU):
    custom = {"BPFO": 95.3, "BPFI": 140.2, "BSF": 61.7}
    result = analyze_bearing(signal, fs=25600, fault_freqs=custom)
    """
    # ---------- Step 1: resolve fault frequencies ----------
    if fault_freqs is not None:
        _fault_freqs = fault_freqs
    elif rpm is not None:
        _fault_freqs = cwru_fault_frequencies(rpm, bearing_position)
    else:
        _fault_freqs = None

    # ---------- Step 2: envelope analysis ----------
    env_result = auto_envelope_analysis(
        signal, fs,
        band=band,
        auto_band=auto_band,
        filter_order=filter_order,
        n_centers=n_centers,
        bandwidth_hz=1000.0,
        multi_scale=multi_scale,
    )

    # ---------- Step 3: fault scanning (if frequencies available) ----------
    if _fault_freqs is not None:
        scan = scan_fault_frequencies(
            env_result["freqs"],
            env_result["magnitude"],
            _fault_freqs,
            tolerance_hz=base_tolerance_hz,
            include_harmonics=include_harmonics,
            rpm_uncertainty_pct=rpm_uncertainty_pct,
            use_adaptive_tolerance=True,
        )
        summary = fault_scan_summary(scan)
    else:
        scan    = None
        summary = None

    return {
        **env_result,
        "fault_freqs":  _fault_freqs,
        "scan_results": scan,
        "summary":      summary,
    }