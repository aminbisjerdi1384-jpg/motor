# Motor Insight (v2) - Vibration-Based Motor & Bearing Condition Monitoring

A Streamlit application for industrial motor vibration analysis and bearing
fault diagnosis: signal preprocessing & filtering, time/frequency/time-frequency
analysis, envelope analysis, two independent fault-diagnosis modes, and a
downloadable PDF report.

## What changed in v2 (and why)

The original v1 used Isolation Forest fit **only on the windows of the
single uploaded file**, so it could only flag windows that looked unusual
*relative to the rest of that same file*. A file that is faulty **from
start to end** (e.g. a real CWRU bearing-fault recording) looked uniformly
"faulty" to the model, so nothing stood out and the file was reported as
Normal. v2 fixes this by adding two diagnosis modes that don't have this
blind spot:

- **Mode A - rule-based**: compares kurtosis/crest factor against fixed,
  physically-motivated thresholds (independent of what else is in the file).
- **Mode B - supervised ML**: trains a Random Forest / SVM on labeled
  examples from multiple files, so it learns what each class actually
  looks like.

Isolation Forest (`anomaly.py`) is kept as a **supplementary, within-file**
novelty detector (e.g. "did something change partway through this
recording?"), not as the primary verdict.

## Project structure

```
motor_insight/
├── app.py                  # Main Streamlit app (8 tabs)
├── preprocessing.py         # Cleaning, normalization, filters, noise/SNR, downsampling
├── features.py              # FFT + 11 time-domain features (RMS, kurtosis, crest factor, ...)
├── spectral.py               # PSD (Welch), Periodogram, Spectrogram (STFT)
├── envelope.py                # Band-pass -> Hilbert -> envelope -> envelope FFT
├── bearing.py                  # BPFO/BPFI/BSF/FTF (official CWRU specs + generic geometry formula)
├── cwru_loader.py                # Load CWRU .mat files; infer fault label from filename
├── diagnosis.py                    # Mode A: rule-based Healthy/Warning/Faulty
├── ml_classifier.py                 # Mode B: Random Forest / SVM training & prediction
├── anomaly.py                        # Isolation Forest (supplementary novelty detector)
├── report.py                          # PDF report (reportlab)
├── utils.py                            # File loading + all matplotlib plot helpers
├── download_cwru_data.py                # Run LOCALLY to fetch the real CWRU dataset
├── generate_sample_data.py               # Generates synthetic 4-class sample CSVs
├── requirements.txt
└── sample_data/                            # Pre-generated synthetic sample files
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Getting data

You have two options, and don't need either to try the app - `sample_data/`
already contains synthetic 4-class CSVs (`Healthy_*.csv`,
`Inner_Race_Fault_*.csv`, `Outer_Race_Fault_*.csv`, `Ball_Fault_*.csv`),
regenerate them anytime with `python generate_sample_data.py`.

**For the real CWRU dataset** (recommended for an academic deliverable),
run this on a machine with internet access (file IDs were taken directly
from the official Case School of Engineering Bearing Data Center,
verified June 2026):

```bash
pip install requests scipy pandas numpy
python download_cwru_data.py --subset quick   # 4 files, fast
python download_cwru_data.py --subset full --loads 0 1 2 3   # everything
```

This downloads the original `.mat` files into `cwru_data/raw_mat/` and
converts them to `cwru_data/csv/` for use as Mode B training files. If you
use this dataset academically, please cite the CWRU Bearing Data Center.

## Using the app

1. Upload a CSV/Excel/`.mat` file and pick the sensor channel.
2. (Optional) Enter the motor RPM to get theoretical BPFO/BPFI/BSF/FTF and
   compare them against the envelope spectrum.
3. (Optional) Enable a Butterworth filter. **Be careful with the band you
   choose** - bearing-fault energy often lives at higher (resonance)
   frequencies, so a narrow low-frequency band can filter out the very
   fault signal you're trying to detect. The Envelope tab has its own,
   separate band-pass control specifically for isolating the resonance
   band before demodulation.
4. Explore the tabs: raw/filtered signal, FFT/PSD/Periodogram, spectrogram,
   envelope analysis, noise/resampling study, full feature table.
5. Pick a diagnosis mode in the sidebar:
   - **Mode A** works immediately, no training needed.
   - **Mode B** needs labeled training files uploaded in the sidebar first
     (use `sample_data/` or your downloaded CWRU CSVs), then click
     "Train Model". **If you change the sidebar filter settings after
     training, retrain** - the app will warn you if it detects a mismatch,
     since the classifier must see the same preprocessing at train and
     predict time.
6. Download the PDF report from the last tab.

## Known limitations / honest caveats

- **PDF report language**: the PDF is in English. `reportlab` does not
  perform Arabic/Persian glyph shaping or bidi reordering, so Persian text
  would render as disconnected, wrongly-ordered letters without extra
  dependencies (a Persian font + `arabic-reshaper` + `python-bidi`). The
  Streamlit UI is in Persian since the browser renders that correctly.
- **Mode A thresholds** (`diagnosis.py`) are illustrative engineering
  heuristics, not a certified standard like ISO 10816. Calibrate them
  against your own machine's healthy baseline for real deployments.
- **Mode B label inference** from filenames (`cwru_loader.py`) is a
  heuristic; always confirm/correct labels in the sidebar before training.
- The bundled `download_cwru_data.py` manifest covers the 12 kHz Drive-End
  Normal + Fault data (all 4 classes). Extend the `MANIFEST` dict in that
  file if you also want the 48 kHz or Fan-End data.
