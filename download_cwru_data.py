"""
download_cwru_data.py
----------------------
Downloads a curated subset of the real CWRU Bearing Data Center dataset
and converts each file to a simple CSV usable by the rest of Motor Insight
(cwru_loader.py / ml_classifier.py).

IMPORTANT: run this script on YOUR machine, not inside an offline sandbox -
it needs outbound internet access to reach engineering.case.edu.

    pip install requests scipy pandas numpy
    python download_cwru_data.py --subset quick
    python download_cwru_data.py --subset full --loads 0 1 2 3

File IDs below were taken directly from the official Case School of
Engineering Bearing Data Center pages (verified June 2026):
    https://engineering.case.edu/bearingdatacenter/download-data-file
    https://engineering.case.edu/bearingdatacenter/normal-baseline-data
    https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data

This subset covers the 12 kHz Drive-End normal + faulty data for all 4
classes used elsewhere in this project (Healthy / Inner Race Fault /
Outer Race Fault / Ball Fault). If you use this dataset in academic work,
please cite the CWRU Bearing Data Center per their usage guidelines.
"""
import argparse
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"
RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# label -> fault_diameter_tag -> {load_hp: file_id}
MANIFEST = {
    "Healthy": {
        "0.000": {0: 97, 1: 98, 2: 99, 3: 100},
    },
    "Inner Race Fault": {
        "0.007": {0: 105, 1: 106, 2: 107, 3: 108},
        "0.014": {0: 169, 1: 170, 2: 171, 3: 172},
        "0.021": {0: 209, 1: 210, 2: 211, 3: 212},
        "0.028": {0: 3001, 1: 3002, 2: 3003, 3: 3004},
    },
    "Ball Fault": {
        "0.007": {0: 118, 1: 119, 2: 120, 3: 121},
        "0.014": {0: 185, 1: 186, 2: 187, 3: 188},
        "0.021": {0: 222, 1: 223, 2: 224, 3: 225},
        "0.028": {0: 3005, 1: 3006, 2: 3007, 3: 3008},
    },
    "Outer Race Fault": {
        "0.007_centered": {0: 130, 1: 131, 2: 132, 3: 133},
        "0.014_centered": {0: 197, 1: 198, 2: 199, 3: 200},
        "0.021_centered": {0: 234, 1: 235, 2: 236, 3: 237},
    },
}

# A small, fast subset: one Healthy file + one file per fault class, load 0 only.
QUICK_SUBSET = {
    "Healthy": ["0.000"],
    "Inner Race Fault": ["0.007"],
    "Outer Race Fault": ["0.007_centered"],
    "Ball Fault": ["0.007"],
}


def iter_manifest(subset: str, loads):
    spec = QUICK_SUBSET if subset == "quick" else {k: list(v.keys()) for k, v in MANIFEST.items()}
    for label, tags in spec.items():
        for tag in tags:
            file_ids = MANIFEST[label][tag]
            for load in loads:
                if load not in file_ids:
                    continue
                yield label, tag, load, file_ids[load], RPM_BY_LOAD[load]


def download_file(file_id: int, dest_path: Path, session: requests.Session, retries: int = 3) -> bool:
    if dest_path.exists():
        print(f"  [skip] already downloaded: {dest_path.name}")
        return True

    url = BASE_URL.format(file_id)
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            dest_path.write_bytes(resp.content)
            print(f"  [ok]   {url} -> {dest_path.name} ({len(resp.content)/1024:.0f} KB)")
            return True
        except requests.RequestException as e:
            print(f"  [retry {attempt}/{retries}] {url} failed: {e}")
            time.sleep(1.5)
    print(f"  [FAILED] could not download {url}")
    return False


def convert_to_csv(mat_path: Path, csv_path: Path, channel: str = "DE", rpm: float = None):
    """Convert a downloaded .mat file to a single-column CSV via cwru_loader."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cwru_loader import load_mat_file  # local import to avoid hard dependency if unused
    import pandas as pd

    channels = load_mat_file(str(mat_path))
    signal = channels.get(channel) or next(iter(channels.values()))
    df = pd.DataFrame({channel.lower(): signal})
    if rpm is not None:
        df["rpm"] = rpm
    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Download & convert CWRU bearing dataset files.")
    parser.add_argument("--subset", choices=["quick", "full"], default="quick",
                         help="'quick' = 4 files (1 per class), 'full' = entire manifest above.")
    parser.add_argument("--loads", type=int, nargs="+", default=[0],
                         help="Motor load(s) in HP to include: any of 0 1 2 3 (default: 0).")
    parser.add_argument("--output-dir", type=str, default="cwru_data",
                         help="Where to store downloaded .mat and converted .csv files.")
    parser.add_argument("--channel", type=str, default="DE", choices=["DE", "FE", "BA"],
                         help="Which accelerometer channel to convert to CSV.")
    parser.add_argument("--skip-csv", action="store_true",
                         help="Only download .mat files, skip CSV conversion.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    mat_dir = out_dir / "raw_mat"
    csv_dir = out_dir / "csv"
    mat_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (MotorInsight academic data download script)"})

    rows = list(iter_manifest(args.subset, args.loads))
    print(f"Planning to download {len(rows)} file(s) (subset={args.subset}, loads={args.loads})\n")

    manifest_rows = []
    for label, tag, load, file_id, rpm in rows:
        safe_label = label.replace(" ", "_")
        mat_name = f"{safe_label}_{tag}_load{load}_{file_id}.mat"
        mat_path = mat_dir / mat_name
        print(f"{label} | diameter={tag} | load={load}HP | rpm={rpm} | id={file_id}")
        ok = download_file(file_id, mat_path, session)
        if ok and not args.skip_csv:
            csv_path = csv_dir / mat_name.replace(".mat", ".csv")
            try:
                convert_to_csv(mat_path, csv_path, channel=args.channel, rpm=rpm)
                print(f"  [csv]  -> {csv_path}")
            except Exception as e:
                print(f"  [csv FAILED] {e}")
        manifest_rows.append((label, tag, load, rpm, file_id, ok))

    print("\nDone. Summary:")
    for label, tag, load, rpm, file_id, ok in manifest_rows:
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {label:<20} {tag:<18} load={load} rpm={rpm} id={file_id}")

    print(f"\nCSV files (ready for the 'Train Classifier' tab in app.py) are in: {csv_dir.resolve()}")


if __name__ == "__main__":
    main()
