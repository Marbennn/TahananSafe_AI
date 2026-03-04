"""
Fit a confidence calibration artifact for IncidentAnalyzer outputs.

The script runs analyzer predictions on CSV records, compares predicted
incident_type vs ground truth, then fits isotonic regression:
raw confidence -> empirical correctness probability.

Output JSON is consumed by inference/analyzer.py.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.isotonic import IsotonicRegression

# Add project root to import path when running as script.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.analyzer import IncidentAnalyzer
from utils.validators import IncidentValidator


def canonical_label(raw: Any) -> str:
    if raw is None:
        return "None / Invalid"
    text = str(raw).strip()
    if not text:
        return "None / Invalid"
    lower = text.lower()
    if lower in {"none / false report", "none/false report", "none / invalid", "none/invalid", "invalid"}:
        return "None / Invalid"
    for label in IncidentValidator.ABUSE_TYPES:
        if lower == label.lower():
            return label
    for label in IncidentValidator.ABUSE_TYPES:
        if label.lower() in lower:
            return label
    return "Unknown"


def load_csv_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            desc = str(row.get("Incident_Description", "")).strip()
            if len(desc) < 10:
                continue
            rows.append(
                {
                    "incident_description": desc,
                    "incident_type": canonical_label(row.get("Incident_Type")),
                }
            )
    return rows


def expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = max(1, len(confidences))
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(confidences[mask]))
        bin_acc = float(np.mean(correctness[mask]))
        ece += (np.sum(mask) / total) * abs(bin_acc - bin_conf)
    return float(ece)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit confidence calibrator from CSV datasets.")
    parser.add_argument(
        "--main-csv",
        type=str,
        default="datasets/Main_Dataset.csv",
        help="Path to Main_Dataset.csv",
    )
    parser.add_argument(
        "--negative-csv",
        type=str,
        default="datasets/Negative_Dataset.csv",
        help="Path to Negative_Dataset.csv",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load fine-tuned model before calibration (slower, GPU recommended).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/confidence_calibrator.json",
        help="Output calibration JSON path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on number of examples (0 = use all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_csv = Path(args.main_csv)
    negative_csv = Path(args.negative_csv)
    if not main_csv.exists():
        raise FileNotFoundError(f"Main CSV not found: {main_csv}")
    if not negative_csv.exists():
        raise FileNotFoundError(f"Negative CSV not found: {negative_csv}")

    records = load_csv_records(main_csv) + load_csv_records(negative_csv)
    if not records:
        raise ValueError("No valid records were loaded from CSV files.")
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    analyzer = IncidentAnalyzer()
    if args.load_model:
        analyzer.load_model()

    raw_conf: List[float] = []
    correct: List[float] = []
    for row in records:
        result = analyzer.analyze(row["incident_description"])
        pred_label = canonical_label(result.get("incident_type"))
        conf = float(result.get("confidence_score", 0.0)) / 100.0
        conf = max(0.0, min(conf, 1.0))
        raw_conf.append(conf)
        correct.append(1.0 if pred_label == row["incident_type"] else 0.0)

    x = np.array(raw_conf, dtype=float)
    y = np.array(correct, dtype=float)
    if len(x) < 20:
        raise ValueError("Need at least 20 examples to fit a useful confidence calibrator.")

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(x, y)
    calibrated = calibrator.predict(x)

    raw_brier = float(np.mean((x - y) ** 2))
    cal_brier = float(np.mean((calibrated - y) ** 2))
    raw_ece = expected_calibration_error(x, y, n_bins=10)
    cal_ece = expected_calibration_error(calibrated, y, n_bins=10)

    payload = {
        "x_thresholds": [float(v) for v in calibrator.X_thresholds_],
        "y_thresholds": [float(v) for v in calibrator.y_thresholds_],
        "metadata": {
            "num_examples": int(len(x)),
            "raw_brier_score": round(raw_brier, 6),
            "calibrated_brier_score": round(cal_brier, 6),
            "raw_ece_10_bins": round(raw_ece, 6),
            "calibrated_ece_10_bins": round(cal_ece, 6),
            "load_model": bool(args.load_model),
            "main_csv": str(main_csv),
            "negative_csv": str(negative_csv),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
    print(f"\nSaved calibrator to: {output_path}")


if __name__ == "__main__":
    main()

