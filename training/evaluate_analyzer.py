"""
Evaluation script for IncidentAnalyzer on CSV datasets.

Reports:
- Accuracy
- Macro precision/recall/F1
- Per-class precision/recall/F1/support
- Confusion matrix
- Confidence calibration (Brier score + ECE)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from bisect import bisect_right

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

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
    if lower in {
        "none / false report",
        "none/false report",
        "none / invalid",
        "none/invalid",
        "none / non-abuse report",
        "none/non-abuse report",
        "none / non abuse report",
        "none/non abuse report",
        "invalid",
    }:
        return "None / Invalid"

    # First exact match
    for label in IncidentValidator.ABUSE_TYPES:
        if lower == label.lower():
            return label

    # Then contains-match for mixed labels like "Physical Abuse + Psychological Abuse"
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


def load_calibration_artifact(path: Path) -> Tuple[List[float], List[float]] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    xs = payload.get("x_thresholds", [])
    ys = payload.get("y_thresholds", [])
    if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) < 2 or len(xs) != len(ys):
        return None
    x_vals = [float(v) for v in xs]
    y_vals = [float(v) for v in ys]
    if any(x_vals[i] > x_vals[i + 1] for i in range(len(x_vals) - 1)):
        return None
    return x_vals, y_vals


def apply_calibration(confidence: float, xs: List[float], ys: List[float]) -> float:
    x = max(0.0, min(float(confidence), 1.0))
    if x <= xs[0]:
        return max(0.0, min(float(ys[0]), 1.0))
    if x >= xs[-1]:
        return max(0.0, min(float(ys[-1]), 1.0))
    idx = max(0, min(len(xs) - 2, bisect_right(xs, x) - 1))
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    if x1 <= x0:
        y = y0
    else:
        y = y0 + ((x - x0) / (x1 - x0)) * (y1 - y0)
    return max(0.0, min(float(y), 1.0))


def evaluate(records: List[Dict[str, Any]], load_model: bool, calibration_json: str = "") -> Dict[str, Any]:
    analyzer = IncidentAnalyzer()
    if load_model:
        analyzer.load_model()

    y_true: List[str] = []
    y_pred: List[str] = []
    confidences: List[float] = []

    for row in records:
        true_label = row["incident_type"]
        result = analyzer.analyze(row["incident_description"])
        pred_label = canonical_label(result.get("incident_type"))
        conf = float(result.get("confidence_score", 0.0)) / 100.0
        conf = max(0.0, min(conf, 1.0))

        y_true.append(true_label)
        y_pred.append(pred_label)
        confidences.append(conf)

    labels = IncidentValidator.ABUSE_TYPES
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    conf_arr = np.array(confidences)
    correct_arr = np.array([1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)])
    brier = float(np.mean((conf_arr - correct_arr) ** 2))
    ece = expected_calibration_error(conf_arr, correct_arr, n_bins=10)

    calibration_applied = False
    calibrated_brier = None
    calibrated_ece = None
    if calibration_json:
        artifact = load_calibration_artifact(Path(calibration_json))
        if artifact is not None:
            xs, ys = artifact
            calibrated_conf = np.array([apply_calibration(c, xs, ys) for c in conf_arr], dtype=float)
            calibrated_brier = float(np.mean((calibrated_conf - correct_arr) ** 2))
            calibrated_ece = expected_calibration_error(calibrated_conf, correct_arr, n_bins=10)
            calibration_applied = True

    per_class: Dict[str, Dict[str, Any]] = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return {
        "num_examples": len(records),
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "brier_score": round(brier, 4),
        "ece_10_bins": round(ece, 4),
        "calibration_applied": calibration_applied,
        "calibrated_brier_score": round(calibrated_brier, 4) if calibrated_brier is not None else None,
        "calibrated_ece_10_bins": round(calibrated_ece, 4) if calibrated_ece is not None else None,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IncidentAnalyzer using CSV datasets.")
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
        help="Load fine-tuned model before evaluation (slower, GPU recommended).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/evaluation_report.json",
        help="Path to save evaluation JSON report.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on number of examples (0 = use all).",
    )
    parser.add_argument(
        "--calibration-json",
        type=str,
        default="",
        help="Optional confidence calibrator JSON to evaluate calibrated confidence metrics.",
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

    report = evaluate(records, load_model=args.load_model, calibration_json=args.calibration_json)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
