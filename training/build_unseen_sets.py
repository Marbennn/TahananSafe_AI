"""
Build strict unseen CSV holdout sets from main/negative training CSVs.

Outputs (default):
- datasets/Unseen_Main.csv
- datasets/Unseen_Negative.csv

These files are meant for final evaluation and should be excluded from training.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List
import os
import sys

import pandas as pd
import yaml

# Allow running as script from project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validators import IncidentValidator


def _read_config_paths(config_path: Path) -> Dict[str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds = cfg.get("dataset", {})
    return {
        "main_csv": str(ds.get("main_dataset_path", "datasets/Main_Dataset.csv")),
        "negative_csv": str(ds.get("negative_dataset_path", "datasets/Negative_Dataset.csv")),
        "unseen_main_csv": str(ds.get("unseen_main_path", "datasets/Unseen_Main.csv")),
        "unseen_negative_csv": str(ds.get("unseen_negative_path", "datasets/Unseen_Negative.csv")),
    }


def _canonical_label(raw: Any) -> str:
    if raw is None:
        return "Unknown"
    text = str(raw).strip()
    if not text:
        return "Unknown"
    lower = text.lower()
    if lower in {"none / invalid", "none/invalid", "none / false report", "none/false report", "invalid", "none"}:
        return "None / Invalid"
    for label in IncidentValidator.ABUSE_TYPES:
        if lower == label.lower():
            return label
    for label in IncidentValidator.ABUSE_TYPES:
        if label.lower() in lower:
            return label
    return text


def _normalize_desc(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def _sample_main_unseen(
    df: pd.DataFrame,
    ratio: float,
    min_per_label: int,
    rng: random.Random,
) -> pd.DataFrame:
    by_label: Dict[str, List[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        by_label[_canonical_label(row.get("Incident_Type"))].append(idx)

    chosen: List[int] = []
    for label, indices in by_label.items():
        if not indices:
            continue
        count = len(indices)
        desired = int(round(count * ratio))
        if count >= min_per_label:
            desired = max(desired, min_per_label)
        desired = max(1, desired)
        desired = min(desired, max(1, count - 1))
        chosen.extend(rng.sample(indices, k=desired))

    return df.loc[sorted(set(chosen))].copy()


def _sample_negative_unseen(
    df: pd.DataFrame,
    ratio: float,
    rng: random.Random,
) -> pd.DataFrame:
    n = len(df)
    if n <= 1:
        return df.copy()
    desired = int(round(n * ratio))
    desired = max(1, desired)
    desired = min(desired, n - 1)
    chosen = rng.sample(list(df.index), k=desired)
    return df.loc[sorted(chosen)].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unseen holdout CSVs from training CSVs.")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to training config")
    parser.add_argument("--main-csv", type=str, default="", help="Override main CSV path")
    parser.add_argument("--negative-csv", type=str, default="", help="Override negative CSV path")
    parser.add_argument("--unseen-main-csv", type=str, default="", help="Output path for unseen main CSV")
    parser.add_argument("--unseen-negative-csv", type=str, default="", help="Output path for unseen negative CSV")
    parser.add_argument("--main-ratio", type=float, default=0.10, help="Holdout ratio for main dataset")
    parser.add_argument("--negative-ratio", type=float, default=0.10, help="Holdout ratio for negative dataset")
    parser.add_argument("--min-per-main-label", type=int, default=20, help="Minimum unseen rows per main label when possible")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _read_config_paths(Path(args.config))
    main_csv = Path(args.main_csv or paths["main_csv"])
    negative_csv = Path(args.negative_csv or paths["negative_csv"])
    unseen_main_csv = Path(args.unseen_main_csv or paths["unseen_main_csv"])
    unseen_negative_csv = Path(args.unseen_negative_csv or paths["unseen_negative_csv"])

    if not main_csv.exists():
        raise FileNotFoundError(f"Main CSV not found: {main_csv}")
    if not negative_csv.exists():
        raise FileNotFoundError(f"Negative CSV not found: {negative_csv}")

    rng = random.Random(int(args.seed))
    main_df = pd.read_csv(main_csv)
    negative_df = pd.read_csv(negative_csv)

    unseen_main = _sample_main_unseen(
        main_df,
        ratio=max(0.01, float(args.main_ratio)),
        min_per_label=max(1, int(args.min_per_main_label)),
        rng=rng,
    )
    unseen_negative = _sample_negative_unseen(
        negative_df,
        ratio=max(0.01, float(args.negative_ratio)),
        rng=rng,
    )

    unseen_main_csv.parent.mkdir(parents=True, exist_ok=True)
    unseen_negative_csv.parent.mkdir(parents=True, exist_ok=True)
    unseen_main.to_csv(unseen_main_csv, index=False, encoding="utf-8")
    unseen_negative.to_csv(unseen_negative_csv, index=False, encoding="utf-8")

    # Basic overlap sanity check (exact normalized description).
    main_descs = {_normalize_desc(x) for x in unseen_main.get("Incident_Description", pd.Series([], dtype=str)).tolist()}
    neg_descs = {_normalize_desc(x) for x in unseen_negative.get("Incident_Description", pd.Series([], dtype=str)).tolist()}
    cross_overlap = len(main_descs.intersection(neg_descs))

    main_labels = Counter(_canonical_label(x) for x in unseen_main.get("Incident_Type", pd.Series([], dtype=str)).tolist())

    print("Built unseen holdout sets:")
    print(f"- Unseen main     : {unseen_main_csv} ({len(unseen_main)} rows)")
    print(f"- Unseen negative : {unseen_negative_csv} ({len(unseen_negative)} rows)")
    print(f"- Cross-set description overlap: {cross_overlap}")
    print(f"- Unseen main label distribution: {dict(main_labels)}")


if __name__ == "__main__":
    main()
