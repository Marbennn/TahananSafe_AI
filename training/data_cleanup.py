"""
Auto-clean common CSV data issues found by data_audit.py.

Fixes:
- Incident_Type normalization for common non-abuse aliases
- Language normalization (e.g., Mixed -> Mixed Language)
- Risk_Level recalculated from Incident_Risk_Percentage
- Priority_Level recalculated from Incident_Risk_Percentage
- Priority code normalization (P1/P2/P3 -> verbose form)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


NONE_TYPE_MAP = {
    "none / non-abuse report": "None / Invalid",
    "none/non-abuse report": "None / Invalid",
    "none / invalid": "None / Invalid",
    "none/invalid": "None / Invalid",
    "none / false report": "None / False Report",
    "none/false report": "None / False Report",
}

LANGUAGE_MAP = {
    "mixed": "Mixed Language",
    "tagalog": "Tagalog",
    "english": "English",
    "ilocano": "Ilocano",
    "pangasinan": "Pangasinan",
    "mixed language": "Mixed Language",
}

PRIORITY_MAP = {
    "P1": "First Priority (P1)",
    "P2": "Second Priority (P2)",
    "P3": "Third Priority (P3)",
}


def _read_config_paths(config_path: Path) -> tuple[str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds = cfg.get("dataset", {})
    return str(ds.get("main_dataset_path", "datasets/Main_Dataset.csv")), str(
        ds.get("negative_dataset_path", "datasets/Negative_Dataset.csv")
    )


def _safe_float(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _risk_level_from_pct(risk: float) -> str:
    if risk >= 80:
        return "Critical"
    if risk >= 60:
        return "High"
    if risk >= 40:
        return "Medium"
    return "Low"


def _priority_from_pct(risk: float) -> str:
    if risk >= 80:
        return "First Priority (P1)"
    if risk >= 60:
        return "Second Priority (P2)"
    return "Third Priority (P3)"


def _normalize_incident_type(raw: Any) -> str:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return "None / Invalid"
    mapped = NONE_TYPE_MAP.get(text.lower())
    return mapped if mapped is not None else text


def _normalize_language(raw: Any) -> str:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return "English"
    mapped = LANGUAGE_MAP.get(text.lower())
    return mapped if mapped is not None else text


def _normalize_priority(raw: Any) -> str:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return "Third Priority (P3)"
    if text in PRIORITY_MAP:
        return PRIORITY_MAP[text]
    return text


def clean_csv(in_path: Path, out_path: Path) -> None:
    df = pd.read_csv(in_path)

    if "Incident_Type" in df.columns:
        df["Incident_Type"] = df["Incident_Type"].map(_normalize_incident_type)

    if "Language" in df.columns:
        df["Language"] = df["Language"].map(_normalize_language)

    if "Priority_Level" in df.columns:
        df["Priority_Level"] = df["Priority_Level"].map(_normalize_priority)

    if "Incident_Risk_Percentage" in df.columns:
        risk_values = df["Incident_Risk_Percentage"].map(_safe_float)
        has_risk = risk_values.notna()

        # Keep original values for missing/non-numeric risks.
        if "Risk_Level" in df.columns:
            df.loc[has_risk, "Risk_Level"] = risk_values[has_risk].map(_risk_level_from_pct)
        if "Priority_Level" in df.columns:
            df.loc[has_risk, "Priority_Level"] = risk_values[has_risk].map(_priority_from_pct)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean common CSV dataset inconsistencies.")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--main-csv", type=str, default="", help="Override main CSV path")
    parser.add_argument("--negative-csv", type=str, default="", help="Override negative CSV path")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="datasets/cleaned",
        help="Output directory for cleaned CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_main, cfg_negative = _read_config_paths(Path(args.config))
    main_csv = Path(args.main_csv or cfg_main)
    negative_csv = Path(args.negative_csv or cfg_negative)
    out_dir = Path(args.out_dir)

    if not main_csv.exists():
        raise FileNotFoundError(f"Main CSV not found: {main_csv}")
    if not negative_csv.exists():
        raise FileNotFoundError(f"Negative CSV not found: {negative_csv}")

    main_out = out_dir / "Main_Dataset.cleaned.csv"
    negative_out = out_dir / "Negative_Dataset.cleaned.csv"
    clean_csv(main_csv, main_out)
    clean_csv(negative_csv, negative_out)

    print("Cleaned files written:")
    print(f"- {main_out}")
    print(f"- {negative_out}")


if __name__ == "__main__":
    main()

