"""
Dataset audit script for TahananSafe AI CSV training data.

Checks:
- Missing/invalid fields
- Incident type parsing/validity (including multi-label strings)
- Risk level/priority consistency versus risk percentage
- Dataset-role consistency (main vs negative)
- Duplicate descriptions with conflicting labels

Outputs:
- training/audit_conflicts.csv
- training/audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

import os
import sys

# Allow running as script from project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validators import IncidentValidator


NONE_LIKE = {
    "none / invalid",
    "none/invalid",
    "none / false report",
    "none/false report",
    "invalid",
    "none",
}

PRIORITY_MAP = {
    "P1": "First Priority (P1)",
    "P2": "Second Priority (P2)",
    "P3": "Third Priority (P3)",
}

YES_VALUES = {"yes", "y", "true", "1"}
NO_VALUES = {"no", "n", "false", "0"}


def _read_config_paths(config_path: Path) -> tuple[str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds = cfg.get("dataset", {})
    return str(ds.get("main_dataset_path", "datasets/Main_Dataset.csv")), str(
        ds.get("negative_dataset_path", "datasets/Negative_Dataset.csv")
    )


def _canonical_priority(raw: Any) -> str:
    if raw is None:
        return "Third Priority (P3)"
    text = str(raw).strip()
    if text in IncidentValidator.PRIORITY_LEVELS:
        return text
    return PRIORITY_MAP.get(text.upper(), text)


def _parse_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _canonical_risk_level(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    mapping = {
        "low": "Low",
        "medium": "Medium",
        "med": "Medium",
        "high": "High",
        "critical": "Critical",
    }
    return mapping.get(text)


def _expected_risk_level(risk_pct: float) -> str:
    if risk_pct >= 80:
        return "Critical"
    if risk_pct >= 60:
        return "High"
    if risk_pct >= 40:
        return "Medium"
    return "Low"


def _expected_priority(risk_pct: float) -> str:
    if risk_pct >= 80:
        return "First Priority (P1)"
    if risk_pct >= 60:
        return "Second Priority (P2)"
    return "Third Priority (P3)"


def _parse_bool_text(raw: Any) -> Optional[bool]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in YES_VALUES:
        return True
    if text in NO_VALUES:
        return False
    return None


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _canonical_incident_labels(raw: Any) -> List[str]:
    """
    Parse incident type text into canonical labels.
    Supports multi-label text such as:
    - "Physical Abuse + Psychological Abuse"
    - "Physical Abuse, Economic Abuse"
    """
    if raw is None:
        return ["None / Invalid"]

    text = str(raw).strip()
    if not text:
        return ["None / Invalid"]

    lower = text.lower()
    if lower in NONE_LIKE:
        return ["None / Invalid"]

    labels: List[str] = []
    seen = set()
    for label in IncidentValidator.ABUSE_TYPES:
        if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", text, flags=re.IGNORECASE):
            if label not in seen:
                seen.add(label)
                labels.append(label)

    if labels:
        return labels

    # Fallback fuzzy mapping for partial labels.
    fuzzy = [
        ("physical", "Physical Abuse"),
        ("sexual", "Sexual Abuse"),
        ("psychological", "Psychological Abuse"),
        ("emotional", "Psychological Abuse"),
        ("economic", "Economic Abuse"),
        ("financial", "Economic Abuse"),
        ("elder", "Elder Abuse"),
        ("senior", "Elder Abuse"),
        ("neglect", "Neglect / Acts of Omission"),
        ("omission", "Neglect / Acts of Omission"),
        ("unknown", "Unknown"),
    ]
    parts = re.split(r"\s*(?:\+|,|;|\band\b)\s*", text, flags=re.IGNORECASE)
    for part in parts:
        p = part.strip().lower()
        if not p:
            continue
        if p in NONE_LIKE and "None / Invalid" not in seen:
            seen.add("None / Invalid")
            labels.append("None / Invalid")
            continue
        for token, mapped in fuzzy:
            if token in p and mapped not in seen:
                seen.add(mapped)
                labels.append(mapped)
                break

    return labels if labels else []


def _issue(
    issues: List[Dict[str, Any]],
    row: Dict[str, Any],
    code: str,
    severity: str,
    details: str,
) -> None:
    issues.append(
        {
            "source": row["source"],
            "row_number": int(row["row_number"]),
            "issue_code": code,
            "severity": severity,
            "details": details,
            "incident_type_raw": row["incident_type_raw"],
            "incident_description": row["incident_description"][:220],
            "risk_level_raw": row["risk_level_raw"],
            "risk_percentage_raw": row["risk_percentage_raw"],
            "priority_raw": row["priority_raw"],
        }
    )


def _load_rows(csv_path: Path, source_name: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    rows: List[Dict[str, Any]] = []
    for idx, r in df.iterrows():
        rows.append(
            {
                "source": source_name,
                "file_path": str(csv_path),
                "row_number": idx + 2,  # CSV header is line 1
                "incident_type_raw": r.get("Incident_Type"),
                "incident_description": _normalize_space(str(r.get("Incident_Description", ""))),
                "language_raw": r.get("Language"),
                "risk_level_raw": r.get("Risk_Level"),
                "risk_percentage_raw": r.get("Incident_Risk_Percentage"),
                "priority_raw": r.get("Priority_Level"),
                "children_raw": r.get("Children_Involved"),
                "weapon_raw": r.get("Weapon_Mentioned"),
                "confidence_raw": r.get("AI_Confidence_Score"),
            }
        )
    return rows


def _audit_rows(rows: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> Dict[str, Counter]:
    label_counter_main: Counter = Counter()
    label_counter_negative: Counter = Counter()

    for row in rows:
        desc = row["incident_description"]
        labels = _canonical_incident_labels(row["incident_type_raw"])
        risk_pct = _parse_float(row["risk_percentage_raw"])
        risk_level = _canonical_risk_level(row["risk_level_raw"])
        priority = _canonical_priority(row["priority_raw"])
        lang = str(row["language_raw"]).strip() if row["language_raw"] is not None else ""
        conf = _parse_float(row["confidence_raw"])
        children = _parse_bool_text(row["children_raw"])
        weapon = _parse_bool_text(row["weapon_raw"])

        if row["source"] == "main":
            for lbl in labels:
                label_counter_main[lbl] += 1
        else:
            for lbl in labels:
                label_counter_negative[lbl] += 1

        if len(desc) < 10:
            _issue(issues, row, "SHORT_DESCRIPTION", "high", "Incident description is shorter than 10 chars.")

        if not labels:
            _issue(
                issues,
                row,
                "INVALID_INCIDENT_TYPE",
                "high",
                f"Could not parse Incident_Type='{row['incident_type_raw']}' into allowed labels.",
            )

        if lang and lang not in IncidentValidator.LANGUAGES:
            _issue(
                issues,
                row,
                "INVALID_LANGUAGE",
                "medium",
                f"Language '{lang}' is not in allowed list.",
            )

        if risk_pct is None:
            _issue(
                issues,
                row,
                "INVALID_RISK_PERCENT",
                "high",
                f"Risk percentage '{row['risk_percentage_raw']}' is not numeric.",
            )
        elif risk_pct < 0 or risk_pct > 100:
            _issue(
                issues,
                row,
                "RISK_OUT_OF_RANGE",
                "high",
                f"Risk percentage {risk_pct} is outside 0-100.",
            )

        if risk_level is None:
            _issue(
                issues,
                row,
                "INVALID_RISK_LEVEL",
                "high",
                f"Risk level '{row['risk_level_raw']}' is invalid.",
            )

        if priority not in IncidentValidator.PRIORITY_LEVELS:
            _issue(
                issues,
                row,
                "INVALID_PRIORITY",
                "high",
                f"Priority '{row['priority_raw']}' is invalid.",
            )

        if conf is None:
            _issue(
                issues,
                row,
                "INVALID_CONFIDENCE",
                "medium",
                f"Confidence '{row['confidence_raw']}' is not numeric.",
            )
        elif conf < 0 or conf > 100:
            _issue(
                issues,
                row,
                "CONFIDENCE_OUT_OF_RANGE",
                "medium",
                f"Confidence {conf} is outside 0-100.",
            )

        if children is None:
            _issue(
                issues,
                row,
                "INVALID_CHILDREN_VALUE",
                "low",
                f"Children_Involved='{row['children_raw']}' should be Yes/No.",
            )
        if weapon is None:
            _issue(
                issues,
                row,
                "INVALID_WEAPON_VALUE",
                "low",
                f"Weapon_Mentioned='{row['weapon_raw']}' should be Yes/No.",
            )

        if risk_pct is not None and 0 <= risk_pct <= 100 and risk_level is not None:
            expected_level = _expected_risk_level(risk_pct)
            if expected_level != risk_level:
                _issue(
                    issues,
                    row,
                    "RISK_LEVEL_MISMATCH",
                    "high",
                    f"Risk {risk_pct} implies {expected_level}, but row has {risk_level}.",
                )

            expected_priority = _expected_priority(risk_pct)
            if expected_priority != priority:
                _issue(
                    issues,
                    row,
                    "PRIORITY_MISMATCH",
                    "high",
                    f"Risk {risk_pct} implies {expected_priority}, but row has {priority}.",
                )

        # Dataset-role consistency.
        core_labels = [x for x in labels if x in IncidentValidator.ABUSE_CORE_TYPES]
        non_abuse_labels = [x for x in labels if x in {"None / Invalid", "None / False Report", "Unknown"}]
        if row["source"] == "negative" and core_labels:
            _issue(
                issues,
                row,
                "NEGATIVE_HAS_ABUSE_LABEL",
                "high",
                f"Negative dataset row contains abuse labels: {', '.join(core_labels)}",
            )
        if row["source"] == "main" and non_abuse_labels:
            _issue(
                issues,
                row,
                "MAIN_HAS_NON_ABUSE_LABEL",
                "medium",
                f"Main dataset row contains non-abuse label(s): {', '.join(non_abuse_labels)}",
            )

    return {"main_labels": label_counter_main, "negative_labels": label_counter_negative}


def _audit_duplicate_conflicts(rows: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> int:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        norm_desc = _normalize_space(row["incident_description"].lower())
        if norm_desc:
            grouped[norm_desc].append(row)

    conflict_groups = 0
    for _, group in grouped.items():
        if len(group) < 2:
            continue

        label_sigs = set()
        risk_vals = []
        priority_vals = set()
        for row in group:
            labels = sorted(_canonical_incident_labels(row["incident_type_raw"]))
            label_sigs.add(" | ".join(labels))
            risk = _parse_float(row["risk_percentage_raw"])
            if risk is not None:
                risk_vals.append(risk)
            priority_vals.add(_canonical_priority(row["priority_raw"]))

        risk_spread = (max(risk_vals) - min(risk_vals)) if len(risk_vals) >= 2 else 0.0
        is_conflict = len(label_sigs) > 1 or len(priority_vals) > 1 or risk_spread >= 20.0
        if not is_conflict:
            continue

        conflict_groups += 1
        details = (
            f"Duplicate description has conflicts: labels={sorted(label_sigs)}, "
            f"priorities={sorted(priority_vals)}, risk_spread={round(risk_spread, 2)}"
        )
        for row in group:
            _issue(issues, row, "DUPLICATE_CONFLICT", "high", details)

    return conflict_groups


def run_audit(
    main_csv: Path,
    negative_csv: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(main_csv, "main") + _load_rows(negative_csv, "negative")
    issues: List[Dict[str, Any]] = []
    counters = _audit_rows(rows, issues)
    duplicate_conflict_groups = _audit_duplicate_conflicts(rows, issues)

    conflicts_path = out_dir / "audit_conflicts.csv"
    pd.DataFrame(issues).to_csv(conflicts_path, index=False, encoding="utf-8")

    issues_by_code = Counter(i["issue_code"] for i in issues)
    issues_by_severity = Counter(i["severity"] for i in issues)
    issues_by_source = Counter(i["source"] for i in issues)

    core_counts = {
        k: int(v)
        for k, v in counters["main_labels"].items()
        if k in IncidentValidator.ABUSE_CORE_TYPES
    }
    imbalance_ratio = None
    if core_counts:
        vals = [v for v in core_counts.values() if v > 0]
        if vals:
            imbalance_ratio = round(max(vals) / min(vals), 3)

    summary = {
        "main_csv": str(main_csv),
        "negative_csv": str(negative_csv),
        "total_rows": len(rows),
        "total_issues": len(issues),
        "issues_by_code": dict(issues_by_code),
        "issues_by_severity": dict(issues_by_severity),
        "issues_by_source": dict(issues_by_source),
        "duplicate_conflict_groups": duplicate_conflict_groups,
        "main_label_distribution": dict(counters["main_labels"]),
        "negative_label_distribution": dict(counters["negative_labels"]),
        "main_core_class_imbalance_ratio_max_over_min": imbalance_ratio,
        "recommended_next_action": [
            "Fix all HIGH severity issues first.",
            "Fix duplicate conflicts and risk/priority mismatches before retraining.",
            "Re-run data_audit.py until HIGH issues are 0.",
        ],
    }

    summary_path = out_dir / "audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "summary": summary,
        "conflicts_path": conflicts_path,
        "summary_path": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CSV training datasets for label consistency.")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to training config.yaml")
    parser.add_argument("--main-csv", type=str, default="", help="Override main dataset CSV path")
    parser.add_argument("--negative-csv", type=str, default="", help="Override negative dataset CSV path")
    parser.add_argument("--out-dir", type=str, default="training", help="Directory for audit outputs")
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

    result = run_audit(main_csv, negative_csv, out_dir)
    summary = result["summary"]

    print("\n=== DATA AUDIT SUMMARY ===")
    print(f"Main CSV: {main_csv}")
    print(f"Negative CSV: {negative_csv}")
    print(f"Total rows: {summary['total_rows']}")
    print(f"Total issues: {summary['total_issues']}")
    print(f"Issues by severity: {summary['issues_by_severity']}")
    print(f"Issues by code: {summary['issues_by_code']}")
    print(f"Duplicate conflict groups: {summary['duplicate_conflict_groups']}")
    print(f"Core class imbalance ratio (max/min): {summary['main_core_class_imbalance_ratio_max_over_min']}")
    print(f"\nSaved: {result['conflicts_path']}")
    print(f"Saved: {result['summary_path']}")


if __name__ == "__main__":
    main()
