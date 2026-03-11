"""
CSV-first case retriever for retrieval-augmented incident analysis.

Indexes historical incident descriptions from datasets and returns
similar examples for a new report.
"""

from __future__ import annotations

import csv
import json
import os
import re
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None

from utils.validators import IncidentValidator


class CaseRetriever:
    """Retrieve similar historical cases from local datasets."""

    NON_ABUSE_TYPES = {
        "none / invalid",
        "none / false report",
        "none/invalid",
        "none/false report",
        "none / non-abuse report",
        "none/non-abuse report",
        "none / non abuse report",
        "none/non abuse report",
        "invalid",
        "none",
    }

    def __init__(
        self,
        main_dataset_path: Optional[str] = None,
        negative_dataset_path: Optional[str] = None,
        default_top_k: int = 5,
    ) -> None:
        self.default_top_k = max(1, int(default_top_k))
        self.main_dataset_path = main_dataset_path or os.getenv("MAIN_DATASET_PATH")
        self.negative_dataset_path = negative_dataset_path or os.getenv("NEGATIVE_DATASET_PATH")

        self.enabled = False
        self.records: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.case_matrix = None
        self._exact_lookup: Dict[str, List[Dict[str, Any]]] = {}
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._query_cache_limit = max(8, int(os.getenv("RETRIEVAL_QUERY_CACHE_SIZE", "256")))

        self._build_index()

    def _default_candidate_paths(self) -> List[Path]:
        cwd = Path.cwd()
        return [
            cwd / "datasets" / "Main_Dataset.csv",
            cwd / "datasets" / "Negative_Dataset.csv",
            cwd / "datasets" / "sample_main_dataset.json",
            cwd / "datasets" / "sample_negative_dataset.json",
        ]

    def _resolve_dataset_paths(self) -> List[Path]:
        paths: List[Path] = []
        for raw in [self.main_dataset_path, self.negative_dataset_path]:
            if raw:
                p = Path(raw)
                if p.exists():
                    paths.append(p)

        if paths:
            return paths

        for p in self._default_candidate_paths():
            if p.exists():
                paths.append(p)
        return paths

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(str(value).replace("%", "").strip())
        except Exception:
            return default

    def _canonical_incident_type(self, value: Any) -> str:
        if value is None:
            return "None / Invalid"
        text = str(value).strip()
        if not text:
            return "None / Invalid"
        lower = text.lower()
        lower = re.sub(r"\s+", " ", lower).strip()
        lower = lower.replace("non abuse", "non-abuse")
        if lower in self.NON_ABUSE_TYPES:
            return "None / Invalid"
        if lower in {"none / false report", "none/false report"}:
            return "None / False Report"
        return text

    def _normalize_incident_types(self, value: Any) -> List[str]:
        """Normalize a raw incident type field into canonical labels."""
        if value is None:
            return ["None / Invalid"]

        raw = str(value).strip()
        if not raw:
            return ["None / Invalid"]
        raw_lower = raw.lower()
        raw_lower = re.sub(r"\s+", " ", raw_lower).strip()
        raw_lower = raw_lower.replace("non abuse", "non-abuse")
        if raw_lower in self.NON_ABUSE_TYPES:
            return ["None / Invalid"]
        if raw_lower in {"none / false report", "none/false report"}:
            return ["None / False Report"]

        labels: List[str] = []
        seen = set()

        for label in IncidentValidator.ABUSE_TYPES:
            if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", raw, re.IGNORECASE):
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
        if labels:
            return labels

        parts = re.split(r"\s*(?:\+|,|;|\band\b)\s*", raw, flags=re.IGNORECASE)
        fuzzy_map = [
            ("sexual", "Sexual Abuse"),
            ("physical", "Physical Abuse"),
            ("psychological", "Psychological Abuse"),
            ("emotional", "Psychological Abuse"),
            ("economic", "Economic Abuse"),
            ("financial", "Economic Abuse"),
            ("elder", "Elder Abuse"),
            ("senior", "Elder Abuse"),
            ("neglect", "Neglect / Acts of Omission"),
            ("omission", "Neglect / Acts of Omission"),
        ]
        for part in parts:
            p = part.strip()
            if not p:
                continue
            p_lower = p.lower()
            if p_lower in self.NON_ABUSE_TYPES:
                canonical = "None / Invalid"
            elif p_lower in {"none / false report", "none/false report"}:
                canonical = "None / False Report"
            else:
                canonical = None
                for label in IncidentValidator.ABUSE_TYPES:
                    if p_lower == label.lower():
                        canonical = label
                        break
                if canonical is None:
                    for token, mapped in fuzzy_map:
                        if token in p_lower:
                            canonical = mapped
                            break

            if canonical and canonical not in seen:
                seen.add(canonical)
                labels.append(canonical)

        if labels:
            return labels

        canonical = self._canonical_incident_type(raw)
        return [canonical]

    def _from_generic_record(self, row: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        desc = row.get("incident_description") or row.get("Incident_Description")
        if desc is None:
            return None
        description = str(desc).strip()
        if len(description) < 10:
            return None

        incident_type_raw = row.get("incident_type") or row.get("Incident_Type") or "None / Invalid"
        risk_pct = row.get("risk_percentage")
        if risk_pct is None:
            risk_pct = row.get("Incident_Risk_Percentage")

        risk_level = row.get("risk_level")
        if risk_level is None:
            risk_level = row.get("Risk_Level")

        priority_level = row.get("priority_level")
        if priority_level is None:
            priority_level = row.get("Priority_Level")

        language = row.get("language")
        if language is None:
            language = row.get("Language")

        children_involved = row.get("children_involved")
        if children_involved is None:
            children_involved = row.get("Children_Involved")

        weapon_mentioned = row.get("weapon_mentioned")
        if weapon_mentioned is None:
            weapon_mentioned = row.get("Weapon_Mentioned")

        incident_types = self._normalize_incident_types(incident_type_raw)
        primary_incident_type = incident_types[0] if incident_types else "None / Invalid"

        return {
            "incident_description": description,
            "incident_type": primary_incident_type,
            "incident_types": incident_types,
            "risk_percentage": self._safe_float(risk_pct, default=0.0),
            "risk_level": str(risk_level).strip() if risk_level is not None else "",
            "priority_level": str(priority_level).strip() if priority_level is not None else "",
            "language": str(language).strip() if language is not None else "",
            "children_involved": self._to_bool(children_involved),
            "weapon_mentioned": self._to_bool(weapon_mentioned),
            "source": source,
        }

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        v = str(value).strip().lower()
        return v in {"true", "1", "yes", "y", "oo", "opo"}

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        if isinstance(content, list):
            rows.extend(content)
        elif isinstance(content, dict):
            rows.append(content)

        out: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            rec = self._from_generic_record(row, source=str(path))
            if rec:
                out.append(rec)
        return out

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                rec = self._from_generic_record(row, source=str(path))
                if rec:
                    out.append(rec)
        return out

    def _load_csv(self, path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = self._from_generic_record(row, source=str(path))
                if rec:
                    out.append(rec)
        return out

    def _load_records_from_path(self, path: Path) -> List[Dict[str, Any]]:
        if path.is_dir():
            out: List[Dict[str, Any]] = []
            for child in sorted(path.glob("*")):
                out.extend(self._load_records_from_path(child))
            return out

        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._load_json(path)
        if suffix == ".jsonl":
            return self._load_jsonl(path)
        if suffix == ".csv":
            return self._load_csv(path)
        return []

    def _build_index(self) -> None:
        if TfidfVectorizer is None or cosine_similarity is None:
            return

        dataset_paths = self._resolve_dataset_paths()
        all_records: List[Dict[str, Any]] = []
        for path in dataset_paths:
            all_records.extend(self._load_records_from_path(path))

        if not all_records:
            return

        self._exact_lookup = {}
        for rec in all_records:
            key = self._normalize_text_key(rec.get("incident_description", ""))
            if not key:
                continue
            bucket = self._exact_lookup.setdefault(key, [])
            bucket.append(rec)

        descriptions = [r["incident_description"] for r in all_records]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
        )
        case_matrix = vectorizer.fit_transform(descriptions)

        self.records = all_records
        self.vectorizer = vectorizer
        self.case_matrix = case_matrix
        self.enabled = True

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        if not text:
            return ""
        t = str(text).strip().lower()
        t = re.sub(r"([a-zA-Z])\1{2,}", r"\1", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _compact_text(text: str, max_chars: int = 180) -> str:
        t = " ".join(str(text).split())
        if len(t) <= max_chars:
            return t
        return t[: max_chars - 3] + "..."

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.05,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not query or self.vectorizer is None or self.case_matrix is None:
            return []

        k = max(1, int(top_k or self.default_top_k))
        cache_key = f"{str(query).strip().lower()}|{k}|{round(float(min_similarity), 4)}"
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

        query_vec = self.vectorizer.transform([str(query)])
        scores = cosine_similarity(query_vec, self.case_matrix).ravel()
        if scores.size == 0:
            return []

        ranked_indices = np.argsort(-scores)[:k]
        out: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            sim = float(scores[idx])
            if sim < min_similarity:
                continue
            rec = self.records[int(idx)]
            out.append(
                {
                    "similarity": round(sim, 4),
                    "incident_type": rec["incident_type"],
                    "incident_types": rec.get("incident_types", [rec["incident_type"]]),
                    "risk_percentage": float(rec["risk_percentage"]),
                    "risk_level": rec.get("risk_level", ""),
                    "priority_level": rec.get("priority_level", ""),
                    "language": rec.get("language", ""),
                    "children_involved": bool(rec.get("children_involved", False)),
                    "weapon_mentioned": bool(rec.get("weapon_mentioned", False)),
                    "incident_description": self._compact_text(rec["incident_description"]),
                    "full_incident_description": str(rec["incident_description"]),
                    "source": rec["source"],
                }
            )
        self._query_cache[cache_key] = copy.deepcopy(out)
        if len(self._query_cache) > self._query_cache_limit:
            oldest_key = next(iter(self._query_cache))
            self._query_cache.pop(oldest_key, None)
        return out

    def retrieve_for_prompt(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.08,
    ) -> List[Dict[str, Any]]:
        """
        Return cleaned retrieval results intended for prompt grounding.
        Slightly stricter than generic retrieve().
        """
        matches = self.retrieve(query=query, top_k=top_k, min_similarity=min_similarity)
        cleaned: List[Dict[str, Any]] = []
        for m in matches:
            cleaned.append(
                {
                    "similarity": float(m.get("similarity", 0.0)),
                    "incident_type": str(m.get("incident_type", "Unknown")),
                    "incident_types": list(m.get("incident_types", [])),
                    "risk_percentage": float(m.get("risk_percentage", 0.0)),
                    "risk_level": str(m.get("risk_level", "")),
                    "priority_level": str(m.get("priority_level", "")),
                    "language": str(m.get("language", "")),
                    "children_involved": bool(m.get("children_involved", False)),
                    "weapon_mentioned": bool(m.get("weapon_mentioned", False)),
                    "incident_description": str(m.get("full_incident_description") or m.get("incident_description") or "").strip(),
                    "source": str(m.get("source", "")),
                }
            )
        return cleaned

    def build_prompt_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.08,
        max_case_chars: int = 220,
    ) -> str:
        """
        Build a compact retrieval context block for the generation prompt.
        """
        matches = self.retrieve_for_prompt(query=query, top_k=top_k, min_similarity=min_similarity)
        if not matches:
            return ""

        lines: List[str] = []
        for i, m in enumerate(matches[: max(1, int(top_k or 3))], start=1):
            desc = self._compact_text(m.get("incident_description", ""), max_chars=max_case_chars)
            lines.append(
                "\n".join(
                    [
                        f"Case {i}:",
                        f"- Similarity: {float(m.get('similarity', 0.0)):.4f}",
                        f"- Incident Type: {m.get('incident_type', 'Unknown')}",
                        f"- Risk Percentage: {float(m.get('risk_percentage', 0.0)):.2f}",
                        f"- Risk Level: {m.get('risk_level', '') or 'Unknown'}",
                        f"- Priority Level: {m.get('priority_level', '') or 'Unknown'}",
                        f"- Children Involved: {'Yes' if bool(m.get('children_involved', False)) else 'No'}",
                        f"- Weapon Mentioned: {'Yes' if bool(m.get('weapon_mentioned', False)) else 'No'}",
                        f"- Description: {desc}",
                    ]
                )
            )

        return "\n\n".join(lines)

    def summarize_consensus(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        exact = self.summarize_exact_match(query, top_k=top_k)
        if exact is not None:
            return exact

        matches = self.retrieve(query, top_k=top_k, min_similarity=min_similarity)
        if not matches:
            return None

        total_weight = sum(float(m["similarity"]) for m in matches)
        if total_weight <= 0:
            return None

        weights_by_type: Dict[str, float] = {}
        weighted_risk = 0.0
        for m in matches:
            w = float(m["similarity"])
            labels = m.get("incident_types") or [str(m.get("incident_type", "Unknown"))]
            labels = [str(x) for x in labels if x]
            if not labels:
                labels = [str(m.get("incident_type", "Unknown"))]
            share = w / max(1, len(labels))
            for t in labels:
                weights_by_type[t] = weights_by_type.get(t, 0.0) + share
            weighted_risk += w * float(m["risk_percentage"])

        best_type, best_weight = max(weights_by_type.items(), key=lambda x: x[1])
        non_abuse_weight = sum(
            w
            for t, w in weights_by_type.items()
            if t.lower() in self.NON_ABUSE_TYPES or t in {"None / Invalid", "None / False Report"}
        )
        type_distribution = {
            t: round(w / total_weight, 4)
            for t, w in sorted(weights_by_type.items(), key=lambda x: x[1], reverse=True)
        }

        return {
            "best_type": best_type,
            "best_type_ratio": round(best_weight / total_weight, 4),
            "non_abuse_ratio": round(non_abuse_weight / total_weight, 4),
            "weighted_risk": round(weighted_risk / total_weight, 2),
            "top_similarity": float(matches[0]["similarity"]),
            "match_count": len(matches),
            "type_distribution": type_distribution,
            "matches": matches[:3],
            "exact_match": False,
        }

    def summarize_exact_match(self, query: str, top_k: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not self.enabled or not query:
            return None
        key = self._normalize_text_key(query)
        records = self._exact_lookup.get(key, [])
        if not records:
            return None

        k = max(1, int(top_k or self.default_top_k))
        selected = records[:k]
        matches: List[Dict[str, Any]] = []
        weights_by_type: Dict[str, float] = {}
        weighted_risk = 0.0
        non_abuse_weight = 0.0

        for rec in selected:
            labels = rec.get("incident_types") or [rec.get("incident_type", "Unknown")]
            labels = [str(x) for x in labels if x]
            if not labels:
                labels = [str(rec.get("incident_type", "Unknown"))]
            share = 1.0 / max(1, len(labels))
            for t in labels:
                weights_by_type[t] = weights_by_type.get(t, 0.0) + share
                t_lower = str(t).lower().strip()
                t_lower = re.sub(r"\s+", " ", t_lower).replace("non abuse", "non-abuse")
                if t_lower in self.NON_ABUSE_TYPES or t in {"None / Invalid", "None / False Report"}:
                    non_abuse_weight += share
            weighted_risk += float(rec.get("risk_percentage", 0.0))
            matches.append(
                {
                    "similarity": 1.0,
                    "incident_type": rec.get("incident_type", "Unknown"),
                    "incident_types": rec.get("incident_types", [rec.get("incident_type", "Unknown")]),
                    "risk_percentage": float(rec.get("risk_percentage", 0.0)),
                    "risk_level": rec.get("risk_level", ""),
                    "priority_level": rec.get("priority_level", ""),
                    "language": rec.get("language", ""),
                    "children_involved": bool(rec.get("children_involved", False)),
                    "weapon_mentioned": bool(rec.get("weapon_mentioned", False)),
                    "incident_description": self._compact_text(rec.get("incident_description", "")),
                    "full_incident_description": str(rec.get("incident_description", "")),
                    "source": rec.get("source", ""),
                    "exact_match": True,
                }
            )

        total_weight = float(len(selected))
        if total_weight <= 0:
            return None
        best_type, best_weight = max(weights_by_type.items(), key=lambda x: x[1])
        type_distribution = {
            t: round(w / total_weight, 4)
            for t, w in sorted(weights_by_type.items(), key=lambda x: x[1], reverse=True)
        }
        return {
            "best_type": best_type,
            "best_type_ratio": round(best_weight / total_weight, 4),
            "non_abuse_ratio": round(non_abuse_weight / total_weight, 4),
            "weighted_risk": round(weighted_risk / total_weight, 2),
            "top_similarity": 1.0,
            "match_count": len(selected),
            "type_distribution": type_distribution,
            "matches": matches[:3],
            "exact_match": True,
        }