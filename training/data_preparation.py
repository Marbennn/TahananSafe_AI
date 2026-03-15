"""
Data Preparation Script for TahananSafe AI
Prepares and processes datasets for fine-tuning the Qwen/Qwen2.5-0.5B-Instruct model.
"""

import json
import os
import math
import random
import re
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter, defaultdict
from datasets import Dataset, DatasetDict
import pandas as pd


class DataPreparator:
    """Prepares datasets for training"""
    
    def __init__(self, config_path: str = "training/config.yaml"):
        """Initialize with configuration"""
        # Use utf-8-sig to safely read YAML files that may include BOM.
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            self.config = yaml.safe_load(f)
        
        self.abuse_types = self.config['abuse_types']
        self.languages = self.config['languages']
        self.risk_levels = self.config['risk_levels']
        self.priority_levels = self.config['priority_levels']
        self.non_abuse_labels = {
            "none / invalid",
            "none/invalid",
            "none / false report",
            "none/false report",
            "none / non-abuse report",
            "none/non-abuse report",
            "invalid",
            "none",
            "unknown",
        }
        self.community_report_types = [
            "Theft / Robbery",
            "Physical Altercation / Assault",
            "Community or Neighbor Disputes",
            "Public Disturbance",
            "Missing Person",
            "Property Damage / Vandalism",
            "Fraud / Scams",
            "Suspicious Activity",
            "Out-of-Scope Reports",
        ]
        self.community_report_types_lower = {x.lower() for x in self.community_report_types}
        # Domestic-only scope signals used to filter abuse rows before retraining.
        self.domestic_relationship_terms = {
            "asawa", "husband", "wife", "partner", "boyfriend", "girlfriend", "kinakasama",
            "live-in", "live in", "mag-asawa", "mag asawa", "pamilya", "family",
            "tatay", "nanay", "ama", "ina", "magulang", "parent", "parents",
            "anak", "child", "children", "minor", "stepfather", "stepmother",
            "lolo", "lola", "elder", "elderly", "senior", "senior citizen",
            "kapatid", "kuya", "ate", "brother", "sister", "tiyo", "tiya", "tiyuhin", "tiyahin",
        }
        self.household_context_terms = {
            "bahay", "loob ng bahay", "sa bahay", "bahay namin", "bahay nila",
            "kwarto", "tahanan", "home", "inside the house", "household",
        }
        self.child_terms = {
            "bata", "mga bata", "child", "children", "minor", "menor de edad", "sanggol", "baby", "anak",
        }
        self.elder_terms = {
            "lolo", "lola", "elder", "elderly", "senior", "senior citizen", "matanda", "bedridden",
        }
        self.abuse_action_terms = {
            "sinaktan", "sinasaktan", "nanakit", "nananakit",
            "sinuntok", "sinipa", "binugbog", "sinakal", "tinulak", "hinampas", "sinampal",
            "pinilit", "ginahasa", "nirape", "rape", "raped", "sexual",
            "pinagbantaan", "binantaan", "threat", "threatened",
            "pinapabayaan", "pinabayaan", "neglect", "neglected",
            "kinuha ang pera", "walang sustento", "hindi nagbibigay ng pera", "economic abuse",
        }

    @staticmethod
    def _normalize_whitespace(text: Any) -> str:
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text).strip())

    def _looks_like_noise_text(self, text: str) -> bool:
        """
        Detect low-information/noisy descriptions that can hurt training.
        Keep this conservative to avoid deleting valid reports.
        """
        t = self._normalize_whitespace(text).lower()
        if not t:
            return True

        # Single-char spam: "aaaaaaa", "kkkkkk", etc.
        if re.fullmatch(r"(.)\1{5,}", t):
            return True

        # Laughter/noise-only fragments.
        if re.fullmatch(r"(ha|he|hi|ho|hu|ah|oh|lol|lmao|h)+", re.sub(r"[^a-z]", "", t)):
            return True

        # Too little lexical signal after removing punctuation/digits.
        letters_only = re.sub(r"[^a-zA-Z\s]", " ", t)
        tokens = [tok for tok in letters_only.split() if tok]
        if len(tokens) <= 1 and len("".join(tokens)) <= 8:
            return True

        return False

    @staticmethod
    def _contains_keyword(text: str, keyword: str) -> bool:
        if not text or not keyword:
            return False
        pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
        return re.search(pattern, text.lower()) is not None

    def _count_keyword_hits(self, text: str, keywords: set[str]) -> int:
        if not text:
            return 0
        return sum(1 for kw in keywords if self._contains_keyword(text, kw))

    def _has_domestic_relationship_context(self, text: Any) -> bool:
        """
        Domestic scope check for abuse-labeled rows.
        A row is considered in-scope when it has clear family/intimate/household
        relationship context, including child/elder protection contexts.
        """
        t = self._normalize_whitespace(text).lower()
        if not t:
            return False

        rel_hits = self._count_keyword_hits(t, self.domestic_relationship_terms)
        household_hits = self._count_keyword_hits(t, self.household_context_terms)
        child_hits = self._count_keyword_hits(t, self.child_terms)
        elder_hits = self._count_keyword_hits(t, self.elder_terms)
        abuse_hits = self._count_keyword_hits(t, self.abuse_action_terms)

        if rel_hits > 0:
            return True
        if child_hits > 0 and (household_hits > 0 or abuse_hits > 0):
            return True
        if elder_hits > 0 and (household_hits > 0 or abuse_hits > 0):
            return True
        return False

    def _apply_domestic_scope_filters(
        self,
        main_data: List[Dict[str, Any]],
        negative_data: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Enforce domestic-only training scope:
        - Main dataset: drop abuse-labeled rows without domestic/household context.
        - Negative dataset: optionally drop rows that are still abuse-labeled.
        """
        ds_cfg = self.config.get("dataset", {})
        enabled = self._to_bool(ds_cfg.get("domestic_scope_filter_enabled", True), default=True)
        if not enabled:
            return main_data, negative_data

        drop_negative_abuse_rows = self._to_bool(ds_cfg.get("drop_negative_abuse_rows", True), default=True)

        filtered_main: List[Dict[str, Any]] = []
        removed_main_non_domestic = 0
        for row in main_data:
            label = self._canonical_abuse_type(row.get("incident_type"))
            if label in self.abuse_types:
                if not self._has_domestic_relationship_context(row.get("incident_description")):
                    removed_main_non_domestic += 1
                    continue
            filtered_main.append(row)

        filtered_negative: List[Dict[str, Any]] = []
        removed_negative_abuse = 0
        for row in negative_data:
            label = self._canonical_abuse_type(row.get("incident_type"))
            if drop_negative_abuse_rows and label in self.abuse_types:
                removed_negative_abuse += 1
                continue
            filtered_negative.append(row)

        print(
            "Domestic scope filter: "
            f"removed main_non_domestic_abuse={removed_main_non_domestic}, "
            f"removed_negative_abuse_labels={removed_negative_abuse}; "
            f"kept_main={len(filtered_main)}, kept_negative={len(filtered_negative)}"
        )
        return filtered_main, filtered_negative

    def _apply_data_quality_guards(
        self,
        data: List[Dict[str, Any]],
        source_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Apply conservative safeguards to reduce overfitting and false confidence:
        - drop empty/too-short/noisy descriptions
        - cap duplicate rows by (incident_type, normalized_description)
        """
        ds_cfg = self.config.get("dataset", {})
        enabled = self._to_bool(ds_cfg.get("quality_filters_enabled", True), default=True)
        if not enabled:
            return data

        min_len = max(1, int(ds_cfg.get("min_description_length", 12)))
        drop_noise = self._to_bool(ds_cfg.get("drop_obvious_noise_rows", True), default=True)
        max_dup = max(1, int(ds_cfg.get("max_duplicates_per_type_description", 1)))

        filtered: List[Dict[str, Any]] = []
        removed_empty = 0
        removed_short = 0
        removed_noise = 0

        for row in data:
            desc = self._normalize_whitespace(row.get("incident_description"))
            if not desc:
                removed_empty += 1
                continue
            if len(desc) < min_len:
                removed_short += 1
                continue
            if drop_noise and self._looks_like_noise_text(desc):
                removed_noise += 1
                continue

            updated = dict(row)
            updated["incident_description"] = desc
            filtered.append(updated)

        counts: Dict[tuple[str, str], int] = defaultdict(int)
        deduped: List[Dict[str, Any]] = []
        removed_dups = 0
        for row in filtered:
            incident_type = self._canonical_abuse_type(row.get("incident_type"))
            desc_key = self._normalize_description_for_leakage(row.get("incident_description"))
            key = (incident_type, desc_key)
            counts[key] += 1
            if counts[key] > max_dup:
                removed_dups += 1
                continue
            deduped.append(row)

        print(
            f"Quality filter ({source_name}): "
            f"removed empty={removed_empty}, short={removed_short}, noise={removed_noise}, duplicates={removed_dups}; "
            f"kept={len(deduped)}"
        )
        return deduped

    def _is_non_abuse_type(self, incident_type: Any) -> bool:
        """Check whether a label should be treated as non-abuse/negative."""
        if incident_type is None:
            return True
        text = str(incident_type).strip().lower()
        return text in self.non_abuse_labels or text in self.community_report_types_lower

    def _canonical_abuse_type(self, incident_type: Any) -> str:
        """Map raw incident type text into known core abuse labels when possible."""
        if incident_type is None:
            return "Unknown"

        text = str(incident_type).strip()
        if not text:
            return "Unknown"
        lower = text.lower()

        if lower in self.non_abuse_labels:
            return "None / Invalid"

        for label in self.community_report_types:
            if lower == label.lower():
                return label
        for label in self.community_report_types:
            if label.lower() in lower:
                return label

        for label in self.abuse_types:
            if lower == label.lower():
                return label
        for label in self.abuse_types:
            if label.lower() in lower:
                return label
        return text

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _max_min_ratio(counter: Dict[str, int]) -> float:
        vals = [v for v in counter.values() if v > 0]
        if len(vals) <= 1:
            return 1.0
        return float(max(vals) / min(vals))

    def _core_label_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {label: 0 for label in self.abuse_types}
        for ex in data:
            label = self._canonical_abuse_type(ex.get("incident_type"))
            if label in counts:
                counts[label] += 1
        return counts

    def _apply_class_balancing(self, main_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Oversample minority core abuse classes so max/min ratio is bounded.
        Only applies to abuse-labeled rows in the main dataset.
        """
        ds_cfg = self.config.get("dataset", {})
        enabled = self._to_bool(ds_cfg.get("class_balance_enabled", True), default=True)
        if not enabled:
            print("Class balancing disabled by config.")
            return main_data

        target_ratio = float(ds_cfg.get("class_balance_target_ratio", 3.0))
        target_ratio = max(1.0, target_ratio)
        seed = int(ds_cfg.get("class_balance_seed", 42))
        rng = random.Random(seed)

        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ex in main_data:
            label = self._canonical_abuse_type(ex.get("incident_type"))
            if label in self.abuse_types:
                groups[label].append(ex)

        if not groups:
            print("No core abuse classes found for balancing. Skipping.")
            return main_data

        before_counts = self._core_label_distribution(main_data)
        nonzero_before = {k: v for k, v in before_counts.items() if v > 0}
        if not nonzero_before:
            print("No non-empty class counts found for balancing. Skipping.")
            return main_data

        max_count = max(nonzero_before.values())
        target_min_count = int(math.ceil(max_count / target_ratio))

        extras: List[Dict[str, Any]] = []
        for label in self.abuse_types:
            current = len(groups.get(label, []))
            if current <= 0:
                print(f"Warning: class '{label}' has 0 samples; cannot oversample from empty class.")
                continue
            desired = max(current, target_min_count)
            needed = desired - current
            if needed <= 0:
                continue
            base = groups[label]
            for _ in range(needed):
                sample = dict(rng.choice(base))
                sample["_balanced_oversample"] = True
                extras.append(sample)

        balanced_data = list(main_data) + extras
        after_counts = self._core_label_distribution(balanced_data)

        print("Class balancing summary (core abuse classes):")
        print(f"  Target max/min ratio: {target_ratio}")
        print(f"  Added oversampled rows: {len(extras)}")
        print(f"  Before counts: {before_counts}")
        print(f"  After counts : {after_counts}")
        print(f"  Before ratio : {round(self._max_min_ratio(before_counts), 3)}")
        print(f"  After ratio  : {round(self._max_min_ratio(after_counts), 3)}")
        return balanced_data

    @staticmethod
    def _normalize_description_for_leakage(text: Any) -> str:
        """Normalize descriptions for exact leakage filtering."""
        if text is None:
            return ""
        cleaned = re.sub(r"\s+", " ", str(text).strip().lower())
        return cleaned

    def _collect_unseen_description_set(self) -> set[str]:
        """
        Load optional unseen CSVs and collect normalized descriptions
        to exclude from training preparation.
        """
        ds_cfg = self.config.get("dataset", {})
        enabled = self._to_bool(ds_cfg.get("exclude_unseen_from_training", True), default=True)
        if not enabled:
            print("Unseen exclusion disabled by config.")
            return set()

        unseen_paths = [
            ds_cfg.get("unseen_main_path", ""),
            ds_cfg.get("unseen_negative_path", ""),
        ]
        unseen_descs: set[str] = set()

        for raw in unseen_paths:
            path_str = str(raw).strip() if raw is not None else ""
            if not path_str:
                continue
            path = Path(path_str)
            if not path.exists():
                continue

            try:
                rows = self.load_dataset_files(path_str)
            except Exception as e:
                print(f"Warning: failed to read unseen dataset {path_str}: {e}")
                continue

            for row in rows:
                norm = self._normalize_description_for_leakage(row.get("incident_description"))
                if norm:
                    unseen_descs.add(norm)

        if unseen_descs:
            print(f"Loaded {len(unseen_descs)} unique unseen descriptions for leakage filtering.")
        else:
            print("No unseen descriptions loaded (unseen files missing/empty).")
        return unseen_descs

    def _exclude_by_unseen_descriptions(
        self,
        data: List[Dict[str, Any]],
        unseen_descs: set[str],
        source_name: str,
    ) -> List[Dict[str, Any]]:
        """Filter out rows whose descriptions appear in unseen holdout CSVs."""
        if not unseen_descs:
            return data
        kept: List[Dict[str, Any]] = []
        removed = 0
        for row in data:
            desc_norm = self._normalize_description_for_leakage(row.get("incident_description"))
            if desc_norm and desc_norm in unseen_descs:
                removed += 1
                continue
            kept.append(row)
        print(f"Excluded {removed} {source_name} rows due to unseen holdout overlap.")
        return kept
        
    def _load_from_json_dir(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Load all JSON/JSONL files from a directory"""
        data: List[Dict[str, Any]] = []

        for file_path in dataset_path.glob("*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    data.extend(content)
                else:
                    data.append(content)

        for file_path in dataset_path.glob("*.jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))

        return data

    def _normalize_priority(self, value: Any) -> str:
        """Normalize priority codes (P1/P2/P3) to full labels."""
        if value is None:
            return "Third Priority (P3)"

        text = str(value).strip()
        mapping = {
            "P1": "First Priority (P1)",
            "P2": "Second Priority (P2)",
            "P3": "Third Priority (P3)",
        }

        # Already in verbose form
        if text in self.priority_levels:
            return text

        return mapping.get(text, "Third Priority (P3)")

    def _load_from_csv_file(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Load dataset from a single CSV file."""
        df = pd.read_csv(csv_path)
        records: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            rec = {
                "incident_type": row.get("Incident_Type"),
                "incident_description": row.get("Incident_Description"),
                "language": row.get("Language"),
                "risk_level": row.get("Risk_Level"),
                "risk_percentage": row.get("Incident_Risk_Percentage"),
                "priority_level": self._normalize_priority(row.get("Priority_Level")),
                "children_involved": str(row.get("Children_Involved"))
                .strip()
                .lower()
                in {"yes", "y", "true", "1"},
                "weapon_mentioned": str(row.get("Weapon_Mentioned"))
                .strip()
                .lower()
                in {"yes", "y", "true", "1"},
                "confidence_score": row.get("AI_Confidence_Score"),
            }
            records.append(rec)

        return records

    def load_dataset_files(self, dataset_path_str: str) -> List[Dict[str, Any]]:
        """
        Load dataset from either:
        - a directory of JSON/JSONL files, or
        - a single CSV file (Main_Dataset.csv / Negative_Dataset.csv).
        """
        dataset_path = Path(dataset_path_str)

        if dataset_path.is_file() and dataset_path.suffix.lower() == ".csv":
            return self._load_from_csv_file(dataset_path)
        elif dataset_path.is_dir():
            return self._load_from_json_dir(dataset_path)
        else:
            raise FileNotFoundError(
                f"Dataset path not found or unsupported format: {dataset_path_str}"
            )
    
    def create_prompt_template(self, example: Dict[str, Any]) -> str:
        """Create a prompt template for fine-tuning"""
        incident_desc = example.get('incident_description', '')
        
        prompt = f"""Analyze the following incident report and provide structured outputs.

Incident Description: {incident_desc}

Required Analysis:
1. Incident Type: {example.get('incident_type', '')}
2. Language Used: {example.get('language', '')}
3. Risk Level: {example.get('risk_level', '')}
4. Risk Percentage: {example.get('risk_percentage', 0)}%
5. Priority Level: {example.get('priority_level', '')}
6. Children Involved: {'Yes' if example.get('children_involved', False) else 'No'}
7. Weapon Mentioned: {'Yes' if example.get('weapon_mentioned', False) else 'No'}
8. AI Confidence Score: {example.get('confidence_score', 0)}%

Analysis complete."""
        
        return prompt
    
    def format_for_training(self, data: List[Dict[str, Any]], is_negative: bool = False) -> List[Dict[str, str]]:
        """Format data for training"""
        formatted_data = []
        
        for example in data:
            # Create input prompt
            incident_desc = example.get('incident_description', '')
            
            if is_negative:
                # Keep negative rows realistic/varied so the model learns robust
                # non-abuse behavior instead of one rigid template.
                incident_type = self._canonical_abuse_type(example.get("incident_type"))
                allowed_negative_types = set(self.community_report_types) | {"None / Invalid", "None / False Report"}
                if incident_type not in allowed_negative_types:
                    incident_type = "None / Invalid"
                language = example.get("language", "English")
                risk_level = example.get("risk_level", "Low")
                risk_percentage = example.get("risk_percentage", 0)
                priority_level = self._normalize_priority(example.get("priority_level"))
                children_involved = "Yes" if example.get("children_involved", False) else "No"
                weapon_mentioned = "Yes" if example.get("weapon_mentioned", False) else "No"
                confidence_score = example.get("confidence_score", 90)

                output = f"""Incident Type: {incident_type}
Language Used: {language}
Risk Level: {risk_level}
Risk Percentage: {risk_percentage}%
Priority Level: {priority_level}
Children Involved: {children_involved}
Weapon Mentioned: {weapon_mentioned}
AI Confidence Score: {confidence_score}%"""
            else:
                # Create structured output
                incident_type = example.get('incident_type', 'Unknown')
                language = example.get('language', 'English')
                risk_level = example.get('risk_level', 'Low')
                risk_percentage = example.get('risk_percentage', 0)
                priority_level = example.get('priority_level', 'Third Priority (P3)')
                children_involved = 'Yes' if example.get('children_involved', False) else 'No'
                weapon_mentioned = 'Yes' if example.get('weapon_mentioned', False) else 'No'
                confidence_score = example.get('confidence_score', 85)
                
                output = f"""Incident Type: {incident_type}
Language Used: {language}
Risk Level: {risk_level}
Risk Percentage: {risk_percentage}%
Priority Level: {priority_level}
Children Involved: {children_involved}
Weapon Mentioned: {weapon_mentioned}
AI Confidence Score: {confidence_score}%"""
            
            # Create training prompt
            input_text = f"Analyze this incident report:\n\n{incident_desc}\n\nProvide structured analysis:"
            target_text = output
            
            formatted_data.append({
                'input': input_text,
                'output': target_text,
                'text': f"{input_text}\n\n{target_text}",
                'metadata': json.dumps(example)
            })
        
        return formatted_data
    
    def prepare_datasets(self):
        """Main method to prepare all datasets"""
        print("Loading main dataset...")
        main_data = self.load_dataset_files(self.config['dataset']['main_dataset_path'])
        print(f"Loaded {len(main_data)} examples from main dataset")
        
        print("Loading negative dataset...")
        negative_data = self.load_dataset_files(self.config['dataset']['negative_dataset_path'])
        print(f"Loaded {len(negative_data)} examples from negative dataset")

        extra_negative_paths = self.config.get("dataset", {}).get("extra_negative_dataset_paths", [])
        if isinstance(extra_negative_paths, str):
            extra_negative_paths = [extra_negative_paths]
        if isinstance(extra_negative_paths, list):
            for raw_path in extra_negative_paths:
                path = str(raw_path).strip()
                if not path:
                    continue
                p = Path(path)
                if not p.exists():
                    print(f"Warning: extra_negative_dataset_paths file not found: {path}")
                    continue
                extra_rows = self.load_dataset_files(path)
                negative_data.extend(extra_rows)
                print(f"Loaded {len(extra_rows)} extra negative/community examples from {path}")

        # Optional hard contrastive dataset for ambiguous wording.
        ambiguous_path = self.config["dataset"].get("ambiguous_pairs_path", "").strip()
        if ambiguous_path:
            ambiguous_file = Path(ambiguous_path)
            if ambiguous_file.exists():
                print("Loading ambiguous pairs dataset...")
                ambiguous_data = self.load_dataset_files(ambiguous_path)
                ambiguous_main = [x for x in ambiguous_data if not self._is_non_abuse_type(x.get("incident_type"))]
                ambiguous_negative = [x for x in ambiguous_data if self._is_non_abuse_type(x.get("incident_type"))]
                main_data.extend(ambiguous_main)
                negative_data.extend(ambiguous_negative)
                print(
                    f"Loaded {len(ambiguous_data)} ambiguous pairs "
                    f"({len(ambiguous_main)} abuse, {len(ambiguous_negative)} non-abuse)"
                )
            else:
                print(f"Warning: ambiguous_pairs_path not found: {ambiguous_path}")

        # Step 4: exclude explicit unseen holdout rows from training data.
        unseen_descs = self._collect_unseen_description_set()
        if unseen_descs:
            main_data = self._exclude_by_unseen_descriptions(main_data, unseen_descs, "main")
            negative_data = self._exclude_by_unseen_descriptions(negative_data, unseen_descs, "negative")

        # Data quality safeguards to prevent overfitting on repeated/noisy rows.
        main_data = self._apply_data_quality_guards(main_data, "main")
        negative_data = self._apply_data_quality_guards(negative_data, "negative")
        main_data, negative_data = self._apply_domestic_scope_filters(main_data, negative_data)

        # Step 3: rebalance core abuse classes before formatting/splitting.
        main_data = self._apply_class_balancing(main_data)
        
        # Format data
        print("Formatting main dataset...")
        formatted_main = self.format_for_training(main_data, is_negative=False)
        
        print("Formatting negative dataset...")
        formatted_negative = self.format_for_training(negative_data, is_negative=True)
        
        # Combine datasets
        all_data = formatted_main + formatted_negative
        print(f"Total training examples: {len(all_data)}")
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(all_data)
        
        # Split dataset
        train_test = dataset.train_test_split(
            test_size=self.config['dataset']['test_split'] + self.config['dataset']['val_split']
        )
        
        val_test = train_test['test'].train_test_split(
            test_size=self.config['dataset']['test_split'] / 
                     (self.config['dataset']['test_split'] + self.config['dataset']['val_split'])
        )
        
        dataset_dict = DatasetDict({
            'train': train_test['train'],
            'validation': val_test['train'],
            'test': val_test['test']
        })
        
        # Save processed dataset
        output_path = Path(self.config['dataset']['processed_path'])
        # Windows can throw OSError 22 when save_to_disk overwrites an existing
        # dataset directory in place. Remove previous generated artifacts first.
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving processed dataset to {output_path}...")
        dataset_dict.save_to_disk(str(output_path))
        
        print(f"Dataset splits:")
        print(f"  Train: {len(dataset_dict['train'])} examples")
        print(f"  Validation: {len(dataset_dict['validation'])} examples")
        print(f"  Test: {len(dataset_dict['test'])} examples")
        
        return dataset_dict


if __name__ == "__main__":
    preparator = DataPreparator()
    dataset_dict = preparator.prepare_datasets()
    print("Data preparation complete!")
