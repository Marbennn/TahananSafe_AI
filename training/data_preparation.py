"""
Data Preparation Script for TahananSafe AI
Prepares and processes datasets for fine-tuning the Qwen/Qwen2.5-0.5B-Instruct model.
"""

import json
import os
import math
import random
import re
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
        with open(config_path, 'r') as f:
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

    def _is_non_abuse_type(self, incident_type: Any) -> bool:
        """Check whether a label should be treated as non-abuse/negative."""
        if incident_type is None:
            return True
        text = str(incident_type).strip().lower()
        return text in self.non_abuse_labels

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
                if incident_type not in {"None / Invalid", "None / False Report"}:
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
