"""
Data Preparation Script for TahananSafe AI
Prepares and processes datasets for fine-tuning the Qwen/Qwen2.5-0.5B-Instruct model.
"""

import json
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any
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
                # For negative dataset, mark as non-abuse
                output = """Incident Type: None / Invalid
Language Used: [Detected Language]
Risk Level: Low
Risk Percentage: 0%
Priority Level: Third Priority (P3)
Children Involved: No
Weapon Mentioned: No
AI Confidence Score: 95%

This report does not contain valid abuse-related content."""
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
