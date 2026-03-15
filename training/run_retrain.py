"""
Retraining helper:
1) Prepare datasets using the selected config.
2) Run LoRA fine-tuning with the same config.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from training.data_preparation import DataPreparator
from training.train import IncidentReportTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets and retrain model.")
    parser.add_argument(
        "--config",
        default="config_retrain.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    preparator = DataPreparator(args.config)
    preparator.prepare_datasets()

    trainer = IncidentReportTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()


