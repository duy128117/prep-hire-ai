from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.calibration_service import CandidateCalibrationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the candidate score calibration model from expert consensus labels."
    )
    parser.add_argument("--dataset", required=True, help="Expert-annotated CSV dataset")
    parser.add_argument(
        "--model-output",
        default="artifacts/candidate_calibrator.joblib",
        help="Path for the trained joblib artifact",
    )
    parser.add_argument(
        "--report-output",
        default="artifacts/candidate_calibration_metrics.json",
        help="Path for the evaluation report",
    )
    parser.add_argument("--minimum-rows", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = CandidateCalibrationTrainer(minimum_rows=args.minimum_rows)
    result = trainer.train_csv(args.dataset, args.model_output, args.report_output)
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()

