from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.weak_supervision_training_service import (
    CandidateAnswerScorerTrainer,
    QuestionQualityRankerTrainer,
)
from app.services.weak_supervision_service import TURN_EVIDENCE_CRITERIA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two multi-LLM weak-supervision baseline models.")
    parser.add_argument(
        "--split-dir",
        default="data/research/weak_supervision_v1_1/splits",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/weak_supervision/v1_1",
    )
    parser.add_argument("--minimum-rows", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    artifact_dir = Path(args.artifact_dir)
    question_result = QuestionQualityRankerTrainer(
        minimum_rows=args.minimum_rows
    ).train_partition_csv(
        split_dir / "question_train.csv",
        split_dir / "question_validation.csv",
        split_dir / "question_test.csv",
        artifact_dir / "question_quality_ranker.joblib",
        artifact_dir / "question_quality_ranker_metrics.json",
    )
    answer_result = CandidateAnswerScorerTrainer(
        minimum_rows=args.minimum_rows,
        criteria=TURN_EVIDENCE_CRITERIA,
    ).train_partition_csv(
        split_dir / "answer_train.csv",
        split_dir / "answer_validation.csv",
        split_dir / "answer_test.csv",
        artifact_dir / "candidate_answer_scorer.joblib",
        artifact_dir / "candidate_answer_scorer_metrics.json",
    )
    print(json.dumps({
        "split_dir": str(split_dir),
        "artifact_dir": str(artifact_dir),
        "question_ranker": question_result.report,
        "answer_scorer": answer_result.report,
    }, indent=2))


if __name__ == "__main__":
    main()
