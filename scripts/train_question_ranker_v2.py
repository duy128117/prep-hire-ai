from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.question_ranking_v2_service import PairwiseQuestionRankerTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the V2 pairwise question ranker."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/research/weak_supervision_v1_1/splits"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/weak_supervision/v1_1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = PairwiseQuestionRankerTrainer().train_partition_csv(
        args.split_dir / "question_train.csv",
        args.split_dir / "question_validation.csv",
        args.split_dir / "question_test.csv",
        args.artifact_dir / "question_pairwise_ranker_v2.joblib",
        args.artifact_dir / "question_pairwise_ranker_v2_metrics.json",
    )
    print(json.dumps(result.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
