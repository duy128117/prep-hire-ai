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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two multi-LLM weak-supervision baseline models.")
    parser.add_argument(
        "--question-dataset",
        default="data/research/question_ranking_dataset.csv",
    )
    parser.add_argument(
        "--answer-dataset",
        default="data/research/answer_scoring_dataset.csv",
    )
    parser.add_argument("--minimum-rows", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_result = QuestionQualityRankerTrainer(minimum_rows=args.minimum_rows).train_csv(
        args.question_dataset,
        "data/research/question_quality_ranker.joblib",
        "data/research/question_quality_ranker_metrics.json",
    )
    answer_result = CandidateAnswerScorerTrainer(minimum_rows=args.minimum_rows).train_csv(
        args.answer_dataset,
        "data/research/candidate_answer_scorer.joblib",
        "data/research/candidate_answer_scorer_metrics.json",
    )
    print(json.dumps({
        "question_ranker": question_result.report,
        "answer_scorer": answer_result.report,
    }, indent=2))


if __name__ == "__main__":
    main()
