from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.question_ranking_v2_service import (
    PairwiseQuestionRankerTrainer,
    pairwise_ranking_metrics,
    read_question_csv,
)
from app.services.weak_supervision_training_service import (
    QuestionQualityRankerTrainer,
    _text_pipeline,
)


def _pointwise_metrics(
    fit_records: list[dict[str, str]],
    evaluation_records: list[dict[str, str]],
) -> dict[str, Any]:
    model = _text_pipeline()
    model.fit(
        [QuestionQualityRankerTrainer._combined_text(row) for row in fit_records],
        [float(row["mean_quality_score"]) for row in fit_records],
    )
    scores = model.predict([
        QuestionQualityRankerTrainer._combined_text(row)
        for row in evaluation_records
    ])
    return pairwise_ranking_metrics(evaluation_records, scores)


def _metric_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | None]:
    names = ("top_1_accuracy_tie_aware", "mrr", "pairwise_accuracy", "ndcg")
    return {
        name: (
            round(float(candidate[name]) - float(baseline[name]), 6)
            if baseline.get(name) is not None and candidate.get(name) is not None
            else None
        )
        for name in names
    }


def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return PairwiseQuestionRankerTrainer._selection_key(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark V1 pointwise and V2 pairwise rankers without split leakage."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/research/weak_supervision_v1_1/splits"),
    )
    parser.add_argument(
        "--v2-report",
        type=Path,
        default=Path(
            "artifacts/weak_supervision/v1_1/question_pairwise_ranker_v2_metrics.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/weak_supervision/v1_1/question_ranker_v1_vs_v2.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_records = read_question_csv(args.split_dir / "question_train.csv")
    validation_records = read_question_csv(
        args.split_dir / "question_validation.csv"
    )
    test_records = read_question_csv(args.split_dir / "question_test.csv")
    v2_report = json.loads(args.v2_report.read_text(encoding="utf-8"))

    v1_validation = _pointwise_metrics(train_records, validation_records)
    v1_test = _pointwise_metrics(train_records + validation_records, test_records)
    v2_validation = v2_report["selected_validation_metrics"]
    v2_test = v2_report["test_metrics"]
    promote_v2 = _selection_key(v2_validation) > _selection_key(v1_validation)
    output = {
        "selection_policy": (
            "Choose on validation only, ordered by tie-aware Top-1, MRR, "
            "pairwise accuracy, then NDCG. Test is reporting-only."
        ),
        "selected_model": "v2_pairwise" if promote_v2 else "v1_pointwise",
        "v2_promoted": promote_v2,
        "validation": {
            "v1_pointwise": v1_validation,
            "v2_pairwise": v2_validation,
            "v2_minus_v1": _metric_delta(v1_validation, v2_validation),
        },
        "held_out_test": {
            "v1_pointwise": v1_test,
            "v2_pairwise": v2_test,
            "v2_minus_v1": _metric_delta(v1_test, v2_test),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
