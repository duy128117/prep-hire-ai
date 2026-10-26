from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.question_ranking_v2_service import (
    PairwiseQuestionRanker,
    pairwise_ranking_metrics,
    read_question_csv,
)
from app.services.weak_supervision_training_service import QuestionQualityRankerTrainer


PREDICTION_COLUMNS = (
    "group_id",
    "case_id",
    "output_id",
    "stage",
    "target_criterion",
    "question_text",
    "generator_model_name",
    "reference_quality_score",
    "model_score",
    "reference_rank",
    "prediction_rank",
    "is_reference_top",
    "is_model_selected",
    "model_selected_reference_gap",
    "group_top_1_correct",
    "needs_human_review",
)
BLIND_REVIEW_COLUMNS = (
    "review_priority",
    "group_id",
    "blind_candidate_label",
    "stage",
    "target_criterion",
    "context_text",
    "question_text",
    "owner_selected_best",
    "owner_quality_score_1_to_5",
    "owner_error_category",
    "owner_notes",
)
REVIEW_KEY_COLUMNS = (
    "review_priority",
    "group_id",
    "blind_candidate_label",
    "output_id",
    "generator_model_name",
    "reference_quality_score",
    "model_score",
    "reference_rank",
    "prediction_rank",
    "is_reference_top",
    "is_model_selected",
)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _model_scores(
    artifact_path: Path,
    records: list[dict[str, str]],
) -> tuple[str, np.ndarray]:
    artifact = joblib.load(artifact_path)
    model_type = str(artifact.get("model_type") or "unknown")
    if model_type == "pairwise_question_quality_ranker_v2":
        return model_type, PairwiseQuestionRanker(artifact_path).score_rows(records)
    texts = [QuestionQualityRankerTrainer._combined_text(row) for row in records]
    lower, upper = artifact.get("score_range", (1.0, 5.0))
    scores = np.clip(artifact["model"].predict(texts), lower, upper)
    return model_type, np.asarray(scores, dtype=float)


def _stable_blind_order(group_id: str, row: dict[str, str]) -> str:
    identity = str(row.get("output_id") or row.get("question_text") or "")
    return hashlib.sha256(f"42:{group_id}:{identity}".encode("utf-8")).hexdigest()


def build_analysis(
    records: list[dict[str, str]],
    scores: np.ndarray,
    max_review_groups: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped_indices[str(row["group_id"])].append(index)

    prediction_rows: list[dict[str, Any]] = []
    review_candidates: list[tuple[float, str, list[int]]] = []
    correct_groups = 0
    for group_id, indices in grouped_indices.items():
        references = np.asarray(
            [float(records[index]["mean_quality_score"]) for index in indices],
            dtype=float,
        )
        estimates = np.asarray([float(scores[index]) for index in indices], dtype=float)
        reference_order = np.argsort(-references, kind="stable")
        prediction_order = np.argsort(-estimates, kind="stable")
        reference_rank = {
            int(candidate): rank
            for rank, candidate in enumerate(reference_order, start=1)
        }
        prediction_rank = {
            int(candidate): rank
            for rank, candidate in enumerate(prediction_order, start=1)
        }
        reference_maximum = float(np.max(references))
        selected_local = int(prediction_order[0])
        selected_gap = reference_maximum - float(references[selected_local])
        top_one_correct = bool(np.isclose(selected_gap, 0.0))
        correct_groups += int(top_one_correct)
        if not top_one_correct:
            review_candidates.append((selected_gap, group_id, indices))

        for local_index, record_index in enumerate(indices):
            row = records[record_index]
            prediction_rows.append({
                "group_id": group_id,
                "case_id": row.get("case_id", ""),
                "output_id": row.get("output_id", ""),
                "stage": row.get("stage", ""),
                "target_criterion": row.get("target_criterion", ""),
                "question_text": row.get("question_text", ""),
                "generator_model_name": row.get("generator_model_name", ""),
                "reference_quality_score": round(float(references[local_index]), 6),
                "model_score": round(float(estimates[local_index]), 6),
                "reference_rank": reference_rank[local_index],
                "prediction_rank": prediction_rank[local_index],
                "is_reference_top": bool(
                    np.isclose(references[local_index], reference_maximum)
                ),
                "is_model_selected": local_index == selected_local,
                "model_selected_reference_gap": round(float(selected_gap), 6),
                "group_top_1_correct": top_one_correct,
                "needs_human_review": row.get("needs_human_review", ""),
            })

    review_candidates.sort(key=lambda item: (-item[0], item[1]))
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    prediction_lookup = {
        str(row["output_id"]): row for row in prediction_rows
    }
    for priority, (_gap, group_id, indices) in enumerate(
        review_candidates[:max_review_groups],
        start=1,
    ):
        ordered_indices = sorted(
            indices,
            key=lambda index: _stable_blind_order(group_id, records[index]),
        )
        for label_index, record_index in enumerate(ordered_indices):
            row = records[record_index]
            blind_label = chr(ord("A") + label_index)
            prediction = prediction_lookup[str(row.get("output_id") or "")]
            blind_rows.append({
                "review_priority": priority,
                "group_id": group_id,
                "blind_candidate_label": blind_label,
                "stage": row.get("stage", ""),
                "target_criterion": row.get("target_criterion", ""),
                "context_text": row.get("context_text", ""),
                "question_text": row.get("question_text", ""),
                "owner_selected_best": "",
                "owner_quality_score_1_to_5": "",
                "owner_error_category": "",
                "owner_notes": "",
            })
            key_rows.append({
                "review_priority": priority,
                "group_id": group_id,
                "blind_candidate_label": blind_label,
                "output_id": row.get("output_id", ""),
                "generator_model_name": row.get("generator_model_name", ""),
                "reference_quality_score": prediction["reference_quality_score"],
                "model_score": prediction["model_score"],
                "reference_rank": prediction["reference_rank"],
                "prediction_rank": prediction["prediction_rank"],
                "is_reference_top": prediction["is_reference_top"],
                "is_model_selected": prediction["is_model_selected"],
            })

    summary = {
        "rows": len(records),
        "context_groups": len(grouped_indices),
        "correct_top_1_groups": correct_groups,
        "incorrect_top_1_groups": len(grouped_indices) - correct_groups,
        "review_groups": min(len(review_candidates), max_review_groups),
        "review_rows": len(blind_rows),
    }
    return prediction_rows, blind_rows, key_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export question-ranker errors and a blind owner-review queue."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/research/weak_supervision_v1_1/splits/question_validation.csv"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "artifacts/weak_supervision/v1_1/question_quality_ranker.joblib"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/research/weak_supervision_v1_1/review"),
    )
    parser.add_argument("--name", default="question_ranker_v1_validation")
    parser.add_argument("--max-review-groups", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_question_csv(args.dataset)
    model_type, scores = _model_scores(args.artifact, records)
    prediction_rows, blind_rows, key_rows, summary = build_analysis(
        records,
        scores,
        args.max_review_groups,
    )
    predictions_path = args.output_dir / f"{args.name}_predictions.csv"
    blind_path = args.output_dir / f"{args.name}_blind_review.csv"
    key_path = args.output_dir / f"{args.name}_review_key.csv"
    summary_path = args.output_dir / f"{args.name}_summary.json"
    _write_csv(predictions_path, PREDICTION_COLUMNS, prediction_rows)
    _write_csv(blind_path, BLIND_REVIEW_COLUMNS, blind_rows)
    _write_csv(key_path, REVIEW_KEY_COLUMNS, key_rows)
    output = {
        "model_type": model_type,
        "dataset": str(args.dataset),
        "artifact": str(args.artifact),
        "ranking_metrics": pairwise_ranking_metrics(records, scores),
        **summary,
        "predictions_output": str(predictions_path),
        "blind_review_output": str(blind_path),
        "review_key_output": str(key_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
