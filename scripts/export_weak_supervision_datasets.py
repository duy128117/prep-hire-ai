from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import get_standalone_db
from app.services.weak_supervision_dataset_service import (
    ANSWER_DATASET_COLUMNS,
    QUESTION_DATASET_COLUMNS,
    TURN_EVIDENCE_DATASET_COLUMNS,
    WeakSupervisionDatasetBuilder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export multi-LLM pseudo-label datasets for question ranking and answer scoring."
    )
    parser.add_argument(
        "--question-output",
        default="data/research/weak_supervision_v1_1/all/question_ranking_all.csv",
    )
    parser.add_argument(
        "--answer-output",
        default="data/research/weak_supervision_v1_1/all/turn_evidence_all.csv",
    )
    parser.add_argument("--prompt-version", default="balanced-weak-v1.1")
    return parser.parse_args()


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    db = get_standalone_db()
    def show_progress(dataset_name: str, row_count: int) -> None:
        print(f"exported {dataset_name} source rows: {row_count}", flush=True)

    try:
        question_rows, question_counters = WeakSupervisionDatasetBuilder.question_rows(
            db,
            prompt_version=args.prompt_version,
            progress_callback=show_progress,
        )
        answer_rows, answer_counters = WeakSupervisionDatasetBuilder.answer_rows(
            db,
            prompt_version=args.prompt_version,
            progress_callback=show_progress,
        )
    finally:
        db.close()

    question_path = Path(args.question_output)
    answer_path = Path(args.answer_output)
    _write_csv(question_path, QUESTION_DATASET_COLUMNS, question_rows)
    answer_columns = (
        TURN_EVIDENCE_DATASET_COLUMNS
        if args.prompt_version == "balanced-weak-v1.1"
        else ANSWER_DATASET_COLUMNS
    )
    _write_csv(answer_path, answer_columns, answer_rows)
    print(json.dumps({
        "question_output": str(question_path),
        "answer_output": str(answer_path),
        "prompt_version": args.prompt_version,
        **question_counters,
        **answer_counters,
    }, indent=2))


if __name__ == "__main__":
    main()
