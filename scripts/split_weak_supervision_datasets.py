from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.weak_supervision_split_service import (
    SPLIT_NAMES,
    split_weak_supervision_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic context-grouped train, validation, and test files."
    )
    parser.add_argument(
        "--question-dataset",
        default="data/research/weak_supervision_v1_1/all/question_ranking_all.csv",
    )
    parser.add_argument(
        "--answer-dataset",
        default="data/research/weak_supervision_v1_1/all/turn_evidence_all.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/research/weak_supervision_v1_1/splits",
    )
    parser.add_argument(
        "--manifest",
        default="data/research/weak_supervision_v1_1/manifest.json",
    )
    parser.add_argument("--dataset-version", default="weak-supervision-v1.1")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    question_source = Path(args.question_dataset)
    answer_source = Path(args.answer_dataset)
    question_columns, question_rows = _read_csv(question_source)
    answer_columns, answer_rows = _read_csv(answer_source)
    splits = split_weak_supervision_rows(
        question_rows,
        answer_rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        random_state=args.random_state,
    )

    output_dir = Path(args.output_dir)
    files: dict[str, dict[str, Any]] = {}
    for dataset_name, columns, rows_by_split in (
        ("question", question_columns, splits.question_rows),
        ("answer", answer_columns, splits.answer_rows),
    ):
        for split_name in SPLIT_NAMES:
            path = output_dir / f"{dataset_name}_{split_name}.csv"
            rows = rows_by_split[split_name]
            _write_csv(path, columns, rows)
            files[f"{dataset_name}_{split_name}"] = {
                "path": str(path),
                "rows": len(rows),
                "groups": len(splits.group_ids[split_name]),
                "sha256": _sha256(path),
            }

    group_sets = {name: set(values) for name, values in splits.group_ids.items()}
    manifest = {
        "schema_version": 1,
        "dataset_version": args.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": args.random_state,
        "split_strategy": (
            "deterministic_group_by_benchmark_case_stratified_by_stage_and_criterion"
        ),
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": round(1 - args.train_ratio - args.validation_ratio, 10),
        },
        "sources": {
            "question": {
                "path": str(question_source),
                "rows": len(question_rows),
                "sha256": _sha256(question_source),
            },
            "answer": {
                "path": str(answer_source),
                "rows": len(answer_rows),
                "sha256": _sha256(answer_source),
            },
        },
        "context_groups": {
            name: len(values) for name, values in splits.group_ids.items()
        },
        "group_overlap": {
            "train_validation": len(group_sets["train"] & group_sets["validation"]),
            "train_test": len(group_sets["train"] & group_sets["test"]),
            "validation_test": len(group_sets["validation"] & group_sets["test"]),
        },
        "files": files,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
