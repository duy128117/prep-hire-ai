from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.question_ranking_v2_service import read_question_csv


PAIRWISE_COLUMNS = (
    "group_id",
    "case_id",
    "stage",
    "target_criterion",
    "context_text",
    "preferred_output_id",
    "preferred_question",
    "preferred_quality_score",
    "rejected_output_id",
    "rejected_question",
    "rejected_quality_score",
    "quality_score_delta",
)


def build_pairwise_rows(
    records: list[dict[str, str]],
    margin: float,
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        grouped[str(row["group_id"])].append(row)

    pairs: list[dict[str, Any]] = []
    skipped_ties = 0
    for group_id in sorted(grouped):
        rows = grouped[group_id]
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                left = rows[left_index]
                right = rows[right_index]
                left_score = float(left["mean_quality_score"])
                right_score = float(right["mean_quality_score"])
                if abs(left_score - right_score) <= margin:
                    skipped_ties += 1
                    continue
                preferred, rejected = (
                    (left, right) if left_score > right_score else (right, left)
                )
                preferred_score = float(preferred["mean_quality_score"])
                rejected_score = float(rejected["mean_quality_score"])
                pairs.append({
                    "group_id": group_id,
                    "case_id": preferred.get("case_id", ""),
                    "stage": preferred.get("stage", ""),
                    "target_criterion": preferred.get("target_criterion", ""),
                    "context_text": preferred.get("context_text", ""),
                    "preferred_output_id": preferred.get("output_id", ""),
                    "preferred_question": preferred.get("question_text", ""),
                    "preferred_quality_score": preferred_score,
                    "rejected_output_id": rejected.get("output_id", ""),
                    "rejected_question": rejected.get("question_text", ""),
                    "rejected_quality_score": rejected_score,
                    "quality_score_delta": round(
                        preferred_score - rejected_score,
                        6,
                    ),
                })
    return pairs, skipped_ties


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRWISE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reproducible preferred/rejected question pairs for Ranker V2."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/research/weak_supervision_v1_1/splits"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/research/weak_supervision_v1_1/pairwise"),
    )
    parser.add_argument("--pair-margin", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        source = args.split_dir / f"question_{split_name}.csv"
        rows, skipped_ties = build_pairwise_rows(
            read_question_csv(source),
            args.pair_margin,
        )
        output = args.output_dir / f"question_pairs_{split_name}.csv"
        files[split_name] = {
            "source": str(source),
            "output": str(output),
            "pairs": len(rows),
            "skipped_tied_or_near_tied_pairs": skipped_ties,
            "sha256": _write_csv(output, rows),
        }
    manifest = {
        "schema_version": 1,
        "dataset_version": "weak-supervision-v1.1-question-pairs-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pair_margin": args.pair_margin,
        "pair_definition": "preferred question has the higher multi-LLM consensus score",
        "files": files,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
