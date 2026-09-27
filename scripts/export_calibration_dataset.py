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
from app.core.config import settings
from app.services.research_dataset_service import CalibrationDatasetBuilder


CSV_COLUMNS = (
    "candidate_id",
    "job_id",
    "session_id",
    "expert_count",
    "cv_years_experience",
    "jd_years_required",
    "cv_skill_count",
    "jd_skill_count",
    "interview_answer_count",
    "interview_total_words",
    "llm_technical_knowledge",
    "llm_problem_solving",
    "llm_experience_evidence",
    "llm_communication",
    "llm_role_relevance",
    "llm_learning_adaptability",
    "llm_overall_readiness",
    "expert_consensus_technical_knowledge",
    "expert_consensus_problem_solving",
    "expert_consensus_experience_evidence",
    "expert_consensus_communication",
    "expert_consensus_role_relevance",
    "expert_consensus_learning_adaptability",
    "expert_consensus_overall_readiness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export consented expert-reference data for one LLM model.")
    parser.add_argument("--model-name", required=True, help="Exact model_name stored in candidate_evaluations")
    parser.add_argument(
        "--output",
        default="data/research/calibration_dataset.csv",
        help="Destination CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(settings.RESEARCH_PSEUDONYM_SALT) < 16:
        raise SystemExit("Set RESEARCH_PSEUDONYM_SALT to at least 16 characters before exporting research data.")
    db = get_standalone_db()
    try:
        rows, counters = CalibrationDatasetBuilder.collect_rows(
            db,
            args.model_name,
            settings.RESEARCH_PSEUDONYM_SALT,
        )
    finally:
        db.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"model_name": args.model_name, "output": str(output_path), **counters}, indent=2))
    if counters["anonymous_candidate_groups"]:
        print(
            "Warning: sessions without user_id are treated as different candidates. "
            "Use a stable pseudonymous user_id to prevent candidate leakage across train/test splits."
        )


if __name__ == "__main__":
    main()
