from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy.orm import selectinload
from sqlalchemy.exc import OperationalError

from app.db.session import get_standalone_db
from app.models.question_benchmark import QuestionBenchmarkOutput, QuestionBenchmarkRun
from app.models.weak_supervision import SyntheticAnswerSample
from app.services.weak_supervision_service import (
    QUESTION_AI_RUBRIC_VERSION,
    TURN_EVIDENCE_AI_RUBRIC_VERSION,
    TURN_EVIDENCE_CRITERIA,
)
from scripts.generate_weak_supervision_dataset import DATASET_PROMPT_VERSION


def _sorted_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): counter[key]
        for key in sorted(counter, key=lambda item: str(item))
    }


def main() -> None:
    db = get_standalone_db()
    try:
        for attempt in range(1, 4):
            try:
                outputs = (
                    db.query(QuestionBenchmarkOutput)
                    .options(
                        selectinload(QuestionBenchmarkOutput.ai_judgments),
                        selectinload(QuestionBenchmarkOutput.synthetic_answers).selectinload(
                            SyntheticAnswerSample.ai_judgments
                        ),
                    )
                    .join(QuestionBenchmarkRun)
                    .filter(QuestionBenchmarkRun.prompt_version == DATASET_PROMPT_VERSION)
                    .all()
                )
                break
            except OperationalError:
                db.rollback()
                if attempt >= 3:
                    raise
                time.sleep(attempt)
    finally:
        db.close()

    complete = 0
    samples: list[SyntheticAnswerSample] = []
    question_types: Counter[str] = Counter()
    difficulties: Counter[int] = Counter()
    generators: Counter[str] = Counter()
    quality_levels: Counter[int] = Counter()
    target_lengths: Counter[str] = Counter()
    actual_word_counts: list[int] = []
    length_compliance: Counter[bool] = Counter()
    applicable_counts: Counter[str] = Counter()
    missing_metadata = 0

    for output in outputs:
        question_metadata = output.generation_metadata_json or {}
        if not question_metadata:
            missing_metadata += 1
        question_types[str(question_metadata.get("question_type") or "missing")] += 1
        difficulties[int(question_metadata.get("difficulty") or 0)] += 1
        generators[output.provider] += 1
        for criterion in question_metadata.get("applicable_answer_criteria") or ():
            if criterion in TURN_EVIDENCE_CRITERIA:
                applicable_counts[criterion] += 1

        question_judge_count = len({
            judgment.judge_model_name
            for judgment in output.ai_judgments
            if judgment.rubric_version == QUESTION_AI_RUBRIC_VERSION
        })
        ready_answer = False
        for sample in output.synthetic_answers:
            metadata = sample.generation_metadata_json or {}
            samples.append(sample)
            quality_levels[int(sample.target_quality_level)] += 1
            target_lengths[str(metadata.get("target_length_band") or "missing")] += 1
            actual_word_counts.append(int(metadata.get("actual_word_count") or 0))
            length_compliance[bool(metadata.get("length_compliance"))] += 1
            answer_judge_count = len({
                judgment.judge_model_name
                for judgment in sample.ai_judgments
                if judgment.rubric_version == TURN_EVIDENCE_AI_RUBRIC_VERSION
            })
            ready_answer = ready_answer or answer_judge_count >= 3
        complete += int(question_judge_count >= 3 and ready_answer)

    report = {
        "dataset_prompt_version": DATASET_PROMPT_VERSION,
        "outputs": len(outputs),
        "complete_sets": complete,
        "answer_samples": len(samples),
        "question_metadata_missing": missing_metadata,
        "generator_counts": _sorted_counter(generators),
        "question_type_counts": _sorted_counter(question_types),
        "difficulty_counts": _sorted_counter(difficulties),
        "quality_level_counts": _sorted_counter(quality_levels),
        "target_length_counts": _sorted_counter(target_lengths),
        "length_compliance": _sorted_counter(length_compliance),
        "actual_word_count": {
            "minimum": min(actual_word_counts, default=0),
            "maximum": max(actual_word_counts, default=0),
            "mean": round(sum(actual_word_counts) / len(actual_word_counts), 2)
            if actual_word_counts
            else 0,
        },
        "applicable_criterion_counts": _sorted_counter(applicable_counts),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
