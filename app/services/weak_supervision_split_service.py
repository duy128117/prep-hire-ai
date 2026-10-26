from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class WeakSupervisionSplits:
    question_rows: dict[str, list[dict[str, Any]]]
    answer_rows: dict[str, list[dict[str, Any]]]
    group_ids: dict[str, tuple[str, ...]]


def _group_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    groups = {str(row.get("group_id", "")).strip() for row in rows}
    groups.discard("")
    return groups


def split_weak_supervision_rows(
    question_rows: Iterable[dict[str, Any]],
    answer_rows: Iterable[dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    random_state: int = 42,
) -> WeakSupervisionSplits:
    questions = list(question_rows)
    answers = list(answer_rows)
    if not questions or not answers:
        raise ValueError("Question and answer datasets must both contain rows")
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Train and validation ratios must be positive")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must leave room for a test split")

    question_groups = _group_ids(questions)
    answer_groups = _group_ids(answers)
    if question_groups != answer_groups:
        only_questions = sorted(question_groups - answer_groups)
        only_answers = sorted(answer_groups - question_groups)
        raise ValueError(
            "Question and answer datasets must contain the same context groups; "
            f"question_only={only_questions[:5]}, answer_only={only_answers[:5]}"
        )
    if len(question_groups) < 7:
        raise ValueError("At least 7 independent context groups are required for three splits")

    group_strata: dict[str, tuple[str, str]] = {}
    for row in questions:
        group_id = str(row["group_id"])
        stratum = (str(row.get("stage", "")), str(row.get("target_criterion", "")))
        existing = group_strata.setdefault(group_id, stratum)
        if existing != stratum:
            raise ValueError(
                f"Context group {group_id} contains multiple stage/criterion strata"
            )

    groups_by_stratum: dict[tuple[str, str], list[str]] = {}
    for group_id, stratum in group_strata.items():
        groups_by_stratum.setdefault(stratum, []).append(group_id)

    randomizer = random.Random(random_state)
    split_group_lists = {name: [] for name in SPLIT_NAMES}
    for stratum in sorted(groups_by_stratum):
        stratum_groups = sorted(groups_by_stratum[stratum])
        randomizer.shuffle(stratum_groups)
        stratum_count = len(stratum_groups)
        train_count = max(1, round(stratum_count * train_ratio))
        validation_count = max(1, round(stratum_count * validation_ratio))
        if train_count + validation_count >= stratum_count:
            validation_count = max(1, stratum_count - train_count - 1)
        if train_count + validation_count >= stratum_count:
            raise ValueError(
                f"Stratum {stratum} needs at least 3 context groups"
            )
        split_group_lists["train"].extend(stratum_groups[:train_count])
        split_group_lists["validation"].extend(
            stratum_groups[train_count:train_count + validation_count]
        )
        split_group_lists["test"].extend(
            stratum_groups[train_count + validation_count:]
        )

    group_ids = {
        name: tuple(sorted(values))
        for name, values in split_group_lists.items()
    }
    group_to_split = {
        group_id: split_name
        for split_name, groups in group_ids.items()
        for group_id in groups
    }

    def assign(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        split_rows = {name: [] for name in SPLIT_NAMES}
        for row in rows:
            group_id = str(row["group_id"])
            split_rows[group_to_split[group_id]].append(row)
        return split_rows

    return WeakSupervisionSplits(
        question_rows=assign(questions),
        answer_rows=assign(answers),
        group_ids=group_ids,
    )
