from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models.interview_evidence_summary import InterviewEvidenceSummary
from app.models.interview_session import InterviewSession


AGGREGATOR_VERSION = "interview-evidence-v1"
BASE_COMPETENCIES = (
    "technical_knowledge",
    "problem_solving",
    "experience_evidence",
    "communication",
    "role_relevance",
    "learning_adaptability",
)
TURN_SCORE_WEIGHTS = {
    "answer_relevance": 0.15,
    "technical_correctness": 0.20,
    "expected_signal_coverage": 0.20,
    "reasoning_quality": 0.15,
    "evidence_quality": 0.15,
    "clarity_conciseness": 0.075,
    "consistency": 0.075,
}
COMPETENCY_SIGNALS = {
    "technical_knowledge": {
        "technical_correctness": 0.40,
        "expected_signal_coverage": 0.25,
        "reasoning_quality": 0.20,
        "answer_relevance": 0.10,
        "consistency": 0.05,
    },
    "problem_solving": {
        "reasoning_quality": 0.35,
        "technical_correctness": 0.25,
        "expected_signal_coverage": 0.20,
        "evidence_quality": 0.10,
        "consistency": 0.10,
    },
    "experience_evidence": {
        "evidence_quality": 0.40,
        "expected_signal_coverage": 0.25,
        "answer_relevance": 0.15,
        "consistency": 0.15,
        "clarity_conciseness": 0.05,
    },
    "communication": {
        "clarity_conciseness": 0.45,
        "answer_relevance": 0.30,
        "consistency": 0.15,
        "expected_signal_coverage": 0.10,
    },
    "role_relevance": {
        "expected_signal_coverage": 0.40,
        "answer_relevance": 0.30,
        "evidence_quality": 0.15,
        "consistency": 0.15,
    },
    "learning_adaptability": {
        "reasoning_quality": 0.30,
        "evidence_quality": 0.30,
        "expected_signal_coverage": 0.20,
        "answer_relevance": 0.10,
        "consistency": 0.10,
    },
}
OVERALL_WEIGHTS = {
    "technical_knowledge": 0.20,
    "problem_solving": 0.20,
    "experience_evidence": 0.15,
    "communication": 0.15,
    "role_relevance": 0.15,
    "learning_adaptability": 0.15,
}


class InterviewEvidenceAggregator:
    @staticmethod
    def _valid_scores(turn: Any) -> tuple[dict[str, float], set[str]] | None:
        payload = turn.answer_scores_json or {}
        if turn.answer_text is None or payload.get("status") != "success":
            return None
        raw_scores = payload.get("scores") or {}
        scores: dict[str, float] = {}
        for name, raw_value in raw_scores.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and 1 <= value <= 5:
                scores[str(name)] = value
        applicable = {
            str(item) for item in payload.get("applicable_criteria") or []
        }
        return (scores, applicable) if scores and applicable else None

    @classmethod
    def _turn_competency_score(cls, turn: Any, competency: str) -> float | None:
        parsed = cls._valid_scores(turn)
        if not parsed:
            return None
        scores, applicable = parsed
        weights = COMPETENCY_SIGNALS.get(competency, TURN_SCORE_WEIGHTS)
        usable = {
            name: weight
            for name, weight in weights.items()
            if name in applicable and name in scores
        }
        total_weight = sum(usable.values())
        if not total_weight:
            return None
        return round(
            sum(scores[name] * weight for name, weight in usable.items())
            / total_weight,
            3,
        )

    @staticmethod
    def _eligible(turn: Any, competency: str) -> tuple[bool, float, str]:
        if turn.criterion == competency:
            return True, 1.0, "direct"
        if competency == "communication":
            return True, 0.6, "cross_turn_communication"
        if competency == "role_relevance":
            return True, 0.6, "cross_turn_role_relevance"
        return False, 0.0, "not_applicable"

    @classmethod
    def _aggregate_competency(
        cls,
        turns: list[Any],
        competency: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        evidence: list[dict[str, Any]] = []
        for turn in turns:
            eligible, evidence_weight, source = cls._eligible(turn, competency)
            if not eligible:
                continue
            score = cls._turn_competency_score(turn, competency)
            if score is None:
                continue
            payload = turn.answer_scores_json or {}
            scores = payload.get("scores") or {}
            quality = scores.get("evidence_quality")
            evidence.append(
                {
                    "turn_index": turn.turn_index,
                    "stage": turn.stage,
                    "target_criterion": turn.criterion,
                    "source": source,
                    "weight": evidence_weight,
                    "score": score,
                    "evidence_quality": quality,
                    "strong": score >= 4.0 and (
                        quality is None or float(quality) >= 3.5
                    ),
                }
            )

        if not evidence:
            return (
                {
                    "score": None,
                    "confidence": 0.0,
                    "status": "insufficient_evidence",
                    "evidence_count": 0,
                    "evidence_turns": [],
                    "strong_evidence_turns": [],
                },
                [],
                [
                    {
                        "type": "insufficient_evidence",
                        "competency": competency,
                        "message": f"No usable turn evidence for {competency}.",
                    }
                ],
            )

        total_weight = sum(item["weight"] for item in evidence)
        score = sum(item["score"] * item["weight"] for item in evidence) / total_weight
        values = [float(item["score"]) for item in evidence]
        spread = max(values) - min(values)
        deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
        coverage_target = 2.0
        coverage = min(1.0, total_weight / coverage_target)
        agreement = max(0.0, 1.0 - deviation / 2.0)
        quality_values = [
            float(item["evidence_quality"])
            for item in evidence
            if item["evidence_quality"] is not None
        ]
        quality = (
            (statistics.fmean(quality_values) - 1.0) / 4.0
            if quality_values
            else 0.5
        )
        confidence = max(0.0, min(1.0, 0.5 * coverage + 0.3 * agreement + 0.2 * quality))
        direct_count = sum(item["source"] == "direct" for item in evidence)
        status = (
            "sufficient"
            if confidence >= 0.6 and (direct_count >= 2 or competency in {"communication", "role_relevance"})
            else "limited"
        )
        flags: list[dict[str, Any]] = []
        if spread > 1.5:
            flags.append(
                {
                    "type": "score_divergence",
                    "competency": competency,
                    "turns": [item["turn_index"] for item in evidence],
                    "spread": round(spread, 3),
                    "message": "Turn evidence varies substantially for this competency.",
                }
            )
        return (
            {
                "score": round(score, 3),
                "confidence": round(confidence, 3),
                "status": status,
                "evidence_count": len(evidence),
                "direct_evidence_count": direct_count,
                "evidence_turns": [item["turn_index"] for item in evidence],
                "strong_evidence_turns": [
                    item["turn_index"] for item in evidence if item["strong"]
                ],
            },
            evidence,
            flags,
        )

    @staticmethod
    def _consistency_flags(turns: list[Any]) -> list[dict[str, Any]]:
        flags: list[dict[str, Any]] = []
        for turn in turns:
            payload = turn.answer_scores_json or {}
            consistency = (payload.get("scores") or {}).get("consistency")
            if consistency is not None and float(consistency) < 2.5:
                flags.append(
                    {
                        "type": "possible_inconsistency",
                        "turns": [turn.turn_index],
                        "score": round(float(consistency), 3),
                        "message": "This turn has a low predicted consistency score.",
                    }
                )
        return flags

    @classmethod
    def aggregate(cls, session: InterviewSession) -> dict[str, Any]:
        answered_turns = [turn for turn in session.turns if turn.answer_text is not None]
        competencies: dict[str, dict[str, Any]] = {}
        evidence: dict[str, list[dict[str, Any]]] = {}
        flags = cls._consistency_flags(answered_turns)
        for competency in BASE_COMPETENCIES:
            result, competency_evidence, competency_flags = cls._aggregate_competency(
                answered_turns,
                competency,
            )
            competencies[competency] = result
            evidence[competency] = competency_evidence
            flags.extend(competency_flags)

        available = {
            name: result
            for name, result in competencies.items()
            if result["score"] is not None
        }
        if available:
            available_weight = sum(OVERALL_WEIGHTS[name] for name in available)
            overall_score = sum(
                float(result["score"]) * OVERALL_WEIGHTS[name]
                for name, result in available.items()
            ) / available_weight
            confidence = sum(
                float(result["confidence"]) * OVERALL_WEIGHTS[name]
                for name, result in available.items()
            ) / available_weight
            confidence *= len(available) / len(BASE_COMPETENCIES)
            sufficient_components = sum(
                result["status"] == "sufficient" for result in available.values()
            )
            overall_status = (
                "sufficient"
                if (
                    len(available) >= 5
                    and sufficient_components >= 4
                    and confidence >= 0.6
                )
                else "limited"
            )
            overall = {
                "score": round(overall_score, 3),
                "confidence": round(confidence, 3),
                "status": overall_status,
                "covered_competencies": sorted(available),
                "sufficient_competency_count": sufficient_components,
                "missing_competencies": sorted(
                    set(BASE_COMPETENCIES).difference(available)
                ),
            }
        else:
            overall = {
                "score": None,
                "confidence": 0.0,
                "status": "insufficient_evidence",
                "covered_competencies": [],
                "missing_competencies": list(BASE_COMPETENCIES),
            }
        competencies["overall_readiness"] = overall
        return {
            "aggregator_version": AGGREGATOR_VERSION,
            "status": "final" if session.status == "completed" else "provisional",
            "competencies": competencies,
            "evidence": evidence,
            "flags": flags,
        }


class InterviewEvidenceSummaryService:
    @staticmethod
    def get(
        db: Session,
        session_id: str,
    ) -> InterviewEvidenceSummary | None:
        return (
            db.query(InterviewEvidenceSummary)
            .filter(
                InterviewEvidenceSummary.session_id == session_id,
                InterviewEvidenceSummary.aggregator_version == AGGREGATOR_VERSION,
            )
            .first()
        )

    @classmethod
    def create_or_update(
        cls,
        db: Session,
        session: InterviewSession,
        *,
        commit: bool = True,
    ) -> InterviewEvidenceSummary:
        result = InterviewEvidenceAggregator.aggregate(session)
        summary = cls.get(db, session.session_id)
        if summary is None:
            summary = InterviewEvidenceSummary(
                session_id=session.session_id,
                aggregator_version=AGGREGATOR_VERSION,
                status=result["status"],
                competency_scores_json=result["competencies"],
                evidence_json=result["evidence"],
                flags_json=result["flags"],
            )
            db.add(summary)
        else:
            summary.status = result["status"]
            summary.competency_scores_json = result["competencies"]
            summary.evidence_json = result["evidence"]
            summary.flags_json = result["flags"]
            summary.updated_at = datetime.now(timezone.utc)
        if commit:
            db.commit()
            db.refresh(summary)
        else:
            db.flush()
        return summary
