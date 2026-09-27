from __future__ import annotations

import hashlib
import hmac
from typing import Any

from sqlalchemy.orm import Session

from app.models.candidate_evaluation import CandidateEvaluation
from app.models.interview_session import InterviewSession
from app.services.calibration_service import RUBRIC_CRITERIA
from app.services.evaluation_service import ExpertConsensusService, RUBRIC_VERSION


class CalibrationDatasetBuilder:
    @staticmethod
    def _job_group_id(job_profile: dict[str, Any]) -> str:
        job_id = job_profile.get("job_id")
        if job_id not in {None, 0, "0", ""}:
            return str(job_id)
        identity = f"{job_profile.get('title', '')}|{job_profile.get('source_url', '')}"
        return f"uploaded-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]}"

    @staticmethod
    def build_row(
        session: InterviewSession,
        evaluation: CandidateEvaluation,
        consensus: dict[str, Any],
        pseudonym_salt: str,
    ) -> dict[str, Any]:
        reference = consensus.get("reference_scores")
        if not reference:
            raise ValueError("A ready expert consensus is required to build a training row")

        cv_profile = session.cv_profile_json or {}
        job_profile = session.job_profile_json or {}
        answered_turns = [turn for turn in session.turns if turn.answer_text]
        raw_candidate_key = str(session.user_id) if session.user_id else session.session_id
        candidate_token = hmac.new(
            pseudonym_salt.encode("utf-8"),
            raw_candidate_key.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:20]
        row: dict[str, Any] = {
            "candidate_id": f"candidate-{candidate_token}",
            "job_id": CalibrationDatasetBuilder._job_group_id(job_profile),
            "session_id": session.session_id,
            "expert_count": consensus["expert_count"],
            "cv_years_experience": float(cv_profile.get("cv_years_experience") or 0),
            "jd_years_required": float(job_profile.get("job_years_required") or 0),
            "cv_skill_count": len(cv_profile.get("cv_skills") or []),
            "jd_skill_count": len(job_profile.get("job_skills") or []),
            "interview_answer_count": len(answered_turns),
            "interview_total_words": sum(turn.answer_word_count or 0 for turn in answered_turns),
        }
        for criterion in RUBRIC_CRITERIA:
            row[f"llm_{criterion}"] = float(evaluation.scores_json[criterion])
            row[f"expert_consensus_{criterion}"] = float(reference[criterion])
        return row

    @classmethod
    def collect_rows(
        cls,
        db: Session,
        model_name: str,
        pseudonym_salt: str,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        if len(pseudonym_salt) < 16:
            raise ValueError("RESEARCH_PSEUDONYM_SALT must contain at least 16 characters")
        sessions = (
            db.query(InterviewSession)
            .filter(
                InterviewSession.status == "completed",
                InterviewSession.research_consent.is_(True),
            )
            .order_by(InterviewSession.created_at)
            .all()
        )
        rows: list[dict[str, Any]] = []
        counters = {
            "eligible_sessions": len(sessions),
            "missing_model_evaluation": 0,
            "insufficient_experts": 0,
            "anonymous_candidate_groups": 0,
        }
        for session in sessions:
            evaluations = [item for item in session.evaluations if item.model_name == model_name]
            if not evaluations:
                counters["missing_model_evaluation"] += 1
                continue
            evaluation = max(evaluations, key=lambda item: (item.created_at, item.evaluation_id))
            annotations = [
                item for item in session.expert_annotations if item.rubric_version == RUBRIC_VERSION
            ]
            consensus = ExpertConsensusService.consensus_from_annotations(annotations)
            if not consensus.get("reference_scores"):
                counters["insufficient_experts"] += 1
                continue
            if session.user_id is None:
                counters["anonymous_candidate_groups"] += 1
            rows.append(cls.build_row(session, evaluation, consensus, pseudonym_salt))
        counters["exported_rows"] = len(rows)
        return rows, counters
