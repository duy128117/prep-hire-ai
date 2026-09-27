from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.interview_session import InterviewSession
from app.models.interview_turn import InterviewTurn
from app.research.rubrics import CANDIDATE_EVALUATION_RUBRIC
from app.services.llm_provider_service import (
    LLMJsonResult,
    LLMProviderError,
    LLMProviderService,
)
from app.services.pdf_processor import CvIngestService


PROMPT_VERSION = "interview-question-v1"
VALID_CRITERIA = tuple(CANDIDATE_EVALUATION_RUBRIC["criteria"])


@dataclass(frozen=True)
class GeneratedQuestion:
    stage: str
    criterion: str
    question: str
    expected_signal: str
    source: str
    model_name: str | None
    prompt_hash: str
    latency_ms: int
    provider: str | None = None


class AdaptiveQuestionEngine:
    JSON_MAX_ATTEMPTS = 3

    @staticmethod
    def redact_personal_data(text: str) -> str:
        value = text or ""
        value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", value)
        value = re.sub(r"https?://\S+|www\.\S+", "[REDACTED_URL]", value, flags=re.IGNORECASE)
        value = re.sub(
            r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){8,10}(?!\d)",
            "[REDACTED_PHONE]",
            value,
        )
        return value

    @staticmethod
    def scheduled_stage(turn_index: int, max_questions: int) -> str:
        if turn_index == 0:
            return "introduction"
        if turn_index == max_questions - 1:
            return "closing"
        progress = turn_index / max(1, max_questions - 1)
        if progress < 0.3:
            return "cv_evidence"
        if progress < 0.55:
            return "technical"
        if progress < 0.8:
            return "problem_solving"
        return "behavioral"

    @classmethod
    def determine_stage(
        cls,
        turn_index: int,
        max_questions: int,
        history: list[dict[str, Any]],
    ) -> str:
        scheduled = cls.scheduled_stage(turn_index, max_questions)
        if scheduled == "closing" or not history:
            return scheduled
        previous = history[-1]
        answer = str(previous.get("answer") or "").strip()
        if previous.get("stage") not in {"introduction", "closing"} and len(answer.split()) < 20:
            return "follow_up"
        return scheduled

    @staticmethod
    def criterion_for_stage(stage: str, history: list[dict[str, Any]]) -> str:
        if stage == "follow_up" and history:
            previous = str(history[-1].get("criterion") or "")
            if previous in VALID_CRITERIA:
                return previous
        return {
            "introduction": "communication",
            "cv_evidence": "experience_evidence",
            "technical": "technical_knowledge",
            "problem_solving": "problem_solving",
            "behavioral": "learning_adaptability",
            "closing": "overall_readiness",
        }.get(stage, "role_relevance")

    @classmethod
    def build_prompt(
        cls,
        stage: str,
        criterion: str,
        language: str,
        cv_text: str,
        jd_text: str,
        cv_profile: dict[str, Any],
        job_profile: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> str:
        recent_history = [
            {
                "stage": item.get("stage"),
                "criterion": item.get("criterion"),
                "question": cls.redact_personal_data(str(item.get("question") or ""))[:800],
                "answer": cls.redact_personal_data(str(item.get("answer") or ""))[:1600],
            }
            for item in history[-4:]
        ]
        context = {
            "cv_profile": cv_profile,
            "job_profile": job_profile,
            "cv_excerpt": cls.redact_personal_data(cv_text)[:5000],
            "jd_excerpt": cls.redact_personal_data(jd_text)[:5000],
            "recent_history": recent_history,
            "next_stage": stage,
            "target_competency": criterion,
        }
        output_language = "Vietnamese" if language == "vi" else "English"
        return (
            "You are conducting a structured, job-related technical interview. "
            "Generate exactly one adaptive question from the supplied context. "
            "Do not score the candidate, repeat a question, or ask about protected personal characteristics. "
            "For follow_up, probe the previous answer for evidence or technical reasoning. "
            "Return valid JSON only, without markdown.\n"
            f"Question language: {output_language}.\n"
            '{"question": string, "criterion": string, "expected_signal": string}\n'
            f"criterion must be one of: {', '.join(VALID_CRITERIA)}.\n"
            f"Context:\n{json.dumps(context, ensure_ascii=False)}"
        )

    @staticmethod
    def fallback_question(
        stage: str,
        criterion: str,
        language: str,
        cv_profile: dict[str, Any],
        job_profile: dict[str, Any],
    ) -> tuple[str, str]:
        cv_skills = cv_profile.get("cv_skills") or []
        job_skills = job_profile.get("job_skills") or []
        cv_skill = str(cv_skills[0].get("name")) if cv_skills else "một công nghệ trong CV"
        job_skill = str(job_skills[0].get("name")) if job_skills else "một yêu cầu chính của vị trí"
        if language == "vi":
            questions = {
                "introduction": "Bạn hãy giới thiệu ngắn gọn về kinh nghiệm phù hợp nhất với vị trí này.",
                "cv_evidence": f"Hãy mô tả một dự án bạn trực tiếp sử dụng {cv_skill}, vai trò của bạn và kết quả đo lường được.",
                "technical": f"Với {job_skill}, bạn sẽ cân nhắc những lựa chọn kỹ thuật và đánh đổi quan trọng nào?",
                "problem_solving": f"Nếu hệ thống dùng {job_skill} gặp lỗi trên production, bạn sẽ điều tra và xử lý theo trình tự nào?",
                "behavioral": "Hãy kể về một lần yêu cầu dự án thay đổi đáng kể và cách bạn thích nghi.",
                "closing": "Bạn muốn bổ sung bằng chứng nào khác để thể hiện mức độ sẵn sàng cho vị trí này?",
                "follow_up": "Câu trả lời vừa rồi còn ngắn; bạn có thể nêu tình huống cụ thể, hành động và kết quả không?",
            }
        else:
            questions = {
                "introduction": "Please briefly introduce the experience most relevant to this role.",
                "cv_evidence": f"Describe a project where you directly used {cv_skill}, your role, and a measurable result.",
                "technical": f"What technical choices and trade-offs would you consider when using {job_skill}?",
                "problem_solving": f"If a production system using {job_skill} failed, how would you investigate and resolve it?",
                "behavioral": "Describe a significant project requirement change and how you adapted to it.",
                "closing": "What additional evidence would you like to share about your readiness for this role?",
                "follow_up": "Your previous answer was brief. Could you give a concrete situation, your actions, and the outcome?",
            }
        return questions.get(stage, questions["technical"]), f"Observable evidence for {criterion.replace('_', ' ')}."

    @classmethod
    def generate(
        cls,
        turn_index: int,
        max_questions: int,
        language: str,
        cv_text: str,
        jd_text: str,
        cv_profile: dict[str, Any],
        job_profile: dict[str, Any],
        history: list[dict[str, Any]],
        allow_external_ai: bool,
        model_name: str | None = None,
        provider: str | None = None,
        strict_external_ai: bool = False,
    ) -> GeneratedQuestion:
        stage = cls.determine_stage(turn_index, max_questions, history)
        criterion = cls.criterion_for_stage(stage, history)
        prompt = cls.build_prompt(stage, criterion, language, cv_text, jd_text, cv_profile, job_profile, history)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        started_at = time.perf_counter()
        data: dict[str, Any] | None = None

        selected_identity = provider or model_name or "inception"
        selected_model = model_name
        selected_provider = provider
        if allow_external_ai and settings.CV_AI_ENRICHMENT_ENABLED:
            try:
                response: LLMJsonResult | None = None
                correction = ""
                for attempt in range(1, cls.JSON_MAX_ATTEMPTS + 1):
                    effective_prompt = prompt + correction
                    try:
                        response = LLMProviderService.generate_json(
                            selected_identity,
                            effective_prompt,
                            operation="question_generation",
                            max_output_tokens=1024,
                            temperature=0.3,
                        )
                        prompt_hash = hashlib.sha256(
                            effective_prompt.encode("utf-8")
                        ).hexdigest()
                        break
                    except LLMProviderError as exc:
                        if exc.error_code != "invalid_json" or attempt >= cls.JSON_MAX_ATTEMPTS:
                            raise
                        correction = (
                            "\nYour previous response was not valid JSON. Return exactly one "
                            "complete JSON object with question, criterion, and expected_signal. "
                            "Do not use markdown or explanatory text."
                        )
                if response is None:
                    raise RuntimeError("Question generator returned no response")
                data = response.data
                selected_model = response.model
                selected_provider = response.provider
            except Exception as exc:
                if strict_external_ai:
                    raise
                print(f"--- INTERVIEW QUESTION AI ERROR: {exc} ---")

        latency_ms = max(0, round((time.perf_counter() - started_at) * 1000))
        if data:
            question = str(data.get("question") or "").strip()
            generated_criterion = str(data.get("criterion") or "").strip()
            expected_signal = str(data.get("expected_signal") or "").strip()
            if question and len(question) <= 1200:
                if generated_criterion in VALID_CRITERIA:
                    criterion = generated_criterion
                return GeneratedQuestion(
                    stage, criterion, question, expected_signal[:1200], "ai",
                    selected_model, prompt_hash, latency_ms, selected_provider,
                )

        question, expected_signal = cls.fallback_question(stage, criterion, language, cv_profile, job_profile)
        return GeneratedQuestion(
            stage, criterion, question, expected_signal, "fallback", None, prompt_hash, latency_ms
        )


class InterviewSessionService:
    @staticmethod
    def history(session: InterviewSession) -> list[dict[str, Any]]:
        return [
            {
                "stage": turn.stage,
                "criterion": turn.criterion,
                "question": turn.question_text,
                "answer": turn.answer_text,
            }
            for turn in session.turns
            if turn.answer_text is not None
        ]

    @staticmethod
    def new_turn(session: InterviewSession, turn_index: int) -> InterviewTurn:
        generated = AdaptiveQuestionEngine.generate(
            turn_index,
            session.max_questions,
            session.language,
            session.cv_text,
            session.jd_text,
            session.cv_profile_json,
            session.job_profile_json,
            InterviewSessionService.history(session),
            session.allow_external_ai,
        )
        return InterviewTurn(
            session_id=session.session_id,
            turn_index=turn_index,
            stage=generated.stage,
            criterion=generated.criterion,
            question_text=generated.question,
            expected_signal=generated.expected_signal,
            question_source=generated.source,
            model_name=generated.model_name,
            prompt_version=PROMPT_VERSION,
            prompt_hash=generated.prompt_hash,
            generation_latency_ms=generated.latency_ms,
        )

    @staticmethod
    def create(
        db: Session,
        cv_text: str,
        jd_text: str,
        language: str,
        max_questions: int,
        allow_external_ai: bool,
        research_consent: bool,
        user_id: int | None = None,
    ) -> InterviewSession:
        extraction_cv_text = AdaptiveQuestionEngine.redact_personal_data(cv_text) if allow_external_ai else cv_text
        extraction_jd_text = AdaptiveQuestionEngine.redact_personal_data(jd_text) if allow_external_ai else jd_text
        cv_profile = CvIngestService.extract_profile(
            db,
            extraction_cv_text,
            use_ai=allow_external_ai,
        )
        job_profile = CvIngestService.build_uploaded_job_context(
            db,
            extraction_jd_text,
            use_ai=allow_external_ai,
        )
        session = InterviewSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            language=language,
            max_questions=max_questions,
            allow_external_ai=allow_external_ai,
            research_consent=research_consent,
            cv_text=cv_text.strip(),
            jd_text=jd_text.strip(),
            cv_profile_json=cv_profile.model_dump(mode="json"),
            job_profile_json=job_profile.model_dump(mode="json"),
        )
        db.add(session)
        db.flush()
        first_turn = InterviewSessionService.new_turn(session, 0)
        session.turns.append(first_turn)
        session.current_stage = first_turn.stage
        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def get(db: Session, session_id: str, lock: bool = False) -> InterviewSession | None:
        query = db.query(InterviewSession).filter(InterviewSession.session_id == session_id)
        return query.with_for_update().first() if lock else query.first()

    @staticmethod
    def submit_answer(db: Session, session_id: str, answer: str) -> InterviewSession:
        session = InterviewSessionService.get(db, session_id, lock=True)
        if not session:
            raise ValueError("Interview session not found")
        if session.status != "active":
            raise ValueError("Interview session is already completed")

        current_turn = next(
            (turn for turn in session.turns if turn.turn_index == session.current_turn_index), None
        )
        if not current_turn or current_turn.answer_text is not None:
            raise ValueError("The current interview question cannot accept an answer")

        normalized = answer.strip()
        current_turn.answer_text = normalized
        current_turn.answer_word_count = len(normalized.split())
        current_turn.answered_at = datetime.now(timezone.utc)
        next_index = current_turn.turn_index + 1

        if next_index >= session.max_questions:
            session.status = "completed"
            session.current_stage = "completed"
            session.completed_at = datetime.now(timezone.utc)
        else:
            db.flush()
            next_turn = InterviewSessionService.new_turn(session, next_index)
            session.turns.append(next_turn)
            session.current_turn_index = next_index
            session.current_stage = next_turn.stage

        session.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)
        return session
