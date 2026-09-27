from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_standalone_db
from app.models.interview_session import InterviewSession
from app.models.job import Job
from app.models.question_benchmark import (
    QuestionBenchmarkCase,
    QuestionBenchmarkOutput,
    QuestionBenchmarkRun,
)
from app.models.weak_supervision import (
    AnswerAIJudgment,
    QuestionAIJudgment,
    SyntheticAnswerSample,
)
from app.services.interview_service import AdaptiveQuestionEngine
from app.services.llm_provider_service import (
    LLMCallRecord,
    LLMProviderError,
    LLMProviderService,
    LLMProviderSpec,
    capture_llm_usage,
)
from app.services.weak_supervision_service import (
    ANSWER_AI_RUBRIC_VERSION,
    ANSWER_GENERATION_PROMPT_VERSION,
    ANSWER_JUDGE_PROMPT_VERSION,
    QUESTION_AI_RUBRIC_VERSION,
    QUESTION_JUDGE_PROMPT_VERSION,
    AnswerAIJudgeEngine,
    QuestionAIJudgeEngine,
    SyntheticAnswerEngine,
)


DATASET_PROMPT_VERSION = "balanced-weak-v1"
DEFAULT_CHECKPOINT = Path("data/research/generation_checkpoint.json")
SYNTHETIC_CASE_PREFIX = "SYNTH-JOB-"
PROFILE_LEVELS = (
    ("intern", 0.0, 0.40),
    ("junior", 1.0, 0.52),
    ("mid", 3.0, 0.66),
    ("senior", 6.0, 0.80),
    ("lead", 9.0, 0.90),
)
ACTIVE_PROVIDER_COUNT = 4
LOGICAL_SLOT_NAMES = ("slot_1", "slot_2", "slot_3", "slot_4")


class ProviderTokenBudgetReached(RuntimeError):
    def __init__(self, provider: str, used_tokens: int, token_budget: int) -> None:
        self.provider = provider
        self.used_tokens = used_tokens
        self.token_budget = token_budget
        super().__init__(
            f"Soft token budget reached for {provider}: "
            f"{used_tokens} >= {token_budget}"
        )


class InsufficientProviderPoolError(RuntimeError):
    def __init__(self, available_count: int) -> None:
        self.available_count = available_count
        super().__init__(
            f"Only {available_count} configured providers remain; "
            f"at least {ACTIVE_PROVIDER_COUNT} are required"
        )


class GenerationLedger:
    def __init__(self, path: Path, target_sets: int, token_budget: int) -> None:
        self.path = path
        self.token_budget = max(0, token_budget)
        self.state = self._load()
        self.state["target_sets"] = target_sets
        self.state.setdefault("status", "ready")
        self.state.setdefault("completed_sets", 0)
        self.state.setdefault("completed_rounds", 0)
        self.state.setdefault("providers", {})
        self.state.setdefault("last_error", None)
        for spec in LLMProviderService.all_specs():
            self.state["providers"].setdefault(
                spec.provider,
                {
                    "model": spec.model,
                    "calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "operations": {},
                },
            )
        self._initialize_provider_pool()
        self.save()

    def _initialize_provider_pool(self) -> None:
        pool = self.state.setdefault("provider_pool", {})
        pool["architecture"] = "dynamic-round-robin-v1"
        pool["candidates"] = [
            {"provider": spec.provider, "model": spec.model}
            for spec in LLMProviderService.pool_specs()
        ]
        pool.setdefault("rotation_cursor", 0)
        pool.setdefault("exhausted", [])
        pool.setdefault("failovers", [])
        pool.setdefault("active_assignments", [])

        transient_markers = (
            "rate limit",
            "http 429",
            "too many requests",
            "network request failed",
            "network_error",
            "connection reset",
            "timed out",
            "http 5",
        )
        pool["exhausted"] = [
            item
            for item in pool["exhausted"]
            if not any(
                marker in str(item.get("reason") or "").lower()
                for marker in transient_markers
            )
        ]

        specs = {spec.provider: spec for spec in LLMProviderService.pool_specs()}
        blocked = {str(item.get("provider")) for item in pool["exhausted"]}
        valid_assignments: list[dict[str, str]] = []
        used: set[str] = set()
        for item in pool["active_assignments"]:
            slot = str(item.get("slot") or "")
            provider = str(item.get("provider") or "")
            spec = specs.get(provider)
            if (
                slot not in LOGICAL_SLOT_NAMES
                or not spec
                or provider in blocked
                or provider in used
                or not LLMProviderService.is_configured(spec)
            ):
                continue
            valid_assignments.append({
                "slot": slot,
                "provider": spec.provider,
                "model": spec.model,
            })
            used.add(provider)
        pool["active_assignments"] = valid_assignments
        self._fill_active_assignments()

    def available_roster(self) -> tuple[LLMProviderSpec, ...]:
        pool = self.state["provider_pool"]
        blocked = {str(item.get("provider")) for item in pool["exhausted"]}
        return tuple(
            spec
            for spec in LLMProviderService.pool_specs()
            if spec.provider not in blocked and LLMProviderService.is_configured(spec)
        )

    def _fill_active_assignments(self) -> None:
        pool = self.state["provider_pool"]
        active = list(pool["active_assignments"])
        used = {str(item["provider"]) for item in active}
        missing_slots = [
            slot
            for slot in LOGICAL_SLOT_NAMES
            if not any(item["slot"] == slot for item in active)
        ]
        available = list(self.available_roster())
        if available:
            cursor = int(pool.get("rotation_cursor") or 0) % len(available)
            available = available[cursor:] + available[:cursor]
        candidates = [spec for spec in available if spec.provider not in used]
        for slot, spec in zip(missing_slots, candidates):
            active.append({
                "slot": slot,
                "provider": spec.provider,
                "model": spec.model,
            })
        active.sort(key=lambda item: LOGICAL_SLOT_NAMES.index(item["slot"]))
        pool["active_assignments"] = active

    def active_roster(self) -> tuple[LLMProviderSpec, ...]:
        return tuple(spec for _, spec in self.active_assignments())

    def active_assignments(self) -> tuple[tuple[str, LLMProviderSpec], ...]:
        self._fill_active_assignments()
        active = self.state["provider_pool"]["active_assignments"]
        if len(active) < ACTIVE_PROVIDER_COUNT:
            raise InsufficientProviderPoolError(len(self.available_roster()))
        assignments = tuple(
            (str(item["slot"]), LLMProviderService.resolve(str(item["provider"])))
            for item in active[:ACTIVE_PROVIDER_COUNT]
        )
        providers = [spec.provider for _, spec in assignments]
        if len(providers) != len(set(providers)):
            raise ValueError("Each active dataset worker must use a different provider")
        return assignments

    def rotate_active_roster(self) -> None:
        pool = self.state["provider_pool"]
        available_count = len(self.available_roster())
        if available_count < ACTIVE_PROVIDER_COUNT:
            raise InsufficientProviderPoolError(available_count)
        pool["rotation_cursor"] = (
            int(pool.get("rotation_cursor") or 0) + 1
        ) % available_count
        pool["active_assignments"] = []
        self._fill_active_assignments()
        self.save()

    def failover_provider(
        self,
        provider: str,
        reason: str,
    ) -> LLMProviderSpec | None:
        pool = self.state["provider_pool"]
        now = datetime.now(timezone.utc).isoformat()
        failed = next(
            (
                item
                for item in pool["active_assignments"]
                if item["provider"] == provider
            ),
            None,
        )
        spec = next(
            (item for item in LLMProviderService.pool_specs() if item.provider == provider),
            None,
        )
        if spec and not any(item.get("provider") == provider for item in pool["exhausted"]):
            pool["exhausted"].append({
                "provider": provider,
                "model": spec.model,
                "reason": reason[:500],
                "at": now,
            })
        pool["active_assignments"] = [
            item
            for item in pool["active_assignments"]
            if item["provider"] != provider
        ]
        self._fill_active_assignments()
        if failed:
            replacement = next(
                (
                    item
                    for item in pool["active_assignments"]
                    if item["slot"] == failed["slot"]
                ),
                None,
            )
            if replacement:
                pool["failovers"].append({
                    "slot": failed["slot"],
                    "from_provider": provider,
                    "from_model": failed["model"],
                    "to_provider": replacement["provider"],
                    "to_model": replacement["model"],
                    "reason": reason[:500],
                    "at": now,
                })
                self.save()
                return LLMProviderService.resolve(str(replacement["provider"]))
            self.save()
        return None

    def pool_status(self) -> dict[str, Any]:
        pool = self.state["provider_pool"]
        available = self.available_roster()
        return {
            "required_active": ACTIVE_PROVIDER_COUNT,
            "configured_count": sum(
                LLMProviderService.is_configured(spec)
                for spec in LLMProviderService.pool_specs()
            ),
            "available_count": len(available),
            "available_providers": [spec.provider for spec in available],
            "active_assignments": list(pool["active_assignments"]),
            "unavailable": list(pool["exhausted"]),
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def record(self, call: LLMCallRecord) -> None:
        provider = self.state["providers"].setdefault(
            call.provider,
            {
                "model": call.model,
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "operations": {},
            },
        )
        provider["calls"] += 1
        provider["successful_calls"] += int(call.status == "success")
        provider["failed_calls"] += int(call.status != "success")
        provider["input_tokens"] += call.input_tokens
        provider["output_tokens"] += call.output_tokens
        provider["total_tokens"] += call.total_tokens
        operation = provider["operations"].setdefault(
            call.operation,
            {"calls": 0, "total_tokens": 0},
        )
        operation["calls"] += 1
        operation["total_tokens"] += call.total_tokens
        if call.status != "success":
            self.state["last_error"] = {
                "provider": call.provider,
                "operation": call.operation,
                "error_code": call.error_code,
                "quota_exhausted": call.quota_exhausted,
            }
        self.save()

        if self.token_budget and provider["total_tokens"] >= self.token_budget:
            raise ProviderTokenBudgetReached(
                call.provider,
                provider["total_tokens"],
                self.token_budget,
            )

    def mark_running(self, completed_sets: int) -> None:
        self.state.update(status="running", completed_sets=completed_sets, last_error=None)
        self.state["completed_rounds"] = completed_sets // 4
        self.save()

    def mark_round_complete(self, completed_sets: int) -> None:
        self.state.update(status="running", completed_sets=completed_sets, last_error=None)
        self.state["completed_rounds"] = completed_sets // 4
        self.save()

    def pause(self, exc: BaseException, completed_sets: int) -> None:
        self.state.update(status="paused", completed_sets=completed_sets)
        self.state["completed_rounds"] = completed_sets // 4
        self.state["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:500],
        }
        self.save()

    def complete(self, completed_sets: int) -> None:
        self.state.update(status="completed", completed_sets=completed_sets, last_error=None)
        self.state["completed_rounds"] = completed_sets // 4
        self.save()


def _complete_outputs(db: Session) -> list[QuestionBenchmarkOutput]:
    outputs = (
        db.query(QuestionBenchmarkOutput)
        .join(QuestionBenchmarkRun)
        .filter(
            QuestionBenchmarkRun.prompt_version == DATASET_PROMPT_VERSION,
            QuestionBenchmarkOutput.status == "success",
        )
        .all()
    )
    complete: list[QuestionBenchmarkOutput] = []
    for output in outputs:
        question_judges = {
            item.judge_model_name
            for item in output.ai_judgments
            if item.rubric_version == QUESTION_AI_RUBRIC_VERSION
        }
        answer_ready = any(
            sample.status == "success"
            and len({
                item.judge_model_name
                for item in sample.ai_judgments
                if item.rubric_version == ANSWER_AI_RUBRIC_VERSION
            }) >= 3
            for sample in output.synthetic_answers
        )
        if len(question_judges) >= 3 and answer_ready:
            complete.append(output)
    return complete


def generation_status(db: Session) -> dict[str, Any]:
    complete = _complete_outputs(db)
    generator_counts = {spec.provider: 0 for spec in LLMProviderService.all_specs()}
    provider_to_slot = {
        "groq": "slot_1",
        "cloudflare": "slot_2",
        "cohere": "slot_3",
        "inception": "slot_3",
        "huggingface": "slot_4",
        "opentyphoon": "slot_4",
    }
    slot_counts = {slot: 0 for slot in LOGICAL_SLOT_NAMES}
    for output in complete:
        generator_counts[output.provider] = generator_counts.get(output.provider, 0) + 1
        slot_name = output.logical_slot or provider_to_slot.get(output.provider)
        if slot_name:
            slot_counts[slot_name] += 1
    return {
        "benchmark_cases": db.query(func.count(QuestionBenchmarkCase.case_id)).scalar() or 0,
        "synthetic_benchmark_cases": (
            db.query(func.count(QuestionBenchmarkCase.case_id))
            .filter(QuestionBenchmarkCase.title.like(f"{SYNTHETIC_CASE_PREFIX}%"))
            .scalar()
            or 0
        ),
        "research_consented_sessions": (
            db.query(func.count(InterviewSession.session_id))
            .filter(InterviewSession.research_consent.is_(True))
            .scalar()
            or 0
        ),
        "available_jobs": db.query(func.count(Job.job_id)).scalar() or 0,
        "complete_sets": len(complete),
        "generator_counts": generator_counts,
        "logical_slot_counts": slot_counts,
        "balanced": len(set(slot_counts.values())) <= 1,
        "provider_pool": [
            {
                "provider": spec.provider,
                "model": spec.model,
                "configured": LLMProviderService.is_configured(spec),
            }
            for spec in LLMProviderService.pool_specs()
        ],
        "required_active_providers": ACTIVE_PROVIDER_COUNT,
    }


def generation_status_with_retry(
    db: Session,
    max_attempts: int = 3,
) -> dict[str, Any]:
    for attempt in range(1, max_attempts + 1):
        try:
            return generation_status(db)
        except OperationalError:
            rollback_or_invalidate(db)
            if attempt >= max_attempts:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


def failover_worthy(exc: BaseException) -> bool:
    if isinstance(exc, ProviderTokenBudgetReached):
        return True
    if not isinstance(exc, LLMProviderError):
        return False
    return (
        exc.quota_exhausted
        or exc.error_code in {"missing_credentials", "network_error", "invalid_json"}
        or exc.status_code in {401, 402, 403, 429}
        or (exc.status_code is not None and exc.status_code >= 500)
    )


def failed_provider(exc: BaseException) -> str | None:
    if isinstance(exc, (LLMProviderError, ProviderTokenBudgetReached)):
        return exc.provider
    return None


def rollback_or_invalidate(db: Session) -> str | None:
    try:
        db.rollback()
        return None
    except BaseException as exc:
        # A long AI round can outlive a pooled database connection. Invalidating
        # lets the same Session acquire a fresh connection for the atomic retry.
        try:
            db.invalidate()
        except BaseException:
            db.close()
        return str(exc)[:500]


def grouped_case_index(
    round_start: int,
    provider_count: int,
    case_count: int,
    grouped_context_start_set: int,
) -> int:
    if provider_count <= 0 or case_count <= 0:
        raise ValueError("provider_count and case_count must be positive")
    if round_start < grouped_context_start_set:
        raise ValueError("round_start cannot precede grouped_context_start_set")
    completed_grouped_sets = round_start - grouped_context_start_set
    if completed_grouped_sets % provider_count != 0:
        raise ValueError("grouped generation must start on a complete provider round")
    grouped_round = completed_grouped_sets // provider_count
    return (grouped_context_start_set + grouped_round) % case_count


def seed_synthetic_cases(db: Session) -> dict[str, int]:
    jobs = db.query(Job).order_by(Job.job_id).all()
    existing_titles = {
        title
        for (title,) in db.query(QuestionBenchmarkCase.title)
        .filter(QuestionBenchmarkCase.title.like(f"{SYNTHETIC_CASE_PREFIX}%"))
        .all()
    }
    created = 0
    skipped = 0

    for job_index, job in enumerate(jobs):
        ranked_skills = sorted(
            (
                {
                    "name": item.skill.name,
                    "importance": float(item.importance or 0.6),
                    "required_proficiency": min(0.9, max(0.5, float(item.importance or 0.6))),
                }
                for item in job.job_skills
                if item.skill and item.skill.name
            ),
            key=lambda item: (item["importance"], item["name"].lower()),
            reverse=True,
        )
        if not ranked_skills:
            ranked_skills = [{
                "name": "Role-specific fundamentals",
                "importance": 0.7,
                "required_proficiency": 0.6,
            }]

        jd_text = (
            job.description_clean
            or job.description_raw
            or job.skills_qualifications
            or job.title
        ).strip()
        job_level = getattr(job.level, "value", None) or "unspecified"
        job_profile = {
            "job_id": job.job_id,
            "title": job.title,
            "job_level": job_level,
            "employment_type": job.employment_type or "",
            "experience": job.experience or "",
            "job_location": job.location or "",
            "role_responsibilities": job.role_responsibilities or "",
            "skills_qualifications": job.skills_qualifications or "",
            "job_skills": ranked_skills[:12],
            "research_context_source": "public_crawled_jd",
        }

        for profile_index, (level, years, proficiency) in enumerate(PROFILE_LEVELS):
            title = f"{SYNTHETIC_CASE_PREFIX}{job.job_id}-PROFILE-{profile_index + 1}"
            if title in existing_titles:
                skipped += 1
                continue

            skill_count = max(
                1,
                min(len(ranked_skills), round(len(ranked_skills) * (0.35 + profile_index * 0.14))),
            )
            cv_skills = [
                {
                    "name": item["name"],
                    "proficiency": min(0.95, proficiency + (index % 3 - 1) * 0.04),
                    "years_of_experience": max(0.0, years - index * 0.5),
                }
                for index, item in enumerate(ranked_skills[:skill_count])
            ]
            skill_names = ", ".join(item["name"] for item in cv_skills)
            cv_text = (
                f"Synthetic {level} candidate profile for research. "
                f"Target role: {job.title}. Experience: {years:g} years. "
                f"Skills demonstrated in coursework or projects: {skill_names}. "
                "Project evidence includes implementation, testing, collaboration, and documented outcomes. "
                "This profile is artificial and contains no real person's data."
            )
            cv_profile = {
                "cv_level": level,
                "cv_years_experience": years,
                "cv_skills": cv_skills,
                "preferred_locations": [],
                "cv_certifications": [],
                "research_context_source": "deterministic_synthetic_profile",
            }
            max_questions = 7
            turn_index = (job_index * len(PROFILE_LEVELS) + profile_index) % max_questions
            history: list[dict[str, Any]] = []
            stage = AdaptiveQuestionEngine.determine_stage(turn_index, max_questions, history)
            criterion = AdaptiveQuestionEngine.criterion_for_stage(stage, history)
            db.add(QuestionBenchmarkCase(
                source_session_id=None,
                title=title,
                language="vi",
                turn_index=turn_index,
                max_questions=max_questions,
                stage=stage,
                criterion=criterion,
                cv_text=cv_text,
                jd_text=jd_text,
                cv_profile_json=cv_profile,
                job_profile_json=job_profile,
                interview_history_json=history,
            ))
            existing_titles.add(title)
            created += 1

    db.commit()
    return {"created": created, "skipped": skipped, "total_jobs": len(jobs)}


def _make_question(
    db: Session,
    benchmark_case: QuestionBenchmarkCase,
    logical_slot: str,
    generator: LLMProviderSpec,
    roster: tuple[LLMProviderSpec, ...],
) -> QuestionBenchmarkOutput:
    run = QuestionBenchmarkRun(
        run_id=str(uuid.uuid4()),
        case_id=benchmark_case.case_id,
        status="running",
        prompt_version=DATASET_PROMPT_VERSION,
    )
    db.add(run)
    db.flush()

    generated = AdaptiveQuestionEngine.generate(
        turn_index=benchmark_case.turn_index,
        max_questions=benchmark_case.max_questions,
        language=benchmark_case.language,
        cv_text=benchmark_case.cv_text,
        jd_text=benchmark_case.jd_text,
        cv_profile=benchmark_case.cv_profile_json,
        job_profile=benchmark_case.job_profile_json,
        history=benchmark_case.interview_history_json,
        allow_external_ai=True,
        model_name=generator.model,
        provider=generator.provider,
        strict_external_ai=True,
    )
    if generated.source != "ai" or not generated.question:
        raise RuntimeError(f"{generator.provider} did not generate a valid question")

    output = QuestionBenchmarkOutput(
        output_id=str(uuid.uuid4()),
        run_id=run.run_id,
        provider=generator.provider,
        logical_slot=logical_slot,
        model_name=generator.model,
        blind_label=f"Q-{uuid.uuid4().hex[:8].upper()}",
        status="success",
        question_text=generated.question,
        criterion=generated.criterion,
        expected_signal=generated.expected_signal,
        latency_ms=generated.latency_ms,
        prompt_hash=generated.prompt_hash,
    )
    db.add(output)
    db.flush()

    for judge in roster:
        if judge.provider == generator.provider:
            continue
        result = QuestionAIJudgeEngine.judge(output, judge.model)
        db.add(QuestionAIJudgment(
            output_id=output.output_id,
            provider=judge.provider,
            judge_model_name=judge.model,
            rubric_version=QUESTION_AI_RUBRIC_VERSION,
            scores_json=result.scores,
            evidence_json=result.evidence,
            recommendation=result.recommendation,
            prompt_version=QUESTION_JUDGE_PROMPT_VERSION,
            prompt_hash=result.prompt_hash,
            latency_ms=result.latency_ms,
        ))
    db.flush()
    return output


def _make_answer(
    db: Session,
    output: QuestionBenchmarkOutput,
    generator: LLMProviderSpec,
    quality_level: int,
    roster: tuple[LLMProviderSpec, ...],
) -> SyntheticAnswerSample:
    generated = SyntheticAnswerEngine.generate(output, generator.model, quality_level)
    sample = SyntheticAnswerSample(
        sample_id=str(uuid.uuid4()),
        question_output_id=output.output_id,
        blind_label=f"A-{uuid.uuid4().hex[:8].upper()}",
        provider=generator.provider,
        generator_model_name=generator.model,
        target_quality_level=quality_level,
        status="success",
        answer_text=generated.answer,
        prompt_version=ANSWER_GENERATION_PROMPT_VERSION,
        prompt_hash=generated.prompt_hash,
        generation_latency_ms=generated.latency_ms,
    )
    db.add(sample)
    db.flush()

    for judge in roster:
        if judge.provider == generator.provider:
            continue
        result = AnswerAIJudgeEngine.judge(sample, judge.model)
        db.add(AnswerAIJudgment(
            sample_id=sample.sample_id,
            provider=judge.provider,
            judge_model_name=judge.model,
            rubric_version=ANSWER_AI_RUBRIC_VERSION,
            scores_json=result.scores,
            evidence_json=result.evidence,
            recommendation=result.recommendation,
            prompt_version=ANSWER_JUDGE_PROMPT_VERSION,
            prompt_hash=result.prompt_hash,
            latency_ms=result.latency_ms,
        ))
    db.flush()
    return sample


def generate(
    db: Session,
    *,
    target_sets: int,
    max_rounds: int | None,
    ledger: GenerationLedger,
) -> int:
    try:
        assignments = ledger.active_assignments()
    except InsufficientProviderPoolError as exc:
        completed_sets = len(_complete_outputs(db))
        ledger.pause(exc, completed_sets)
        print(json.dumps({
            "status": "paused",
            "completed_sets": completed_sets,
            "reason": "fewer_than_four_available_providers",
            "provider_pool": ledger.pool_status(),
        }, ensure_ascii=False, indent=2))
        return 1
    providers = tuple(spec for _, spec in assignments)
    if target_sets <= 0 or target_sets % ACTIVE_PROVIDER_COUNT != 0:
        raise ValueError("target_sets must be positive and divisible by four")
    LLMProviderService.require_configured(providers)

    cases = db.query(QuestionBenchmarkCase).order_by(QuestionBenchmarkCase.case_id).all()
    if not cases:
        raise ValueError("No benchmark cases exist. Create consented CV/JD benchmark cases first")

    completed_sets = len(_complete_outputs(db))
    if completed_sets % len(providers) != 0:
        raise ValueError(
            f"Existing generated data is unbalanced ({completed_sets} complete sets); refusing to continue"
        )
    grouped_context_start_set = ledger.state.get("grouped_context_start_set")
    if grouped_context_start_set is None:
        grouped_context_start_set = completed_sets
        ledger.state["grouped_context_start_set"] = grouped_context_start_set
        ledger.state["context_grouping"] = "four_generators_same_case"
        ledger.save()
    grouped_context_start_set = int(grouped_context_start_set)
    ledger.mark_running(completed_sets)
    rounds_run = 0

    while completed_sets < target_sets:
        if max_rounds is not None and rounds_run >= max_rounds:
            ledger.pause(RuntimeError("max_rounds reached"), completed_sets)
            return 2

        try:
            assignments = ledger.active_assignments()
        except InsufficientProviderPoolError as exc:
            ledger.pause(exc, completed_sets)
            print(json.dumps({
                "status": "paused",
                "completed_sets": completed_sets,
                "reason": "fewer_than_four_available_providers",
                "provider_pool": ledger.pool_status(),
            }, ensure_ascii=False, indent=2))
            return 1
        providers = tuple(spec for _, spec in assignments)
        round_start = completed_sets
        try:
            with capture_llm_usage(ledger.record):
                case_index = grouped_case_index(
                    round_start,
                    len(providers),
                    len(cases),
                    grouped_context_start_set,
                )
                benchmark_case = cases[case_index]
                for offset, (logical_slot, generator) in enumerate(assignments):
                    set_index = round_start + offset
                    quality_level = (set_index % 5) + 1
                    output = _make_question(
                        db,
                        benchmark_case,
                        logical_slot,
                        generator,
                        providers,
                    )
                    _make_answer(db, output, generator, quality_level, providers)
                    output.run.status = "completed"
            db.commit()
        except BaseException as exc:
            rollback_error = rollback_or_invalidate(db)
            provider = failed_provider(exc)
            if provider and failover_worthy(exc):
                replacement = ledger.failover_provider(provider, str(exc))
                if replacement:
                    cases = (
                        db.query(QuestionBenchmarkCase)
                        .order_by(QuestionBenchmarkCase.case_id)
                        .all()
                    )
                    assignments = ledger.active_assignments()
                    providers = tuple(spec for _, spec in assignments)
                    print(json.dumps({
                        "status": "provider_failover",
                        "failed_provider": provider,
                        "replacement_provider": replacement.provider,
                        "replacement_model": replacement.model,
                        "retrying_round_start": round_start,
                        "active_providers": [spec.provider for spec in providers],
                        "database_connection_recovered": bool(rollback_error),
                    }, ensure_ascii=False))
                    continue
            ledger.pause(exc, completed_sets)
            print(json.dumps({
                "status": "paused",
                "completed_sets": completed_sets,
                "failed_round_start": round_start,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "rollback_error": rollback_error,
                "provider_pool": ledger.pool_status(),
            }, ensure_ascii=False, indent=2))
            return 1

        completed_sets += len(providers)
        rounds_run += 1
        ledger.mark_round_complete(completed_sets)
        ledger.rotate_active_roster()
        print(json.dumps({
            "status": "running",
            "completed_sets": completed_sets,
            "target_sets": target_sets,
            "completed_rounds": completed_sets // ACTIVE_PROVIDER_COUNT,
            "next_active_providers": [
                spec.provider for spec in ledger.active_roster()
            ],
        }, ensure_ascii=False))

    ledger.complete(completed_sets)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a balanced, resumable multi-provider weak-supervision dataset."
    )
    parser.add_argument("--target-sets", type=int, default=1000)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--seed-synthetic-cases", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--provider-token-budget",
        type=int,
        default=settings.DATASET_PROVIDER_TOKEN_BUDGET,
        help="Pause when any provider reaches this cumulative token count; 0 disables the soft cap.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db = get_standalone_db()
    try:
        status = generation_status_with_retry(db)
        if args.seed_synthetic_cases:
            result = seed_synthetic_cases(db)
            print(json.dumps({
                "seed": result,
                "status": generation_status_with_retry(db),
            }, ensure_ascii=False, indent=2))
            return 0
        if args.status:
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0
        print(json.dumps({"preflight": status}, ensure_ascii=False, indent=2))
        ledger = GenerationLedger(
            args.checkpoint,
            args.target_sets,
            args.provider_token_budget,
        )
        return generate(
            db,
            target_sets=args.target_sets,
            max_rounds=args.max_rounds,
            ledger=ledger,
        )
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
