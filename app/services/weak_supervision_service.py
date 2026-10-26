from __future__ import annotations

import hashlib
import json
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.question_benchmark import QuestionBenchmarkOutput, QuestionBenchmarkRun
from app.models.weak_supervision import (
    AnswerAIJudgment,
    AnswerHumanAudit,
    QuestionAIJudgment,
    SyntheticAnswerSample,
)
from app.research.rubrics import (
    ANSWER_QUALITY_RUBRIC,
    QUESTION_QUALITY_RUBRIC,
    TURN_EVIDENCE_RUBRIC,
)
from app.services.evaluation_service import ExpertConsensusService
from app.services.llm_provider_service import LLMProviderError, LLMProviderService


QUESTION_AI_RUBRIC_VERSION = "question-quality-ai-v1"
ANSWER_AI_RUBRIC_VERSION = "answer-quality-ai-v1"
QUESTION_JUDGE_PROMPT_VERSION = "question-ai-judge-v1"
ANSWER_GENERATION_PROMPT_VERSION = "synthetic-answer-v1"
ANSWER_JUDGE_PROMPT_VERSION = "answer-ai-judge-v1"
TURN_EVIDENCE_AI_RUBRIC_VERSION = "turn-evidence-ai-v2"
TURN_EVIDENCE_JUDGE_PROMPT_VERSION = "turn-evidence-ai-judge-v2"
DEBIASED_ANSWER_GENERATION_PROMPT_VERSION = "synthetic-answer-v1.1"
MINIMUM_AI_JUDGES = 3
QUESTION_CRITERIA = tuple(QUESTION_QUALITY_RUBRIC["criteria"])
ANSWER_CRITERIA = tuple(ANSWER_QUALITY_RUBRIC["criteria"])
TURN_EVIDENCE_CRITERIA = tuple(TURN_EVIDENCE_RUBRIC["criteria"])


@dataclass(frozen=True)
class AIJudgmentResult:
    scores: dict[str, float]
    evidence: dict[str, str]
    recommendation: str
    prompt_hash: str
    latency_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedAnswerResult:
    answer: str
    prompt_hash: str
    latency_ms: int
    actual_word_count: int = 0
    length_compliance: bool = True


class WeakSupervisionEngine:
    JSON_MAX_ATTEMPTS = 3
    JUDGMENT_MAX_ATTEMPTS = 3

    @staticmethod
    def call_json(
        model_name: str,
        prompt: str,
        operation: str = "weak_supervision",
    ) -> tuple[dict[str, Any], str, int]:
        if not settings.CV_AI_ENRICHMENT_ENABLED:
            raise ValueError("External AI is not configured")
        correction = ""
        total_latency_ms = 0
        for attempt in range(1, WeakSupervisionEngine.JSON_MAX_ATTEMPTS + 1):
            effective_prompt = prompt + correction
            prompt_hash = hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest()
            try:
                response = LLMProviderService.generate_json(
                    model_name,
                    effective_prompt,
                    operation=operation,
                    max_output_tokens=2048,
                    temperature=0.1,
                )
                total_latency_ms += response.latency_ms
                return response.data, prompt_hash, total_latency_ms
            except LLMProviderError as exc:
                if exc.error_code != "invalid_json" or attempt >= WeakSupervisionEngine.JSON_MAX_ATTEMPTS:
                    raise
                correction = (
                    "\nYour previous response was not valid JSON. Retry the same task and "
                    "return exactly one complete JSON object with double-quoted keys and strings. "
                    "Do not use markdown fences, comments, trailing commas, or explanatory text."
                )
        raise AssertionError("unreachable")

    @staticmethod
    def parse_judgment(data: dict[str, Any], criteria: tuple[str, ...]) -> tuple[dict[str, float], dict[str, str], str]:
        raw_scores = data.get("scores")
        if not isinstance(raw_scores, dict):
            raise ValueError("AI judgment is missing scores")
        scores: dict[str, float] = {}
        for criterion in criteria:
            value = raw_scores.get(criterion)
            if isinstance(value, bool):
                raise ValueError(f"AI judgment is missing a valid score for {criterion}")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"AI judgment is missing a valid score for {criterion}"
                ) from exc
            if numeric < 1 or numeric > 5:
                raise ValueError(f"AI judgment score for {criterion} must be between 1 and 5")
            scores[criterion] = numeric

        raw_evidence = data.get("evidence")
        evidence = {
            criterion: str(raw_evidence.get(criterion) or "").strip()[:1200]
            for criterion in criteria
        } if isinstance(raw_evidence, dict) else {criterion: "" for criterion in criteria}
        recommendation = str(data.get("recommendation") or "review").strip().lower()
        if recommendation not in {"accept", "review", "reject"}:
            recommendation = "review"
        return scores, evidence, recommendation

    @classmethod
    def call_judgment(
        cls,
        model_name: str,
        prompt: str,
        operation: str,
        criteria: tuple[str, ...],
    ) -> AIJudgmentResult:
        correction = ""
        total_latency_ms = 0
        for attempt in range(1, cls.JUDGMENT_MAX_ATTEMPTS + 1):
            data, prompt_hash, latency_ms = cls.call_json(
                model_name,
                prompt + correction,
                operation,
            )
            total_latency_ms += latency_ms
            try:
                scores, evidence, recommendation = cls.parse_judgment(data, criteria)
                return AIJudgmentResult(
                    scores,
                    evidence,
                    recommendation,
                    prompt_hash,
                    total_latency_ms,
                    {
                        key: data.get(key)
                        for key in (
                            "applicability",
                            "missing_signals",
                            "incorrect_claims",
                        )
                        if key in data
                    },
                )
            except ValueError as exc:
                if attempt >= cls.JUDGMENT_MAX_ATTEMPTS:
                    raise
                correction = (
                    "\nYour previous JSON violated the rubric validation: "
                    f"{exc}. Return a corrected complete JSON object only."
                )
        raise AssertionError("unreachable")


class QuestionAIJudgeEngine:
    @staticmethod
    def build_prompt(output: QuestionBenchmarkOutput) -> str:
        benchmark_case = output.run.case
        context = {
            "cv_profile": benchmark_case.cv_profile_json,
            "job_profile": benchmark_case.job_profile_json,
            "cv_excerpt": benchmark_case.cv_text[:5000],
            "jd_excerpt": benchmark_case.jd_text[:5000],
            "interview_history": benchmark_case.interview_history_json[-4:],
            "stage": benchmark_case.stage,
            "target_criterion": benchmark_case.criterion,
            "candidate_question": output.question_text,
        }
        return (
            "You are an independent blind judge of one AI-generated interview question. "
            "The generator identity is intentionally hidden. Judge only the supplied evidence. "
            "Do not infer or mention which model generated it. Use the complete 1-5 rubric and "
            "return valid JSON only. A score of 3 means acceptable, not excellent. "
            "Every score is mandatory and must be a number from 1 through 5 inclusive; never use "
            "0, null, N/A, or omit a criterion. If there is no prior interview answer, score "
            "adaptivity by how well the question fits the scheduled stage and supplied context.\n"
            f"Rubric: {json.dumps(QUESTION_QUALITY_RUBRIC, ensure_ascii=False)}\n"
            '{"scores": {"context_relevance": number, "technical_depth": number, '
            '"clarity": number, "evidence_elicitation": number, "adaptivity": number, '
            '"non_redundancy": number, "fairness": number}, '
            '"evidence": {"criterion_name": "brief evidence"}, '
            '"recommendation": "accept|review|reject"}\n'
            f"Evaluation context: {json.dumps(context, ensure_ascii=False)}"
        )

    @classmethod
    def judge(cls, output: QuestionBenchmarkOutput, model_name: str) -> AIJudgmentResult:
        prompt = cls.build_prompt(output)
        return WeakSupervisionEngine.call_judgment(
            model_name,
            prompt,
            "question_judgment",
            QUESTION_CRITERIA,
        )


class SyntheticAnswerEngine:
    QUALITY_GUIDANCE = {
        1: "Incorrect, irrelevant, or unsupported; it should clearly fail to demonstrate the target competency.",
        2: "Partially relevant but vague, shallow, or materially flawed.",
        3: "Basically correct and relevant, but lacking depth, evidence, or important trade-offs.",
        4: "Correct, structured, and supported by a concrete example or sound reasoning.",
        5: "Excellent depth, precise reasoning, concrete evidence, trade-offs, and measurable outcomes.",
    }
    LENGTH_GUIDANCE = {
        "short": (20, 80),
        "medium": (81, 170),
        "long": (171, 320),
    }

    @classmethod
    def build_prompt(
        cls,
        output: QuestionBenchmarkOutput,
        target_quality_level: int,
        generation_metadata: dict[str, Any] | None = None,
    ) -> str:
        benchmark_case = output.run.case
        context = {
            "cv_profile": benchmark_case.cv_profile_json,
            "job_profile": benchmark_case.job_profile_json,
            "interview_history": benchmark_case.interview_history_json[-4:],
            "stage": benchmark_case.stage,
            "target_criterion": benchmark_case.criterion,
            "question": output.question_text,
            "question_metadata": output.generation_metadata_json or {},
        }
        language = "Vietnamese" if benchmark_case.language == "vi" else "English"
        metadata = generation_metadata or {}
        length_band = str(metadata.get("target_length_band") or "").strip()
        length_instruction = ""
        if length_band in cls.LENGTH_GUIDANCE:
            minimum, maximum = cls.LENGTH_GUIDANCE[length_band]
            length_instruction = (
                f" Target length band: {length_band}, approximately {minimum}-{maximum} words. "
                "Length and quality are independent: a short answer can be excellent and a long "
                "answer can be poor. Do not improve quality merely to fill the requested length."
            )
        return (
            "Act as a synthetic job candidate for research data generation. Generate one answer only. "
            "Do not reveal the requested quality level and do not add grading commentary. "
            "Stay consistent with the supplied CV and never invent protected personal attributes. "
            f"Answer language: {language}. Target quality behavior: {cls.QUALITY_GUIDANCE[target_quality_level]}"
            f"{length_instruction}\n"
            'Return valid JSON only: {"answer": string}.\n'
            f"Context: {json.dumps(context, ensure_ascii=False)}"
        )

    @classmethod
    def generate(
        cls,
        output: QuestionBenchmarkOutput,
        model_name: str,
        target_quality_level: int,
        generation_metadata: dict[str, Any] | None = None,
    ) -> GeneratedAnswerResult:
        prompt = cls.build_prompt(output, target_quality_level, generation_metadata)
        length_band = str((generation_metadata or {}).get("target_length_band") or "")
        target_range = cls.LENGTH_GUIDANCE.get(length_band)
        total_latency_ms = 0
        last_result: GeneratedAnswerResult | None = None
        correction = ""
        for attempt in range(1, 3):
            data, prompt_hash, latency_ms = WeakSupervisionEngine.call_json(
                model_name,
                prompt + correction,
                "answer_generation",
            )
            total_latency_ms += latency_ms
            answer = str(data.get("answer") or "").strip()
            if len(answer) < 2 or len(answer) > 10000:
                raise ValueError("Generator model returned an invalid synthetic answer")
            word_count = len(answer.split())
            compliant = not target_range or target_range[0] <= word_count <= target_range[1]
            last_result = GeneratedAnswerResult(
                answer,
                prompt_hash,
                total_latency_ms,
                word_count,
                compliant,
            )
            if compliant or not target_range:
                return last_result
            if attempt == 1:
                correction = (
                    f"\nThe previous answer had {word_count} words and missed the required "
                    f"{target_range[0]}-{target_range[1]} word range. Regenerate the same target "
                    "quality without mentioning these instructions. Return JSON only."
                )
        if last_result is None:
            raise AssertionError("unreachable")
        return last_result


class AnswerAIJudgeEngine:
    @staticmethod
    def build_prompt(sample: SyntheticAnswerSample) -> str:
        output = sample.question_output
        benchmark_case = output.run.case
        context = {
            "cv_profile": benchmark_case.cv_profile_json,
            "job_profile": benchmark_case.job_profile_json,
            "interview_history": benchmark_case.interview_history_json[-4:],
            "stage": benchmark_case.stage,
            "target_criterion": benchmark_case.criterion,
            "question": output.question_text,
            "candidate_answer": sample.answer_text,
        }
        return (
            "You are an independent blind judge of one synthetic interview answer. "
            "The answer generator and requested target quality are intentionally hidden. "
            "Judge only the answer against the supplied context. Return valid JSON only. "
            "A score of 3 means acceptable, not excellent. Every score is mandatory and must be "
            "a number from 1 through 5 inclusive; never use 0, null, N/A, or omit a criterion.\n"
            f"Rubric: {json.dumps(ANSWER_QUALITY_RUBRIC, ensure_ascii=False)}\n"
            '{"scores": {"answer_relevance": number, "technical_correctness": number, '
            '"reasoning_depth": number, "evidence_specificity": number, "clarity": number, '
            '"consistency": number, "criterion_score": number}, '
            '"evidence": {"criterion_name": "brief evidence"}, '
            '"recommendation": "accept|review|reject"}\n'
            f"Evaluation context: {json.dumps(context, ensure_ascii=False)}"
        )

    @classmethod
    def judge(cls, sample: SyntheticAnswerSample, model_name: str) -> AIJudgmentResult:
        prompt = cls.build_prompt(sample)
        return WeakSupervisionEngine.call_judgment(
            model_name,
            prompt,
            "answer_judgment",
            ANSWER_CRITERIA,
        )


class TurnEvidenceAIJudgeEngine:
    @staticmethod
    def build_prompt(sample: SyntheticAnswerSample) -> str:
        output = sample.question_output
        benchmark_case = output.run.case
        question_metadata = output.generation_metadata_json or {}
        applicable = set(question_metadata.get("applicable_answer_criteria") or ())
        applicability = {
            criterion: criterion in applicable
            for criterion in TURN_EVIDENCE_CRITERIA
        }
        context = {
            "cv_profile": benchmark_case.cv_profile_json,
            "job_profile": benchmark_case.job_profile_json,
            "interview_history": benchmark_case.interview_history_json[-4:],
            "stage": benchmark_case.stage,
            "target_criterion": benchmark_case.criterion,
            "question": output.question_text,
            "question_type": question_metadata.get("question_type"),
            "difficulty": question_metadata.get("difficulty"),
            "required_signals": question_metadata.get("required_signals") or [],
            "bonus_signals": question_metadata.get("bonus_signals") or [],
            "contradiction_rules": question_metadata.get("contradiction_rules") or [],
            "applicability": applicability,
            "candidate_answer": sample.answer_text,
        }
        return (
            "You are an independent blind judge of one interview turn. The answer generator, "
            "requested quality, and requested length are hidden. Never reward verbosity by itself. "
            "A concise answer can score 5 when it fully satisfies a concise question. Judge every "
            "criterion from 1 through 5, but downstream training will mask criteria marked false in "
            "the supplied applicability map. A score of 3 means acceptable basic evidence. Return "
            "valid JSON only.\n"
            f"Rubric: {json.dumps(TURN_EVIDENCE_RUBRIC, ensure_ascii=False)}\n"
            '{"scores": {"answer_relevance": number, "technical_correctness": number, '
            '"expected_signal_coverage": number, "reasoning_quality": number, '
            '"evidence_quality": number, "clarity_conciseness": number, '
            '"consistency": number}, "evidence": {"criterion_name": "brief evidence"}, '
            '"applicability": {"criterion_name": boolean}, '
            '"missing_signals": [string], "incorrect_claims": [string], '
            '"recommendation": "accept|review|reject"}\n'
            f"Evaluation context: {json.dumps(context, ensure_ascii=False)}"
        )

    @classmethod
    def judge(cls, sample: SyntheticAnswerSample, model_name: str) -> AIJudgmentResult:
        return WeakSupervisionEngine.call_judgment(
            model_name,
            cls.build_prompt(sample),
            "turn_evidence_judgment",
            TURN_EVIDENCE_CRITERIA,
        )

class WeakSupervisionService:
    @staticmethod
    def _unique_models(model_names: list[str]) -> list[str]:
        normalized = []
        for name in model_names:
            value = name.strip()
            try:
                value = LLMProviderService.resolve(value).model
            except ValueError:
                pass
            normalized.append(value)
        if len({name.lower() for name in normalized}) != len(normalized):
            raise ValueError("Judge model names must be unique")
        return normalized

    @staticmethod
    def _provider_for(model_name: str) -> str:
        try:
            return LLMProviderService.resolve(model_name).provider
        except ValueError:
            return "unknown"

    @staticmethod
    def _consensus(annotations: list[Any], criteria: tuple[str, ...]) -> dict[str, Any]:
        judge_count = len(annotations)
        if judge_count < MINIMUM_AI_JUDGES:
            return {
                "status": "insufficient_ai_judges",
                "ai_judge_count": judge_count,
                "pseudo_label_scores": None,
                "criterion_stats": {},
                "mean_quality_score": None,
                "inter_judge_icc_2_k": None,
                "needs_human_review": True,
                "decision": "review",
            }

        stats: dict[str, dict[str, float]] = {}
        disagreement = False
        for criterion in criteria:
            values = [float(item.scores_json[criterion]) for item in annotations]
            score_range = max(values) - min(values)
            disagreement = disagreement or score_range >= 2
            stats[criterion] = {
                "median": round(float(statistics.median(values)), 3),
                "mean": round(float(statistics.mean(values)), 3),
                "standard_deviation": round(float(statistics.pstdev(values)), 3),
                "minimum": round(min(values), 3),
                "maximum": round(max(values), 3),
            }
        pseudo_labels = {criterion: stats[criterion]["median"] for criterion in criteria}
        mean_score = round(float(statistics.mean(pseudo_labels.values())), 3)
        matrix = np.asarray(
            [[float(item.scores_json[criterion]) for item in annotations] for criterion in criteria],
            dtype=float,
        )
        if "fairness" in pseudo_labels and (pseudo_labels["fairness"] < 3 or pseudo_labels["clarity"] < 3):
            decision = "reject"
        elif mean_score >= 3:
            decision = "accept"
        else:
            decision = "review"
        return {
            "status": "pseudo_label_ready",
            "ai_judge_count": judge_count,
            "pseudo_label_scores": pseudo_labels,
            "criterion_stats": stats,
            "mean_quality_score": mean_score,
            "inter_judge_icc_2_k": ExpertConsensusService._icc_2_k(matrix),
            "needs_human_review": disagreement or decision != "accept",
            "decision": decision,
        }

    @classmethod
    def judge_question_run(cls, db: Session, run_id: str, judge_model_names: list[str]) -> dict[str, Any]:
        run = db.query(QuestionBenchmarkRun).filter(QuestionBenchmarkRun.run_id == run_id).first()
        if not run:
            raise ValueError("Benchmark run not found")
        models = cls._unique_models(judge_model_names)
        successful_outputs = [output for output in run.outputs if output.status == "success"]
        if not successful_outputs:
            raise ValueError("Benchmark run has no successful question outputs")
        for output in successful_outputs:
            eligible = [name for name in models if name.lower() != output.model_name.lower()]
            if len(eligible) < MINIMUM_AI_JUDGES:
                raise ValueError(
                    f"Output {output.blind_label} needs at least three judge models excluding its generator"
                )

        created = 0
        skipped = 0
        failures: list[dict[str, str]] = []
        for output in successful_outputs:
            for judge_model in models:
                if judge_model.lower() == output.model_name.lower():
                    continue
                existing = db.query(QuestionAIJudgment).filter(
                    QuestionAIJudgment.output_id == output.output_id,
                    QuestionAIJudgment.judge_model_name == judge_model,
                    QuestionAIJudgment.rubric_version == QUESTION_AI_RUBRIC_VERSION,
                ).first()
                if existing:
                    skipped += 1
                    continue
                try:
                    result = QuestionAIJudgeEngine.judge(output, judge_model)
                    db.add(QuestionAIJudgment(
                        output_id=output.output_id,
                        provider=cls._provider_for(judge_model),
                        judge_model_name=judge_model,
                        rubric_version=QUESTION_AI_RUBRIC_VERSION,
                        scores_json=result.scores,
                        evidence_json=result.evidence,
                        recommendation=result.recommendation,
                        prompt_version=QUESTION_JUDGE_PROMPT_VERSION,
                        prompt_hash=result.prompt_hash,
                        latency_ms=result.latency_ms,
                    ))
                    db.commit()
                    created += 1
                except Exception as exc:
                    db.rollback()
                    failures.append({"blind_label": output.blind_label, "judge_model": judge_model, "error": str(exc)})
        return {"run_id": run_id, "created": created, "skipped": skipped, "failures": failures}

    @classmethod
    def question_consensus(cls, output: QuestionBenchmarkOutput) -> dict[str, Any]:
        annotations = [
            item for item in output.ai_judgments if item.rubric_version == QUESTION_AI_RUBRIC_VERSION
        ]
        return cls._consensus(annotations, QUESTION_CRITERIA)

    @classmethod
    def generate_answers(
        cls,
        db: Session,
        output_id: str,
        generator_model_name: str,
        target_quality_levels: list[int],
    ) -> list[SyntheticAnswerSample]:
        output = db.query(QuestionBenchmarkOutput).filter(QuestionBenchmarkOutput.output_id == output_id).first()
        if not output or output.status != "success" or not output.question_text:
            raise ValueError("A successful benchmark question output is required")
        samples: list[SyntheticAnswerSample] = []
        for quality_level in target_quality_levels:
            sample = SyntheticAnswerSample(
                sample_id=str(uuid.uuid4()),
                question_output_id=output.output_id,
                blind_label=f"A-{uuid.uuid4().hex[:8].upper()}",
                provider=cls._provider_for(generator_model_name),
                generator_model_name=generator_model_name,
                target_quality_level=quality_level,
                status="generating",
                prompt_version=ANSWER_GENERATION_PROMPT_VERSION,
                prompt_hash="",
                generation_latency_ms=0,
            )
            try:
                result = SyntheticAnswerEngine.generate(output, generator_model_name, quality_level)
                sample.answer_text = result.answer
                sample.prompt_hash = result.prompt_hash
                sample.generation_latency_ms = result.latency_ms
                sample.status = "success"
            except Exception as exc:
                sample.prompt_hash = hashlib.sha256(
                    SyntheticAnswerEngine.build_prompt(output, quality_level).encode("utf-8")
                ).hexdigest()
                sample.status = "failed"
                sample.error_message = str(exc)[:2000]
            db.add(sample)
            samples.append(sample)
        db.commit()
        for sample in samples:
            db.refresh(sample)
        return samples

    @staticmethod
    def get_answer_sample(db: Session, sample_id: str) -> SyntheticAnswerSample | None:
        return db.query(SyntheticAnswerSample).filter(SyntheticAnswerSample.sample_id == sample_id).first()

    @classmethod
    def judge_answer_sample(cls, db: Session, sample_id: str, judge_model_names: list[str]) -> dict[str, Any]:
        sample = cls.get_answer_sample(db, sample_id)
        if not sample or sample.status != "success" or not sample.answer_text:
            raise ValueError("A successful synthetic answer sample is required")
        models = cls._unique_models(judge_model_names)
        eligible = [name for name in models if name.lower() != sample.generator_model_name.lower()]
        if len(eligible) < MINIMUM_AI_JUDGES:
            raise ValueError("At least three judge models excluding the answer generator are required")

        created = 0
        skipped = 0
        failures: list[dict[str, str]] = []
        for judge_model in eligible:
            existing = db.query(AnswerAIJudgment).filter(
                AnswerAIJudgment.sample_id == sample_id,
                AnswerAIJudgment.judge_model_name == judge_model,
                AnswerAIJudgment.rubric_version == ANSWER_AI_RUBRIC_VERSION,
            ).first()
            if existing:
                skipped += 1
                continue
            try:
                result = AnswerAIJudgeEngine.judge(sample, judge_model)
                db.add(AnswerAIJudgment(
                    sample_id=sample.sample_id,
                    provider=cls._provider_for(judge_model),
                    judge_model_name=judge_model,
                    rubric_version=ANSWER_AI_RUBRIC_VERSION,
                    scores_json=result.scores,
                    evidence_json=result.evidence,
                    recommendation=result.recommendation,
                    prompt_version=ANSWER_JUDGE_PROMPT_VERSION,
                    prompt_hash=result.prompt_hash,
                    latency_ms=result.latency_ms,
                ))
                db.commit()
                created += 1
            except Exception as exc:
                db.rollback()
                failures.append({"judge_model": judge_model, "error": str(exc)})
        return {"sample_id": sample_id, "created": created, "skipped": skipped, "failures": failures}

    @classmethod
    def answer_consensus(cls, sample: SyntheticAnswerSample) -> dict[str, Any]:
        annotations = [
            item for item in sample.ai_judgments if item.rubric_version == ANSWER_AI_RUBRIC_VERSION
        ]
        return cls._consensus(annotations, ANSWER_CRITERIA)

    @classmethod
    def turn_evidence_consensus(cls, sample: SyntheticAnswerSample) -> dict[str, Any]:
        annotations = [
            item
            for item in sample.ai_judgments
            if item.rubric_version == TURN_EVIDENCE_AI_RUBRIC_VERSION
        ]
        if len(annotations) < MINIMUM_AI_JUDGES:
            return {
                "status": "insufficient_ai_judges",
                "ai_judge_count": len(annotations),
                "pseudo_label_scores": None,
                "criterion_stats": {},
                "applicability": {},
                "mean_quality_score": None,
                "needs_human_review": True,
                "decision": "review",
            }

        output_metadata = sample.question_output.generation_metadata_json or {}
        applicable_names = set(
            output_metadata.get("applicable_answer_criteria") or ()
        )
        applicability = {
            criterion: criterion in applicable_names
            for criterion in TURN_EVIDENCE_CRITERIA
        }
        stats: dict[str, dict[str, float]] = {}
        pseudo_labels: dict[str, float | None] = {}
        disagreement = False
        for criterion in TURN_EVIDENCE_CRITERIA:
            if not applicability[criterion]:
                pseudo_labels[criterion] = None
                continue
            values = [float(item.scores_json[criterion]) for item in annotations]
            score_range = max(values) - min(values)
            disagreement = disagreement or score_range >= 2
            stats[criterion] = {
                "median": round(float(statistics.median(values)), 3),
                "mean": round(float(statistics.mean(values)), 3),
                "standard_deviation": round(float(statistics.pstdev(values)), 3),
                "minimum": round(min(values), 3),
                "maximum": round(max(values), 3),
            }
            pseudo_labels[criterion] = stats[criterion]["median"]

        usable_scores = [
            float(score) for score in pseudo_labels.values() if score is not None
        ]
        mean_score = (
            round(float(statistics.mean(usable_scores)), 3)
            if usable_scores
            else None
        )
        if mean_score is None:
            decision = "review"
        elif mean_score >= 3:
            decision = "accept"
        else:
            decision = "review"
        return {
            "status": "pseudo_label_ready",
            "ai_judge_count": len(annotations),
            "pseudo_label_scores": pseudo_labels,
            "criterion_stats": stats,
            "applicability": applicability,
            "mean_quality_score": mean_score,
            "needs_human_review": disagreement or decision != "accept",
            "decision": decision,
        }

    @classmethod
    def create_answer_human_audit(
        cls,
        db: Session,
        sample_id: str,
        reviewer_id: str,
        scores: dict[str, float],
        notes: str | None,
    ) -> AnswerHumanAudit:
        sample = cls.get_answer_sample(db, sample_id)
        if not sample or sample.status != "success":
            raise ValueError("A successful synthetic answer sample is required")
        normalized_id = reviewer_id.strip()
        existing = db.query(AnswerHumanAudit).filter(
            AnswerHumanAudit.sample_id == sample_id,
            AnswerHumanAudit.reviewer_id == normalized_id,
            AnswerHumanAudit.rubric_version == ANSWER_AI_RUBRIC_VERSION,
        ).first()
        if existing:
            raise ValueError("Human audit already exists and is locked")
        audit = AnswerHumanAudit(
            sample_id=sample_id,
            reviewer_id=normalized_id,
            rubric_version=ANSWER_AI_RUBRIC_VERSION,
            scores_json=scores,
            notes=notes,
        )
        db.add(audit)
        db.commit()
        db.refresh(audit)
        return audit
