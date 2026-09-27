from __future__ import annotations

import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.v1.endpoints.question_benchmark import get_blind_answer_sample, get_blind_benchmark
from app.db.base_class import Base
from app.models.interview_session import InterviewSession
from app.models.interview_turn import InterviewTurn
from app.models.question_benchmark import (
    QuestionBenchmarkCase,
    QuestionBenchmarkOutput,
    QuestionBenchmarkRun,
    QuestionQualityAnnotation,
)
from app.models.weak_supervision import (
    AnswerAIJudgment,
    AnswerHumanAudit,
    QuestionAIJudgment,
    SyntheticAnswerSample,
)
from app.services.interview_service import GeneratedQuestion
from app.services.question_benchmark_service import QUESTION_CRITERIA, QuestionBenchmarkService
from app.services.weak_supervision_service import (
    ANSWER_CRITERIA,
    AIJudgmentResult,
    GeneratedAnswerResult,
    WeakSupervisionService,
)
from app.services.weak_supervision_dataset_service import WeakSupervisionDatasetBuilder


def _scores(offset: float = 0.0) -> dict[str, float]:
    return {criterion: 3.0 + offset for criterion in QUESTION_CRITERIA}


def _answer_scores(offset: float = 0.0) -> dict[str, float]:
    return {criterion: 3.0 + offset for criterion in ANSWER_CRITERIA}


class QuestionBenchmarkServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(
            self.engine,
            tables=[
                InterviewSession.__table__,
                InterviewTurn.__table__,
                QuestionBenchmarkCase.__table__,
                QuestionBenchmarkRun.__table__,
                QuestionBenchmarkOutput.__table__,
                QuestionQualityAnnotation.__table__,
                QuestionAIJudgment.__table__,
                SyntheticAnswerSample.__table__,
                AnswerAIJudgment.__table__,
                AnswerHumanAudit.__table__,
            ],
        )
        self.db = sessionmaker(bind=self.engine)()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def _add_session(self, consent: bool = True) -> InterviewSession:
        session = InterviewSession(
            session_id="benchmark-source-session",
            status="active",
            language="vi",
            max_questions=5,
            current_turn_index=1,
            current_stage="cv_evidence",
            allow_external_ai=True,
            research_consent=consent,
            cv_text="Backend intern, email candidate@example.com, dùng Python và FastAPI.",
            jd_text="Liên hệ 0912 345 678. Tuyển Backend Developer dùng Python, API và SQL.",
            cv_profile_json={"summary": "Portfolio https://candidate.example", "cv_skills": []},
            job_profile_json={"title": "Backend Developer", "job_skills": []},
        )
        session.turns.append(
            InterviewTurn(
                turn_index=0,
                stage="introduction",
                criterion="communication",
                question_text="Hãy giới thiệu về bạn.",
                expected_signal="Kinh nghiệm liên quan.",
                question_source="fallback",
                model_name=None,
                prompt_version="interview-question-v1",
                prompt_hash="a" * 64,
                generation_latency_ms=0,
                answer_text="Tôi là intern, email candidate@example.com và đã làm API bằng Python.",
                answer_word_count=12,
            )
        )
        self.db.add(session)
        self.db.commit()
        return session

    def test_case_requires_consent(self) -> None:
        session = self._add_session(consent=False)
        with self.assertRaisesRegex(ValueError, "Research consent"):
            QuestionBenchmarkService.create_case_from_session(self.db, session.session_id, 1)

    def test_case_snapshot_redacts_contact_data(self) -> None:
        session = self._add_session()
        benchmark_case = QuestionBenchmarkService.create_case_from_session(
            self.db,
            session.session_id,
            1,
        )

        self.assertNotIn("candidate@example.com", benchmark_case.cv_text)
        self.assertNotIn("0912 345 678", benchmark_case.jd_text)
        self.assertNotIn("candidate.example", str(benchmark_case.cv_profile_json))
        self.assertNotIn("candidate@example.com", str(benchmark_case.interview_history_json))
        self.assertEqual(benchmark_case.stage, "cv_evidence")

    def test_multi_llm_workflow_hides_models_until_three_ai_judges(self) -> None:
        session = self._add_session()
        benchmark_case = QuestionBenchmarkService.create_case_from_session(
            self.db,
            session.session_id,
            1,
        )

        def fake_generate(**kwargs):
            model_name = kwargs["model_name"]
            return GeneratedQuestion(
                stage="cv_evidence",
                criterion="experience_evidence",
                question=f"Câu hỏi do {model_name} tạo trong dữ liệu quản trị.",
                expected_signal="Bằng chứng dự án.",
                source="ai",
                model_name=model_name,
                prompt_hash="b" * 64,
                latency_ms=10,
            )

        with patch(
            "app.services.question_benchmark_service.AdaptiveQuestionEngine.generate",
            side_effect=fake_generate,
        ):
            run = QuestionBenchmarkService.run_models(
                self.db,
                benchmark_case.case_id,
                ["model-alpha", "model-beta"],
            )

        self.assertEqual(run.status, "completed")
        self.assertEqual(len(run.outputs), 2)
        blind_payload = get_blind_benchmark(run.run_id, self.db).model_dump_json()
        self.assertNotIn('"model_name"', blind_payload)
        self.assertNotIn('"provider"', blind_payload)

        early_summary = QuestionBenchmarkService.summarize_run(run)
        self.assertEqual(early_summary["status"], "insufficient_ai_judgments")
        self.assertEqual(early_summary["model_results"], [])

        def fake_judge(output, model_name):
            offset = {"judge-a": -0.5, "judge-b": 0.0, "judge-c": 0.5}[model_name]
            return AIJudgmentResult(
                scores=_scores(offset),
                evidence={criterion: "Blind evidence" for criterion in QUESTION_CRITERIA},
                recommendation="accept",
                prompt_hash="c" * 64,
                latency_ms=12,
            )

        with patch(
            "app.services.weak_supervision_service.QuestionAIJudgeEngine.judge",
            side_effect=fake_judge,
        ):
            judge_result = WeakSupervisionService.judge_question_run(
                self.db,
                run.run_id,
                ["judge-a", "judge-b", "judge-c"],
            )

        self.assertEqual(judge_result["created"], 6)
        self.assertEqual(judge_result["failures"], [])

        ready_summary = QuestionBenchmarkService.summarize_run(run)
        self.assertEqual(ready_summary["status"], "ready")
        self.assertEqual(len(ready_summary["model_results"]), 2)
        self.assertEqual(ready_summary["model_results"][0]["ai_judge_count"], 3)
        self.assertEqual(ready_summary["model_results"][0]["human_audit_count"], 0)
        self.assertEqual(ready_summary["model_results"][0]["mean_quality_score"], 3.0)
        question_rows, counters = WeakSupervisionDatasetBuilder.question_rows(self.db)
        self.assertEqual(counters["exported_question_rows"], 2)
        self.assertEqual(len(question_rows), 2)

    def test_generator_cannot_judge_its_own_question(self) -> None:
        session = self._add_session()
        benchmark_case = QuestionBenchmarkService.create_case_from_session(self.db, session.session_id, 1)

        with patch(
            "app.services.question_benchmark_service.AdaptiveQuestionEngine.generate",
            return_value=GeneratedQuestion(
                stage="cv_evidence",
                criterion="experience_evidence",
                question="Hãy mô tả bằng chứng dự án.",
                expected_signal="Bằng chứng dự án.",
                source="ai",
                model_name="generator-model",
                prompt_hash="d" * 64,
                latency_ms=10,
            ),
        ):
            run = QuestionBenchmarkService.run_models(
                self.db,
                benchmark_case.case_id,
                ["generator-model", "second-generator"],
            )

        with patch(
            "app.services.weak_supervision_service.QuestionAIJudgeEngine.judge",
            return_value=AIJudgmentResult(
                scores=_scores(),
                evidence={criterion: "Evidence" for criterion in QUESTION_CRITERIA},
                recommendation="accept",
                prompt_hash="e" * 64,
                latency_ms=5,
            ),
        ) as judge:
            WeakSupervisionService.judge_question_run(
                self.db,
                run.run_id,
                ["generator-model", "second-generator", "judge-c", "judge-d"],
            )

        for call in judge.call_args_list:
            output, judge_model = call.args
            self.assertNotEqual(output.model_name.lower(), judge_model.lower())

    def test_synthetic_answer_has_ai_labels_and_locked_human_audit(self) -> None:
        session = self._add_session()
        benchmark_case = QuestionBenchmarkService.create_case_from_session(self.db, session.session_id, 1)
        with patch(
            "app.services.question_benchmark_service.AdaptiveQuestionEngine.generate",
            return_value=GeneratedQuestion(
                stage="technical",
                criterion="technical_knowledge",
                question="Bạn xử lý transaction như thế nào?",
                expected_signal="Technical reasoning.",
                source="ai",
                model_name="question-generator",
                prompt_hash="f" * 64,
                latency_ms=10,
            ),
        ):
            run = QuestionBenchmarkService.run_models(
                self.db,
                benchmark_case.case_id,
                ["question-generator", "question-generator-2"],
            )
        output = run.outputs[0]

        with patch(
            "app.services.weak_supervision_service.SyntheticAnswerEngine.generate",
            return_value=GeneratedAnswerResult(
                answer="Tôi sử dụng transaction và rollback khi một thao tác thất bại.",
                prompt_hash="1" * 64,
                latency_ms=8,
            ),
        ):
            samples = WeakSupervisionService.generate_answers(
                self.db,
                output.output_id,
                "answer-generator",
                [2],
            )
        sample = samples[0]
        blind_json = get_blind_answer_sample(sample.sample_id, self.db).model_dump_json()
        self.assertNotIn("answer-generator", blind_json)
        self.assertNotIn("target_quality_level", blind_json)

        with patch(
            "app.services.weak_supervision_service.AnswerAIJudgeEngine.judge",
            return_value=AIJudgmentResult(
                scores=_answer_scores(),
                evidence={criterion: "Answer evidence" for criterion in ANSWER_CRITERIA},
                recommendation="accept",
                prompt_hash="2" * 64,
                latency_ms=7,
            ),
        ):
            result = WeakSupervisionService.judge_answer_sample(
                self.db,
                sample.sample_id,
                ["answer-generator", "judge-a", "judge-b", "judge-c"],
            )
        self.assertEqual(result["created"], 3)
        consensus = WeakSupervisionService.answer_consensus(sample)
        self.assertEqual(consensus["status"], "pseudo_label_ready")
        self.assertEqual(consensus["ai_judge_count"], 3)

        answer_rows, counters = WeakSupervisionDatasetBuilder.answer_rows(self.db)
        self.assertEqual(counters["exported_answer_rows"], 1)
        self.assertEqual(answer_rows[0]["target_quality_level_metadata"], 2)

        WeakSupervisionService.create_answer_human_audit(
            self.db,
            sample.sample_id,
            "reviewer-1",
            _answer_scores(),
            "Checked against the rubric.",
        )
        with self.assertRaisesRegex(ValueError, "locked"):
            WeakSupervisionService.create_answer_human_audit(
                self.db,
                sample.sample_id,
                "reviewer-1",
                _answer_scores(),
                None,
            )


if __name__ == "__main__":
    unittest.main()
