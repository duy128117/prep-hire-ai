from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.session import SessionLocal
from app.services.interview_service import InterviewSessionService
from app.services.synthetic_interview_context_service import (
    SyntheticInterviewContextService,
)


def synthetic_answer(stage: str, question: str, seed: int) -> str:
    evidence = {
        "introduction": (
            "My most relevant experience is building and testing a production-style "
            "project in the target stack. I owned implementation, documented trade-offs, "
            "and tracked delivery and reliability outcomes."
        ),
        "cv_evidence": (
            "In the project described in my profile, I implemented the main workflow, "
            "added automated tests, and reviewed failures after deployment. I measured "
            "the result against request volume and the improvement metric in the CV."
        ),
        "technical": (
            "I would begin with requirements and measurable constraints, compare at least "
            "two implementation options, and choose based on reliability, maintainability, "
            "security, and operational cost. I would validate the decision with tests and monitoring."
        ),
        "problem_solving": (
            "I would first confirm impact and recent changes, inspect logs and metrics, "
            "reproduce the issue, isolate the failing layer, and test one hypothesis at a time. "
            "After recovery I would add a regression test and a prevention action."
        ),
        "behavioral": (
            "When requirements changed, I clarified the new acceptance criteria, split the "
            "work into critical and optional parts, communicated the delivery risk, and "
            "updated tests before implementation. The team delivered the critical scope first."
        ),
        "follow_up": (
            "A concrete example is the synthetic project in my CV: I owned the implementation, "
            "measured the baseline, introduced the change, verified it with tests, and compared "
            "the resulting metric with the baseline before release."
        ),
        "closing": (
            "I would add evidence from code reviews, automated tests, incident analysis, and "
            "measured project outcomes. These show both technical readiness and an ability to learn."
        ),
    }
    return f"{evidence.get(stage, evidence['technical'])} Synthetic run seed: {seed}."


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description="Create and complete clearly labeled synthetic interview sessions."
    )
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--language", choices=("vi", "en"), default="vi")
    parser.add_argument("--allow-external-ai", action="store_true")
    parser.add_argument("--dry-run-contexts", action="store_true")
    args = parser.parse_args()
    if args.sessions < 1:
        raise ValueError("--sessions must be at least 1")
    if not 5 <= args.max_questions <= 15:
        raise ValueError("--max-questions must be between 5 and 15")

    results: list[dict[str, object]] = []
    for offset in range(args.sessions):
        seed = args.start_seed + offset
        context = SyntheticInterviewContextService.generate(seed)
        if args.dry_run_contexts:
            results.append(
                {
                    "seed": seed,
                    "role": context.role,
                    "level": context.level,
                    "cv_text": context.cv_text,
                    "jd_text": context.jd_text,
                }
            )
            continue

        db = SessionLocal()
        try:
            session = InterviewSessionService.create(
                db=db,
                cv_text=context.cv_text,
                jd_text=context.jd_text,
                language=args.language,
                max_questions=args.max_questions,
                allow_external_ai=args.allow_external_ai,
                research_consent=False,
                context_source="synthetic",
            )
            while session.status == "active":
                turn = next(
                    item
                    for item in session.turns
                    if item.turn_index == session.current_turn_index
                )
                session = InterviewSessionService.submit_answer(
                    db,
                    session.session_id,
                    synthetic_answer(turn.stage, turn.question_text, seed),
                )
            results.append(
                {
                    "seed": seed,
                    "session_id": session.session_id,
                    "status": session.status,
                    "turns": len(session.turns),
                    "role": context.role,
                    "level": context.level,
                    "context_source": session.context_source,
                }
            )
        finally:
            db.close()
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
