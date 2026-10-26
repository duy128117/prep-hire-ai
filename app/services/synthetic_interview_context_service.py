from __future__ import annotations

import random
from dataclasses import dataclass


ROLE_SKILLS = {
    "backend": ("Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Git"),
    "frontend": ("TypeScript", "React", "CSS", "REST API", "Testing", "Git"),
    "data": ("Python", "SQL", "Pandas", "Airflow", "Data Modeling", "Docker"),
    "devops": ("Linux", "Docker", "Kubernetes", "CI/CD", "AWS", "Monitoring"),
    "qa": ("Test Design", "API Testing", "Playwright", "SQL", "CI/CD", "Git"),
    "mobile": ("Flutter", "Dart", "REST API", "SQLite", "Testing", "Git"),
}
ROLE_TITLES = {
    "backend": "Backend Developer",
    "frontend": "Frontend Developer",
    "data": "Data Engineer",
    "devops": "DevOps Engineer",
    "qa": "QA Automation Engineer",
    "mobile": "Mobile Developer",
}
LEVEL_YEARS = {"intern": 0, "junior": 1, "middle": 3, "senior": 6}


@dataclass(frozen=True)
class SyntheticInterviewContext:
    seed: int
    role: str
    level: str
    cv_text: str
    jd_text: str


class SyntheticInterviewContextService:
    @classmethod
    def generate(
        cls,
        seed: int,
        role: str | None = None,
        level: str | None = None,
    ) -> SyntheticInterviewContext:
        rng = random.Random(seed)
        selected_role = role or rng.choice(sorted(ROLE_SKILLS))
        selected_level = level or rng.choice(sorted(LEVEL_YEARS))
        if selected_role not in ROLE_SKILLS:
            raise ValueError(f"Unsupported synthetic role: {selected_role}")
        if selected_level not in LEVEL_YEARS:
            raise ValueError(f"Unsupported synthetic level: {selected_level}")

        skills = list(ROLE_SKILLS[selected_role])
        rng.shuffle(skills)
        years = LEVEL_YEARS[selected_level]
        title = ROLE_TITLES[selected_role]
        project_scale = rng.choice((1200, 3500, 8000, 15000))
        improvement = rng.choice((18, 24, 31, 40))
        candidate_skills = skills[:4] if selected_level in {"intern", "junior"} else skills[:5]
        required_skills = skills[:5]

        cv_text = (
            f"SYNTHETIC PROFILE - seed {seed}.\n"
            f"Target role: {selected_level.title()} {title}.\n"
            f"Experience: {years} years of project experience.\n"
            f"Skills: {', '.join(candidate_skills)}.\n"
            f"Project: Built a {selected_role} system serving approximately "
            f"{project_scale} requests or records per day. I implemented core features, "
            f"tests, and deployment checks. The work improved the selected reliability "
            f"or delivery metric by {improvement} percent.\n"
            "Collaboration: Used code review, issue tracking, and short technical "
            "retrospectives to communicate decisions and risks."
        )
        jd_text = (
            f"SYNTHETIC JOB DESCRIPTION - seed {seed}.\n"
            f"Position: {selected_level.title()} {title}.\n"
            f"Required experience: {years} or more years, with equivalent project "
            "experience accepted for intern and junior candidates.\n"
            f"Core skills: {', '.join(required_skills)}.\n"
            "Responsibilities: design and implement maintainable features, diagnose "
            "production issues, write tests, explain technical trade-offs, and work "
            "with a cross-functional delivery team.\n"
            "Evaluation focus: technical correctness, problem solving, concrete "
            "evidence, communication, and learning adaptability."
        )
        return SyntheticInterviewContext(
            seed=seed,
            role=selected_role,
            level=selected_level,
            cv_text=cv_text,
            jd_text=jd_text,
        )
