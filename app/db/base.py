from app.db.base_class import Base
from app.models.company import Company
from app.models.candidate_evaluation import CandidateEvaluation
from app.models.cv_analysis_result import CvAnalysisResult
from app.models.cv_skill import CvSkill
from app.models.job import Job
from app.models.job_upload import JobUpload
from app.models.job_skill import JobSkill
from app.models.interview_session import InterviewSession
from app.models.interview_turn import InterviewTurn
from app.models.expert_annotation import ExpertAnnotation
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
from app.models.skill import Skill
from app.models.skill_course import SkillCourse
from app.models.skill_gap import SkillGap
