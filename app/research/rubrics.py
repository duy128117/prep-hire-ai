QUESTION_QUALITY_RUBRIC = {
    "scale": {"min": 1, "max": 5},
    "criteria": {
        "context_relevance": "The question is grounded in the candidate CV and target JD.",
        "technical_depth": "The question can distinguish shallow recall from real competence.",
        "clarity": "The question is specific, understandable, and asks one main thing.",
        "evidence_elicitation": "The question encourages concrete examples and verifiable evidence.",
        "adaptivity": "The question appropriately follows the previous answer and interview state.",
        "non_redundancy": "The question adds information instead of repeating covered content.",
        "fairness": "The question is job-related and avoids irrelevant or discriminatory assumptions.",
    },
}


ANSWER_QUALITY_RUBRIC = {
    "scale": {"min": 1, "max": 5},
    "criteria": {
        "answer_relevance": "The answer directly addresses the interview question.",
        "technical_correctness": "Technical claims are accurate and appropriate for the target role.",
        "reasoning_depth": "The answer explains reasoning, trade-offs, or a structured approach.",
        "evidence_specificity": "The answer gives concrete actions, examples, and outcomes.",
        "clarity": "The answer is understandable, focused, and logically organized.",
        "consistency": "The answer is consistent with the CV and prior interview history.",
        "criterion_score": "The answer demonstrates the competency targeted by this question.",
    },
}


TURN_EVIDENCE_RUBRIC = {
    "scale": {"min": 1, "max": 5},
    "criteria": {
        "answer_relevance": (
            "How directly the answer addresses the actual question without irrelevant padding."
        ),
        "technical_correctness": (
            "Whether role-relevant technical claims are accurate and appropriately qualified."
        ),
        "expected_signal_coverage": (
            "How many required answer signals are demonstrated correctly."
        ),
        "reasoning_quality": (
            "Whether the answer explains a logical approach, choices, and trade-offs when applicable."
        ),
        "evidence_quality": (
            "Whether examples, actions, and outcomes are concrete and credible when applicable."
        ),
        "clarity_conciseness": (
            "Whether the answer is clear and sufficiently detailed without rewarding length itself."
        ),
        "consistency": (
            "Whether the answer is internally consistent and compatible with the CV and prior turns."
        ),
    },
    "anchors": {
        "1": "Fails the criterion or contains a serious error.",
        "2": "Shows limited evidence with major gaps.",
        "3": "Acceptable basic evidence with meaningful omissions.",
        "4": "Strong, correct, and well-supported evidence.",
        "5": "Excellent evidence that is precise, complete, and appropriately nuanced.",
    },
}


CANDIDATE_EVALUATION_RUBRIC = {
    "scale": {"min": 1, "max": 5},
    "criteria": {
        "technical_knowledge": "Accuracy and depth of role-relevant technical knowledge.",
        "problem_solving": "Quality and structure of reasoning through unfamiliar problems.",
        "experience_evidence": "Specificity and credibility of evidence from prior work or projects.",
        "communication": "Clarity, concision, and organization of interview answers.",
        "role_relevance": "Relevance of demonstrated capabilities to the responsibilities in the JD.",
        "learning_adaptability": "Ability to learn, respond to feedback, and handle changing constraints.",
        "overall_readiness": "Overall readiness for the role based only on job-relevant evidence.",
    },
}
