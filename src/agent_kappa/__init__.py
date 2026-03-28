"""
agent-kappa: Measure diversity in multi-agent LLM teams.

Adapts classical ensemble diversity metrics (Kuncheva & Whitaker, 2003)
for LLM agent outputs. Predicts whether majority voting will help your team.

Usage:
    from agent_kappa import team_diagnosis, kappa_correct, fleiss_kappa

    diagnosis = team_diagnosis(
        agent_outputs=[agent1_answers, agent2_answers, ...],
        ground_truth=correct_answers,
    )
    print(diagnosis)
"""

from agent_kappa.metrics import (
    cohens_kappa,
    fleiss_kappa,
    kappa_correct,
    q_statistic,
    disagreement_measure,
    double_fault_measure,
    all_diversity_metrics,
)
from agent_kappa.diagnosis import team_diagnosis, DiagnosisResult
from agent_kappa.voting import majority_vote, vote_accuracy

__version__ = "0.1.0"
__all__ = [
    "team_diagnosis",
    "DiagnosisResult",
    "cohens_kappa",
    "fleiss_kappa",
    "kappa_correct",
    "q_statistic",
    "disagreement_measure",
    "double_fault_measure",
    "all_diversity_metrics",
    "majority_vote",
    "vote_accuracy",
]
