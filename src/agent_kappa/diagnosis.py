"""
High-level team diagnosis — the main user-facing API.

    from agent_kappa import team_diagnosis
    diagnosis = team_diagnosis(agent_outputs, ground_truth)
    print(diagnosis)
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_kappa.metrics import all_diversity_metrics, kappa_correct
from agent_kappa.voting import (
    individual_accuracies,
    majority_vote,
    vote_accuracy,
    vote_boost,
)


@dataclass
class DiagnosisResult:
    """Result of a team diversity diagnosis."""

    # Core metric
    kappa_correct: float

    # Supporting metrics
    kappa_raw: float
    q_statistic: float
    disagreement: float
    double_fault: float

    # Accuracy info
    individual_accuracies: list[float]
    avg_individual_accuracy: float
    vote_accuracy: float
    vote_boost: float

    # Interpretation
    verdict: str
    recommendation: str

    def __str__(self) -> str:
        lines = [
            "=" * 55,
            "AGENT TEAM DIAGNOSIS",
            "=" * 55,
            "",
            f"  κ_correct:  {self.kappa_correct:>8.3f}   ← primary metric",
            f"  κ_raw:      {self.kappa_raw:>8.3f}",
            f"  Q-statistic:{self.q_statistic:>8.3f}",
            f"  Disagreement:{self.disagreement:>7.3f}",
            f"  Double-fault:{self.double_fault:>7.3f}",
            "",
            f"  Individual accuracy (avg): {self.avg_individual_accuracy:.1%}",
        ]
        for i, acc in enumerate(self.individual_accuracies):
            lines.append(f"    Agent {i}: {acc:.1%}")
        lines += [
            f"  Majority vote accuracy:    {self.vote_accuracy:.1%}",
            f"  Vote boost:                {self.vote_boost:+.1%}",
            "",
            f"  VERDICT: {self.verdict}",
            f"  {self.recommendation}",
            "",
            "=" * 55,
        ]
        return "\n".join(lines)


def team_diagnosis(
    agent_outputs: list[list[str]],
    ground_truth: list[str],
) -> DiagnosisResult:
    """
    Run a full diversity diagnosis on an agent team.

    Args:
        agent_outputs: List of agents, each containing answers per item.
                       agent_outputs[agent_idx][item_idx] = answer string.
        ground_truth: Correct answer for each item.

    Returns:
        DiagnosisResult with metrics, accuracy, verdict, and recommendation.
    """
    metrics = all_diversity_metrics(agent_outputs, ground_truth)
    kc = metrics["kappa_correct"]
    ind_accs = individual_accuracies(agent_outputs, ground_truth)
    avg_ind = sum(ind_accs) / len(ind_accs)
    v_acc = vote_accuracy(agent_outputs, ground_truth)
    v_boost = vote_boost(agent_outputs, ground_truth)

    # Determine verdict and recommendation
    # Thresholds based on Landis & Koch (1977) kappa interpretation scale
    # and our empirical observations across 6 model architectures.
    # These are guidelines, not hard boundaries — run the benchmark
    # on your own models to calibrate.
    if kc < 0.21:
        verdict = "DIVERSE — voting will likely help"
        recommendation = (
            "Your agents fail on different problems (κ below 'fair' agreement). "
            "Majority voting should improve accuracy."
        )
    elif kc < 0.41:
        verdict = "MODERATE — voting may help"
        recommendation = (
            "Some error correlation ('fair' agreement on errors). "
            "Voting may help. Consider adding prompt or model diversity."
        )
    else:
        verdict = "REDUNDANT — voting probably won't help"
        if avg_ind < 0.5:
            recommendation = (
                "Your agents fail on the same problems AND individual accuracy is low. "
                "You need a stronger base model, not more agents."
            )
        else:
            recommendation = (
                "Your agents fail on the same problems. "
                "Adding more copies won't help. Try different model architectures, "
                "diverse prompts, or different temperatures to reduce error correlation."
            )

    return DiagnosisResult(
        kappa_correct=kc,
        kappa_raw=metrics["kappa_raw"],
        q_statistic=metrics["q_statistic"],
        disagreement=metrics["disagreement"],
        double_fault=metrics["double_fault"],
        individual_accuracies=[round(a, 4) for a in ind_accs],
        avg_individual_accuracy=round(avg_ind, 4),
        vote_accuracy=round(v_acc, 4),
        vote_boost=round(v_boost, 4),
        verdict=verdict,
        recommendation=recommendation,
    )
