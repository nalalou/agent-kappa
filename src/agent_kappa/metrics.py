"""
Ensemble diversity metrics adapted for LLM agent outputs.

Implements the 6 most useful metrics from Kuncheva & Whitaker (2003),
plus kappa_correct which measures error correlation specifically.

All functions accept lists of string answers (not numeric predictions),
making them directly usable with LLM outputs.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations


def cohens_kappa(rater1: list[str], rater2: list[str]) -> float:
    """
    Cohen's kappa between two raters (agents).

    Measures agreement beyond chance on categorical outputs.
    Range: [-1, 1]. 1 = perfect agreement, 0 = chance, <0 = adversarial.
    """
    assert len(rater1) == len(rater2), "Raters must have same number of items"
    n = len(rater1)
    if n == 0:
        return 0.0

    agree = sum(1 for a, b in zip(rater1, rater2) if a == b)
    p_o = agree / n

    cats1: dict[str, int] = {}
    cats2: dict[str, int] = {}
    for a, b in zip(rater1, rater2):
        cats1[a] = cats1.get(a, 0) + 1
        cats2[b] = cats2.get(b, 0) + 1

    all_cats = set(cats1) | set(cats2)
    p_e = sum((cats1.get(c, 0) / n) * (cats2.get(c, 0) / n) for c in all_cats)

    if p_e >= 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """
    Fleiss' kappa for multiple raters (agents).

    Args:
        ratings: List of agents, each containing a list of answers per item.
                 ratings[agent_idx][item_idx] = answer string.

    Returns:
        Fleiss' kappa. Range: [-1, 1].
    """
    n_raters = len(ratings)
    n_items = len(ratings[0])

    if n_raters < 2 or n_items == 0:
        return 0.0

    all_cats = sorted(set(c for rater in ratings for c in rater))
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}
    n_cats = len(all_cats)

    # Count matrix: items × categories
    counts = [[0] * n_cats for _ in range(n_items)]
    for rater in ratings:
        for i, label in enumerate(rater):
            counts[i][cat_to_idx[label]] += 1

    # Per-item agreement
    P_items = []
    for i in range(n_items):
        sum_sq = sum(c * c for c in counts[i])
        P_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1))
        P_items.append(P_i)

    P_bar = sum(P_items) / n_items

    # Category proportions
    p_cats = []
    for j in range(n_cats):
        total_j = sum(counts[i][j] for i in range(n_items))
        p_cats.append(total_j / (n_items * n_raters))

    P_e_bar = sum(pj * pj for pj in p_cats)

    if P_e_bar >= 1.0:
        return 1.0

    return (P_bar - P_e_bar) / (1 - P_e_bar)


def kappa_correct(
    agent_outputs: list[list[str]],
    ground_truth: list[str],
) -> float:
    """
    Kappa on correctness — the key diagnostic metric.

    Converts each agent's outputs to binary correct/incorrect,
    then computes Fleiss' kappa on those binary labels.

    Low kappa_correct = agents fail on DIFFERENT problems = vote helps.
    High kappa_correct = agents fail on the SAME problems = vote won't help.

    Interpretation (Landis & Koch 1977 scale + our empirical observations):
        κ_correct < 0.21 (slight) → errors are independent → voting helps
        κ_correct 0.21-0.41 (fair) → some error correlation → voting may help
        κ_correct > 0.41 (moderate+) → correlated errors → voting probably won't help

    These are guidelines from 6 model architectures, not hard thresholds.
    Run the benchmark on your own setup to calibrate.

    Args:
        agent_outputs: List of agents, each containing answers per item.
        ground_truth: Correct answers for each item.
    """
    binary = []
    for agent_answers in agent_outputs:
        labels = [
            "correct" if a == g else "incorrect"
            for a, g in zip(agent_answers, ground_truth)
        ]
        binary.append(labels)

    return fleiss_kappa(binary)


# --- Pairwise metrics from Kuncheva & Whitaker (2003) ---
# These operate on binary correct/incorrect vectors.


def _to_binary(
    agent_outputs: list[list[str]], ground_truth: list[str]
) -> list[list[int]]:
    """Convert to binary: 1 = correct, 0 = incorrect."""
    return [
        [1 if a == g else 0 for a, g in zip(agent, ground_truth)]
        for agent in agent_outputs
    ]


def _contingency(y1: list[int], y2: list[int]) -> tuple[int, int, int, int]:
    """
    2×2 contingency table for two binary classifiers.
    Returns (a, b, c, d) where:
        a = both correct
        b = y1 correct, y2 incorrect
        c = y1 incorrect, y2 correct
        d = both incorrect
    """
    a = b = c = d = 0
    for v1, v2 in zip(y1, y2):
        if v1 == 1 and v2 == 1:
            a += 1
        elif v1 == 1 and v2 == 0:
            b += 1
        elif v1 == 0 and v2 == 1:
            c += 1
        else:
            d += 1
    return a, b, c, d


def q_statistic(
    agent_outputs: list[list[str]], ground_truth: list[str]
) -> float:
    """
    Yule's Q-statistic (averaged over all pairs).

    Q = 1: agents always agree. Q = 0: independent. Q < 0: negatively correlated.
    Range: [-1, 1].

    Reference: Kuncheva & Whitaker (2003), Section 3.1.
    """
    binary = _to_binary(agent_outputs, ground_truth)
    qs = []
    for i, j in combinations(range(len(binary)), 2):
        a, b, c, d = _contingency(binary[i], binary[j])
        denom = a * d + b * c
        if denom == 0:
            qs.append(0.0)
        else:
            qs.append((a * d - b * c) / denom)
    return sum(qs) / len(qs) if qs else 0.0


def disagreement_measure(
    agent_outputs: list[list[str]], ground_truth: list[str]
) -> float:
    """
    Disagreement measure (averaged over all pairs).

    Proportion of items where exactly one of two agents is correct.
    Higher = more diverse. Range: [0, 1].

    Reference: Kuncheva & Whitaker (2003), Section 3.1.
    """
    binary = _to_binary(agent_outputs, ground_truth)
    n = len(ground_truth)
    disags = []
    for i, j in combinations(range(len(binary)), 2):
        a, b, c, d = _contingency(binary[i], binary[j])
        disags.append((b + c) / n)
    return sum(disags) / len(disags) if disags else 0.0


def double_fault_measure(
    agent_outputs: list[list[str]], ground_truth: list[str]
) -> float:
    """
    Double-fault measure (averaged over all pairs).

    Proportion of items where BOTH agents are wrong simultaneously.
    Lower = better (less correlated failure). Range: [0, 1].

    Reference: Kuncheva & Whitaker (2003), Section 3.1.
    """
    binary = _to_binary(agent_outputs, ground_truth)
    n = len(ground_truth)
    dfs = []
    for i, j in combinations(range(len(binary)), 2):
        a, b, c, d = _contingency(binary[i], binary[j])
        dfs.append(d / n)
    return sum(dfs) / len(dfs) if dfs else 0.0


def all_diversity_metrics(
    agent_outputs: list[list[str]], ground_truth: list[str]
) -> dict[str, float]:
    """
    Compute all diversity metrics at once.

    Returns dict with keys:
        kappa_correct, kappa_raw, q_statistic,
        disagreement, double_fault
    """
    return {
        "kappa_correct": round(kappa_correct(agent_outputs, ground_truth), 4),
        "kappa_raw": round(fleiss_kappa(agent_outputs), 4),
        "q_statistic": round(q_statistic(agent_outputs, ground_truth), 4),
        "disagreement": round(disagreement_measure(agent_outputs, ground_truth), 4),
        "double_fault": round(double_fault_measure(agent_outputs, ground_truth), 4),
    }
