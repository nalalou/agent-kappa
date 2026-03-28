"""Majority voting utilities for agent teams."""

from __future__ import annotations

from collections import Counter


def majority_vote(agent_outputs: list[list[str]]) -> list[str]:
    """
    Compute majority vote across agents for each item.

    Args:
        agent_outputs: List of agents, each containing answers per item.

    Returns:
        List of majority-voted answers (one per item).
    """
    n_items = len(agent_outputs[0])
    votes = []
    for i in range(n_items):
        answers = [agent[i] for agent in agent_outputs]
        winner = Counter(answers).most_common(1)[0][0]
        votes.append(winner)
    return votes


def vote_accuracy(
    agent_outputs: list[list[str]], ground_truth: list[str]
) -> float:
    """Accuracy of majority-voted answers against ground truth."""
    votes = majority_vote(agent_outputs)
    correct = sum(1 for v, g in zip(votes, ground_truth) if v == g)
    return correct / len(ground_truth)


def individual_accuracies(
    agent_outputs: list[list[str]], ground_truth: list[str]
) -> list[float]:
    """Per-agent accuracy."""
    return [
        sum(1 for a, g in zip(agent, ground_truth) if a == g) / len(ground_truth)
        for agent in agent_outputs
    ]


def vote_boost(
    agent_outputs: list[list[str]], ground_truth: list[str]
) -> float:
    """Vote accuracy minus average individual accuracy."""
    v_acc = vote_accuracy(agent_outputs, ground_truth)
    i_accs = individual_accuracies(agent_outputs, ground_truth)
    avg_individual = sum(i_accs) / len(i_accs)
    return v_acc - avg_individual
