"""Tests for diversity metrics — verify the math is correct."""

import pytest

from agent_kappa.metrics import (
    cohens_kappa,
    disagreement_measure,
    double_fault_measure,
    fleiss_kappa,
    kappa_correct,
    q_statistic,
    all_diversity_metrics,
)
from agent_kappa.voting import majority_vote, vote_accuracy, vote_boost
from agent_kappa.diagnosis import team_diagnosis


class TestCohensKappa:
    def test_perfect_agreement(self):
        assert cohens_kappa(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_no_agreement(self):
        # All different — kappa should be negative or zero
        k = cohens_kappa(["a", "b", "c"], ["c", "a", "b"])
        assert k < 0.5

    def test_partial_agreement(self):
        k = cohens_kappa(
            ["a", "a", "b", "b", "c"],
            ["a", "a", "b", "c", "c"],
        )
        assert 0 < k < 1

    def test_empty_raises(self):
        assert cohens_kappa([], []) == 0.0


class TestFleissKappa:
    def test_perfect_agreement(self):
        ratings = [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]
        assert fleiss_kappa(ratings) == 1.0

    def test_complete_disagreement(self):
        # Each rater gives different answers for each item
        ratings = [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]]
        k = fleiss_kappa(ratings)
        assert k < 0.1

    def test_two_raters_matches_cohens(self):
        r1 = ["a", "b", "a", "b", "a"]
        r2 = ["a", "b", "b", "b", "a"]
        fk = fleiss_kappa([r1, r2])
        ck = cohens_kappa(r1, r2)
        assert abs(fk - ck) < 0.02  # small difference expected between Fleiss' and Cohen's


class TestKappaCorrect:
    def test_independent_errors(self):
        # Agents fail on different problems → low kappa_correct
        gt = ["1", "2", "3", "4", "5", "6", "7", "8"]
        a1 = ["1", "2", "3", "4", "X", "X", "7", "8"]  # fails 5,6
        a2 = ["1", "2", "X", "X", "5", "6", "7", "8"]  # fails 3,4
        kc = kappa_correct([a1, a2], gt)
        assert kc < 0.3

    def test_correlated_errors(self):
        # Agents fail on same problems → high kappa_correct
        gt = ["1", "2", "3", "4", "5", "6", "7", "8"]
        a1 = ["1", "2", "3", "4", "X", "Y", "7", "8"]  # fails 5,6
        a2 = ["1", "2", "3", "4", "Z", "W", "7", "8"]  # fails 5,6
        kc = kappa_correct([a1, a2], gt)
        assert kc > 0.5

    def test_all_correct(self):
        gt = ["1", "2", "3"]
        a1 = ["1", "2", "3"]
        a2 = ["1", "2", "3"]
        kc = kappa_correct([a1, a2], gt)
        assert kc == 1.0


class TestQStatistic:
    def test_perfect_agreement(self):
        gt = ["1", "2", "3", "4"]
        a1 = ["1", "2", "X", "X"]
        a2 = ["1", "2", "X", "X"]
        q = q_statistic([a1, a2], gt)
        assert q == 1.0

    def test_independent(self):
        gt = ["1", "2", "3", "4", "5", "6", "7", "8"]
        a1 = ["1", "2", "3", "4", "X", "X", "X", "X"]
        a2 = ["X", "X", "X", "X", "5", "6", "7", "8"]
        q = q_statistic([a1, a2], gt)
        assert q < 0  # negatively correlated


class TestVoting:
    def test_majority_vote(self):
        outputs = [["a", "b", "c"], ["a", "b", "a"], ["a", "c", "a"]]
        assert majority_vote(outputs) == ["a", "b", "a"]

    def test_vote_accuracy(self):
        outputs = [["1", "2", "3"], ["1", "X", "3"], ["1", "2", "X"]]
        gt = ["1", "2", "3"]
        assert vote_accuracy(outputs, gt) == 1.0  # majority corrects errors

    def test_vote_boost_positive(self):
        # Independent errors → vote should help
        gt = ["1", "2", "3", "4"]
        a1 = ["1", "X", "3", "4"]
        a2 = ["1", "2", "X", "4"]
        a3 = ["1", "2", "3", "X"]
        boost = vote_boost([a1, a2, a3], gt)
        assert boost > 0


class TestDiagnosis:
    def test_diverse_team(self):
        gt = ["1", "2", "3", "4", "5", "6", "7", "8"]
        a1 = ["1", "2", "3", "4", "X", "X", "7", "8"]
        a2 = ["1", "2", "X", "X", "5", "6", "7", "8"]
        a3 = ["X", "X", "3", "4", "5", "6", "7", "8"]
        d = team_diagnosis([a1, a2, a3], gt)
        assert "DIVERSE" in d.verdict
        assert d.kappa_correct < 0.3

    def test_redundant_team(self):
        gt = ["1", "2", "3", "4", "5", "6", "7", "8"]
        a1 = ["1", "2", "3", "4", "X", "Y", "7", "8"]
        a2 = ["1", "2", "3", "4", "Z", "W", "7", "8"]
        a3 = ["1", "2", "3", "4", "Q", "R", "7", "8"]
        d = team_diagnosis([a1, a2, a3], gt)
        assert "REDUNDANT" in d.verdict
        assert d.kappa_correct > 0.5

    def test_str_output(self):
        gt = ["1", "2", "3", "4"]
        a1 = ["1", "X", "3", "4"]
        a2 = ["1", "2", "X", "4"]
        d = team_diagnosis([a1, a2], gt)
        s = str(d)
        assert "DIAGNOSIS" in s
        assert "κ_correct" in s
