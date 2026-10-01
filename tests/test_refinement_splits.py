"""Split-role enforcement in RuleLearner.evaluate_and_refine()."""

from unittest.mock import MagicMock

import pytest

from rulechef.core import Dataset, Example, Rule, RuleFormat, Task, TaskType
from rulechef.evaluation import EvalResult
from rulechef.learner import RuleLearner


def _dataset(n):
    task = Task(
        name="tiny",
        description="d",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
    )
    ds = Dataset(name="tiny", task=task)
    for i in range(n):
        ds.examples.append(
            Example(id=f"e{i}", input={"text": f"t{i}"}, expected_output={"label": "a"}, source="test")
        )
    return ds


def _rule():
    return Rule(id="r1", name="r", description="d", format=RuleFormat.REGEX, content="a")


class TestEvaluateAndRefineSplitRoles:
    def test_marks_exploratory_when_selection_too_small(self, monkeypatch):
        learner = RuleLearner(llm=MagicMock())
        dataset = _dataset(3)

        perfect = EvalResult(micro_f1=1.0, exact_match=1.0, total_docs=3, failures=[])
        monkeypatch.setattr(learner, "_evaluate_rules", lambda rules, ds: perfect)
        monkeypatch.setattr(learner, "_stamp_validated_stats", lambda rules, ds: None)

        _, best_eval = learner.evaluate_and_refine([_rule()], dataset, max_iterations=1)

        assert best_eval.exploratory is True

    def test_selection_split_drives_acceptance_not_train(self, monkeypatch):
        """Candidate patches must be judged against `selection` split."""
        learner = RuleLearner(llm=MagicMock())
        dataset = _dataset(60)

        def fake_evaluate(rules, ds):
            if ds.name.endswith("_selection"):
                return EvalResult(
                    micro_f1=0.2,
                    exact_match=0.2,
                    total_docs=len(ds.examples),
                    failures=[{"input": {}, "expected": {}, "got":{}, "is_correction": False}],
                )
            return EvalResult(micro_f1=1.0, exact_match=1.0, total_docs=len(ds.examples), failures=[])

        monkeypatch.setattr(learner, "_evaluate_rules", fake_evaluate)
        monkeypatch.setattr(learner, "_stamp_validated_stats", lambda rules, ds: None)
        monkeypatch.setattr(learner, "synthesize_patch_ruleset", lambda *a, **k: ([], set()))

        _, best_eval = learner.evaluate_and_refine([_rule()], dataset, max_iterations=1)

        assert best_eval.micro_f1 == pytest.approx(0.2)
        assert best_eval.exploratory is False

    def test_calibration_used_for_validated_stats_not_selection(self, monkeypatch):
        """validate_precision must come from `calibration` split."""
        learner = RuleLearner(llm=MagicMock())
        dataset = _dataset(60)

        stamped_on = []
        monkeypatch.setattr(
            learner, "_evaluate_rules",
            lambda rules, ds: EvalResult(micro_f1=0.5, exact_match=0.5, total_docs=len(ds.examples), failures=[]),
        )
        monkeypatch.setattr(
            learner, "_stamp_validated_stats",
            lambda rules, ds: stamped_on.append(ds.name),
        )
        monkeypatch.setattr(learner, "synthesize_patch_ruleset", lambda *a, **k: ([], set()))

        learner.evaluate_and_refine([_rule()], dataset, max_iterations=1)

        assert stamped_on == ["tiny_calibration"]
