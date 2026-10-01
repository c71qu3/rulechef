"""Split-role enforcement in LearningPipeline.run()"""

from unittest.mock import MagicMock

from rulechef import RuleChef
from rulechef.core import Rule, RuleFormat, Task, TaskType


def _clf_task():
    return Task(
        name="clf",
        description="classify",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
    )


def _dummy_rule():
    return Rule(id="r1", name="dummy", description="d", format=RuleFormat.REGEX, content="a")


class TestSynthesisSeesTrainOnly:
    def test_bulk_synthesis_never_sees_non_train_ids(self, monkeypatch, tmp_path):
        chef = RuleChef(
            task=_clf_task(),
            client=MagicMock(),
            storage_path=tmp_path,
            dataset_name="pipeline_splits",
        )
        chef.synthesis_strategy = "bulk"  # force bulk path

        for i in range(60):
            chef.add_example({"text": f"a text {i}"}, {"label": "a"})

        seen_ids = []

        def fake_synthesize(dataset, max_rules=None):
            seen_ids.extend(e.id for e in dataset.examples)
            return [_dummy_rule()]

        monkeypatch.setattr(chef.learner, "synthesize_ruleset", fake_synthesize)

        chef.learn_rules(run_evaluation=False, max_refinement_iterations=0)

        assert seen_ids, "synthesize_ruleset was never called"
        train_ids = {e.id for e in chef.dataset.examples if e.split == "train"}
        non_train_ids = {e.id for e in chef.dataset.examples if e.split != "train"}
        assert set(seen_ids) <= train_ids
        assert not (set(seen_ids) & non_train_ids)

    def test_per_class_synthesis_never_sees_non_train_ids(self, monkeypatch, tmp_path):
        chef = RuleChef(
            task=_clf_task(),
            client=MagicMock(),
            storage_path=tmp_path,
            dataset_name="pipeline_splits"
        )
        chef.synthesis_strategy = "per_class"

        for i in range(40):
            chef.add_example({"text": f"a text {i}"}, {"label": "a"})
            chef.add_example({"text": f"b text {i}"}, {"label": "b"})

        seen_ids = []

        def fake_per_class(dataset, max_rules_per_class=None, max_counter_examples=None):
            seen_ids.extend(e.id for e in dataset.examples)
            return [_dummy_rule()]

        monkeypatch.setattr(chef.learner, "synthesize_ruleset_per_class", fake_per_class)

        chef.learn_rules(run_evaluation=False, max_refinement_iterations=0)

        train_ids = {e.id for e in chef.dataset.examples if e.split == "train"}
        assert seen_ids
        assert set(seen_ids) <= train_ids

    def test_split_assignment_is_sticky_across_learn_rules_calls(self, tmp_path):
        chef = RuleChef(
            task=_clf_task(),
            client=MagicMock(),
            storage_path=tmp_path,
            dataset_name="pipeline_splits",
        )
        chef.synthesis_strategy = "bulk"
        chef.learner.synthesize_ruleset = lambda dataset, max_rules=None: [_dummy_rule()]

        for i in range(60):
            chef.add_example({"text": f"a text {i}"}, {"label": "a"})
        chef.learn_rules(run_evaluation=False, max_refinement_iterations=0)
        before = {e.id: e.split for e in chef.dataset.examples}

        for i in range(60, 70):
            chef.add_example({"text": f"a text {i}"}, {"label": "a"})
        chef.learn_rules(run_evaluation=False, max_refinement_iterations=0)

        for e in chef.dataset.examples:
            if e.id in before:
                assert e.split == before[e.id]
