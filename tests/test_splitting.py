"""Tests for rulechef.splitting — sticky, stratified train/selection/calibration/test splits."""

from rulechef.core import Correction, Dataset, Example, Task, TaskType
from rulechef.splitting import MIN_SPLIT_SIZE, assign_splits, select_split


N_PER_CLASS = 20


def _classification_dataset(n_per_class=N_PER_CLASS, classes=("a", "b", "c"), group_fn=None):
    task = Task(
        name="clf",
        description="classify",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
    )
    dataset = Dataset(name="clf-data", task=task)
    for label in classes:
        for i in range(n_per_class):
            dataset.examples.append(
                Example(
                    id=f"{label}{i}",
                    input={"text": f"example {label} {i}"},
                    expected_output={"label": label},
                    source="test",
                    group=group_fn(label, i) if group_fn else None,
                )
            )
    return dataset


class TestAssignSplits:
    def test_assigns_all_four_roles(self):
        dataset = _classification_dataset()
        assign_splits(dataset, seed=1)
        seen = {e.split for e in dataset.examples}
        assert seen <= {"train", "selection", "calibration", "test"}
        assert "unassigned" not in seen

    def test_corrections_always_train(self):
        dataset = _classification_dataset()
        dataset.corrections.append(
            Correction(
                id="c1",
                input={"text": "fix me"},
                model_output={"label": "a"},
                expected_output={"label": "b"},
            )
        )
        assign_splits(dataset, seed=1)
        assert dataset.corrections[0].split == "train"

    def test_no_overlap_between_splits(self):
        dataset = _classification_dataset()
        assign_splits(dataset, seed=1)
        by_split = {}
        for e in dataset.examples:
            by_split.setdefault(e.split, set()).add(e.id)
        all_ids = set()
        for ids in by_split.values():
            assert not (all_ids & ids)
            all_ids |= ids

    def test_stratified_by_label(self):
        dataset = _classification_dataset()
        assign_splits(dataset, seed=1)
        for split in ("selection", "calibration", "test"):
            labels = {e.expected_output["label"] for e in dataset.examples if e.split == split}
            assert labels == {"a", "b", "c"}

    def test_sticky_assignment_ignores_already_split_examples(self):
        """Calling assign_splits after adding examples must not reshuffle original split."""
        dataset = _classification_dataset()
        assign_splits(dataset, seed=1)
        before = {e.id: e.split for e in dataset.examples}

        for i in range(N_PER_CLASS, N_PER_CLASS + 10):
            dataset.examples.append(
                Example(
                    id=f"a{i}",
                    input={"text": f"example a {i}"},
                    expected_output={"label": "a"},
                    source="test",
                )
            )
        assign_splits(dataset, seed=1)

        for e in dataset.examples:
            if e.id in before:
                assert e.split == before[e.id]
            else:
                assert e.split != "unassigned"

    def test_group_never_straddles_a_split(self):
        dataset = _classification_dataset(
            n_per_class=40, group_fn=lambda label, i: f"{label}-doc{i // 4}"
        )
        assign_splits(dataset, seed=1)
        by_group = {}
        for e in dataset.examples:
            by_group.setdefault(e.group, set()).add(e.split)
        for group, splits in by_group.items():
            assert len(splits) == 1, f"group {group} split across {splits}"

    def test_exploratory_when_selection_split_too_small(self):
        dataset = _classification_dataset(n_per_class=2, classes=("a", "b"))
        report = assign_splits(dataset, seed=1)
        assert report.exploratory is True
        assert "selection" in report.reason or "calibration" in report.reason or "test" in report.reason


class TestSelectSplit:
    def test_select_split_filters_examples(self):
        dataset = _classification_dataset()
        assign_splits(dataset, seed=1)
        train = select_split(dataset, "train")
        assert all(e.split == "train" for e in train.examples)
        assert len(train.examples) < len(dataset.examples)

    def test_select_split_train_includes_correction_others_dont(self):
        dataset = _classification_dataset()
        dataset.corrections.append(
            Correction(
                id="c1",
                input={"text": "fix me"},
                model_output={"label": "a"},
                expected_output={"label": "b"},
            )
        )
        assign_splits(dataset, seed=1)
        train = select_split(dataset, "train")
        selection = select_split(dataset, "selection")
        assert len(train.corrections) == 1
        assert len(selection.corrections) == 0

    def test_select_split_rejects_unknown_name(self):
        dataset = _classification_dataset()
        assign_splits(dataset, seed=1)
        try:
            select_split(dataset, "bogus")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_min_split_size_constant_is_exported(self):
        assert MIN_SPLIT_SIZE >= 1
