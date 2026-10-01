"""Tests for rulechef.storage and dataset persistence round-trips."""

from rulechef.core import Correction, Dataset, Example, Task, TaskType
from rulechef.storage import DatasetStore


def _task():
    return Task(
        name="clf",
        description="classify",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
    )


def test_split_and_group_round_trip(tmp_path):
    store = DatasetStore(tmp_path)
    dataset = Dataset(name="rt", task=_task())
    dataset.examples.append(
        Example(
            id="e1",
            input={"text": "hi"},
            expected_output={"label": "a"},
            source="human_labeled",
            split="selection",
            group="doc-1",
        )
    )
    dataset.corrections.append(
        Correction(
            id="c1",
            input={"text": "fix"},
            model_output={"label": "a"},
            expected_output={"label": "b"},
        )
    )
    store.save(dataset)

    loaded = Dataset(name="rt", task=_task())
    store.load(loaded)

    assert loaded.examples[0].split == "selection"
    assert loaded.examples[0].group == "doc-1"
    assert loaded.corrections[0].split == "train"


def test_load_defaults_split_for_legacy_files_without_it(tmp_path):
    """Datasets saved before this feature existed have no 'split' key."""
    store = DatasetStore(tmp_path)
    filepath = tmp_path / "legacy.json"
    filepath.write_text(
        '{"name": "legacy", "task": %s, "examples": '
        '[{"id": "e1", "input": {"text": "hi"}, "expected_output": {"label": "a"}, '
        '"source": "human_labeled", "confidence": 0.8}], '
        '"corrections": [], "feedback": [], "structured_feedback": [], "rules": []}'
        % __import__("json").dumps(_task().to_dict())
    )

    loaded = Dataset(name="legacy", task=_task())
    store.load(loaded)

    assert loaded.examples[0].split == "unassigned"
    assert loaded.examples[0].group is None
