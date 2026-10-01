"""Persistent train/selection/calibration/test split assignment.

Every example gets a *permanent* role once assigned:

- train: seen by synthesis, per-class, synthesis, and patch generation.
- selection: used only to pick the best candidate rule set / accept or
  reject a path during refinement.
- calibration: used only to compute each rule's validated_precision /
  validated_support (rank_rules, post-audit re-stamping). Kept separate
  from selection so tthe same examples that picked the winner don't also
  report on how good the winner is.
- test: never touched by synthesis, selection, or calibration. Reserved
  for final reporting.

assign_splits() is sticky: it only assigns examples whose `.split` is
still "unassigned". Re-running it as new examples are committed to a
datasest does now reshuffle prior assignments, so split identity survives
across learn_rules() calls, refinement, audit, and reporting.

Examples sharing a `.group` are always assigned to the same split, so a
group never straddles a split boundary.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field

from rulechef.core import Dataset, Example, TaskType

SPLIT_NAMES = ("train", "selection", "calibration", "test")

# Below this size a split is considered too small to trus for decisions
MIN_SPLIT_SIZE = 5

DEFAULT_FRACTIONS = {"selection": 0.15, "calibration": 0.15, "test": 0.15}


def _class_signature(example: Example, task_type: TaskType) -> str:
    """Bucket key used for stratification.

    CLASSIFICATION stratifies by label, NER by the set of entity types in
    the example. Other task types fall into a single bucket (random split).
    """
    output = example.expected_output or {}
    if task_type == TaskType.CLASSIFICATION:
        return str(output.get("label", ""))
    if task_type == TaskType.NER:
        entities = output.get("entities") or []
        types = sorted({e.get("type", "") for e in entities if isinstance(e, dict)})
        return "|".join(types) if types else "_no_entities"
    return "_all"


@dataclass
class SplitReport:
    """Summary of the current split assignment.

    Attributes:
        counts: Number of examples (+corrections, counted under "train")
            per split name.
        exploratory: True when selection, calibratioin, or test is below
            MIN_SPLIT_SIZE, meaning decisions made against it should be
            treated as exploratory, not as validated generalization.
        reason: Human-readable explanation when exploratory is True.
    """
    
    counts: dict[str, int] = field(default_factory=dict)
    exploratory: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {"counts": self.counts, "exploratory": self.exploratory, "reason": self.reason}


def assign_splits(
    dataset: Dataset,
    fractions: dict[str, float] | None = None,
    seed: int = 42,
) -> SplitReport:
    """Assign a persistent split to every "unassigned" example.

    Stratified by class signature, grouped by `.group` (defaults to the
    example's own id so ungrouped data splits per-example). Corrections
    are always pinned to "train". Mutates `dataset` in place.

    Args:
        dataset: Dataset whose examples/corrections get a `.split`.
        fractions: Target share of *groups* assigned to selection/
            calibration/test; the remainder is train. Defaults to 15%
            each, 55% train.
        seed: Random seed for the deterministic shuffle within each
        stratum. Only affects newly-assigned examples.

    Returns:
        SplitReport describing the resulting split sizes.
    """
    fractions = fractions or DEFAULT_FRACTIONS

    for correction in dataset.corrections:
        correction.split = "train"

    # Map already assigned groups to their split in order to attach new
    # examples from that group into the same split
    existing_group_splits: dict[str, str] = {}
    for ex in dataset.examples:
        if ex.split != "unassigned":
            group = ex.group or ex.id
            prev = existing_group_splits.get(group)
            if prev is not None and prev != ex.split:
                raise ValueError(
                    f"Group {group!r} already assigned to split {prev!r} "
                    f"but found example {ex.id!r} in split {ex.split!r}"
                )
            existing_group_splits[group] = ex.split

    pending = [ex for ex in dataset.examples if ex.split == "unassigned"]
    if pending:
        by_signature: dict[str, dict[str, list[Example]]] = defaultdict(lambda: defaultdict(list))
        for ex in pending:
            group = ex.group or ex.id

            # If a group already exists in a split force new examples in
            # that group into the same split
            existing_split = existing_group_splits.get(group)
            if existing_split is not None:
                ex.split = existing_split
                continue

            signature = _class_signature(ex, dataset.task.type)
            by_signature[signature][group].append(ex)
            

        rng = random.Random(seed)
        for signature in sorted(by_signature):
            groups = sorted(by_signature[signature])
            rng.shuffle(groups)
            n = len(groups)

            remaining = n
            group_splits: dict[str, str] = {}
            for split_name in ("test", "calibration", "selection"):
                n_split = min(round(n * fractions.get(split_name, 0)), max(remaining - 1, 0))
                take, groups = groups[:n_split], groups[n_split:]
                for group in take:
                    group_splits[group] = split_name
                remaining -= n_split
            for group in groups:
                group_splits[group] = "train"

            for group, split_name in group_splits.items():
                for ex in by_signature[signature][group]:
                    ex.split = split_name

    return _build_report(dataset)


def _build_report(dataset: Dataset) -> SplitReport:
    counts: dict[str, int] = defaultdict(int)
    for ex in dataset.examples:
        counts[ex.split] += 1
    counts["train"] += len(dataset.corrections)

    small = [s for s in ("selection", "calibration", "test") if counts.get(s, 0) < MIN_SPLIT_SIZE]
    exploratory = bool(small)
    reason = (
        f"splits below {MIN_SPLIT_SIZE} examples, not validated: "
        + ", ".join(f"{s}={counts.get(s, 0)}" for s in small)
        if exploratory
        else ""
    )
    return SplitReport(counts=dict(counts), exploratory=exploratory, reason=reason)


def select_split(dataset: Dataset, split: str) -> Dataset:
    """Build a Dataset view containing only examples assigned to `split`.

    Corrections are included only in the "train" view.

    Args:
        dataset: Source dataset (not mutated).
        split: One of SPLIT_NAMES.

    Returns:
        A new Dataset sharing the source's task/feedbask/rules but scoped
        to the requested split's examples.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}, expected one of {SPLIT_NAMES}")

    examples = [ex for ex in dataset.examples if ex.split == split]
    corrections = dataset.corrections if split == "train" else []
    return Dataset(
        name=f"{dataset.name}_{split}",
        task=dataset.task,
        description=dataset.description,
        examples=examples,
        corrections=corrections,
        feedback=dataset.feedback,
        structured_feedback=dataset.structured_feedback,
        rules=dataset.rules,
    )


# TODO: Replace with new methods
def split_dataset(
    dataset: Dataset,
    holdout_fraction: float = 0.2,
    seed: int = 42,
    min_dev_size: int = 5,
) -> tuple[Dataset, Dataset | None]:
    """Split a dataset into train and held-out dev portions.
    Examples are split stratified by class signature. Corrections always
    stay in train: they are explicit user fixes that must drive patching,
    and holding them out would hide the highest-value signal from the
    learner. Feedback and existing rules are shared with the train split.
    Args:
        dataset: Source dataset (not mutated).
        holdout_fraction: Fraction of examples to hold out (0 < f < 1).
        seed: Random seed for the shuffle within each stratum.
        min_dev_size: If the resulting dev set would be smaller than this,
            no split is performed and (dataset, None) is returned.
    Returns:
        Tuple of (train_dataset, dev_dataset). dev_dataset is None when the
        dataset is too small to split safely.
    """
    if not 0 < holdout_fraction < 1:
        return dataset, None

    by_signature: dict[str, list[Example]] = defaultdict(list)
    for ex in dataset.examples:
        by_signature[_class_signature(ex, dataset.task.type)].append(ex)

    rng = random.Random(seed)
    train_examples: list[Example] = []
    dev_examples: list[Example] = []

    for signature in sorted(by_signature):
        examples = list(by_signature[signature])
        rng.shuffle(examples)
        # Hold out a proportional share, but never the entire stratum
        n_dev = min(round(len(examples) * holdout_fraction), len(examples) - 1)
        dev_examples.extend(examples[: max(n_dev, 0)])
        train_examples.extend(examples[max(n_dev, 0) :])

    if len(dev_examples) < min_dev_size:
        return dataset, None

    train = Dataset(
        name=f"{dataset.name}_train",
        task=dataset.task,
        description=dataset.description,
        examples=train_examples,
        corrections=dataset.corrections,
        feedback=dataset.feedback,
        structured_feedback=dataset.structured_feedback,
        rules=dataset.rules,
    )
    dev = Dataset(
        name=f"{dataset.name}_dev",
        task=dataset.task,
        description=dataset.description,
        examples=dev_examples,
    )
    return train, dev
