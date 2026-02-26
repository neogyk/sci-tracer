# https://huggingface.co/datasets/xw27/scibench
import dspy
from datasets import load_dataset


SOURCES = [
    "atkins",
    "calculus",
    "chemmc",
    "class",
    "diff",
    "fund",
    "matter",
    "quan",
    "stat",
    "thermo",
]


class SciBenchSignature(dspy.Signature):
    """Solve a scientific or mathematical problem. Return the numeric answer and its unit."""

    problem_text = dspy.InputField(
        desc="The scientific/math problem statement (may contain LaTeX)"
    )
    source = dspy.InputField(
        desc="Source textbook domain (e.g. atkins, calculus, thermo)"
    )
    answer = dspy.OutputField(desc="The numeric answer value")
    unit = dspy.OutputField(desc="The unit of the answer (LaTeX format)")
    solution = dspy.OutputField(desc="Step-by-step solution to the problem")


def load_scibench(train_ratio=0.8, shuffle_seed=42, source_filter=None):
    """Load SciBench as DSPy train/dev Example lists.

    Args:
        train_ratio: fraction of data used for training (rest goes to dev).
        shuffle_seed: random seed for reproducible splits.
        source_filter: optional list of source names to keep (e.g. ["atkins", "thermo"]).

    Returns:
        (trainset, devset) — each a list of dspy.Example with .with_inputs("problem_text", "source").
    """
    ds = load_dataset("xw27/scibench", split="train")
    print(f"Loaded {len(ds)} SciBench problems")

    examples = []
    for row in ds:
        if source_filter and row["source"].strip() not in source_filter:
            continue

        answer_str = str(row["answer_number"]).strip()
        unit_str = (row.get("unit") or "").strip()
        solution_str = (row.get("solution") or "").strip()

        ex = dspy.Example(
            problem_text=row["problem_text"],
            source=row["source"].strip(),
            answer=answer_str,
            unit=unit_str,
            solution=solution_str,
        ).with_inputs("problem_text", "source")
        examples.append(ex)

    # Deterministic shuffle + split
    import random

    rng = random.Random(shuffle_seed)
    rng.shuffle(examples)
    split = int(len(examples) * train_ratio)
    trainset = examples[:split]
    devset = examples[split:]

    print(f"SciBench split: {len(trainset)} train / {len(devset)} dev")
    return trainset, devset


if __name__ == "__main__":
    trainset, devset = load_scibench()
    print(f"\n--- Sample train example ---")
    print(trainset[0])
    print(f"\n--- Sample dev example ---")
    _len = len(trainset)
    for i in range(_len):
        problem_text = trainset[i].problem_text
        solution = trainset[i].solution
        answer = trainset[i].answer
