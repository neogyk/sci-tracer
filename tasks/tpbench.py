import dspy
from datasets import load_dataset

##https://huggingface.co/datasets/ZhiqiGao/TPBench

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


class TBBenchSignature(dspy.Signature):
    """Solve a scientific or mathematical problem. Return the numeric answer and its unit."""

    problem_id = dspy.InputField(
        desc="The scientific/math problem statement (may contain LaTeX)"
    )
    domain = dspy.InputField(
        desc="Source textbook domain (e.g. atkins, calculus, thermo)"
    )
    difficulty_level = dspy.InputField(desc="The numeric answer value")
    problem = dspy.InputField(desc="The unit of the answer (LaTeX format)")
    solution = dspy.InputField(desc="Step-by-step solution to the problem")
    answer = dspy.InputField(desc="Step-by-step solution to the problem")
    code_answer_requirements = dspy.InputField(
        desc="Step-by-step solution to the problem"
    )
    reference_implementation = dspy.InputField(
        desc="Step-by-step solution to the problem"
    )


def load_tpbench(train_ratio=0.8, shuffle_seed=42, source_filter=None):
    """Load SciBench as DSPy train/dev Example lists.

    Args:
        train_ratio: fraction of data used for training (rest goes to dev).
        shuffle_seed: random seed for reproducible splits.
        source_filter: optional list of source names to keep (e.g. ["atkins", "thermo"]).

    Returns:
        (trainset, devset) — each a list of dspy.Example with .with_inputs("problem_text", "source").
    """
    ds = load_dataset("ZhiqiGao/TPBench")
    print(f"Loaded {len(ds)} TPBench problems")

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
    print(devset[0])
