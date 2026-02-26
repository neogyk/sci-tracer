# https://huggingface.co/datasets/zjunlp/Mol-Instructions
import json
import zipfile
from pathlib import Path

import dspy
from huggingface_hub import hf_hub_download


# ── Available configs and their task splits ──────────────────────────────────

# Maps config name -> (zip filename in repo, list of task JSON files inside zip)
CONFIGS = {
    "Molecule-oriented Instructions": {
        "zip": "data/Molecule-oriented_Instructions.zip",
        "tasks": [
            "description_guided_molecule_design",
            "forward_reaction_prediction",
            "molecular_description_generation",
            "property_prediction",
            "reagent_prediction",
            "retrosynthesis",
        ],
    },
    "Protein-oriented Instructions": {
        "zip": "data/Protein-oriented_Instructions.zip",
        "tasks": [
            "catalytic_activity",
            "domain_motif",
            "protein_function",
            "general_function",
            "protein_design",
        ],
    },
    "Biomolecular Text Instructions": {
        "zip": "data/Biomolecular_Text_Instructions.zip",
        "tasks": [
            "chemical_disease_interaction_extraction",
            "chemical_entity_recognition",
            "chemical_protein_interaction_extraction",
            "multi_choice_question",
            "open_question",
            "true_or_false_question",
        ],
    },
}


# ── DSPy Signatures ─────────────────────────────────────────────────────────


class MoleculeTask(dspy.Signature):
    """Follow the instruction to solve a molecule-related task given the molecular input."""

    instruction = dspy.InputField(
        desc="Task directive (e.g. predict product, describe molecule)"
    )
    input = dspy.InputField(
        desc="Molecular representation (SELFIES/SMILES) or reaction string"
    )
    output = dspy.OutputField(
        desc="Expected answer: description, prediction, or molecule"
    )


class ProteinTask(dspy.Signature):
    """Follow the instruction to solve a protein-related task given the protein input."""

    instruction = dspy.InputField(
        desc="Task directive (e.g. predict function, design protein)"
    )
    input = dspy.InputField(desc="Amino acid sequence or protein description")
    output = dspy.OutputField(
        desc="Expected answer: function annotation, sequence, or property"
    )


class BioTextTask(dspy.Signature):
    """Follow the instruction to solve a biomedical text task."""

    instruction = dspy.InputField(
        desc="Task directive (e.g. extract entities, answer question)"
    )
    input = dspy.InputField(desc="Biomedical text passage")
    output = dspy.OutputField(
        desc="Expected answer: extracted entities, answer text, or label"
    )


SIGNATURE_MAP = {
    "Molecule-oriented Instructions": MoleculeTask,
    "Protein-oriented Instructions": ProteinTask,
    "Biomolecular Text Instructions": BioTextTask,
}


# ── Internal helpers ─────────────────────────────────────────────────────────


def _load_json_from_zip(zip_path, task_name):
    """Read a single task JSON from inside a downloaded zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        # The JSON file may be at the root or inside a subdirectory
        target = f"{task_name}.json"
        matching = [n for n in zf.namelist() if n.endswith(target)]
        if not matching:
            raise FileNotFoundError(
                f"'{target}' not found in {zip_path}. Available: {zf.namelist()[:10]}..."
            )
        with zf.open(matching[0]) as f:
            return json.load(f)


# ── Dataset loader ───────────────────────────────────────────────────────────


def load_mol_instruct(
    config, split=None, train_ratio=0.8, shuffle_seed=42, max_examples=None
):
    """Load Mol-Instructions as DSPy train/dev Example lists.

    Downloads the zip from HuggingFace Hub (cached after first call) and
    reads the JSON task files directly.

    Args:
        config: one of the keys in CONFIGS (e.g. "Molecule-oriented Instructions").
        split: optional task split name (e.g. "forward_reaction_prediction").
               If None, loads all tasks for the given config.
        train_ratio: fraction of data used for training.
        shuffle_seed: random seed for reproducible splits.
        max_examples: cap total examples (useful for large configs like Protein).

    Returns:
        (trainset, devset) — each a list of dspy.Example with .with_inputs("instruction", "input").
    """
    if config not in CONFIGS:
        raise ValueError(
            f"Unknown config '{config}'. Choose from: {list(CONFIGS.keys())}"
        )

    cfg = CONFIGS[config]
    tasks_to_load = [split] if split else cfg["tasks"]

    # Download zip (huggingface_hub caches it automatically)
    print(f"Downloading {cfg['zip']} ...")
    zip_path = hf_hub_download(
        repo_id="zjunlp/Mol-Instructions",
        filename=cfg["zip"],
        repo_type="dataset",
    )

    examples = []
    for task_name in tasks_to_load:
        print(f"Loading {config} / {task_name} ...")
        rows = _load_json_from_zip(zip_path, task_name)

        for row in rows:
            ex = dspy.Example(
                instruction=row["instruction"],
                input=row["input"],
                output=row["output"],
                metadata=row.get("metadata", ""),
                task=task_name,
            ).with_inputs("instruction", "input")
            examples.append(ex)

        if max_examples and len(examples) >= max_examples:
            examples = examples[:max_examples]
            break

    print(f"Loaded {len(examples)} examples total")

    # Deterministic shuffle + split
    import random

    rng = random.Random(shuffle_seed)
    rng.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)
    trainset = examples[:split_idx]
    devset = examples[split_idx:]

    print(f"Mol-Instructions split: {len(trainset)} train / {len(devset)} dev")
    return trainset, devset


if __name__ == "__main__":
    # Quick smoke test with a small molecule task
    trainset, devset = load_mol_instruct(
        config="Molecule-oriented Instructions",
        split="molecular_description_generation",
        max_examples=100,
    )
    print(f"\n--- Sample train example ---")
    print(trainset[0])
    print(f"\n--- Sample dev example ---")
    print(devset[0])
