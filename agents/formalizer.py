import os
import re
import json
import uuid
import subprocess
import tempfile
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Optional
import dspy
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from agents.researcher import ResearchPlan, InstructionPlan

class HoareFormalizer(dspy.Signature):
    """Convert natural language requirements into a formal Hoare Triple {P} C {Q}."""
    requirement: str = dspy.InputField(desc="The human-readable requirement")
    precondition: str = dspy.OutputField(desc="Logical state before (P)")
    command: str = dspy.OutputField(desc="The action or code (C)")
    postcondition: str = dspy.OutputField(desc="Logical state after (Q)")

class Kripke(BaseModel):
    worlds: str = Field(
        desc=(
            "The set S of possible worlds / states as a comma-separated list. "
            "Each world should be a short identifier (e.g. s0, s1, idle, running)."
        )
    )
    initial_world: str = Field(
        desc="The designated initial world / state (must be a member of S)."
    )
    accessibility_relation: str = Field(
        desc=(
            "The relation R as a set of ordered pairs, e.g. {(s0,s1),(s1,s2),(s2,s0)}. "
            "Each pair (si, sj) means world sj is accessible from world si."
        )
    )
    atomic_propositions: str = Field(
        desc=(
            "Comma-separated list of atomic propositions (boolean facts) relevant to the system, "
            "e.g. door_open, user_authenticated, buffer_empty."
        )
    )
    valuation: str = Field(
        desc=(
            "The valuation function V as a mapping from each atomic proposition to the worlds "
            "where it holds, e.g. {door_open: {s1, s2}, user_authenticated: {s2}}."
        )
    )
    modal_properties: str = Field(
        desc=(
            "Key modal-logic properties to verify over the structure, expressed in CTL or LTL, "
            "e.g. AG(safe), EF(goal), G(p -> F q)."
        )
    )

class KripkeFormalizer(dspy.Signature):
    """Extract a Kripke semantic structure (S, R, V) from natural language text.

    A Kripke structure M = (S, R, V) where:
      S = finite set of possible worlds / states
      R ⊆ S × S = accessibility (transition) relation between worlds
      V : Prop → 2^S = valuation mapping each atomic proposition to the worlds where it holds
    """
    proposed_plan:ResearchPlan = dspy.InputField(desc="Plan proposed by the research agent that has to be executed")
    #reasoning_steps:List[str] = dspy.InputField(desc="Reasoning steps or thought of the agents")
    
    #requirement: str = dspy.InputField(
    #    desc="Natural language description of a system, protocol, or requirement"
    #)
    kripke:Kripke = dspy.OutputField(desc="KripkeStructure of the plan")
    

class Z3Formalization(dspy.Signature):
    """Generate a self-contained Z3Py program for a natural language plan.

    Each call produces ONE independent formalization path.  Use *path_id* to
    vary the encoding strategy so that multiple calls yield diverse candidates
    suitable for majority voting:
      path_1  — direct variable-per-step Bool encoding
      path_2  — integer/bitvector encoding with arithmetic constraints
      path_3  — function-symbol encoding with universal quantifiers
      (and so on for higher path numbers)
    The program must end with a print statement reporting sat/unsat/unknown
    and, when sat, the model values relevant to the question.
    """
    kripke: Kripke = dspy.InputField(
        desc="The research question or problem to verify"
    )
    proposed_plan: ResearchPlan = dspy.InputField(
        desc="Natural language instruction plan to encode as Z3 constraints"
    )
    #path_id: str = dspy.InputField(
    #    desc="Path identifier (e.g. 'path_1').  Use a distinct encoding strategy per path."
    #)
    z3_code: str = dspy.OutputField(
        desc=(
            "Self-contained executable Python using z3-solver.  Must import z3, "
            "declare sorts/variables, assert all constraints from the plan, call "
            "solver.check(), and print a human-readable verdict + model when sat."
        )
    )

class NuSMVFormalization(dspy.Signature):
    """Generate a self-contained NuSMV model for a natural language research plan.

    The model must encode the plan's states and transitions as a Kripke structure
    expressed in NuSMV syntax, then assert CTL/LTL specifications derived from the
    plan's goals so that NuSMV can verify them.
    """
    kripke: Kripke = dspy.InputField(
        desc="Kripke structure (S, R, V) derived from the plan"
    )
    proposed_plan: ResearchPlan = dspy.InputField(
        desc="Natural language instruction plan to encode as NuSMV constraints"
    )
    nusmv_code: str = dspy.OutputField(
        desc=(
            "Self-contained NuSMV model (.smv format). Must declare a MODULE main, "
            "define VAR section with state variables, ASSIGN section encoding transitions "
            "from the Kripke structure, and SPEC / LTLSPEC section asserting properties "
            "from the plan's modal_properties. The model must be complete and runnable "
            "by the nusmv CLI with no additional input."
        )
    )


class Formalizer(dspy.Signature):
    def __init__(self):
        return
    
    def forward(self, proposed_plan:ResearchPlan):
        
        k_formalizer = dspy.ChainOfThought(KripkeFormalizer)
        z3_formalizer = dspy.ChainOfThought(Z3Formalization)
        formalized_plan = k_formalizer.forward(proposed_plan=proposed_plan)
        z3_code = z3_formalizer.forward(proposed_plan=formalized_plan, kripke=formalized_plan.kripke)
        
        #Identify the level of the formatlization 
        
        #TODO run it with code if it's necessary

        # 1. Deductive verification

        # 2. Epistemic Temporal Logic

        # 3. Symbolic representation (Lean, sympy, ans)

        # Make the formalization -> verify correctness of it;

        return z3_code.z3_code
            