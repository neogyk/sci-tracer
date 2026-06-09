"""SciTracer planning agents.

Provides tree-search and evolutionary planning modules that can replace or
augment the flat InstructionPlan planner used in the main RAG pipeline.

Available
---------
PlanOrchestrator        — two-phase planner: CoT macro plan + LATS per milestone
OrchestratorResult      — result returned by PlanOrchestrator.forward()

LATS                    — Language Agent Tree Search (Zhou et al., 2023)
LATSNode                — individual node in the LATS search tree
LATSResult              — result returned by LATS.forward()

RHEA                    — Rolling Horizon Evolutionary Algorithm
RHEAChromosome          — one individual in the RHEA population
RHEAResult              — result returned by RHEA.forward()

PlanVerifierAgent       — agent-driven plan/solution verifier
AgentVerificationResult — per-check result (mirrors tools.kripke.VerificationResult)
AgentVerificationReport — full structured report from PlanVerifierAgent
"""
from agents.planning.lats import LATS, LATSNode, LATSResult
from agents.planning.rhea import RHEA, RHEAChromosome, RHEAResult
from agents.planning.orchestrator import PlanOrchestrator, OrchestratorResult
from agents.planning.agent_verifier import (
    PlanVerifierAgent,
    AgentVerificationResult,
    AgentVerificationReport,
)

__all__ = [
    "PlanOrchestrator",
    "OrchestratorResult",
    "LATS",
    "LATSNode",
    "LATSResult",
    "RHEA",
    "RHEAChromosome",
    "RHEAResult",
    "PlanVerifierAgent",
    "AgentVerificationResult",
    "AgentVerificationReport",
]
