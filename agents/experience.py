"""Experience decorator for agent forward() methods.

Usage
-----
Apply to any agent's forward method::

    from agents.experience import with_experience, default_store

    class Researcher(dspy.Module):
        @with_experience(enable=True)
        def forward(self, prompt, ...):
            ...

The decorator can also be toggled at runtime::

    Researcher.forward.experience_enabled = False   # disable
    Researcher.forward.experience_enabled = True    # re-enable

The global default_store is shared across all decorated agents so every
agent can learn from the failures and successes of others.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import dspy


# ── Experience entry ──────────────────────────────────────────────────────────

@dataclass
class ExperienceEntry:
    name:  str   # "<ClassName>.forward"
    task:  str   # the prompt / task string
    value: str   # "SUCCESS" or "<ErrorType>: <message>"


# ── Experience store ──────────────────────────────────────────────────────────

class ExperienceStore:
    """Ordered collection of agent forward() call outcomes."""

    def __init__(self) -> None:
        self._entries: list[ExperienceEntry] = []

    # ── write ──────────────────────────────────────────────────────────────────

    def record(self, name: str, task: str, value: str) -> None:
        self._entries.append(ExperienceEntry(name=name, task=task, value=value))

    # ── read ───────────────────────────────────────────────────────────────────

    def recent(self, n: int = 5) -> list[ExperienceEntry]:
        return self._entries[-n:]

    def as_context(self, n: int = 5) -> str:
        """Format the n most recent entries as a prompt prefix."""
        entries = self.recent(n)
        if not entries:
            return ""
        lines = ["=== Past Agent Experience (take into account) ==="]
        for e in entries:
            status = "SUCCESS" if e.value == "SUCCESS" else f"FAILED — {e.value}"
            lines.append(f"[{e.name}] {e.task[:150].strip()} → {status}")
        lines.append("=== End of Experience ===")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ExperienceStore({len(self._entries)} entries)"


# ── Global shared store ───────────────────────────────────────────────────────

default_store = ExperienceStore()


# ── DSPy history helper ───────────────────────────────────────────────────────

def _last_dspy_history_summary() -> str:
    """Return a one-line summary of the most recent DSPy LM call, if available."""
    try:
        lm = dspy.settings.lm
        history = getattr(lm, "history", None)
        if not history:
            return ""
        last = history[-1]
        # DSPy stores dicts with keys: prompt, response, usage, etc.
        response = last.get("response") or last.get("outputs") or ""
        if isinstance(response, list):
            response = response[0] if response else ""
        return str(response)[:200]
    except Exception:
        return ""


# ── Decorator ─────────────────────────────────────────────────────────────────

def with_experience(
    store: ExperienceStore | None = None,
    enable: bool = True,
    context_entries: int = 5,
) -> Callable:
    """Decorator factory for agent forward() methods.

    Parameters
    ----------
    store:
        ExperienceStore to read from and write to.
        Defaults to the module-level ``default_store``.
    enable:
        Initial enabled state.  Can be toggled at runtime via
        ``SomeAgent.forward.experience_enabled = False``.
    context_entries:
        How many past experience entries to prepend to the prompt.
    """
    _store = store or default_store

    def decorator(fn: Callable) -> Callable:

        @wraps(fn)
        def wrapper(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
            if not wrapper.experience_enabled:
                return fn(self, prompt, *args, **kwargs)

            name = f"{type(self).__name__}.{fn.__name__}"

            # ── inject past experience into the prompt ─────────────────────────
            exp_ctx = _store.as_context(context_entries)
            enriched_prompt = f"{exp_ctx}\n\n{prompt}" if exp_ctx else prompt

            # ── call the real forward ──────────────────────────────────────────
            try:
                result = fn(self, enriched_prompt, *args, **kwargs)

                # Check result.success when present (e.g. ExecutionResult).
                # A returned object with success=False is a logical failure even
                # though no exception was raised.
                if hasattr(result, "success") and not result.success:
                    stderr_snippet = getattr(result, "stderr", "")[:200]
                    repairs = getattr(result, "repair_attempts", 0)
                    value = (
                        f"FAIL — execution failed after {repairs} repair(s)"
                        + (f": {stderr_snippet}" if stderr_snippet else "")
                    )
                else:
                    lm_note = _last_dspy_history_summary()
                    value = f"SUCCESS — {lm_note}" if lm_note else "SUCCESS"

                _store.record(name, prompt, value)
                return result

            except Exception as exc:
                tb_last = traceback.format_exc().strip().splitlines()[-1]
                error_desc = f"{type(exc).__name__}: {str(exc)[:200]} | {tb_last}"
                _store.record(name, prompt, error_desc)
                raise

        # runtime-toggleable flag
        wrapper.experience_enabled = enable
        wrapper.experience_store   = _store
        return wrapper

    return decorator
