"""Re-export the authoritative tool registry from core.tool_registry."""
from core.tool_registry import (
    _TOOL_REGISTRY    as TOOL_REGISTRY,
    _tool_names_str   as registry_names,
    _tool_descriptions_str as registry_descriptions,
)

__all__ = ["TOOL_REGISTRY", "registry_names", "registry_descriptions"]
