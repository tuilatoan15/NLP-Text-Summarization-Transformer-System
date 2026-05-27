"""Service package exports.

Keep package initialization side-effect free to avoid circular imports.
Import concrete services from their modules directly when possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["CacheService"]

if TYPE_CHECKING:
    from backend.services.cache_service import CacheService


def __getattr__(name: str) -> Any:
    """Lazily expose selected service classes for compatibility."""
    if name == "CacheService":
        from backend.services.cache_service import CacheService

        return CacheService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
