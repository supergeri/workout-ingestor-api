"""Platform adapter registry for unified workout ingestion."""
from typing import Dict, Optional, Type
from .base import PlatformAdapter, MediaContent, PlatformFetchError

_ADAPTER_REGISTRY: Dict[str, Type[PlatformAdapter]] = {}


def register_adapter(adapter_class: Type[PlatformAdapter]) -> None:
    """Register a platform adapter class."""
    _ADAPTER_REGISTRY[adapter_class.platform_name()] = adapter_class


def get_adapter(platform: str) -> PlatformAdapter:
    """Get an instantiated adapter for the given platform.

    Raises:
        KeyError: If no adapter is registered for the platform.
    """
    cls = _ADAPTER_REGISTRY[platform]
    return cls()


__all__ = [
    "register_adapter",
    "get_adapter",
    "PlatformAdapter",
    "MediaContent",
    "PlatformFetchError",
]
