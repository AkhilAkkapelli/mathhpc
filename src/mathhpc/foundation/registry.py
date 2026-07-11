"""Frozen-type test registry (Erratum E-3: incremental per-type invariant coverage).

The module that *defines* a frozen foundation type registers it here, at module
bottom, together with a fixture factory producing representative instances. The
generic invariant suite in ``tests/foundation/test_registry.py`` parametrizes over
``registered_frozen_types()``, and a discovery test fails CI if any frozen
dataclass under ``src/`` is unregistered. No import-time scanning magic: all
registration is an explicit call.
"""

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

__all__ = ["FrozenTypeEntry", "register_frozen_type", "registered_frozen_types"]


class FrozenTypeEntry(NamedTuple):
    """Registry infrastructure (a NamedTuple, deliberately not a dataclass, so the
    discovery scan for frozen *dataclasses* has nothing to exempt)."""

    cls: type
    fixture_factory: Callable[[], tuple[object, ...]]


_REGISTRY: dict[type, FrozenTypeEntry] = {}


def _is_frozen_dataclass_type(cls: type) -> bool:
    params = getattr(cls, "__dataclass_params__", None)
    return dataclasses.is_dataclass(cls) and bool(getattr(params, "frozen", False))


def register_frozen_type[T](cls: type[T], *, fixture: Callable[[], tuple[object, ...]]) -> type[T]:
    """Register a frozen foundation dataclass with its fixture factory.

    Raises ``TypeError`` if ``cls`` is not a frozen dataclass and ``ValueError`` on
    duplicate registration (a duplicate means two modules claim to define the type,
    which is a bug either way). Returns ``cls`` so decorator use is possible, though
    an explicit call at module bottom is the canonical style.
    """
    if not _is_frozen_dataclass_type(cls):
        raise TypeError(f"{cls.__qualname__} is not a frozen dataclass")
    if cls in _REGISTRY:
        raise ValueError(f"{cls.__qualname__} is already registered")
    _REGISTRY[cls] = FrozenTypeEntry(cls, fixture)
    return cls


def registered_frozen_types() -> tuple[FrozenTypeEntry, ...]:
    """All registered entries, in registration order."""
    return tuple(_REGISTRY.values())
