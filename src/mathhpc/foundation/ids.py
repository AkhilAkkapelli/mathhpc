"""Strongly distinct identity types: ``ClaimId`` and ``EvidenceId``.

Each ID is a one-field frozen dataclass. There is deliberately no shared base
class — a common base would invite ``isinstance`` erosion of exactly the kind
separation these types exist to provide. The six-line pattern is duplicated on
purpose; future ID types (Task 002+) copy it and register themselves the same way.

IDs are identities, not quantities: equality is kind + value, hashing mixes the
kind, and ordering does not exist. Serialization is the Task 002 serializer's
responsibility; ``value`` is public for it.
"""

from dataclasses import dataclass
from typing import ClassVar, Final

from mathhpc.foundation.registry import register_frozen_type

__all__ = ["ClaimId", "EvidenceId"]

_U64_MAX: Final = 2**64 - 1


@dataclass(frozen=True, slots=True)
class ClaimId:
    """Opaque identity of a Claim."""

    value: int
    _KIND: ClassVar[str] = "ClaimId"

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise TypeError(
                f"{self._KIND}.value must be int (exactly), got {type(self.value).__qualname__}"
            )
        if not 0 <= self.value <= _U64_MAX:
            raise ValueError(f"{self._KIND}.value out of u64 range: {self.value}")

    def __hash__(self) -> int:
        return hash((self._KIND, self.value))

    def __repr__(self) -> str:
        return f"{self._KIND}({self.value})"


@dataclass(frozen=True, slots=True)
class EvidenceId:
    """Opaque identity of an Evidence record."""

    value: int
    _KIND: ClassVar[str] = "EvidenceId"

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise TypeError(
                f"{self._KIND}.value must be int (exactly), got {type(self.value).__qualname__}"
            )
        if not 0 <= self.value <= _U64_MAX:
            raise ValueError(f"{self._KIND}.value out of u64 range: {self.value}")

    def __hash__(self) -> int:
        return hash((self._KIND, self.value))

    def __repr__(self) -> str:
        return f"{self._KIND}({self.value})"


register_frozen_type(
    ClaimId,
    fixture=lambda: (ClaimId(0), ClaimId(0), ClaimId(1), ClaimId(_U64_MAX)),
)
register_frozen_type(
    EvidenceId,
    fixture=lambda: (EvidenceId(0), EvidenceId(0), EvidenceId(7)),
)
