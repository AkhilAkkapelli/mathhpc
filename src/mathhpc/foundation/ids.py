"""Strongly distinct identity types: the opaque-u64 IDs of the core spec (§7).

Each ID is a one-field frozen dataclass. There is deliberately no shared base
class — a common base would invite ``isinstance`` erosion of exactly the kind
separation these types exist to provide. The six-line pattern is duplicated on
purpose; every ID type copies it and registers itself the same way. Task 001
shipped ``ClaimId``/``EvidenceId``; Task 002 adds ``DecisionId``, ``PlanId``,
``ContractId`` (the remaining IDs of spec §7 line "opaque u64"); Task 003 adds
``GuardId`` alongside the ``RuntimeChecked`` evidence status that references it.
Later record IDs (``HirId``, …) arrive in the PR introducing their record.

IDs are identities, not quantities: equality is kind + value, hashing mixes the
kind, and ordering does not exist. JSON serialization goes through the Task 002
harness (``mathhpc.foundation.serialize``); ``value`` is public for it.
"""

from dataclasses import dataclass
from typing import ClassVar, Final

from mathhpc.foundation.registry import register_frozen_type

__all__ = ["ClaimId", "ContractId", "DecisionId", "EvidenceId", "GuardId", "PlanId"]

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


@dataclass(frozen=True, slots=True)
class DecisionId:
    """Opaque identity of a Decision record."""

    value: int
    _KIND: ClassVar[str] = "DecisionId"

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
class PlanId:
    """Opaque identity of a compiled Plan."""

    value: int
    _KIND: ClassVar[str] = "PlanId"

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
class ContractId:
    """Opaque identity of a Contract (referenced by ``ContractNumeric`` domains)."""

    value: int
    _KIND: ClassVar[str] = "ContractId"

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
class GuardId:
    """Opaque identity of a runtime Guard (referenced by ``RuntimeChecked`` evidence;
    the Guard record itself arrives with the Plan IR / guard-library tasks)."""

    value: int
    _KIND: ClassVar[str] = "GuardId"

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
    schema_version=1,
)
register_frozen_type(
    EvidenceId,
    fixture=lambda: (EvidenceId(0), EvidenceId(0), EvidenceId(7)),
    schema_version=1,
)
register_frozen_type(
    DecisionId,
    fixture=lambda: (DecisionId(0), DecisionId(0), DecisionId(11), DecisionId(_U64_MAX)),
    schema_version=1,
)
register_frozen_type(
    PlanId,
    fixture=lambda: (PlanId(0), PlanId(0), PlanId(13)),
    schema_version=1,
)
register_frozen_type(
    ContractId,
    fixture=lambda: (ContractId(0), ContractId(0), ContractId(17)),
    schema_version=1,
)
register_frozen_type(
    GuardId,
    fixture=lambda: (GuardId(0), GuardId(0), GuardId(23)),
    schema_version=1,
)
