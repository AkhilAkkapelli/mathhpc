"""Minimal ``Scope`` and ``Validity`` (spec §7) — the Task 003 subset.

The frozen definition is:

    Scope:    extent: DataExtent; at: ProgramExtent
    DataExtent    = Universal | Shaped(shape_class) | Instance(id) | Machine(topo_hash)
    ProgramExtent = Value(ValueRef) | Region(IrRegionRef) | Window(start, kill)
    Validity: topo: Hash|ANY; module: Hash|ANY; killed_by: tuple[EffectPattern,...]

Task 003 implements the variants its own tests exercise — ``Universal`` data
extent and ``Value`` program extent — and a ``Validity`` with ``topo``/``module``
hash pins (``None`` = ANY). The remaining extent variants, the ``killed_by``
effect patterns, and fold liveness arrive with Task 013/024 by widening these
unions and migrating the ``Validity`` schema — coverage deferred, discipline not.

``Value`` references its subject by surface symbol name (a ``str``); richer
reference kinds (HIR value refs) arrive with the HIR tasks.
"""

from dataclasses import dataclass

from mathhpc.foundation.immutable import validate_frozen_instance
from mathhpc.foundation.registry import register_frozen_type

__all__ = [
    "ANY_VALIDITY",
    "DataExtent",
    "ProgramExtent",
    "Scope",
    "Universal",
    "Validity",
    "Value",
]


@dataclass(frozen=True, slots=True)
class Universal:
    """Data extent: holds for every datum (shape-, instance-, machine-independent)."""

    def __post_init__(self) -> None:
        validate_frozen_instance(self)


@dataclass(frozen=True, slots=True)
class Value:
    """Program extent: attached to one program value, named by surface symbol."""

    ref: str

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.ref) is not str:
            raise TypeError(f"Value.ref must be str, got {type(self.ref).__qualname__}")
        if not self.ref:
            raise ValueError("Value.ref must be non-empty")


type DataExtent = Universal
type ProgramExtent = Value


@dataclass(frozen=True, slots=True)
class Scope:
    """Where a claim or evidence applies: a data extent at a program extent."""

    extent: DataExtent
    at: ProgramExtent

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.extent) is not Universal:
            raise TypeError(
                f"Scope.extent must be a DataExtent, got {type(self.extent).__qualname__}"
            )
        if type(self.at) is not Value:
            raise TypeError(f"Scope.at must be a ProgramExtent, got {type(self.at).__qualname__}")


@dataclass(frozen=True, slots=True)
class Validity:
    """Environmental pins under which evidence stays admissible.

    ``topo`` and ``module`` are content-hash pins; ``None`` means ANY (the
    evidence does not depend on that axis). ``killed_by`` effect patterns are a
    Task 024 schema extension.
    """

    topo: str | None
    module: str | None

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if self.topo is not None and type(self.topo) is not str:
            raise TypeError(
                f"Validity.topo must be str or None, got {type(self.topo).__qualname__}"
            )
        if self.module is not None and type(self.module) is not str:
            raise TypeError(
                f"Validity.module must be str or None, got {type(self.module).__qualname__}"
            )
        if self.topo == "":
            raise ValueError("Validity.topo must be a non-empty hash or None (ANY)")
        if self.module == "":
            raise ValueError("Validity.module must be a non-empty hash or None (ANY)")


ANY_VALIDITY = Validity(topo=None, module=None)


register_frozen_type(
    Universal,
    fixture=lambda: (Universal(), Universal()),
    schema_version=1,
)
register_frozen_type(
    Value,
    fixture=lambda: (Value("A"), Value("A"), Value("C")),
    schema_version=1,
)
register_frozen_type(
    Scope,
    fixture=lambda: (
        Scope(Universal(), Value("A")),
        Scope(Universal(), Value("A")),
        Scope(Universal(), Value("b")),
    ),
    schema_version=1,
)
register_frozen_type(
    Validity,
    fixture=lambda: (
        ANY_VALIDITY,
        Validity(None, None),
        Validity("ab12", None),
        Validity("ab12", "cd34"),
    ),
    schema_version=1,
)
