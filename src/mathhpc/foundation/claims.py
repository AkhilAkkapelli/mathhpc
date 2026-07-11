"""``Statement`` mini-AST, ``ClaimKey``, and ``Claim`` (spec §7, Task 003 subset).

The frozen ``Statement`` comment reads ``Property(obj, prop) | Equivalence(a, b)
| Bound(...) | ContractSat(...)``. Task 003 implements ``Property`` and
``Equivalence``; ``ContractSat`` arrives with ``Contract`` (Task 005+) and
``Bound`` with the task that first produces one — union widening, per the
incremental registry discipline. Subjects and operands are surface symbol /
candidate names (``str``), matching the report rendering
``Equivalence(L3, matmul, MathematicalReal)``; richer references arrive with HIR.

The load-bearing rule (Foundation 2.2, CLAUDE.md governing rule): **equivalence
claims always carry a semantic domain.** ``ClaimKey`` enforces it at
construction — an ``Equivalence`` statement with ``domain=None`` is a
``TypeError``, the Task 003 acceptance negative test.

``ClaimKey`` is the dedup/aggregation identity of a claim; ``Claim`` binds an
identity to a key at a scope.
"""

from dataclasses import dataclass
from enum import Enum, unique

from mathhpc.foundation.domains import (
    IEEE754,
    ContractNumeric,
    ExceptionProfile,
    FpFormat,
    IntegerExact,
    MathematicalComplex,
    MathematicalReal,
    RoundingMode,
    SemanticDomain,
)
from mathhpc.foundation.ids import ClaimId
from mathhpc.foundation.immutable import validate_frozen_instance
from mathhpc.foundation.registry import register_frozen_type
from mathhpc.foundation.scope import Scope, Universal, Value

__all__ = [
    "Claim",
    "ClaimKey",
    "Equivalence",
    "Property",
    "PropertyName",
    "Statement",
]

_DOMAIN_CLASSES = (MathematicalReal, MathematicalComplex, IntegerExact, IEEE754, ContractNumeric)


@unique
class PropertyName(Enum):
    """The MVP property vocabulary (spec §8); ``shape(...)`` statements arrive
    with the dim solver."""

    SYMMETRIC = "symmetric"
    POSITIVE_DEFINITE = "positive_definite"


@dataclass(frozen=True, slots=True)
class Property:
    """``prop`` holds of the object named ``subject``."""

    subject: str
    prop: PropertyName

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.subject) is not str:
            raise TypeError(f"Property.subject must be str, got {type(self.subject).__qualname__}")
        if not self.subject:
            raise ValueError("Property.subject must be non-empty")
        if type(self.prop) is not PropertyName:
            raise TypeError(
                f"Property.prop must be PropertyName, got {type(self.prop).__qualname__}"
            )


@dataclass(frozen=True, slots=True)
class Equivalence:
    """The computations named ``lhs`` and ``rhs`` are equivalent — meaningful only
    within the semantic domain carried by the enclosing ``ClaimKey``."""

    lhs: str
    rhs: str

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        for name in ("lhs", "rhs"):
            operand = getattr(self, name)
            if type(operand) is not str:
                raise TypeError(f"Equivalence.{name} must be str, got {type(operand).__qualname__}")
            if not operand:
                raise ValueError(f"Equivalence.{name} must be non-empty")


type Statement = Property | Equivalence


@dataclass(frozen=True, slots=True)
class ClaimKey:
    """Dedup/aggregation identity: a statement plus the domain it speaks about.

    ``domain`` is required for ``Equivalence`` (Foundation 2.2: an equivalence
    without a domain is not a proposition) and optional for ``Property``.
    """

    stmt: Statement
    domain: SemanticDomain | None

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.stmt) not in (Property, Equivalence):
            raise TypeError(
                f"ClaimKey.stmt must be a Statement, got {type(self.stmt).__qualname__}"
            )
        if self.domain is not None and type(self.domain) not in _DOMAIN_CLASSES:
            raise TypeError(
                f"ClaimKey.domain must be a SemanticDomain or None, "
                f"got {type(self.domain).__qualname__}"
            )
        if type(self.stmt) is Equivalence and self.domain is None:
            raise TypeError(
                "equivalence claims require a semantic domain (Foundation 2.2): "
                f"Equivalence({self.stmt.lhs!r}, {self.stmt.rhs!r}) with domain=None"
            )


@dataclass(frozen=True, slots=True)
class Claim:
    """An identified claim: a key asserted at a scope."""

    id: ClaimId
    key: ClaimKey
    scope: Scope

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.id) is not ClaimId:
            raise TypeError(f"Claim.id must be ClaimId, got {type(self.id).__qualname__}")
        if type(self.key) is not ClaimKey:
            raise TypeError(f"Claim.key must be ClaimKey, got {type(self.key).__qualname__}")
        if type(self.scope) is not Scope:
            raise TypeError(f"Claim.scope must be Scope, got {type(self.scope).__qualname__}")


def _property_keys() -> tuple[object, ...]:
    sym = ClaimKey(Property("A", PropertyName.SYMMETRIC), MathematicalReal())
    return (
        sym,
        ClaimKey(Property("A", PropertyName.SYMMETRIC), MathematicalReal()),
        ClaimKey(Property("A", PropertyName.POSITIVE_DEFINITE), None),
        ClaimKey(
            Equivalence("L3", "dgemm-lowering"),
            IEEE754(FpFormat.B64, RoundingMode.RNE, ExceptionProfile.E2),
        ),
    )


register_frozen_type(
    Property,
    fixture=lambda: (
        Property("A", PropertyName.SYMMETRIC),
        Property("A", PropertyName.SYMMETRIC),
        Property("B", PropertyName.POSITIVE_DEFINITE),
    ),
    schema_version=1,
)
register_frozen_type(
    Equivalence,
    fixture=lambda: (
        Equivalence("L3", "matmul"),
        Equivalence("L3", "matmul"),
        Equivalence("solve", "dpotrf-chain"),
    ),
    schema_version=1,
)
register_frozen_type(ClaimKey, fixture=_property_keys, schema_version=1)
register_frozen_type(
    Claim,
    fixture=lambda: (
        Claim(
            ClaimId(1),
            ClaimKey(Property("A", PropertyName.SYMMETRIC), MathematicalReal()),
            Scope(Universal(), Value("A")),
        ),
        Claim(
            ClaimId(1),
            ClaimKey(Property("A", PropertyName.SYMMETRIC), MathematicalReal()),
            Scope(Universal(), Value("A")),
        ),
        Claim(
            ClaimId(2),
            ClaimKey(Equivalence("L3", "matmul"), MathematicalReal()),
            Scope(Universal(), Value("C")),
        ),
    ),
    schema_version=1,
)
