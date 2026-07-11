"""``SemanticDomain`` (Foundation v1.0, spec §7): the tag every equivalence and
claim carries — *which arithmetic* a statement is about.

The union is exactly the frozen definition:

    SemanticDomain = MathematicalReal | MathematicalComplex | IntegerExact(width, ovf)
                   | IEEE754(format, rounding, exc_profile) | ContractNumeric(ContractId)

Consumers take the union through ``match`` with ``typing.assert_never``; the
union/consumer registry test that polices folds arrives with Task 015. The
``ExceptionProfile`` members E1–E4 are the named exceptional-behavior profiles of
spec §7; their expansion into the ``ExceptionalBehavior`` frozenset belongs to
``Contract`` (Task 005), not here.
"""

from dataclasses import dataclass
from enum import Enum, unique

from mathhpc.foundation.ids import ContractId
from mathhpc.foundation.immutable import validate_frozen_instance
from mathhpc.foundation.registry import register_frozen_type

__all__ = [
    "IEEE754",
    "ContractNumeric",
    "ExceptionProfile",
    "FpFormat",
    "IntegerExact",
    "MathematicalComplex",
    "MathematicalReal",
    "OverflowBehavior",
    "RoundingMode",
    "SemanticDomain",
]


@unique
class FpFormat(Enum):
    """IEEE 754 binary interchange formats."""

    B16 = "b16"
    B32 = "b32"
    B64 = "b64"
    B128 = "b128"


@unique
class RoundingMode(Enum):
    """The five IEEE 754 rounding-direction attributes."""

    RNE = "rne"  # roundTiesToEven
    RNA = "rna"  # roundTiesToAway
    RTZ = "rtz"  # roundTowardZero
    RUP = "rup"  # roundTowardPositive
    RDN = "rdn"  # roundTowardNegative


@unique
class ExceptionProfile(Enum):
    """Named exceptional-behavior profiles (spec §7, "E1..E4 named profiles")."""

    E1 = "e1"
    E2 = "e2"
    E3 = "e3"
    E4 = "e4"


@unique
class OverflowBehavior(Enum):
    """What ``IntegerExact`` arithmetic does on overflow."""

    WRAP = "wrap"
    SATURATE = "saturate"
    TRAP = "trap"
    UNDEFINED = "undefined"


@dataclass(frozen=True, slots=True)
class MathematicalReal:
    """Exact arithmetic over the mathematical reals."""

    def __post_init__(self) -> None:
        validate_frozen_instance(self)


@dataclass(frozen=True, slots=True)
class MathematicalComplex:
    """Exact arithmetic over the mathematical complex numbers."""

    def __post_init__(self) -> None:
        validate_frozen_instance(self)


@dataclass(frozen=True, slots=True)
class IntegerExact:
    """Exact machine-integer arithmetic of a given bit width and overflow rule."""

    width: int
    ovf: OverflowBehavior

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.width) is not int:
            raise TypeError(
                f"IntegerExact.width must be int (exactly), got {type(self.width).__qualname__}"
            )
        if self.width < 1:
            raise ValueError(f"IntegerExact.width must be >= 1, got {self.width}")
        if type(self.ovf) is not OverflowBehavior:
            raise TypeError(
                f"IntegerExact.ovf must be OverflowBehavior, got {type(self.ovf).__qualname__}"
            )


@dataclass(frozen=True, slots=True)
class IEEE754:
    """Floating-point arithmetic under a format, rounding mode, and exception profile."""

    format: FpFormat
    rounding: RoundingMode
    exc_profile: ExceptionProfile

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.format) is not FpFormat:
            raise TypeError(
                f"IEEE754.format must be FpFormat, got {type(self.format).__qualname__}"
            )
        if type(self.rounding) is not RoundingMode:
            raise TypeError(
                f"IEEE754.rounding must be RoundingMode, got {type(self.rounding).__qualname__}"
            )
        if type(self.exc_profile) is not ExceptionProfile:
            raise TypeError(
                f"IEEE754.exc_profile must be ExceptionProfile, "
                f"got {type(self.exc_profile).__qualname__}"
            )


@dataclass(frozen=True, slots=True)
class ContractNumeric:
    """Arithmetic as promised by an active numerical contract."""

    contract: ContractId

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.contract) is not ContractId:
            raise TypeError(
                f"ContractNumeric.contract must be ContractId, "
                f"got {type(self.contract).__qualname__}"
            )


type SemanticDomain = (
    MathematicalReal | MathematicalComplex | IntegerExact | IEEE754 | ContractNumeric
)


register_frozen_type(
    MathematicalReal,
    fixture=lambda: (MathematicalReal(), MathematicalReal()),
    schema_version=1,
)
register_frozen_type(
    MathematicalComplex,
    fixture=lambda: (MathematicalComplex(), MathematicalComplex()),
    schema_version=1,
)
register_frozen_type(
    IntegerExact,
    fixture=lambda: (
        IntegerExact(64, OverflowBehavior.UNDEFINED),
        IntegerExact(64, OverflowBehavior.UNDEFINED),
        IntegerExact(32, OverflowBehavior.WRAP),
    ),
    schema_version=1,
)
register_frozen_type(
    IEEE754,
    fixture=lambda: (
        IEEE754(FpFormat.B64, RoundingMode.RNE, ExceptionProfile.E2),
        IEEE754(FpFormat.B64, RoundingMode.RNE, ExceptionProfile.E2),
        IEEE754(FpFormat.B32, RoundingMode.RTZ, ExceptionProfile.E1),
    ),
    schema_version=1,
)
register_frozen_type(
    ContractNumeric,
    fixture=lambda: (
        ContractNumeric(ContractId(1)),
        ContractNumeric(ContractId(1)),
        ContractNumeric(ContractId(2)),
    ),
    schema_version=1,
)
