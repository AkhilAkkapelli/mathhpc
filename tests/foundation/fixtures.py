"""Representative frozen fixture types exercising the canonical validation pattern."""

from dataclasses import dataclass

from mathhpc.foundation import (
    ClaimId,
    ContractId,
    ContractNumeric,
    FrozenDict,
    MathematicalReal,
    RoundingMode,
    SemanticDomain,
    register_frozen_type,
    validate_frozen_instance,
)

__all__ = ["Box", "ExampleNested", "ExampleRecord", "SerializationProbe"]


@dataclass(frozen=True, slots=True)
class Box:
    """Non-validating frozen wrapper for validator and strategy tests.

    Deliberately has no ``__post_init__`` so tests can construct it around mutable
    contaminants and cycles. Defined under ``tests/`` — the discovery scan covers
    ``src/`` only — and deliberately unregistered.
    """

    item: object


@dataclass(frozen=True, slots=True)
class ExampleRecord:
    """Canonical frozen-record pattern: explicit ID-kind check + field validation."""

    claim: ClaimId
    name: str
    meta: FrozenDict[str, int]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.claim) is not ClaimId:
            raise TypeError(
                f"ExampleRecord.claim must be ClaimId, got {type(self.claim).__qualname__}"
            )
        validate_frozen_instance(self)


@dataclass(frozen=True, slots=True)
class ExampleNested:
    """Frozen record nesting another frozen record."""

    record: ExampleRecord
    extras: frozenset[int]

    def __post_init__(self) -> None:
        validate_frozen_instance(self)


@dataclass(frozen=True, slots=True)
class SerializationProbe:
    """Exercises the serializer's edge encodings: float (incl. infinities), bytes,
    an optional field, an enum field, and a tagged-union (``SemanticDomain``) field."""

    ratio: float
    blob: bytes
    domain: SemanticDomain
    note: int | None
    rounding: RoundingMode

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.ratio) is not float:
            raise TypeError(f"ratio must be float, got {type(self.ratio).__qualname__}")
        if type(self.blob) is not bytes:
            raise TypeError(f"blob must be bytes, got {type(self.blob).__qualname__}")
        if self.note is not None and type(self.note) is not int:
            raise TypeError(f"note must be int or None, got {type(self.note).__qualname__}")
        if type(self.rounding) is not RoundingMode:
            raise TypeError(
                f"rounding must be RoundingMode, got {type(self.rounding).__qualname__}"
            )


def _example_records() -> tuple[object, ...]:
    first = ExampleRecord(ClaimId(1), "alpha", FrozenDict({"x": 1, "y": 2}), ("t1", "t2"))
    duplicate = ExampleRecord(ClaimId(1), "alpha", FrozenDict({"x": 1, "y": 2}), ("t1", "t2"))
    other = ExampleRecord(ClaimId(2), "beta", FrozenDict({}), ())
    return (first, duplicate, other)


def _example_nested() -> tuple[object, ...]:
    record = ExampleRecord(ClaimId(3), "gamma", FrozenDict({"n": 9}), ("t",))
    return (
        ExampleNested(record, frozenset({1, 2})),
        ExampleNested(record, frozenset({1, 2})),
        ExampleNested(record, frozenset()),
    )


def _serialization_probes() -> tuple[object, ...]:
    real = MathematicalReal()
    return (
        SerializationProbe(0.5, b"\x00\x01", real, 7, RoundingMode.RNE),
        SerializationProbe(0.5, b"\x00\x01", real, 7, RoundingMode.RNE),
        SerializationProbe(
            float("inf"), b"", ContractNumeric(ContractId(3)), None, RoundingMode.RTZ
        ),
        SerializationProbe(float("-inf"), b"z", real, 0, RoundingMode.RUP),
    )


register_frozen_type(ExampleRecord, fixture=_example_records, schema_version=1)
register_frozen_type(ExampleNested, fixture=_example_nested, schema_version=1)
register_frozen_type(SerializationProbe, fixture=_serialization_probes, schema_version=1)
