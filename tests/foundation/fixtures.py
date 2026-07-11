"""Representative frozen fixture types exercising the canonical validation pattern."""

from dataclasses import dataclass

from mathhpc.foundation import ClaimId, FrozenDict, register_frozen_type, validate_frozen_instance

__all__ = ["Box", "ExampleNested", "ExampleRecord"]


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


register_frozen_type(ExampleRecord, fixture=_example_records)
register_frozen_type(ExampleNested, fixture=_example_nested)
