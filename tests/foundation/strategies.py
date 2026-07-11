"""Hypothesis strategies for the Task 001 property suite and the Task 002
JSON round-trip suite (invariant 4)."""

from collections.abc import Callable

import hypothesis.strategies as st

from mathhpc.foundation import (
    IEEE754,
    ClaimId,
    ContractId,
    ContractNumeric,
    DecisionId,
    EvidenceId,
    ExceptionProfile,
    FpFormat,
    FrozenDict,
    IntegerExact,
    MathematicalComplex,
    MathematicalReal,
    OverflowBehavior,
    PlanId,
    RoundingMode,
    SourceSpan,
)
from tests.foundation.fixtures import Box, ExampleNested, ExampleRecord, SerializationProbe

__all__ = ["contaminated", "immutable_values", "serialization_strategies"]


def _leaves() -> st.SearchStrategy[object]:
    return st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True),
        st.text(max_size=6),
        st.binary(max_size=6),
    )


def immutable_values() -> st.SearchStrategy[object]:
    """Arbitrarily nested values built only from allowed immutable forms."""

    def extend(children: st.SearchStrategy[object]) -> st.SearchStrategy[object]:
        tuples = st.lists(children, max_size=3).map(tuple)
        fsets = st.lists(children, max_size=3).map(frozenset)
        keys = st.text(max_size=4)
        pairs = st.lists(st.tuples(keys, children), max_size=3, unique_by=lambda kv: kv[0])
        fds = pairs.map(FrozenDict)
        return st.one_of(tuples, fsets, fds)

    return st.recursive(_leaves(), extend, max_leaves=15)


_MUTANT_BUILDERS: tuple[Callable[[], object], ...] = (
    list,
    dict,
    set,
    lambda: bytearray(b"m"),
)


_U64 = st.integers(min_value=0, max_value=2**64 - 1)


@st.composite
def _source_spans(draw: st.DrawFn) -> SourceSpan:
    file = draw(st.text(min_size=1, max_size=12))
    line0 = draw(st.integers(min_value=1, max_value=500))
    extra_lines = draw(st.integers(min_value=0, max_value=5))
    col0 = draw(st.integers(min_value=0, max_value=120))
    if extra_lines == 0:
        col1 = draw(st.integers(min_value=col0, max_value=col0 + 120))
    else:
        col1 = draw(st.integers(min_value=0, max_value=120))
    return SourceSpan(file, line0, col0, line0 + extra_lines, col1)


def _semantic_domains() -> st.SearchStrategy[object]:
    return st.one_of(
        st.builds(MathematicalReal),
        st.builds(MathematicalComplex),
        st.builds(
            IntegerExact,
            width=st.integers(min_value=1, max_value=128),
            ovf=st.sampled_from(OverflowBehavior),
        ),
        st.builds(
            IEEE754,
            format=st.sampled_from(FpFormat),
            rounding=st.sampled_from(RoundingMode),
            exc_profile=st.sampled_from(ExceptionProfile),
        ),
        st.builds(ContractNumeric, contract=_U64.map(ContractId)),
    )


def serialization_strategies() -> dict[type, st.SearchStrategy[object]]:
    """Per-type instance strategies for the JSON round-trip invariant (invariant 4).

    Types absent here fall back to their registry fixture instances in the
    property suite; adding a strategy upgrades a type from example-based to
    generative coverage.
    """
    example_records = st.builds(
        ExampleRecord,
        claim=_U64.map(ClaimId),
        name=st.text(max_size=8),
        meta=st.dictionaries(st.text(max_size=4), st.integers(), max_size=4).map(FrozenDict),
        tags=st.lists(st.text(max_size=4), max_size=4).map(tuple),
    )
    return {
        ClaimId: _U64.map(ClaimId),
        EvidenceId: _U64.map(EvidenceId),
        DecisionId: _U64.map(DecisionId),
        PlanId: _U64.map(PlanId),
        ContractId: _U64.map(ContractId),
        SourceSpan: _source_spans(),
        MathematicalReal: st.builds(MathematicalReal),
        MathematicalComplex: st.builds(MathematicalComplex),
        IntegerExact: st.builds(
            IntegerExact,
            width=st.integers(min_value=1, max_value=128),
            ovf=st.sampled_from(OverflowBehavior),
        ),
        IEEE754: st.builds(
            IEEE754,
            format=st.sampled_from(FpFormat),
            rounding=st.sampled_from(RoundingMode),
            exc_profile=st.sampled_from(ExceptionProfile),
        ),
        ContractNumeric: st.builds(ContractNumeric, contract=_U64.map(ContractId)),
        ExampleRecord: example_records,
        ExampleNested: st.builds(
            ExampleNested,
            record=example_records,
            extras=st.frozensets(st.integers(), max_size=4),
        ),
        SerializationProbe: st.builds(
            SerializationProbe,
            ratio=st.floats(allow_nan=False),
            blob=st.binary(max_size=8),
            domain=_semantic_domains(),
            note=st.none() | st.integers(),
            rounding=st.sampled_from(RoundingMode),
        ),
    }


@st.composite
def contaminated(draw: st.DrawFn) -> tuple[object, str]:
    """An immutable-looking structure with exactly one mutable value planted at a
    known location; returns ``(value, expected_rendered_path)``."""
    value: object = draw(st.sampled_from(_MUTANT_BUILDERS))()
    path = ""
    depth = draw(st.integers(min_value=0, max_value=4))
    for _ in range(depth):
        wrapper = draw(st.sampled_from(("tuple", "frozendict", "box")))
        if wrapper == "tuple":
            pad = draw(st.integers(min_value=0, max_value=2))
            value = (*(("pad",) * pad), value)
            path = f"[{pad}]{path}"
        elif wrapper == "frozendict":
            key = draw(st.text(alphabet="abcd", min_size=1, max_size=3))
            value = FrozenDict({key: value})
            path = f"[{key!r}]{path}"
        else:
            value = Box(item=value)
            path = f".item{path}"
    return value, f"root{path}"
