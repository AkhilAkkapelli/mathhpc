"""Unit tests for the ``SemanticDomain`` variants."""

from typing import assert_never

import pytest

from mathhpc.foundation import (
    IEEE754,
    ClaimId,
    ContractId,
    ContractNumeric,
    ExceptionProfile,
    FpFormat,
    IntegerExact,
    MathematicalComplex,
    MathematicalReal,
    OverflowBehavior,
    RoundingMode,
    SemanticDomain,
)


def test_singleton_domains_equal_within_and_distinct_across_kinds() -> None:
    assert MathematicalReal() == MathematicalReal()
    assert MathematicalComplex() == MathematicalComplex()
    assert MathematicalReal() != MathematicalComplex()


def test_integer_exact_identity() -> None:
    a = IntegerExact(64, OverflowBehavior.WRAP)
    b = IntegerExact(64, OverflowBehavior.WRAP)
    assert a == b and hash(a) == hash(b)
    assert a != IntegerExact(32, OverflowBehavior.WRAP)
    assert a != IntegerExact(64, OverflowBehavior.TRAP)


def test_integer_exact_rejects_bad_width() -> None:
    with pytest.raises(ValueError):
        IntegerExact(0, OverflowBehavior.WRAP)
    with pytest.raises(TypeError):
        IntegerExact(True, OverflowBehavior.WRAP)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_ieee754_identity_and_field_kinds() -> None:
    a = IEEE754(FpFormat.B64, RoundingMode.RNE, ExceptionProfile.E2)
    b = IEEE754(FpFormat.B64, RoundingMode.RNE, ExceptionProfile.E2)
    assert a == b and hash(a) == hash(b)
    with pytest.raises(TypeError):
        IEEE754(RoundingMode.RNE, RoundingMode.RNE, ExceptionProfile.E2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_contract_numeric_requires_contract_id_exactly() -> None:
    assert ContractNumeric(ContractId(1)) == ContractNumeric(ContractId(1))
    with pytest.raises(TypeError):
        ContractNumeric(ClaimId(1))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def _describe(domain: SemanticDomain) -> str:
    """Exhaustive-match consumer: pyright strict fails this file if a variant is added
    without extending the match (the runtime union/consumer registry is Task 015)."""
    match domain:
        case MathematicalReal():
            return "R"
        case MathematicalComplex():
            return "C"
        case IntegerExact(width=w):
            return f"int{w}"
        case IEEE754(format=f):
            return f.value
        case ContractNumeric(contract=c):
            return repr(c)
        case _:
            assert_never(domain)


def test_semantic_domain_union_is_exhaustively_matchable() -> None:
    assert _describe(MathematicalReal()) == "R"
    assert _describe(IEEE754(FpFormat.B64, RoundingMode.RNE, ExceptionProfile.E2)) == "b64"
    assert _describe(ContractNumeric(ContractId(9))) == "ContractId(9)"
    assert _describe(IntegerExact(16, OverflowBehavior.SATURATE)) == "int16"
    assert _describe(MathematicalComplex()) == "C"
