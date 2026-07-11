"""Unit and negative tests for ``Statement``/``ClaimKey``/``Claim`` and the minimal
``Scope``/``Validity`` (Task 003).

The acceptance-critical case is ``test_equivalence_without_domain_is_a_type_error``.
"""

import pytest

from mathhpc.foundation import (
    IEEE754,
    Claim,
    ClaimId,
    ClaimKey,
    ContractId,
    ContractNumeric,
    Equivalence,
    EvidenceId,
    ExceptionProfile,
    FpFormat,
    IntegerExact,
    MathematicalComplex,
    MathematicalReal,
    OverflowBehavior,
    Property,
    PropertyName,
    RoundingMode,
    Scope,
    Universal,
    Validity,
    Value,
)

_B64 = IEEE754(FpFormat.B64, RoundingMode.RNE, ExceptionProfile.E2)


# --------------------------------------------------------------------------------------
# The Task 003 acceptance test (Foundation 2.2)
# --------------------------------------------------------------------------------------


def test_equivalence_without_domain_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="equivalence claims require a semantic domain"):
        ClaimKey(Equivalence("L3", "matmul"), None)


@pytest.mark.parametrize(
    "domain",
    [
        MathematicalReal(),
        MathematicalComplex(),
        IntegerExact(64, OverflowBehavior.UNDEFINED),
        _B64,
        ContractNumeric(ContractId(1)),
    ],
)
def test_equivalence_with_any_domain_kind_is_legal(domain: object) -> None:
    key = ClaimKey(Equivalence("L3", "matmul"), domain)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    assert key.domain == domain


def test_property_domain_is_optional() -> None:
    ClaimKey(Property("A", PropertyName.SYMMETRIC), None)
    ClaimKey(Property("A", PropertyName.SYMMETRIC), MathematicalReal())


# --------------------------------------------------------------------------------------
# ClaimKey identity and validation
# --------------------------------------------------------------------------------------


def test_claim_key_is_a_dedup_identity() -> None:
    a = ClaimKey(Equivalence("L3", "dgemm-lowering"), _B64)
    b = ClaimKey(Equivalence("L3", "dgemm-lowering"), _B64)
    assert a == b and hash(a) == hash(b)


def test_same_statement_different_domain_is_a_different_claim() -> None:
    real = ClaimKey(Equivalence("L3", "matmul"), MathematicalReal())
    fp = ClaimKey(Equivalence("L3", "matmul"), _B64)
    assert real != fp


def test_claim_key_rejects_non_statement_and_non_domain() -> None:
    with pytest.raises(TypeError, match="stmt must be a Statement"):
        ClaimKey("symmetric(A)", None)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="domain must be a SemanticDomain"):
        ClaimKey(Property("A", PropertyName.SYMMETRIC), RoundingMode.RNE)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_statement_operand_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Property("", PropertyName.SYMMETRIC)
    with pytest.raises(TypeError, match="must be PropertyName"):
        Property("A", "symmetric")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="non-empty"):
        Equivalence("L3", "")
    with pytest.raises(TypeError, match="must be str"):
        Equivalence(3, "matmul")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------------------
# Claim
# --------------------------------------------------------------------------------------


def _key() -> ClaimKey:
    return ClaimKey(Property("A", PropertyName.SYMMETRIC), MathematicalReal())


def _scope() -> Scope:
    return Scope(Universal(), Value("A"))


def test_claim_construction_and_identity() -> None:
    a = Claim(ClaimId(1), _key(), _scope())
    b = Claim(ClaimId(1), _key(), _scope())
    assert a == b and hash(a) == hash(b)


def test_claim_rejects_wrong_id_kind() -> None:
    with pytest.raises(TypeError, match="must be ClaimId"):
        Claim(EvidenceId(1), _key(), _scope())  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_claim_rejects_wrong_field_types() -> None:
    with pytest.raises(TypeError, match="must be ClaimKey"):
        Claim(ClaimId(1), Property("A", PropertyName.SYMMETRIC), _scope())  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="must be Scope"):
        Claim(ClaimId(1), _key(), Universal())  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------------------
# Minimal Scope / Validity
# --------------------------------------------------------------------------------------


def test_scope_construction_and_identity() -> None:
    assert _scope() == _scope() and hash(_scope()) == hash(_scope())


def test_scope_rejects_wrong_extents() -> None:
    with pytest.raises(TypeError, match="extent must be a DataExtent"):
        Scope(Value("A"), Value("A"))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="at must be a ProgramExtent"):
        Scope(Universal(), Universal())  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_value_ref_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Value("")


def test_validity_pins() -> None:
    assert Validity(None, None) == Validity(None, None)
    Validity("ab12", None)
    with pytest.raises(ValueError, match="non-empty hash or None"):
        Validity("", None)
    with pytest.raises(TypeError, match="str or None"):
        Validity(3, None)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
