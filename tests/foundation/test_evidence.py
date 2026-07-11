"""Unit and negative tests for ``ArtifactRef``, the full ``EvidenceStatus`` union,
``Evidence``, and the two Task 003 producers."""

import hashlib
from typing import assert_never

import pytest

from mathhpc.foundation import (
    EMPTY_ARTIFACT,
    AiInferred,
    ArtifactRef,
    Assumed,
    Bundled,
    ClaimId,
    Evidence,
    EvidenceId,
    EvidenceStatus,
    FormallyProved,
    GuardId,
    Measured,
    Predicted,
    RuntimeChecked,
    Scope,
    SmtProved,
    Specified,
    StaticallyDerived,
    Universal,
    Validity,
    Value,
    specified_evidence,
    statically_derived_evidence,
)

_SCOPE = Scope(Universal(), Value("A"))


# --------------------------------------------------------------------------------------
# ArtifactRef
# --------------------------------------------------------------------------------------


def test_artifact_ref_of_computes_matching_hash() -> None:
    ref = ArtifactRef.of(b"payload")
    assert ref.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert ref == ArtifactRef.of(b"payload")
    assert EMPTY_ARTIFACT == ArtifactRef.of(b"")


def test_artifact_ref_rejects_hash_mismatch() -> None:
    good = ArtifactRef.of(b"a").sha256
    with pytest.raises(ValueError, match="content hash mismatch"):
        ArtifactRef(good, b"b")


def test_artifact_ref_rejects_malformed_hash() -> None:
    with pytest.raises(ValueError, match="64 lowercase hex"):
        ArtifactRef("abc", b"")
    with pytest.raises(ValueError, match="64 lowercase hex"):
        ArtifactRef(ArtifactRef.of(b"").sha256.upper(), b"")


# --------------------------------------------------------------------------------------
# The full EvidenceStatus union
# --------------------------------------------------------------------------------------


def _one_of_each_status() -> tuple[EvidenceStatus, ...]:
    return (
        Assumed("user"),
        Specified("blas-reference", Bundled()),
        StaticallyDerived("spd_implies_symmetric"),
        SmtProved("z3", EMPTY_ARTIFACT),
        FormallyProved("lean4", EMPTY_ARTIFACT),
        RuntimeChecked(GuardId(1), 0),
        Measured("topo:ab12", 200, (0.9, 1.1)),
        Predicted("cost-v0", "calib-2026-07"),
        AiInferred("proposer-v0", 0.5),
    )


def _classify(status: EvidenceStatus) -> str:
    """Exhaustive-match consumer: pyright strict fails this file if a variant is
    ever added without extending the match (the fold itself is Task 004)."""
    match status:
        case Assumed(who=w):
            return f"assumed:{w}"
        case Specified(pack=p):
            return f"specified:{p}"
        case StaticallyDerived(rule=r):
            return f"derived:{r}"
        case SmtProved(solver=s):
            return f"smt:{s}"
        case FormallyProved(system=s):
            return f"formal:{s}"
        case RuntimeChecked(guard=g):
            return f"runtime:{g!r}"
        case Measured(n=n):
            return f"measured:{n}"
        case Predicted(model=m):
            return f"predicted:{m}"
        case AiInferred(score=s):
            return f"ai:{s}"
        case _:
            assert_never(status)


def test_all_nine_variants_exist_and_are_matchable() -> None:
    statuses = _one_of_each_status()
    assert len(statuses) == 9
    assert len({_classify(s) for s in statuses}) == 9


def test_no_scores_on_proof_classes() -> None:
    """Structural: the proof-grade variants cannot hold a numeric confidence —
    every field is a name, a provenance record, or a proof artifact."""
    import typing

    from mathhpc.foundation import SpecProvenance

    allowed = {str, Bundled, ArtifactRef, SpecProvenance}
    for cls in (Assumed, Specified, StaticallyDerived, SmtProved, FormallyProved):
        annotations = set(typing.get_type_hints(cls).values())
        assert annotations <= allowed, (cls.__name__, annotations)


def test_runtime_checked_validation() -> None:
    with pytest.raises(TypeError, match="must be GuardId"):
        RuntimeChecked(ClaimId(1), 0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="must be int"):
        RuntimeChecked(GuardId(1), True)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=">= 0"):
        RuntimeChecked(GuardId(1), -1)


def test_measured_validation() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        Measured("m", 0, (0.0, 1.0))
    with pytest.raises(TypeError, match=r"\(float, float\)"):
        Measured("m", 1, (0, 1))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="lo <= hi"):
        Measured("m", 1, (2.0, 1.0))
    with pytest.raises(ValueError, match="lo <= hi"):
        Measured("m", 1, (float("nan"), 1.0))


def test_ai_inferred_score_bounds() -> None:
    AiInferred("m", 0.0)
    AiInferred("m", 1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        AiInferred("m", 1.5)
    with pytest.raises(TypeError, match="must be float"):
        AiInferred("m", 1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------


def test_evidence_construction_and_identity() -> None:
    a = Evidence(
        EvidenceId(1),
        ClaimId(1),
        Assumed("user"),
        (),
        _SCOPE,
        Validity(None, None),
        EMPTY_ARTIFACT,
    )
    b = Evidence(
        EvidenceId(1),
        ClaimId(1),
        Assumed("user"),
        (),
        _SCOPE,
        Validity(None, None),
        EMPTY_ARTIFACT,
    )
    assert a == b and hash(a) == hash(b)


def test_evidence_rejects_wrong_id_kinds() -> None:
    with pytest.raises(TypeError, match="id must be EvidenceId"):
        Evidence(
            ClaimId(1),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            ClaimId(1),
            Assumed("u"),
            (),
            _SCOPE,
            Validity(None, None),
            EMPTY_ARTIFACT,
        )
    with pytest.raises(TypeError, match="claim must be ClaimId"):
        Evidence(
            EvidenceId(1),
            EvidenceId(1),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            Assumed("u"),
            (),
            _SCOPE,
            Validity(None, None),
            EMPTY_ARTIFACT,
        )


def test_evidence_rejects_non_status_and_bad_provenance() -> None:
    with pytest.raises(TypeError, match="must be an EvidenceStatus"):
        Evidence(
            EvidenceId(1),
            ClaimId(1),
            "assumed",  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            (),
            _SCOPE,
            Validity(None, None),
            EMPTY_ARTIFACT,
        )
    with pytest.raises(TypeError, match=r"provenance\[0\] must be EvidenceId"):
        Evidence(
            EvidenceId(1),
            ClaimId(1),
            Assumed("u"),
            (ClaimId(2),),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            _SCOPE,
            Validity(None, None),
            EMPTY_ARTIFACT,
        )


def test_evidence_rejects_self_reference_in_provenance() -> None:
    with pytest.raises(ValueError, match="own id"):
        Evidence(
            EvidenceId(7),
            ClaimId(1),
            Assumed("u"),
            (EvidenceId(7),),
            _SCOPE,
            Validity(None, None),
            EMPTY_ARTIFACT,
        )


# --------------------------------------------------------------------------------------
# The two Task 003 producers
# --------------------------------------------------------------------------------------


def test_specified_producer_emits_bundled_specpack_evidence() -> None:
    ev = specified_evidence(
        EvidenceId(1), ClaimId(1), pack="fortran-language-semantics", scope=_SCOPE
    )
    assert ev.status == Specified("fortran-language-semantics", Bundled())
    assert ev.provenance == ()
    assert ev.artifact == EMPTY_ARTIFACT


def test_statically_derived_producer_requires_premises() -> None:
    ev = statically_derived_evidence(
        EvidenceId(2),
        ClaimId(2),
        rule="spd_implies_symmetric",
        provenance=(EvidenceId(1),),
        scope=_SCOPE,
    )
    assert ev.status == StaticallyDerived("spd_implies_symmetric")
    assert ev.provenance == (EvidenceId(1),)
    with pytest.raises(ValueError, match="non-empty provenance"):
        statically_derived_evidence(
            EvidenceId(3), ClaimId(2), rule="spd_implies_symmetric", provenance=(), scope=_SCOPE
        )
