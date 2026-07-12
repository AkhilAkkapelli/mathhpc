"""Unit tests for the minimal ``FoundationStore`` (Task 004): epochs, snapshot
reads, registration discipline, and the three-state fold rules of v3.1 §5.1."""

import pytest

from mathhpc.foundation import (
    EMPTY_ARTIFACT,
    AiInferred,
    ArtifactRef,
    Assumed,
    Claim,
    ClaimId,
    ClaimKey,
    ContestedConflictError,
    Established,
    Evidence,
    EvidenceId,
    FormallyProved,
    FoundationStore,
    InMemoryFoundationStore,
    MathematicalReal,
    Measured,
    Property,
    PropertyName,
    Refuted,
    Scope,
    StaticallyDerived,
    Universal,
    Unknown,
    Validity,
    Value,
    specified_evidence,
)

_KEY = ClaimKey(Property("A", PropertyName.SYMMETRIC), MathematicalReal())
_SCOPE = Scope(Universal(), Value("A"))
_OTHER_SCOPE = Scope(Universal(), Value("B"))
_CLAIM = Claim(ClaimId(1), _KEY, _SCOPE)


def _store() -> InMemoryFoundationStore:
    store = InMemoryFoundationStore()
    store.register_claim(_CLAIM)
    return store


def _ev(n: int, status: object = None) -> Evidence:
    return Evidence(
        id=EvidenceId(n),
        claim=ClaimId(1),
        status=status if status is not None else Assumed("user"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        provenance=(),
        scope=_SCOPE,
        validity=Validity(None, None),
        artifact=EMPTY_ARTIFACT,
    )


# --------------------------------------------------------------------------------------
# Protocol, epochs, snapshot
# --------------------------------------------------------------------------------------


def test_in_memory_store_satisfies_the_frozen_protocol() -> None:
    store: FoundationStore = InMemoryFoundationStore()
    assert store.snapshot() == 0


def test_append_returns_monotone_epochs_and_snapshot_tracks() -> None:
    store = _store()
    assert store.snapshot() == 0
    assert store.append(_ev(1)) == 1
    assert store.append(_ev(2)) == 2
    assert store.snapshot() == 2


def test_state_is_a_snapshot_read_at_an_epoch() -> None:
    store = _store()
    store.append(_ev(1, StaticallyDerived("r")))
    assert store.state(_KEY, _SCOPE, 0) == Unknown(None)
    assert store.state(_KEY, _SCOPE, 1) == Established(StaticallyDerived("r"))


def test_state_rejects_nonexistent_epochs() -> None:
    store = _store()
    with pytest.raises(ValueError, match="does not exist"):
        store.state(_KEY, _SCOPE, 1)
    with pytest.raises(ValueError, match="does not exist"):
        store.state(_KEY, _SCOPE, -1)


# --------------------------------------------------------------------------------------
# Registration and grow-only discipline
# --------------------------------------------------------------------------------------


def test_append_requires_registered_claim() -> None:
    store = InMemoryFoundationStore()
    with pytest.raises(ValueError, match="unregistered claim"):
        store.append(_ev(1))


def test_register_claim_is_idempotent_but_rejects_rebinding() -> None:
    store = _store()
    store.register_claim(_CLAIM)  # identical: fine
    with pytest.raises(ValueError, match="already bound"):
        store.register_claim(Claim(ClaimId(1), _KEY, _OTHER_SCOPE))


def test_duplicate_evidence_id_rejected() -> None:
    store = _store()
    store.append(_ev(1))
    with pytest.raises(ValueError, match="already appended"):
        store.append(_ev(1, StaticallyDerived("r")))


# --------------------------------------------------------------------------------------
# Fold rules (v3.1 §5.1, three-state v0)
# --------------------------------------------------------------------------------------


def test_empty_and_unmatched_folds_are_unknown() -> None:
    store = _store()
    assert store.state(_KEY, _SCOPE, 0) == Unknown(None)
    other_key = ClaimKey(Property("Z", PropertyName.SYMMETRIC), MathematicalReal())
    assert store.state(other_key, _SCOPE, 0) == Unknown(None)


def test_scope_filter_is_exact_in_v0() -> None:
    store = _store()
    store.append(_ev(1, StaticallyDerived("r")))
    assert store.state(_KEY, _OTHER_SCOPE, store.snapshot()) == Unknown(None)


def test_established_records_strongest_status() -> None:
    store = _store()
    store.append(_ev(1, Assumed("user")))
    store.append(_ev(2, FormallyProved("lean4", EMPTY_ARTIFACT)))
    store.append(_ev(3, StaticallyDerived("r")))
    state = store.state(_KEY, _SCOPE, store.snapshot())
    assert state == Established(FormallyProved("lean4", EMPTY_ARTIFACT))


def test_established_strength_ties_break_by_least_evidence_id() -> None:
    store = _store()
    store.append(_ev(7, StaticallyDerived("rule-b")))
    store.append(_ev(3, StaticallyDerived("rule-a")))
    state = store.state(_KEY, _SCOPE, store.snapshot())
    assert state == Established(StaticallyDerived("rule-a"))


def test_hypotheses_are_annotated_never_counted() -> None:
    store = _store()
    store.append(_ev(9, AiInferred("proposer-v0", 0.99)))
    store.append(_ev(4, AiInferred("proposer-v0", 0.5)))
    assert store.state(_KEY, _SCOPE, store.snapshot()) == Unknown(EvidenceId(4))
    # A proof arriving later establishes regardless of hypothesis presence.
    store.append(_ev(1, StaticallyDerived("r")))
    assert store.state(_KEY, _SCOPE, store.snapshot()) == Established(StaticallyDerived("r"))


def test_measured_is_corroboration_only() -> None:
    store = _store()
    store.append(_ev(1, Measured("topo:ab", 100, (0.9, 1.1))))
    assert store.state(_KEY, _SCOPE, store.snapshot()) == Unknown(None)


def test_refutation_yields_refuted() -> None:
    store = _store()
    store.append_refutation(_ev(5, StaticallyDerived("negation:witness")))
    store.append_refutation(_ev(2, StaticallyDerived("negation:witness")))
    assert store.state(_KEY, _SCOPE, store.snapshot()) == Refuted(EvidenceId(2))


def test_refutation_cannot_carry_hypothesis_status() -> None:
    store = _store()
    with pytest.raises(ValueError, match="hypothesis"):
        store.append_refutation(_ev(1, AiInferred("proposer-v0", 0.9)))


def test_proof_vs_refutation_alarms_instead_of_picking_a_side() -> None:
    store = _store()
    store.append(_ev(4, StaticallyDerived("r")))
    store.append_refutation(_ev(6, StaticallyDerived("negation:witness")))
    with pytest.raises(ContestedConflictError) as exc:
        store.state(_KEY, _SCOPE, store.snapshot())
    assert exc.value.pair == (EvidenceId(4), EvidenceId(6))


def test_refutation_out_of_scope_does_not_meet_proof() -> None:
    store = _store()
    store.append(_ev(1, StaticallyDerived("r")))
    out_of_scope = Evidence(
        id=EvidenceId(2),
        claim=ClaimId(1),
        status=StaticallyDerived("negation:witness"),
        provenance=(),
        scope=_OTHER_SCOPE,
        validity=Validity(None, None),
        artifact=ArtifactRef.of(b"cex"),
    )
    store.append_refutation(out_of_scope)
    assert store.state(_KEY, _SCOPE, store.snapshot()) == Established(StaticallyDerived("r"))
    assert store.state(_KEY, _OTHER_SCOPE, store.snapshot()) == Refuted(EvidenceId(2))


def test_producer_evidence_folds_to_established() -> None:
    store = _store()
    ev = specified_evidence(EvidenceId(1), ClaimId(1), pack="blas-reference", scope=_SCOPE)
    store.append(ev)
    state = store.state(_KEY, _SCOPE, store.snapshot())
    assert isinstance(state, Established)
    assert state.strongest == ev.status
