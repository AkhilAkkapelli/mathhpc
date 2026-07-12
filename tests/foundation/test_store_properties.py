"""Task 004 acceptance: ``state()`` is a deterministic, order-independent fold —
appending any permutation of the same evidence set yields the identical
``ClaimState`` (or the identical ``ContestedConflictError`` pair) at every key
and scope (v3.1 §7: "commutative and associative by construction, since the fold
consumes the *set* of live evidence, not an arrival order")."""

import hypothesis.strategies as st
from hypothesis import given

from mathhpc.foundation import (
    EMPTY_ARTIFACT,
    AiInferred,
    Claim,
    ClaimId,
    ClaimKey,
    ClaimState,
    ContestedConflictError,
    Equivalence,
    Evidence,
    EvidenceId,
    InMemoryFoundationStore,
    MathematicalReal,
    Measured,
    Property,
    PropertyName,
    Scope,
    Universal,
    Validity,
    Value,
)
from tests.foundation.strategies import proof_statuses

_KEY_A = ClaimKey(Property("A", PropertyName.SYMMETRIC), MathematicalReal())
_KEY_B = ClaimKey(Equivalence("L3", "matmul"), MathematicalReal())
_SCOPE_A = Scope(Universal(), Value("A"))
_SCOPE_B = Scope(Universal(), Value("B"))
_CLAIMS = (Claim(ClaimId(1), _KEY_A, _SCOPE_A), Claim(ClaimId(2), _KEY_B, _SCOPE_B))

# (claim id, scope) targets; evidence is spread across both so the fold's
# key/scope filtering is exercised under permutation too.
_TARGETS = ((ClaimId(1), _SCOPE_A), (ClaimId(2), _SCOPE_B))

_Entry = tuple[Evidence, bool]  # (record, is_refutation)


def _statuses() -> st.SearchStrategy[tuple[object, str]]:
    hypothesis_class = st.builds(
        AiInferred, model=st.just("m"), score=st.floats(min_value=0.0, max_value=1.0)
    ) | st.builds(
        Measured,
        machine=st.just("topo"),
        n=st.integers(min_value=1, max_value=100),
        ci=st.just((0.5, 1.5)),
    )
    return st.one_of(
        st.tuples(proof_statuses(), st.sampled_from(["positive", "refutation"])),
        st.tuples(hypothesis_class, st.just("positive")),
    )


@st.composite
def _entries(draw: st.DrawFn) -> list[_Entry]:
    specs = draw(st.lists(_statuses(), max_size=8))
    ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=2**32),
            min_size=len(specs),
            max_size=len(specs),
            unique=True,
        )
    )
    entries: list[_Entry] = []
    for (status, polarity), raw_id in zip(specs, ids, strict=True):
        claim_id, scope = draw(st.sampled_from(_TARGETS))
        record = Evidence(
            id=EvidenceId(raw_id),
            claim=claim_id,
            status=status,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            provenance=(),
            scope=scope,
            validity=Validity(None, None),
            artifact=EMPTY_ARTIFACT,
        )
        entries.append((record, polarity == "refutation"))
    return entries


def _build(entries: list[_Entry]) -> InMemoryFoundationStore:
    store = InMemoryFoundationStore()
    for claim in _CLAIMS:
        store.register_claim(claim)
    for record, is_refutation in entries:
        if is_refutation:
            store.append_refutation(record)
        else:
            store.append(record)
    return store


def _outcome(
    store: InMemoryFoundationStore, key: ClaimKey, scope: Scope
) -> ClaimState | tuple[EvidenceId, EvidenceId]:
    try:
        return store.state(key, scope, store.snapshot())
    except ContestedConflictError as exc:
        return exc.pair


@given(data=st.data())
def test_state_is_permutation_invariant(data: st.DataObject) -> None:
    entries = data.draw(_entries())
    permuted = data.draw(st.permutations(entries))
    one = _build(entries)
    two = _build(permuted)
    for key, scope in ((_KEY_A, _SCOPE_A), (_KEY_B, _SCOPE_B), (_KEY_A, _SCOPE_B)):
        assert _outcome(one, key, scope) == _outcome(two, key, scope)


@given(data=st.data())
def test_prefix_reads_are_stable_under_later_appends(data: st.DataObject) -> None:
    """A state read at an earlier epoch is unchanged by anything appended later
    (append-only: history is immutable)."""
    entries = data.draw(_entries())
    cut = data.draw(st.integers(min_value=0, max_value=len(entries)))
    prefix_store = _build(entries[:cut])
    full_store = _build(entries)
    epoch = prefix_store.snapshot()
    for key, scope in ((_KEY_A, _SCOPE_A), (_KEY_B, _SCOPE_B)):
        try:
            expected: object = prefix_store.state(key, scope, epoch)
        except ContestedConflictError as exc:
            expected = exc.pair
        try:
            actual: object = full_store.state(key, scope, epoch)
        except ContestedConflictError as exc:
            actual = exc.pair
        assert actual == expected
