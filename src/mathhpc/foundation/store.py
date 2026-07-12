"""Minimal ``FoundationStore`` (Task 004): the frozen ``append/state/snapshot``
concurrency contract (v3.1 §7) over an in-memory grow-only log.

The frozen interfaces: ``append(evidence) -> Epoch``, ``state(key, scope, at) ->
ClaimState`` (a deterministic, order-independent fold over the live evidence
*set*), ``snapshot() -> Epoch``. The MVP implementation is exactly the sanctioned
one: a grow-only list plus one counter — a grow-only-set CRDT in one process.

Fold v0 (three states; v3.1 §5.1 rules 2/3/6, restricted per Addendum A1 §6.3):

1. Restrict to evidence for the queried ``ClaimKey`` whose scope **equals** the
   queried scope (scope *intersection* is Task 014's job; with only
   ``Universal × Value`` scopes today, equality is intersection).
2. Refutation-class present and no proof-class positive → ``Refuted{by}``.
3. Proof-class positive present and no refutation → ``Established{strongest}``.
4. Both present → ``ContestedConflictError`` — the ``Contested`` state arrives at
   Task 014; picking a side is never an option (v3.1 §5.2), so v0 alarms.
5. Otherwise → ``Unknown``; ``AiInferred``/``Predicted`` support is annotated in
   ``hypothesized`` but contributes nothing (``Measured`` is corroboration and
   contributes nothing at all).

Deterministic set functions (permutation invariance is the Task 004 acceptance):
``strongest`` = highest ladder strength, ties broken by least ``EvidenceId``
value; ``Refuted.by`` and ``Unknown.hypothesized`` = least ``EvidenceId`` value
of their class; the conflict pair = (least proof id, least refutation id).

The strength ladder over proof-class statuses is **implementation-local** (the
frozen docs mandate "strongest" but define no total order — v2 §7.2 refuses one
that embeds hypothesis-class): FormallyProved > SmtProved > StaticallyDerived >
RuntimeChecked > Specified > Assumed — machine-checked proof above solver proof
above audited rule application above in-window runtime check above trusted
specification above unverified user trust.

v0 deferrals (coverage, never discipline):
- Liveness (validity expiry, kill events, demotion supersession) — Tasks 013/024;
  every appended record is live today.
- ``append`` takes ``Evidence`` only; the frozen signature's ``FeedbackEvent |
  SuppressionEvent`` alternatives widen the parameter at Task 017.
- Refutation-class records enter through the implementation-local
  ``append_refutation`` seam. Its future callers are the ``Counterexample``
  demotion path (Task 017) and guards (Task 026), which also fix the frozen
  "negation-with-witness" statement encoding; hypothesis-class statuses are
  refused ("a hypothesis cannot contest anything").
- ``register_claim`` is implementation-local: it binds ``ClaimId -> Claim`` so
  the fold can associate evidence (which carries a ``ClaimId``) with the queried
  ``ClaimKey``.
"""

from typing import Protocol

from mathhpc.foundation.claim_state import (
    HYPOTHESIS_CLASS,
    PROOF_CLASS,
    ClaimState,
    Established,
    Refuted,
    Unknown,
)
from mathhpc.foundation.claims import Claim, ClaimKey
from mathhpc.foundation.evidence import Evidence
from mathhpc.foundation.ids import ClaimId, EvidenceId
from mathhpc.foundation.scope import Scope

__all__ = [
    "ContestedConflictError",
    "Epoch",
    "FoundationStore",
    "InMemoryFoundationStore",
]

type Epoch = int

_STRENGTH: dict[type[object], int] = {
    cls: rank for rank, cls in enumerate(reversed(PROOF_CLASS), 1)
}


class FoundationStore(Protocol):
    """The frozen store contract (v3.1 §7). ``state`` is a commutative fold."""

    def append(self, ev: Evidence) -> Epoch: ...
    def state(self, key: ClaimKey, scope: Scope, at: Epoch) -> ClaimState: ...
    def snapshot(self) -> Epoch: ...


class ContestedConflictError(RuntimeError):
    """Live proof-class positive AND live refutation on the same key/scope.

    v0 stand-in for the ``Contested`` state (Task 014). Carries the deterministic
    contradiction pair: (least proof-class EvidenceId, least refutation EvidenceId).
    """

    def __init__(self, pair: tuple[EvidenceId, EvidenceId]) -> None:
        self.pair = pair
        super().__init__(
            f"contested claim: proof-class {pair[0]!r} vs refutation {pair[1]!r}; "
            "the Contested state arrives with Task 014 — a side is never picked"
        )


class _LogEntry:
    __slots__ = ("epoch", "evidence", "is_refutation")

    def __init__(self, epoch: int, evidence: Evidence, is_refutation: bool) -> None:
        self.epoch = epoch
        self.evidence = evidence
        self.is_refutation = is_refutation


class InMemoryFoundationStore:
    """Grow-only in-memory event log + one epoch counter (the sanctioned MVP impl)."""

    def __init__(self) -> None:
        self._log: list[_LogEntry] = []
        self._claims: dict[ClaimId, Claim] = {}
        self._seen_evidence: set[EvidenceId] = set()
        self._epoch: int = 0

    # -- implementation-local surface ---------------------------------------------------

    def register_claim(self, claim: Claim) -> None:
        """Bind ``claim.id`` to its key/scope. Idempotent for an identical record;
        rebinding an id to a different record is an error."""
        if type(claim) is not Claim:
            raise TypeError(f"register_claim expects Claim, got {type(claim).__qualname__}")
        existing = self._claims.get(claim.id)
        if existing is not None and existing != claim:
            raise ValueError(f"{claim.id!r} is already bound to a different claim")
        self._claims[claim.id] = claim

    def append_refutation(self, ev: Evidence) -> Epoch:
        """Append refutation-class evidence (falsifies its claim at its scope).

        Hypothesis-class statuses are refused: a hypothesis cannot contest
        anything (v3.1 §5.1).
        """
        if type(ev.status) in HYPOTHESIS_CLASS:
            raise ValueError(
                f"refutation cannot carry hypothesis-class status "
                f"{type(ev.status).__qualname__} (a hypothesis cannot contest anything)"
            )
        return self._append(ev, is_refutation=True)

    # -- the frozen protocol -------------------------------------------------------------

    def append(self, ev: Evidence) -> Epoch:
        return self._append(ev, is_refutation=False)

    def snapshot(self) -> Epoch:
        return self._epoch

    def state(self, key: ClaimKey, scope: Scope, at: Epoch) -> ClaimState:
        if type(key) is not ClaimKey:
            raise TypeError(f"state expects ClaimKey, got {type(key).__qualname__}")
        if type(scope) is not Scope:
            raise TypeError(f"state expects Scope, got {type(scope).__qualname__}")
        if type(at) is not int:
            raise TypeError(f"state expects an int Epoch, got {type(at).__qualname__}")
        if not 0 <= at <= self._epoch:
            raise ValueError(f"epoch {at} does not exist (store is at epoch {self._epoch})")

        proofs: list[Evidence] = []
        refutations: list[Evidence] = []
        hypotheses: list[Evidence] = []
        for entry in self._log:
            if entry.epoch > at:
                continue
            ev = entry.evidence
            claim = self._claims[ev.claim]
            if claim.key != key or ev.scope != scope:
                continue
            if entry.is_refutation:
                refutations.append(ev)
            elif type(ev.status) in PROOF_CLASS:
                proofs.append(ev)
            elif type(ev.status) in HYPOTHESIS_CLASS:
                hypotheses.append(ev)
            # Measured: corroboration only — contributes nothing to the fold.

        if refutations and proofs:
            raise ContestedConflictError(
                (_least_id(proofs), _least_id(refutations)),
            )
        if refutations:
            return Refuted(by=_least_id(refutations))
        if proofs:
            return Established(strongest=_strongest(proofs).status)
        if hypotheses:
            return Unknown(hypothesized=_least_id(hypotheses))
        return Unknown(hypothesized=None)

    # -- internals -----------------------------------------------------------------------

    def _append(self, ev: Evidence, *, is_refutation: bool) -> Epoch:
        if type(ev) is not Evidence:
            raise TypeError(f"append expects Evidence, got {type(ev).__qualname__}")
        if ev.claim not in self._claims:
            raise ValueError(f"evidence {ev.id!r} references unregistered claim {ev.claim!r}")
        if ev.id in self._seen_evidence:
            raise ValueError(f"evidence id {ev.id!r} was already appended (grow-only set)")
        self._epoch += 1
        self._log.append(_LogEntry(self._epoch, ev, is_refutation))
        self._seen_evidence.add(ev.id)
        return self._epoch


def _least_id(records: list[Evidence]) -> EvidenceId:
    return min(records, key=lambda ev: ev.id.value).id


def _strongest(proofs: list[Evidence]) -> Evidence:
    return min(proofs, key=lambda ev: (-_STRENGTH[type(ev.status)], ev.id.value))
