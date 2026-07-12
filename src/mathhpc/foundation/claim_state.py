"""``ClaimState`` — fold v0 codomain (Task 004): {Established, Refuted, Unknown}.

The frozen five-state definition (v3.1 §5.1) is:

    Established { strongest: EvidenceStatus }   // live admissible positive, no refutation
    Refuted     { by: EvidenceId }              // live refutation, no positive proof-class
    Conditional { on: Vec<ClaimId> }            // Task 014
    Contested   { pair }                        // Task 014
    Unknown     { hypothesized: Option<EvidenceId> }

Task 004 implements the fold-v0 subset (Addendum A1 §5 / §6.3: "Task 004 =
three-state fold + permutation invariance"); ``Conditional`` and ``Contested``
arrive with Task 014 by union widening. Until then the store refuses to pick a
side on a proof-vs-refutation conflict by raising (v3.1 §5.2: Contested is an
alarm, never resolved silently).

``PROOF_CLASS`` is v3.1 §5.1 rule 3 verbatim: the statuses whose presence can
establish a claim. ``HYPOTHESIS_CLASS`` (``AiInferred``/``Predicted``) is
annotated by the fold but contributes nothing to establishment or refutation —
"a hypothesis cannot contest anything". ``Measured`` is corroboration: neither
class. ``Established`` enforces the rule structurally: it cannot be constructed
around a non-proof-class status.
"""

from dataclasses import dataclass
from typing import Final

from mathhpc.foundation.evidence import (
    AiInferred,
    Assumed,
    EvidenceStatus,
    FormallyProved,
    Predicted,
    RuntimeChecked,
    SmtProved,
    Specified,
    StaticallyDerived,
)
from mathhpc.foundation.ids import EvidenceId
from mathhpc.foundation.immutable import validate_frozen_instance
from mathhpc.foundation.registry import register_frozen_type

__all__ = [
    "HYPOTHESIS_CLASS",
    "PROOF_CLASS",
    "ClaimState",
    "Established",
    "Refuted",
    "Unknown",
]

PROOF_CLASS: Final = (
    FormallyProved,
    SmtProved,
    StaticallyDerived,
    RuntimeChecked,
    Specified,
    Assumed,
)

HYPOTHESIS_CLASS: Final = (AiInferred, Predicted)


@dataclass(frozen=True, slots=True)
class Established:
    """Live proof-class positive evidence and no live refutation; ``strongest``
    records the highest-strength status held (per-gate admissibility still
    applies on top of Established)."""

    strongest: EvidenceStatus

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.strongest) not in PROOF_CLASS:
            raise TypeError(
                "Established.strongest must be a proof-class EvidenceStatus "
                f"(v3.1 §5.1 rule 3), got {type(self.strongest).__qualname__}"
            )


@dataclass(frozen=True, slots=True)
class Refuted:
    """Live refutation-class evidence and no live positive proof-class."""

    by: EvidenceId

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.by) is not EvidenceId:
            raise TypeError(f"Refuted.by must be EvidenceId, got {type(self.by).__qualname__}")


@dataclass(frozen=True, slots=True)
class Unknown:
    """No proof-class positive and no refutation; hypothesis-class support (if
    any) is noted in ``hypothesized`` but was never counted."""

    hypothesized: EvidenceId | None

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if self.hypothesized is not None and type(self.hypothesized) is not EvidenceId:
            raise TypeError(
                f"Unknown.hypothesized must be EvidenceId or None, "
                f"got {type(self.hypothesized).__qualname__}"
            )


type ClaimState = Established | Refuted | Unknown


register_frozen_type(
    Established,
    fixture=lambda: (
        Established(StaticallyDerived("spd_implies_symmetric")),
        Established(StaticallyDerived("spd_implies_symmetric")),
        Established(Assumed("user")),
    ),
    schema_version=1,
)
register_frozen_type(
    Refuted,
    fixture=lambda: (Refuted(EvidenceId(1)), Refuted(EvidenceId(1)), Refuted(EvidenceId(9))),
    schema_version=1,
)
register_frozen_type(
    Unknown,
    fixture=lambda: (Unknown(None), Unknown(None), Unknown(EvidenceId(3))),
    schema_version=1,
)
