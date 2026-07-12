"""Minimal ``Decision`` and the full ``DecisionControl`` union (spec §7, Task 006).

Frozen definitions:

    class DecisionControl:
        Auto | Prefer(value, cls: PriorityClass, rank: int | None)
             | Require(value) | Use(value, params)
    @frozen class Decision:
        id: DecisionId; layer: str; subject: IrRef; control: DecisionControl
        chosen: AltRef; considered: tuple[AltRef, ...]
        excluded: tuple[ExclusionCertificate, ...]
        depends_on: tuple[tuple[ClaimId, ClaimState], ...]; epoch: Epoch

Task 006 ships the **full** ``DecisionControl`` union (all four variants defined,
serializable, registry-covered) with only ``Auto`` exercised by the Wave-1 slice,
and a **minimal** ``Decision`` whose acceptance is: *epoch + ``depends_on``
recorded and serialized* (Addendum A1 §5). It carries every field whose type is
representable today; two deferrals, each following an established precedent:

- ``subject`` / ``chosen`` / ``considered`` reference IR and plan alternatives by
  surface name (``str``) — the frozen ``IrRef`` / ``AltRef`` reference types
  arrive with the HIR (008) and planner (010) tasks, exactly as ``Value.ref``
  and the ``Equivalence`` operands reference their subjects by ``str`` today.
- ``excluded: tuple[ExclusionCertificate, ...]`` is **deferred**:
  ``ExclusionCertificate`` (with its ``Feedback`` reference) is a genuine frozen
  type introduced at Task 017 and cannot be stubbed. The field is added there by
  the schema-migration scaffold (v1 → v2, default empty tuple), mirroring the
  ``Validity.killed_by`` deferral — coverage deferred, discipline (epoch +
  ``depends_on`` provenance) preserved.

``depends_on`` records the ``(ClaimId, ClaimState)`` pairs the decision rested on
at ``epoch``; ``ClaimState`` is the Task 004 fold-v0 union, so a decision's
dependency snapshot is a real, serialized record from day one.
"""

from dataclasses import dataclass
from enum import Enum, unique
from typing import Final

from mathhpc.foundation.claim_state import ClaimState, Established, Refuted, Unknown
from mathhpc.foundation.evidence import StaticallyDerived
from mathhpc.foundation.ids import ClaimId, DecisionId, EvidenceId
from mathhpc.foundation.immutable import FrozenDict, validate_frozen_instance
from mathhpc.foundation.registry import register_frozen_type
from mathhpc.foundation.store import Epoch

__all__ = [
    "Auto",
    "Decision",
    "DecisionControl",
    "PriorityClass",
    "Prefer",
    "Require",
    "Use",
]

_CLAIM_STATE_CLASSES: Final = (Established, Refuted, Unknown)


@unique
class PriorityClass(Enum):
    """Program-declared preference class, totally ordered strong > normal > hint
    (v3 §1.2). The ordering is exercised by the planner's preference resolution
    (Task 023); here it is only the tag a ``Prefer`` control carries."""

    STRONG = "strong"
    NORMAL = "normal"
    HINT = "hint"


@dataclass(frozen=True, slots=True)
class Auto:
    """No user control: the planner chooses freely. The only variant the Wave-1
    slice exercises."""

    def __post_init__(self) -> None:
        validate_frozen_instance(self)


@dataclass(frozen=True, slots=True)
class Prefer:
    """A soft preference for ``value`` in priority class ``cls``; ``rank`` is the
    explicit ``prefer@N`` position (``None`` when unranked)."""

    value: str
    cls: PriorityClass
    rank: int | None

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.value) is not str:
            raise TypeError(f"Prefer.value must be str, got {type(self.value).__qualname__}")
        if not self.value:
            raise ValueError("Prefer.value must be non-empty")
        if type(self.cls) is not PriorityClass:
            raise TypeError(f"Prefer.cls must be PriorityClass, got {type(self.cls).__qualname__}")
        if self.rank is not None:
            if type(self.rank) is not int:
                raise TypeError(
                    f"Prefer.rank must be int or None, got {type(self.rank).__qualname__}"
                )
            if self.rank < 1:
                raise ValueError(f"Prefer.rank must be >= 1 (1-based), got {self.rank}")


@dataclass(frozen=True, slots=True)
class Require:
    """A hard requirement that the chosen plan use ``value``."""

    value: str

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.value) is not str:
            raise TypeError(f"Require.value must be str, got {type(self.value).__qualname__}")
        if not self.value:
            raise ValueError("Require.value must be non-empty")


@dataclass(frozen=True, slots=True)
class Use:
    """A hard pin of ``value`` with fixed ``params`` (verifies, does not search)."""

    value: str
    params: FrozenDict[str, str]

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.value) is not str:
            raise TypeError(f"Use.value must be str, got {type(self.value).__qualname__}")
        if not self.value:
            raise ValueError("Use.value must be non-empty")
        if type(self.params) is not FrozenDict:
            raise TypeError(
                f"Use.params must be FrozenDict[str, str], got {type(self.params).__qualname__}"
            )


type DecisionControl = Auto | Prefer | Require | Use

_CONTROL_CLASSES: Final = (Auto, Prefer, Require, Use)


@dataclass(frozen=True, slots=True)
class Decision:
    """A recorded planning choice, provenance-linked to the claim states it rested
    on at a snapshot epoch."""

    id: DecisionId
    layer: str
    subject: str
    control: DecisionControl
    chosen: str
    considered: tuple[str, ...]
    depends_on: tuple[tuple[ClaimId, ClaimState], ...]
    epoch: Epoch

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.id) is not DecisionId:
            raise TypeError(f"Decision.id must be DecisionId, got {type(self.id).__qualname__}")
        for name in ("layer", "subject", "chosen"):
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"Decision.{name} must be str, got {type(value).__qualname__}")
            if not value:
                raise ValueError(f"Decision.{name} must be non-empty")
        if type(self.control) not in _CONTROL_CLASSES:
            raise TypeError(
                f"Decision.control must be a DecisionControl, got {type(self.control).__qualname__}"
            )
        if type(self.considered) is not tuple:
            raise TypeError("Decision.considered must be a tuple of str")
        for i, alt in enumerate(self.considered):
            if type(alt) is not str:
                raise TypeError(
                    f"Decision.considered[{i}] must be str, got {type(alt).__qualname__}"
                )
        if type(self.depends_on) is not tuple:
            raise TypeError("Decision.depends_on must be a tuple of (ClaimId, ClaimState) pairs")
        for i, pair in enumerate(self.depends_on):
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError(f"Decision.depends_on[{i}] must be a (ClaimId, ClaimState) pair")
            claim, state = pair
            if type(claim) is not ClaimId:
                raise TypeError(
                    f"Decision.depends_on[{i}][0] must be ClaimId, got {type(claim).__qualname__}"
                )
            if type(state) not in _CLAIM_STATE_CLASSES:
                raise TypeError(
                    f"Decision.depends_on[{i}][1] must be a ClaimState, "
                    f"got {type(state).__qualname__}"
                )
        if type(self.epoch) is not int:
            raise TypeError(
                f"Decision.epoch must be an int Epoch, got {type(self.epoch).__qualname__}"
            )
        if self.epoch < 0:
            raise ValueError(f"Decision.epoch must be >= 0, got {self.epoch}")


register_frozen_type(Auto, fixture=lambda: (Auto(), Auto()), schema_version=1)
register_frozen_type(
    Prefer,
    fixture=lambda: (
        Prefer("dgemm", PriorityClass.NORMAL, None),
        Prefer("dgemm", PriorityClass.NORMAL, None),
        Prefer("cholesky", PriorityClass.STRONG, 2),
    ),
    schema_version=1,
)
register_frozen_type(
    Require,
    fixture=lambda: (Require("cholesky"), Require("cholesky"), Require("dgemm")),
    schema_version=1,
)
register_frozen_type(
    Use,
    fixture=lambda: (
        Use("blocked_gemm", FrozenDict({"block_size": "100"})),
        Use("blocked_gemm", FrozenDict({"block_size": "100"})),
        Use("anchor", FrozenDict[str, str]({})),
    ),
    schema_version=1,
)
register_frozen_type(
    Decision,
    fixture=lambda: (
        Decision(
            DecisionId(1),
            layer="plan",
            subject="C",
            control=Auto(),
            chosen="dgemm",
            considered=("dgemm", "anchor_loop"),
            depends_on=((ClaimId(1), Established(StaticallyDerived("spd_implies_symmetric"))),),
            epoch=7,
        ),
        Decision(
            DecisionId(1),
            layer="plan",
            subject="C",
            control=Auto(),
            chosen="dgemm",
            considered=("dgemm", "anchor_loop"),
            depends_on=((ClaimId(1), Established(StaticallyDerived("spd_implies_symmetric"))),),
            epoch=7,
        ),
        Decision(
            DecisionId(2),
            layer="hir",
            subject="x",
            control=Require("cholesky"),
            chosen="cholesky",
            considered=("cholesky",),
            depends_on=((ClaimId(3), Unknown(None)), (ClaimId(4), Refuted(EvidenceId(9)))),
            epoch=0,
        ),
    ),
    schema_version=1,
)
