"""``EvidenceStatus`` (the full frozen sum type) and ``Evidence`` (spec §7, Task 003).

The union is defined in full — all nine variants, so the vocabulary of
justification kinds is fixed from here on — but Task 003 ships **producers for
``StaticallyDerived`` and ``Specified`` only** (the two kinds the Wave-1 slice
emits). The other seven variants are constructible (and round-trip, and are
registry-covered) but nothing in the system produces them until their tasks:
guards produce ``RuntimeChecked`` (016/026), the machine probe ``Measured``
(019), the AI/SMT stubs ``AiInferred``/``SmtProved`` (025), and so on.

Structural discipline enforced here, not by convention:

- *No scores on proof classes*: only ``AiInferred``/``Measured``/``Predicted``
  carry numeric confidence-like payloads; the proof-grade variants structurally
  cannot hold a score. (That ``AiInferred``/``Predicted`` never gate legality is
  the fold's rule — Task 004.)
- ``Specified`` provenance is ``SpecProvenance`` — ``Bundled`` today; the
  ``LocalFile`` (sha256-pinned) and reserved ``Signed`` variants arrive with the
  SpecPack loader tasks by union widening.
- ``Evidence`` is an immutable event: identified, claim-linked, provenance-
  chained (prior evidence only — never itself), scoped, validity-pinned, and
  carrying a content-addressed artifact (``EMPTY_ARTIFACT`` when the status and
  provenance carry the whole justification).
"""

import math
from dataclasses import dataclass

from mathhpc.foundation.artifact import EMPTY_ARTIFACT, ArtifactRef
from mathhpc.foundation.ids import ClaimId, EventId, EvidenceId, GuardId
from mathhpc.foundation.immutable import validate_frozen_instance
from mathhpc.foundation.registry import register_frozen_type
from mathhpc.foundation.scope import ANY_VALIDITY, Scope, Universal, Validity, Value

__all__ = [
    "AiInferred",
    "Assumed",
    "Bundled",
    "Evidence",
    "EvidenceStatus",
    "FormallyProved",
    "Measured",
    "Predicted",
    "RuntimeChecked",
    "SmtProved",
    "SpecProvenance",
    "Specified",
    "StaticallyDerived",
    "specified_evidence",
    "statically_derived_evidence",
]


def _require_nonempty_str(owner: str, field: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{owner}.{field} must be str, got {type(value).__qualname__}")
    if not value:
        raise ValueError(f"{owner}.{field} must be non-empty")


# --------------------------------------------------------------------------------------
# SpecPack provenance (trust levels; spec §10)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bundled:
    """The SpecPack ships with the compiler distribution."""

    def __post_init__(self) -> None:
        validate_frozen_instance(self)


type SpecProvenance = Bundled


# --------------------------------------------------------------------------------------
# The nine EvidenceStatus variants
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Assumed:
    """Declared true by ``who`` (e.g. a user ``assume`` directive) — unverified trust."""

    who: str

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        _require_nonempty_str("Assumed", "who", self.who)


@dataclass(frozen=True, slots=True)
class Specified:
    """Stated by a SpecPack entry (a standard, a library contract)."""

    pack: str
    provenance: SpecProvenance

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        _require_nonempty_str("Specified", "pack", self.pack)
        if type(self.provenance) is not Bundled:
            raise TypeError(
                f"Specified.provenance must be a SpecProvenance, "
                f"got {type(self.provenance).__qualname__}"
            )


@dataclass(frozen=True, slots=True)
class StaticallyDerived:
    """Produced by a named static rule (property algebra, bridge instantiation)."""

    rule: str

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        _require_nonempty_str("StaticallyDerived", "rule", self.rule)


@dataclass(frozen=True, slots=True)
class SmtProved:
    """Discharged by an SMT solver; the proof object lives in ``artifact``."""

    solver: str
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        _require_nonempty_str("SmtProved", "solver", self.solver)
        if type(self.artifact) is not ArtifactRef:
            raise TypeError(
                f"SmtProved.artifact must be ArtifactRef, got {type(self.artifact).__qualname__}"
            )


@dataclass(frozen=True, slots=True)
class FormallyProved:
    """Discharged by a proof assistant; unused in the MVP, frozen anyway."""

    system: str
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        _require_nonempty_str("FormallyProved", "system", self.system)
        if type(self.artifact) is not ArtifactRef:
            raise TypeError(
                f"FormallyProved.artifact must be ArtifactRef, "
                f"got {type(self.artifact).__qualname__}"
            )


@dataclass(frozen=True, slots=True)
class RuntimeChecked:
    """Established by a guard at execution; ``when`` is the identity of the runtime
    event at which the check ran — the same event space ``Window`` scope bounds
    index (spec §7)."""

    guard: GuardId
    when: EventId

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.guard) is not GuardId:
            raise TypeError(
                f"RuntimeChecked.guard must be GuardId, got {type(self.guard).__qualname__}"
            )
        if type(self.when) is not EventId:
            raise TypeError(
                f"RuntimeChecked.when must be EventId, got {type(self.when).__qualname__}"
            )


@dataclass(frozen=True, slots=True)
class Measured:
    """An empirical measurement: ``n`` samples on ``machine`` with confidence
    interval ``ci = (lo, hi)``."""

    machine: str
    n: int
    ci: tuple[float, float]

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        _require_nonempty_str("Measured", "machine", self.machine)
        if type(self.n) is not int:
            raise TypeError(f"Measured.n must be int, got {type(self.n).__qualname__}")
        if self.n < 1:
            raise ValueError(f"Measured.n must be >= 1, got {self.n}")
        if (
            type(self.ci) is not tuple
            or len(self.ci) != 2
            or any(type(bound) is not float for bound in self.ci)
        ):
            raise TypeError("Measured.ci must be a (float, float) tuple")
        lo, hi = self.ci
        if math.isnan(lo) or math.isnan(hi) or lo > hi:
            raise ValueError(f"Measured.ci must satisfy lo <= hi without NaN, got {self.ci}")


@dataclass(frozen=True, slots=True)
class Predicted:
    """A model prediction; ``calib`` names the calibration record it rests on."""

    model: str
    calib: str

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        _require_nonempty_str("Predicted", "model", self.model)
        _require_nonempty_str("Predicted", "calib", self.calib)


@dataclass(frozen=True, slots=True)
class AiInferred:
    """An AI proposal with a confidence score in [0, 1]. Confidence is never
    proof: this status never gates legality (fold rule, Task 004)."""

    model: str
    score: float

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        _require_nonempty_str("AiInferred", "model", self.model)
        if type(self.score) is not float:
            raise TypeError(f"AiInferred.score must be float, got {type(self.score).__qualname__}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"AiInferred.score must be in [0, 1], got {self.score}")


type EvidenceStatus = (
    Assumed
    | Specified
    | StaticallyDerived
    | SmtProved
    | FormallyProved
    | RuntimeChecked
    | Measured
    | Predicted
    | AiInferred
)

_STATUS_CLASSES = (
    Assumed,
    Specified,
    StaticallyDerived,
    SmtProved,
    FormallyProved,
    RuntimeChecked,
    Measured,
    Predicted,
    AiInferred,
)


# --------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Evidence:
    """An immutable, append-only justification event for a claim."""

    id: EvidenceId
    claim: ClaimId
    status: EvidenceStatus
    provenance: tuple[EvidenceId, ...]
    scope: Scope
    validity: Validity
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.id) is not EvidenceId:
            raise TypeError(f"Evidence.id must be EvidenceId, got {type(self.id).__qualname__}")
        if type(self.claim) is not ClaimId:
            raise TypeError(f"Evidence.claim must be ClaimId, got {type(self.claim).__qualname__}")
        if type(self.status) not in _STATUS_CLASSES:
            raise TypeError(
                f"Evidence.status must be an EvidenceStatus, got {type(self.status).__qualname__}"
            )
        if type(self.provenance) is not tuple:
            raise TypeError("Evidence.provenance must be a tuple of EvidenceId")
        for i, ref in enumerate(self.provenance):
            if type(ref) is not EvidenceId:
                raise TypeError(
                    f"Evidence.provenance[{i}] must be EvidenceId, got {type(ref).__qualname__}"
                )
        if self.id in self.provenance:
            raise ValueError(f"Evidence.provenance must not contain the record's own id {self.id}")
        if type(self.scope) is not Scope:
            raise TypeError(f"Evidence.scope must be Scope, got {type(self.scope).__qualname__}")
        if type(self.validity) is not Validity:
            raise TypeError(
                f"Evidence.validity must be Validity, got {type(self.validity).__qualname__}"
            )
        if type(self.artifact) is not ArtifactRef:
            raise TypeError(
                f"Evidence.artifact must be ArtifactRef, got {type(self.artifact).__qualname__}"
            )


# --------------------------------------------------------------------------------------
# Task 003 producers: the only two evidence kinds the system emits at this stage
# --------------------------------------------------------------------------------------


def specified_evidence(
    id: EvidenceId,
    claim: ClaimId,
    *,
    pack: str,
    scope: Scope,
    validity: Validity = ANY_VALIDITY,
    provenance: tuple[EvidenceId, ...] = (),
    artifact: ArtifactRef = EMPTY_ARTIFACT,
) -> Evidence:
    """Evidence that a bundled SpecPack states the claim (``Specified{Bundled}``)."""
    return Evidence(
        id=id,
        claim=claim,
        status=Specified(pack, Bundled()),
        provenance=provenance,
        scope=scope,
        validity=validity,
        artifact=artifact,
    )


def statically_derived_evidence(
    id: EvidenceId,
    claim: ClaimId,
    *,
    rule: str,
    provenance: tuple[EvidenceId, ...],
    scope: Scope,
    validity: Validity = ANY_VALIDITY,
    artifact: ArtifactRef = EMPTY_ARTIFACT,
) -> Evidence:
    """Evidence produced by applying a named static rule to prior evidence.

    A derivation must cite what it derived from: empty ``provenance`` is refused
    here (an unpremised "derivation" is an assumption and must say so).
    """
    if not provenance:
        raise ValueError(
            f"statically_derived_evidence({rule!r}): a derivation requires non-empty provenance"
        )
    return Evidence(
        id=id,
        claim=claim,
        status=StaticallyDerived(rule),
        provenance=provenance,
        scope=scope,
        validity=validity,
        artifact=artifact,
    )


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

_SCOPE_A = Scope(Universal(), Value("A"))


def _evidence_fixture() -> tuple[object, ...]:
    base = specified_evidence(
        EvidenceId(1), ClaimId(1), pack="fortran-language-semantics", scope=_SCOPE_A
    )
    derived = statically_derived_evidence(
        EvidenceId(2),
        ClaimId(2),
        rule="spd_implies_symmetric",
        provenance=(EvidenceId(1),),
        scope=_SCOPE_A,
        validity=Validity("ab12", None),
        artifact=ArtifactRef.of(b"derivation-note"),
    )
    return (base, base, derived)


register_frozen_type(Bundled, fixture=lambda: (Bundled(), Bundled()), schema_version=1)
register_frozen_type(
    Assumed, fixture=lambda: (Assumed("user"), Assumed("user"), Assumed("dev")), schema_version=1
)
register_frozen_type(
    Specified,
    fixture=lambda: (
        Specified("blas-reference", Bundled()),
        Specified("blas-reference", Bundled()),
        Specified("lapack-reference", Bundled()),
    ),
    schema_version=1,
)
register_frozen_type(
    StaticallyDerived,
    fixture=lambda: (
        StaticallyDerived("spd_implies_symmetric"),
        StaticallyDerived("spd_implies_symmetric"),
        StaticallyDerived("bridge:B1"),
    ),
    schema_version=1,
)
register_frozen_type(
    SmtProved,
    fixture=lambda: (
        SmtProved("z3", EMPTY_ARTIFACT),
        SmtProved("z3", EMPTY_ARTIFACT),
        SmtProved("z3", ArtifactRef.of(b"unsat-core")),
    ),
    schema_version=1,
)
register_frozen_type(
    FormallyProved,
    fixture=lambda: (
        FormallyProved("lean4", EMPTY_ARTIFACT),
        FormallyProved("lean4", EMPTY_ARTIFACT),
        FormallyProved("coq", ArtifactRef.of(b"proof-term")),
    ),
    schema_version=1,
)
register_frozen_type(
    RuntimeChecked,
    fixture=lambda: (
        RuntimeChecked(GuardId(1), EventId(0)),
        RuntimeChecked(GuardId(1), EventId(0)),
        RuntimeChecked(GuardId(2), EventId(41)),
    ),
    schema_version=1,
)
register_frozen_type(
    Measured,
    fixture=lambda: (
        Measured("topo:ab12", 200, (0.9, 1.1)),
        Measured("topo:ab12", 200, (0.9, 1.1)),
        Measured("topo:cd34", 1, (2.0, 2.0)),
    ),
    schema_version=1,
)
register_frozen_type(
    Predicted,
    fixture=lambda: (
        Predicted("cost-v0", "calib-2026-07"),
        Predicted("cost-v0", "calib-2026-07"),
        Predicted("cost-v0", "calib-2026-01"),
    ),
    schema_version=1,
)
register_frozen_type(
    AiInferred,
    fixture=lambda: (
        AiInferred("proposer-v0", 0.5),
        AiInferred("proposer-v0", 0.5),
        AiInferred("proposer-v0", 1.0),
    ),
    schema_version=1,
)
register_frozen_type(Evidence, fixture=_evidence_fixture, schema_version=1)
