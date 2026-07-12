"""``Contract`` — the three-axis ⟨D, A, E⟩ numerical contract (Task 005).

Frozen definitions (spec §7; v2 §6.1):

    Contract:      det: Determinism; acc: Accuracy; exc: ExceptionalBehavior
    Determinism  = D1_BitwisePortable | D2_BitwiseConfig | D3_Deterministic | D4_AnyOrder
    Accuracy     = A1_IeeeStrict | A2_BoundedError(eps, norm) | A3_BackwardStable(cls)
                 | A4_Statistical | A5_BestEffort
    ExceptionalBehavior = frozenset drawn from
                 {NAN_PROP, INF, SIGNED_ZERO, SUBNORMAL, FP_FLAGS}   (E1..E4 named)

Ordering and meet (v2 §6.1: "⊑ is componentwise; meet = componentwise
strongest"): D and A are strength chains (D1 strongest … D4 weakest; A1
strongest … A5 weakest); the E axis is ordered by feature-set inclusion — more
preserved features is stronger, and comparison is the superset test the E-bridge
uses ("SpecPack E-profile ⊇ required E-profile", spec §12). ``contract_meet`` is
the componentwise strongest; on the E axis that is set union.

Two payload-bearing accuracy classes make the A chain a chain *of classes*, not
of instances: distinct payloads within one class (two different A2 bounds, two
different A3 stability classes) are incomparable without the symbolic γ-bound
algebra, which arrives with the bridge tasks. Task 005 is therefore
conservative in the safe direction (spec §20 risk table: "reject more"):
``satisfies`` treats same-class-different-payload as *not* satisfied, and
``contract_meet`` raises ``ContractMeetError`` rather than inventing a bound.

Named profiles (spec §3.1): ``strict = ⟨D2, A1, E1⟩`` (anchor-order-preserving),
``faithful = ⟨D2, A2(classical γ-bounds), E2⟩`` (default), ``fast = ⟨D3, A3, E3⟩``.

E1..E4 licensing, classified precisely:

- **Specified by the accepted design**: the five-feature universe
  {NAN_PROP, INF, SIGNED_ZERO, SUBNORMAL, FP_FLAGS}; the profile *names*
  E1..E4; exceptional behavior represented as a feature ``frozenset``
  (spec §7); superset-based satisfaction (spec §12); and the assignment of
  E1/E2/E3 to strict/faithful/fast respectively (spec §3.1).
- **Natural structural interpretation** (index order parallel to
  "D1 strongest … D4 weakest"): nesting E1 ⊇ E2 ⊇ E3 ⊇ E4, with E1 the
  strongest extreme and E4 the weakest.
- **Implementation-local choices**: the exact contents of E2 and of E3. The
  concrete sets implemented here are E1 = all five features; E2 = {NAN_PROP,
  INF, SIGNED_ZERO, SUBNORMAL}; E3 = {NAN_PROP, INF}; E4 = ∅. The exact E2/E3
  membership is **not textually fixed by the frozen documents**; these sets are
  motivated by (not licensed by) the MVP's adversarial diff-check inputs and
  the unobservability of FP flags in the generated drivers.

Compatibility posture: serialized ``Contract`` values contain the explicit
feature set, never an E-profile name — so a later revision of a named profile
constant affects newly constructed contracts only and never reinterprets a
previously serialized contract.

Task 005 ships the **profile-compatibility gate only** (``Contract.satisfies``);
the per-transformation gating table of v2 §6.3 is implemented by the tasks that
perform those transformations.
"""

from dataclasses import dataclass
from enum import Enum, unique
from typing import Final

from mathhpc.foundation.domains import ExceptionProfile
from mathhpc.foundation.immutable import validate_frozen_instance
from mathhpc.foundation.registry import register_frozen_type

__all__ = [
    "FAITHFUL_PROFILE",
    "FAST_PROFILE",
    "STRICT_PROFILE",
    "A1IeeeStrict",
    "A2BoundedError",
    "A3BackwardStable",
    "A4Statistical",
    "A5BestEffort",
    "Accuracy",
    "Contract",
    "ContractMeetError",
    "Determinism",
    "ExceptionalBehavior",
    "FpFeature",
    "NormKind",
    "contract_meet",
    "exception_profile_features",
]


@unique
class Determinism(Enum):
    """D1 strongest … D4 weakest (v2 §6.1)."""

    D1_BITWISE_PORTABLE = "d1"
    D2_BITWISE_CONFIG = "d2"
    D3_DETERMINISTIC = "d3"
    D4_ANY_ORDER = "d4"


@unique
class FpFeature(Enum):
    """The exceptional-behavior features a contract may require preserved."""

    NAN_PROP = "nan_prop"
    INF = "inf"
    SIGNED_ZERO = "signed_zero"
    SUBNORMAL = "subnormal"
    FP_FLAGS = "fp_flags"


@unique
class NormKind(Enum):
    """Norm sense of an A2 bounded-error certificate."""

    NORMWISE = "normwise"
    COMPONENTWISE = "componentwise"


@dataclass(frozen=True, slots=True)
class A1IeeeStrict:
    """The specified op sequence, correctly rounded; no reassociation, no FMA
    contraction, no flush-to-zero, no precision changes."""

    def __post_init__(self) -> None:
        validate_frozen_instance(self)


@dataclass(frozen=True, slots=True)
class A2BoundedError:
    """Certified forward bound; ``eps`` is a symbolic bound expression (e.g.
    ``"classical_gamma"``) compared only for equality until the γ-bound algebra
    tasks land."""

    eps: str
    norm: NormKind

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.eps) is not str:
            raise TypeError(f"A2BoundedError.eps must be str, got {type(self.eps).__qualname__}")
        if not self.eps:
            raise ValueError("A2BoundedError.eps must be non-empty")
        if type(self.norm) is not NormKind:
            raise TypeError(
                f"A2BoundedError.norm must be NormKind, got {type(self.norm).__qualname__}"
            )


@dataclass(frozen=True, slots=True)
class A3BackwardStable:
    """Algorithm-level backward-error guarantee for a named stability class.

    ``cls`` is an opaque token compared only for equality; it does not itself
    constitute theorem evidence or a discharged numerical guarantee (theorem
    references live in SpecPack entries, per Erratum E-1). Under the
    safe-direction rule, distinct classes never satisfy each other.
    """

    cls: str

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.cls) is not str:
            raise TypeError(f"A3BackwardStable.cls must be str, got {type(self.cls).__qualname__}")
        if not self.cls:
            raise ValueError("A3BackwardStable.cls must be non-empty")


@dataclass(frozen=True, slots=True)
class A4Statistical:
    """Distributional equivalence (stochastic algorithms)."""

    test: str
    tol: str

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        for name in ("test", "tol"):
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"A4Statistical.{name} must be str, got {type(value).__qualname__}")
            if not value:
                raise ValueError(f"A4Statistical.{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class A5BestEffort:
    """Aggressive; sanity checks only."""

    def __post_init__(self) -> None:
        validate_frozen_instance(self)


type Accuracy = A1IeeeStrict | A2BoundedError | A3BackwardStable | A4Statistical | A5BestEffort
type ExceptionalBehavior = frozenset[FpFeature]

_ACCURACY_CLASSES: Final = (
    A1IeeeStrict,
    A2BoundedError,
    A3BackwardStable,
    A4Statistical,
    A5BestEffort,
)

_DET_STRENGTH: Final[dict[Determinism, int]] = {
    Determinism.D1_BITWISE_PORTABLE: 4,
    Determinism.D2_BITWISE_CONFIG: 3,
    Determinism.D3_DETERMINISTIC: 2,
    Determinism.D4_ANY_ORDER: 1,
}

_ACC_CLASS_STRENGTH: Final[dict[type[object], int]] = {
    A1IeeeStrict: 5,
    A2BoundedError: 4,
    A3BackwardStable: 3,
    A4Statistical: 2,
    A5BestEffort: 1,
}


class ContractMeetError(ValueError):
    """Two contracts demand distinct payloads of the same accuracy class; no
    componentwise-strongest exists without the symbolic bound algebra (later
    tasks). Conservative in the safe direction: refuse rather than invent."""


@dataclass(frozen=True, slots=True)
class Contract:
    """A required or offered numerical contract ⟨det, acc, exc⟩."""

    det: Determinism
    acc: Accuracy
    exc: frozenset[FpFeature]

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.det) is not Determinism:
            raise TypeError(f"Contract.det must be Determinism, got {type(self.det).__qualname__}")
        if type(self.acc) not in _ACCURACY_CLASSES:
            raise TypeError(f"Contract.acc must be an Accuracy, got {type(self.acc).__qualname__}")
        if type(self.exc) is not frozenset:
            raise TypeError(f"Contract.exc must be a frozenset, got {type(self.exc).__qualname__}")
        for feature in self.exc:
            if type(feature) is not FpFeature:
                raise TypeError(
                    f"Contract.exc members must be FpFeature, got {type(feature).__qualname__}"
                )

    def satisfies(self, required: "Contract") -> bool:
        """The profile-compatibility gate: does this (offered) contract meet
        ``required`` on all three axes?

        D and A compare on the strength chains; equal accuracy *class* requires
        an identical payload (distinct bounds are incomparable until the γ-bound
        algebra lands — reject in the safe direction); E compares by superset.
        """
        if type(required) is not Contract:
            raise TypeError(f"satisfies expects Contract, got {type(required).__qualname__}")
        if _DET_STRENGTH[self.det] < _DET_STRENGTH[required.det]:
            return False
        offered_rank = _ACC_CLASS_STRENGTH[type(self.acc)]
        required_rank = _ACC_CLASS_STRENGTH[type(required.acc)]
        if offered_rank < required_rank:
            return False
        if offered_rank == required_rank and self.acc != required.acc:
            return False
        return self.exc >= required.exc


def contract_meet(a: Contract, b: Contract) -> Contract:
    """Componentwise strongest (v2 §6.1) — the required contract of a producer is
    the meet over its consumers' requirements.

    **This is a partial operation on the full representable ``Contract``
    domain.** It is undefined (``ContractMeetError``) for equal-rank
    ``Accuracy`` values with distinct payloads, because the current MVP has no
    lawful payload ordering: no γ-bound algebra over symbolic ``eps`` tokens, no
    norm bridge between ``NORMWISE`` and ``COMPONENTWISE`` bounds, no order on
    backward-stability classes, and no statistical-test-specific comparison
    semantics. ``contract_meet`` is NOT a total lattice operation over the full
    representable domain, and definedness itself is not associative: with

        a = ⟨D4, A1, ∅⟩
        b = ⟨D4, A2("x", norm), ∅⟩
        c = ⟨D4, A2("y", norm), ∅⟩

    ``contract_meet(contract_meet(a, b), c)`` is defined (A1 outranks both A2
    payloads) while ``contract_meet(a, contract_meet(b, c))`` cannot be
    evaluated because ``contract_meet(b, c)`` is undefined. The regression test
    ``test_meet_definedness_is_not_associative`` pins this.

    Whenever ``contract_meet(a, b)`` *is* defined, it returns the least contract
    that satisfies both ``a`` and ``b`` under ``Contract.satisfies``, and the
    universal property

        c.satisfies(contract_meet(a, b)) == (c.satisfies(a) and c.satisfies(b))

    holds for every representable ``c``. The weakest element ``⟨D4, A5, ∅⟩`` is
    the identity and the strongest ``⟨D1, A1, all features⟩`` is absorbing.
    """
    if type(a) is not Contract or type(b) is not Contract:
        raise TypeError("contract_meet expects two Contracts")
    det = a.det if _DET_STRENGTH[a.det] >= _DET_STRENGTH[b.det] else b.det
    rank_a = _ACC_CLASS_STRENGTH[type(a.acc)]
    rank_b = _ACC_CLASS_STRENGTH[type(b.acc)]
    if rank_a > rank_b:
        acc = a.acc
    elif rank_b > rank_a:
        acc = b.acc
    elif a.acc == b.acc:
        acc = a.acc
    else:
        raise ContractMeetError(
            f"no componentwise-strongest of {a.acc!r} and {b.acc!r}: distinct "
            "payloads of one accuracy class are incomparable until the symbolic "
            "bound algebra lands"
        )
    return Contract(det, acc, a.exc | b.exc)


_E_FEATURES: Final[dict[ExceptionProfile, frozenset[FpFeature]]] = {
    ExceptionProfile.E1: frozenset(FpFeature),
    ExceptionProfile.E2: frozenset(
        {FpFeature.NAN_PROP, FpFeature.INF, FpFeature.SIGNED_ZERO, FpFeature.SUBNORMAL}
    ),
    ExceptionProfile.E3: frozenset({FpFeature.NAN_PROP, FpFeature.INF}),
    ExceptionProfile.E4: frozenset(),
}


def exception_profile_features(profile: ExceptionProfile) -> frozenset[FpFeature]:
    """The feature set named by an E1..E4 profile."""
    if type(profile) is not ExceptionProfile:
        raise TypeError(
            f"exception_profile_features expects ExceptionProfile, got {type(profile).__qualname__}"
        )
    return _E_FEATURES[profile]


STRICT_PROFILE: Final = Contract(
    Determinism.D2_BITWISE_CONFIG,
    A1IeeeStrict(),
    exception_profile_features(ExceptionProfile.E1),
)
FAITHFUL_PROFILE: Final = Contract(
    Determinism.D2_BITWISE_CONFIG,
    A2BoundedError("classical_gamma", NormKind.COMPONENTWISE),
    exception_profile_features(ExceptionProfile.E2),
)
# FAST_PROFILE's accuracy payload "classical" is an opaque placeholder token:
# the frozen MVP specification names the fast profile as A3 *without* specifying
# a stability-class payload, and the frozen A3 shape requires one. The token is
# compared only for equality and is not theorem evidence. Consequence, under the
# safe-direction compatibility rule: another A3 payload such as "cholesky" does
# NOT satisfy FAST_PROFILE's "classical" requirement. The policy for matching
# concrete backward-stability classes against the generic fast profile must be
# confronted explicitly in the SpecPack/planner tasks (009/010/027), not
# inferred silently from this placeholder.
FAST_PROFILE: Final = Contract(
    Determinism.D3_DETERMINISTIC,
    A3BackwardStable("classical"),
    exception_profile_features(ExceptionProfile.E3),
)


register_frozen_type(
    A1IeeeStrict, fixture=lambda: (A1IeeeStrict(), A1IeeeStrict()), schema_version=1
)
register_frozen_type(
    A2BoundedError,
    fixture=lambda: (
        A2BoundedError("classical_gamma", NormKind.COMPONENTWISE),
        A2BoundedError("classical_gamma", NormKind.COMPONENTWISE),
        A2BoundedError("gamma_2n", NormKind.NORMWISE),
    ),
    schema_version=1,
)
register_frozen_type(
    A3BackwardStable,
    fixture=lambda: (
        A3BackwardStable("classical"),
        A3BackwardStable("classical"),
        A3BackwardStable("cholesky"),
    ),
    schema_version=1,
)
register_frozen_type(
    A4Statistical,
    fixture=lambda: (
        A4Statistical("ks", "1e-3"),
        A4Statistical("ks", "1e-3"),
        A4Statistical("chi2", "1e-2"),
    ),
    schema_version=1,
)
register_frozen_type(
    A5BestEffort, fixture=lambda: (A5BestEffort(), A5BestEffort()), schema_version=1
)
register_frozen_type(
    Contract,
    fixture=lambda: (
        STRICT_PROFILE,
        Contract(
            Determinism.D2_BITWISE_CONFIG,
            A1IeeeStrict(),
            exception_profile_features(ExceptionProfile.E1),
        ),
        FAITHFUL_PROFILE,
        FAST_PROFILE,
    ),
    schema_version=1,
)
