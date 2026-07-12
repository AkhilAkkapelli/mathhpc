"""Task 005 acceptance: meet-lattice unit tests for ``Contract`` ⟨D, A, E⟩, plus
the profile-compatibility gate."""

import itertools

import pytest

from mathhpc.foundation import (
    FAITHFUL_PROFILE,
    FAST_PROFILE,
    STRICT_PROFILE,
    A1IeeeStrict,
    A2BoundedError,
    A3BackwardStable,
    A4Statistical,
    A5BestEffort,
    Contract,
    ContractMeetError,
    Determinism,
    ExceptionProfile,
    FpFeature,
    NormKind,
    contract_meet,
    exception_profile_features,
)

_TOP = Contract(Determinism.D1_BITWISE_PORTABLE, A1IeeeStrict(), frozenset(FpFeature))
_BOTTOM = Contract(Determinism.D4_ANY_ORDER, A5BestEffort(), frozenset())

# Meet-compatible sample lattice: distinct accuracy payloads only across classes,
# so every pairwise meet is defined and the lattice laws are total on the sample.
_SAMPLES = (
    _TOP,
    STRICT_PROFILE,
    FAITHFUL_PROFILE,
    FAST_PROFILE,
    Contract(
        Determinism.D4_ANY_ORDER,
        A4Statistical("ks", "1e-3"),
        frozenset({FpFeature.NAN_PROP}),
    ),
    _BOTTOM,
)


# --------------------------------------------------------------------------------------
# Named profiles are exactly the frozen definitions (spec §3.1)
# --------------------------------------------------------------------------------------


def test_named_profiles_match_the_frozen_definitions() -> None:
    assert STRICT_PROFILE == Contract(
        Determinism.D2_BITWISE_CONFIG,
        A1IeeeStrict(),
        exception_profile_features(ExceptionProfile.E1),
    )
    assert FAITHFUL_PROFILE.det is Determinism.D2_BITWISE_CONFIG
    assert type(FAITHFUL_PROFILE.acc) is A2BoundedError
    assert FAITHFUL_PROFILE.exc == exception_profile_features(ExceptionProfile.E2)
    assert FAST_PROFILE.det is Determinism.D3_DETERMINISTIC
    assert type(FAST_PROFILE.acc) is A3BackwardStable
    assert FAST_PROFILE.exc == exception_profile_features(ExceptionProfile.E3)


def test_exception_profiles_nest() -> None:
    e1 = exception_profile_features(ExceptionProfile.E1)
    e2 = exception_profile_features(ExceptionProfile.E2)
    e3 = exception_profile_features(ExceptionProfile.E3)
    e4 = exception_profile_features(ExceptionProfile.E4)
    assert e1 > e2 > e3 > e4
    assert e1 == frozenset(FpFeature)
    assert e2 == frozenset(
        {FpFeature.NAN_PROP, FpFeature.INF, FpFeature.SIGNED_ZERO, FpFeature.SUBNORMAL}
    )
    assert e4 == frozenset()


# --------------------------------------------------------------------------------------
# Meet-lattice laws (acceptance)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("c", _SAMPLES)
def test_meet_is_idempotent(c: Contract) -> None:
    assert contract_meet(c, c) == c


@pytest.mark.parametrize(("a", "b"), list(itertools.combinations(_SAMPLES, 2)))
def test_meet_is_commutative(a: Contract, b: Contract) -> None:
    assert contract_meet(a, b) == contract_meet(b, a)


@pytest.mark.parametrize(("a", "b", "c"), list(itertools.combinations(_SAMPLES, 3)))
def test_meet_is_associative(a: Contract, b: Contract, c: Contract) -> None:
    assert contract_meet(a, contract_meet(b, c)) == contract_meet(contract_meet(a, b), c)


def test_strongest_absorbs_and_weakest_is_identity() -> None:
    for c in _SAMPLES:
        assert contract_meet(_TOP, c) == _TOP
        assert contract_meet(_BOTTOM, c) == c


@pytest.mark.parametrize(("a", "b"), list(itertools.combinations(_SAMPLES, 2)))
def test_meet_is_the_greatest_lower_bound_wrt_satisfies(a: Contract, b: Contract) -> None:
    """The universal property tying the gate to the lattice: a contract meets
    both requirements iff it meets their meet."""
    both = contract_meet(a, b)
    assert both.satisfies(a) and both.satisfies(b)
    for c in _SAMPLES:
        assert c.satisfies(both) == (c.satisfies(a) and c.satisfies(b))


def test_meet_of_the_named_profiles() -> None:
    assert contract_meet(STRICT_PROFILE, FAITHFUL_PROFILE) == STRICT_PROFILE
    assert contract_meet(FAITHFUL_PROFILE, FAST_PROFILE) == FAITHFUL_PROFILE
    assert contract_meet(STRICT_PROFILE, FAST_PROFILE) == Contract(
        Determinism.D2_BITWISE_CONFIG,
        A1IeeeStrict(),
        exception_profile_features(ExceptionProfile.E1),
    )


def test_meet_refuses_distinct_payloads_of_one_accuracy_class() -> None:
    a = Contract(
        Determinism.D3_DETERMINISTIC,
        A2BoundedError("classical_gamma", NormKind.COMPONENTWISE),
        frozenset(),
    )
    b = Contract(
        Determinism.D3_DETERMINISTIC,
        A2BoundedError("gamma_2n", NormKind.NORMWISE),
        frozenset(),
    )
    with pytest.raises(ContractMeetError, match="incomparable"):
        contract_meet(a, b)


# --------------------------------------------------------------------------------------
# Profile-compatibility gate
# --------------------------------------------------------------------------------------


def test_stronger_profiles_satisfy_weaker_ones() -> None:
    assert STRICT_PROFILE.satisfies(FAITHFUL_PROFILE)
    assert STRICT_PROFILE.satisfies(FAST_PROFILE)
    assert FAITHFUL_PROFILE.satisfies(FAST_PROFILE)
    for profile in (STRICT_PROFILE, FAITHFUL_PROFILE, FAST_PROFILE):
        assert profile.satisfies(profile)
        assert _TOP.satisfies(profile)
        assert profile.satisfies(_BOTTOM)


def test_weaker_profiles_do_not_satisfy_stronger_ones() -> None:
    assert not FAITHFUL_PROFILE.satisfies(STRICT_PROFILE)  # A2 < A1
    assert not FAST_PROFILE.satisfies(FAITHFUL_PROFILE)  # D3 < D2 and A3 < A2
    assert not _BOTTOM.satisfies(FAST_PROFILE)


def test_threaded_d3_pack_fails_a_required_d2() -> None:
    """The spec §9 planner example: a threaded pack carrying D3 cannot meet a
    required D2 even when accuracy and E are fine."""
    threaded = Contract(Determinism.D3_DETERMINISTIC, A1IeeeStrict(), frozenset(FpFeature))
    required = Contract(
        Determinism.D2_BITWISE_CONFIG,
        A5BestEffort(),
        frozenset(),
    )
    assert not threaded.satisfies(required)
    single_thread = Contract(Determinism.D2_BITWISE_CONFIG, A1IeeeStrict(), frozenset(FpFeature))
    assert single_thread.satisfies(required)


def test_e_axis_gate_is_superset_inclusion() -> None:
    offered = Contract(
        Determinism.D1_BITWISE_PORTABLE,
        A1IeeeStrict(),
        exception_profile_features(ExceptionProfile.E3),
    )
    requires_e2 = Contract(
        Determinism.D4_ANY_ORDER,
        A5BestEffort(),
        exception_profile_features(ExceptionProfile.E2),
    )
    assert not offered.satisfies(requires_e2)  # E3 does not cover SIGNED_ZERO/SUBNORMAL


def test_same_accuracy_class_distinct_payload_rejected_in_safe_direction() -> None:
    offered = Contract(
        Determinism.D2_BITWISE_CONFIG,
        A2BoundedError("gamma_2n", NormKind.NORMWISE),
        frozenset(FpFeature),
    )
    required = Contract(
        Determinism.D4_ANY_ORDER,
        A2BoundedError("classical_gamma", NormKind.COMPONENTWISE),
        frozenset(),
    )
    assert not offered.satisfies(required)
    assert offered.satisfies(
        Contract(
            Determinism.D4_ANY_ORDER, A2BoundedError("gamma_2n", NormKind.NORMWISE), frozenset()
        )
    )


# --------------------------------------------------------------------------------------
# Construction validation
# --------------------------------------------------------------------------------------


def test_contract_rejects_wrong_axis_types() -> None:
    with pytest.raises(TypeError, match="det must be Determinism"):
        Contract("d2", A1IeeeStrict(), frozenset())  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="acc must be an Accuracy"):
        Contract(Determinism.D2_BITWISE_CONFIG, "a1", frozenset())  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="exc members must be FpFeature"):
        Contract(Determinism.D2_BITWISE_CONFIG, A1IeeeStrict(), frozenset({"nan"}))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_accuracy_payload_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        A2BoundedError("", NormKind.NORMWISE)
    with pytest.raises(TypeError, match="must be NormKind"):
        A2BoundedError("g", "normwise")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="non-empty"):
        A3BackwardStable("")
