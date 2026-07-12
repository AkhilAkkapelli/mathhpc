"""Unit and acceptance tests for ``Decision`` and the full ``DecisionControl``
union (Task 006). Acceptance: epoch + ``depends_on`` recorded and serialized."""

from typing import assert_never, cast

import pytest

from mathhpc.foundation import (
    AiInferred,
    Auto,
    ClaimId,
    Decision,
    DecisionControl,
    DecisionId,
    Established,
    EvidenceId,
    FrozenDict,
    Prefer,
    PriorityClass,
    Refuted,
    Require,
    StaticallyDerived,
    Unknown,
    Use,
    from_json,
    to_json,
)

_ESTABLISHED = Established(StaticallyDerived("spd_implies_symmetric"))


def _auto_decision(**overrides: object) -> Decision:
    base: dict[str, object] = {
        "id": DecisionId(1),
        "layer": "plan",
        "subject": "C",
        "control": Auto(),
        "chosen": "dgemm",
        "considered": ("dgemm", "anchor_loop"),
        "depends_on": ((ClaimId(1), _ESTABLISHED),),
        "epoch": 7,
    }
    base.update(overrides)
    return Decision(**base)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------------------
# The full DecisionControl union
# --------------------------------------------------------------------------------------


def _describe(control: DecisionControl) -> str:
    """Exhaustive-match consumer: pyright strict fails this file if a variant is
    added to DecisionControl without extending the match."""
    match control:
        case Auto():
            return "auto"
        case Prefer(value=v, cls=c, rank=r):
            return f"prefer:{v}:{c.value}:{r}"
        case Require(value=v):
            return f"require:{v}"
        case Use(value=v):
            return f"use:{v}"
        case _:
            assert_never(control)


def test_all_four_control_variants_exist_and_match() -> None:
    controls: tuple[DecisionControl, ...] = (
        Auto(),
        Prefer("cholesky", PriorityClass.STRONG, 2),
        Require("dgemm"),
        Use("blocked", FrozenDict({"block_size": "100"})),
    )
    assert [_describe(c).split(":")[0] for c in controls] == [
        "auto",
        "prefer",
        "require",
        "use",
    ]


def test_priority_class_members() -> None:
    assert {p.value for p in PriorityClass} == {"strong", "normal", "hint"}


def test_prefer_validation() -> None:
    Prefer("x", PriorityClass.HINT, None)
    with pytest.raises(ValueError, match="non-empty"):
        Prefer("", PriorityClass.HINT, None)
    with pytest.raises(TypeError, match="must be PriorityClass"):
        Prefer("x", "strong", None)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=">= 1"):
        Prefer("x", PriorityClass.HINT, 0)
    with pytest.raises(TypeError, match="rank must be int or None"):
        Prefer("x", PriorityClass.HINT, True)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_require_and_use_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Require("")
    # A raw dict is caught by the deep-immutability validator (mutable container)
    # before the explicit FrozenDict check — either way, construction is refused.
    with pytest.raises(TypeError, match="mutable container: dict"):
        Use("x", {"k": "v"})  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="params must be FrozenDict"):
        Use("x", ("k", "v"))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------------------
# Decision: construction, identity, validation
# --------------------------------------------------------------------------------------


def test_auto_decision_is_exercised() -> None:
    d = _auto_decision()
    assert type(d.control) is Auto
    assert d.chosen == "dgemm"
    assert d == _auto_decision()
    assert hash(d) == hash(_auto_decision())


def test_decision_records_epoch_and_depends_on() -> None:
    d = _auto_decision(
        epoch=42,
        depends_on=(
            (ClaimId(1), _ESTABLISHED),
            (ClaimId(2), Unknown(None)),
            (ClaimId(3), Refuted(EvidenceId(9))),
        ),
    )
    assert d.epoch == 42
    assert d.depends_on[0] == (ClaimId(1), _ESTABLISHED)
    assert d.depends_on[1] == (ClaimId(2), Unknown(None))
    assert d.depends_on[2] == (ClaimId(3), Refuted(EvidenceId(9)))


def test_decision_rejects_wrong_id_kind() -> None:
    with pytest.raises(TypeError, match="id must be DecisionId"):
        _auto_decision(id=ClaimId(1))


def test_decision_rejects_non_control() -> None:
    with pytest.raises(TypeError, match="control must be a DecisionControl"):
        _auto_decision(control="auto")


def test_decision_rejects_bad_depends_on() -> None:
    with pytest.raises(TypeError, match=r"depends_on\[0\] must be a"):
        _auto_decision(depends_on=((ClaimId(1),),))
    with pytest.raises(TypeError, match=r"depends_on\[0\]\[0\] must be ClaimId"):
        _auto_decision(depends_on=((EvidenceId(1), _ESTABLISHED),))
    with pytest.raises(TypeError, match=r"depends_on\[0\]\[1\] must be a ClaimState"):
        _auto_decision(depends_on=((ClaimId(1), AiInferred("m", 0.5)),))


def test_decision_rejects_bad_epoch() -> None:
    with pytest.raises(TypeError, match="epoch must be an int"):
        _auto_decision(epoch=True)
    with pytest.raises(ValueError, match=">= 0"):
        _auto_decision(epoch=-1)


def test_decision_rejects_empty_string_fields() -> None:
    with pytest.raises(ValueError, match="layer must be non-empty"):
        _auto_decision(layer="")
    with pytest.raises(TypeError, match=r"considered\[0\] must be str"):
        _auto_decision(considered=(1,))


# --------------------------------------------------------------------------------------
# Acceptance: epoch + depends_on serialized (round-trip)
# --------------------------------------------------------------------------------------


def test_decision_json_roundtrip_preserves_epoch_and_depends_on() -> None:
    d = _auto_decision(
        epoch=99,
        depends_on=(
            (ClaimId(1), _ESTABLISHED),
            (ClaimId(2), Refuted(EvidenceId(5))),
            (ClaimId(3), Unknown(EvidenceId(8))),
        ),
    )
    restored = from_json(to_json(d), Decision)
    assert restored == d
    assert restored.epoch == 99
    assert restored.depends_on == d.depends_on


def test_decision_document_shape_carries_epoch_and_depends_on() -> None:
    doc = to_json(_auto_decision(epoch=12))
    fields = cast("dict[str, object]", doc["fields"])
    assert fields["epoch"] == 12
    depends_on = cast("list[object]", fields["depends_on"])
    # each dependency is a [claim, state] pair; the state is a tagged record.
    pair = cast("list[dict[str, object]]", depends_on[0])
    assert len(pair) == 2
    claim_doc, state_doc = pair
    assert claim_doc["$type"] == "ClaimId"
    assert state_doc["$type"] == "Established"


@pytest.mark.parametrize(
    "control",
    [
        Auto(),
        Prefer("cholesky", PriorityClass.STRONG, 3),
        Require("dgemm"),
        Use("blocked", FrozenDict({"nb": "8"})),
    ],
)
def test_every_control_variant_roundtrips_inside_a_decision(control: DecisionControl) -> None:
    d = _auto_decision(control=control)
    assert from_json(to_json(d), Decision) == d
