"""ID kind separation: runtime semantics of ClaimId and EvidenceId."""

from typing import cast

import pytest

from mathhpc.foundation import ClaimId, EventId, EvidenceId, FrozenDict, GuardId
from tests.foundation.fixtures import ExampleRecord


def test_claim_id_not_equal_to_evidence_id() -> None:
    assert ClaimId(1) != EvidenceId(1)
    assert (ClaimId(1) == EvidenceId(1)) is False
    assert ClaimId(1) == ClaimId(1)
    assert ClaimId(1) != ClaimId(2)


def test_event_id_not_equal_to_guard_id() -> None:
    assert EventId(1) != GuardId(1)
    assert (EventId(1) == GuardId(1)) is False
    assert EventId(1) == EventId(1)


def test_id_value_exact_int_bool_rejected() -> None:
    with pytest.raises(TypeError, match="must be int"):
        ClaimId(True)
    with pytest.raises(TypeError, match="must be int"):
        EvidenceId(cast(int, "7"))


def test_id_range_validation() -> None:
    assert ClaimId(0).value == 0
    assert ClaimId(2**64 - 1).value == 2**64 - 1
    with pytest.raises(ValueError, match="u64 range"):
        ClaimId(-1)
    with pytest.raises(ValueError, match="u64 range"):
        EvidenceId(2**64)


def test_id_no_ordering() -> None:
    with pytest.raises(TypeError):
        _ = ClaimId(1) < ClaimId(2)  # type: ignore[operator]


def test_id_repr() -> None:
    assert repr(ClaimId(7)) == "ClaimId(7)"
    assert repr(EvidenceId(9)) == "EvidenceId(9)"


def test_id_hash_mixes_kind() -> None:
    assert hash(ClaimId(5)) == hash(("ClaimId", 5))
    assert hash(EvidenceId(5)) == hash(("EvidenceId", 5))


def test_wrong_id_kind_rejected_at_runtime() -> None:
    with pytest.raises(TypeError, match="must be ClaimId"):
        ExampleRecord(
            claim=cast(ClaimId, EvidenceId(1)),
            name="x",
            meta=FrozenDict({}),
            tags=(),
        )
