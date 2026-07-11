"""Validator tests: leaves, rejections, exact paths, cycles, shared substructure."""

import enum
from dataclasses import dataclass

import pytest
from hypothesis import given

from mathhpc.foundation import (
    FrozenDict,
    MutableFieldError,
    assert_deeply_immutable,
    validate_frozen_instance,
)
from tests.foundation.fixtures import Box
from tests.foundation.strategies import contaminated, immutable_values


class _Color(enum.Enum):
    RED = 1


class _Level(enum.IntEnum):
    LOW = 0


class _BigInt(int):
    pass


class _Custom:
    pass


@dataclass(frozen=True, slots=True)
class _Payload:
    metadata: FrozenDict[str, object]


@dataclass(frozen=True, slots=True)
class _Holder:
    payload: _Payload


@dataclass
class _MutableRecord:
    x: int


def test_validator_reports_nested_path() -> None:
    holder = _Holder(payload=_Payload(metadata=FrozenDict({"shape": (1, 2, [3])})))
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable(holder)
    assert excinfo.value.rendered == "root.payload.metadata['shape'][2]"
    assert excinfo.value.reason == "mutable container: list"


def test_validator_accepts_allowed_leaves() -> None:
    for leaf in (None, True, False, 3, -7, 3.5, float("nan"), "s", b"b"):
        assert_deeply_immutable(leaf)


def test_validator_exact_type_leaves_reject_subclasses() -> None:
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable(_BigInt(3))
    assert "not registered as immutable" in excinfo.value.reason


def test_validator_accepts_enum_members() -> None:
    assert_deeply_immutable(_Color.RED)
    assert_deeply_immutable(_Level.LOW)


def test_validator_accepts_nested_immutables() -> None:
    value = (
        FrozenDict({"a": (1, frozenset({2, 3})), "b": FrozenDict({"c": b"x"})}),
        _Color.RED,
        None,
    )
    assert_deeply_immutable(value)


def test_validator_rejects_list() -> None:
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable([1])
    assert excinfo.value.reason == "mutable container: list"
    assert excinfo.value.rendered == "root"


def test_validator_rejects_dict() -> None:
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable(({"a": 1},))
    assert excinfo.value.reason == "mutable container: dict"
    assert excinfo.value.rendered == "root[0]"


def test_validator_rejects_set() -> None:
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable({1, 2})
    assert excinfo.value.reason == "mutable container: set"


def test_validator_rejects_bytearray_and_memoryview() -> None:
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable(bytearray(b"x"))
    assert excinfo.value.reason == "mutable container: bytearray"
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable(memoryview(b"x"))
    assert excinfo.value.reason == "mutable container: memoryview"


def test_validator_rejects_mutable_dataclass() -> None:
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable(Box(item=_MutableRecord(x=1)))
    assert excinfo.value.rendered == "root.item"
    assert excinfo.value.reason == "mutable (non-frozen) dataclass"


def test_validator_rejects_unknown_custom_class() -> None:
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable(_Custom())
    assert "type _Custom is not registered as immutable" == excinfo.value.reason


def test_validator_reports_frozendict_key_path() -> None:
    key = _Custom()
    fd = FrozenDict({key: 1})
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable(fd)
    assert excinfo.value.rendered == f"root[{key!r}]#key"


def test_validator_detects_cycle() -> None:
    box = Box(item=None)
    object.__setattr__(box, "item", box)
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable(box)
    assert excinfo.value.reason == "cycle detected"
    assert excinfo.value.rendered == "root.item"


def test_validator_accepts_shared_dag() -> None:
    shared = ("x", ("y",))
    assert_deeply_immutable((shared, shared, Box(item=shared)))


def test_validate_frozen_instance_fields_only() -> None:
    with pytest.raises(MutableFieldError) as excinfo:
        validate_frozen_instance(Box(item=[1]))
    assert excinfo.value.rendered == "Box.item"


def test_validate_frozen_instance_rejects_non_dataclass() -> None:
    with pytest.raises(TypeError):
        validate_frozen_instance("not a dataclass")


def test_validate_frozen_instance_rejects_mutable_dataclass() -> None:
    with pytest.raises(TypeError):
        validate_frozen_instance(_MutableRecord(x=1))


@given(immutable_values())
def test_prop_immutable_values_pass(value: object) -> None:
    assert_deeply_immutable(value)


@given(contaminated())
def test_prop_contaminated_raises_with_exact_path(case: tuple[object, str]) -> None:
    value, expected = case
    with pytest.raises(MutableFieldError) as excinfo:
        assert_deeply_immutable(value)
    assert excinfo.value.rendered == expected
