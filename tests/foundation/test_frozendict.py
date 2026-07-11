"""FrozenDict mapping semantics: lookup, equality, hashing, immutability, copy, pickle."""

import copy
import pickle

import hypothesis.strategies as st
import pytest
from hypothesis import given

from mathhpc.foundation import FrozenDict


def test_frozendict_lookup() -> None:
    fd = FrozenDict({"a": 1, "b": 2})
    assert fd["a"] == 1
    assert fd["b"] == 2
    with pytest.raises(KeyError):
        fd["missing"]
    assert fd.get("missing") is None


def test_frozendict_len_iter_insertion_order() -> None:
    fd = FrozenDict([("b", 2), ("a", 1), ("c", 3)])
    assert len(fd) == 3
    assert list(fd) == ["b", "a", "c"]
    assert list(fd.items()) == [("b", 2), ("a", 1), ("c", 3)]


def test_frozendict_contains() -> None:
    fd = FrozenDict({"a": 1})
    assert "a" in fd
    assert "z" not in fd


def test_frozendict_from_mapping_and_pairs() -> None:
    assert FrozenDict({"a": 1}) == FrozenDict([("a", 1)])
    empty: FrozenDict[str, int] = FrozenDict()
    assert len(empty) == 0


def test_frozendict_duplicate_key_raises() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        FrozenDict([("a", 1), ("a", 2)])


def test_frozendict_unhashable_value_raises_on_hash() -> None:
    fd = FrozenDict({"a": [1, 2]})  # construction succeeds: immutability != hashability
    assert fd["a"] == [1, 2]
    with pytest.raises(TypeError, match="unhashable"):
        hash(fd)


def test_frozendict_equality_ignores_insertion_order() -> None:
    left = FrozenDict([("a", 1), ("b", 2)])
    right = FrozenDict([("b", 2), ("a", 1)])
    assert left == right
    assert hash(left) == hash(right)


def test_frozendict_equals_plain_dict() -> None:
    fd = FrozenDict({"a": 1, "b": 2})
    assert fd == {"b": 2, "a": 1}
    assert {"a": 1, "b": 2} == fd
    assert fd != {"a": 1}
    assert fd != {"a": 1, "b": 3}


def test_frozendict_not_equal_non_mapping() -> None:
    fd = FrozenDict({"a": 1})
    assert (fd == 5) is False
    assert (fd != 5) is True
    assert (fd == [("a", 1)]) is False


@given(st.lists(st.tuples(st.text(max_size=4), st.integers()), unique_by=lambda kv: kv[0]))
def test_frozendict_hash_matches_equality(pairs: list[tuple[str, int]]) -> None:
    forward = FrozenDict(pairs)
    backward = FrozenDict(list(reversed(pairs)))
    assert forward == backward
    assert hash(forward) == hash(backward)


def test_frozendict_has_no_mutating_api() -> None:
    fd = FrozenDict({"a": 1})
    for name in ("update", "pop", "popitem", "clear", "setdefault", "__setitem__", "__delitem__"):
        assert not hasattr(fd, name)
    with pytest.raises(TypeError):
        fd["b"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        del fd["a"]  # type: ignore[attr-defined]


def test_frozendict_setattr_blocked() -> None:
    fd = FrozenDict({"a": 1})
    with pytest.raises(AttributeError):
        fd.extra = 1  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        fd._items = ()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        del fd._items  # type: ignore[attr-defined]


def test_frozendict_copy_returns_self() -> None:
    fd = FrozenDict({"a": 1})
    assert copy.copy(fd) is fd
    assert copy.deepcopy(fd) is fd


def test_frozendict_pickle_roundtrip() -> None:
    fd = FrozenDict({"a": 1, "b": 2})
    clone = pickle.loads(pickle.dumps(fd))
    assert clone == fd
    assert hash(clone) == hash(fd)
    assert list(clone.items()) == list(fd.items())


def test_frozendict_repr() -> None:
    assert repr(FrozenDict({"a": 1})) == "FrozenDict({'a': 1})"
