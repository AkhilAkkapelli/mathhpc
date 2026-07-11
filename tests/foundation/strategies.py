"""Hypothesis strategies for the Task 001 property suite."""

from collections.abc import Callable

import hypothesis.strategies as st

from mathhpc.foundation import FrozenDict
from tests.foundation.fixtures import Box

__all__ = ["contaminated", "immutable_values"]


def _leaves() -> st.SearchStrategy[object]:
    return st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True),
        st.text(max_size=6),
        st.binary(max_size=6),
    )


def immutable_values() -> st.SearchStrategy[object]:
    """Arbitrarily nested values built only from allowed immutable forms."""

    def extend(children: st.SearchStrategy[object]) -> st.SearchStrategy[object]:
        tuples = st.lists(children, max_size=3).map(tuple)
        fsets = st.lists(children, max_size=3).map(frozenset)
        keys = st.text(max_size=4)
        pairs = st.lists(st.tuples(keys, children), max_size=3, unique_by=lambda kv: kv[0])
        fds = pairs.map(FrozenDict)
        return st.one_of(tuples, fsets, fds)

    return st.recursive(_leaves(), extend, max_leaves=15)


_MUTANT_BUILDERS: tuple[Callable[[], object], ...] = (
    list,
    dict,
    set,
    lambda: bytearray(b"m"),
)


@st.composite
def contaminated(draw: st.DrawFn) -> tuple[object, str]:
    """An immutable-looking structure with exactly one mutable value planted at a
    known location; returns ``(value, expected_rendered_path)``."""
    value: object = draw(st.sampled_from(_MUTANT_BUILDERS))()
    path = ""
    depth = draw(st.integers(min_value=0, max_value=4))
    for _ in range(depth):
        wrapper = draw(st.sampled_from(("tuple", "frozendict", "box")))
        if wrapper == "tuple":
            pad = draw(st.integers(min_value=0, max_value=2))
            value = (*(("pad",) * pad), value)
            path = f"[{pad}]{path}"
        elif wrapper == "frozendict":
            key = draw(st.text(alphabet="abcd", min_size=1, max_size=3))
            value = FrozenDict({key: value})
            path = f"[{key!r}]{path}"
        else:
            value = Box(item=value)
            path = f".item{path}"
    return value, f"root{path}"
