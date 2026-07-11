"""Invariant 4 (Addendum A1 §2): JSON round-trip equality and hash stability for
Hypothesis-generated instances of every registered serializable frozen type.

Parametrized over the frozen-type registry (Erratum E-3): a type registered with a
``schema_version`` is covered automatically — generatively where
``serialization_strategies`` provides a strategy, from its fixture instances
otherwise.
"""

import hypothesis.strategies as st
import pytest
from hypothesis import given

from mathhpc.foundation import canonical_json_bytes, from_json, registered_frozen_types, to_json
from mathhpc.foundation.registry import FrozenTypeEntry
from tests.foundation.strategies import serialization_strategies

_STRATEGIES = serialization_strategies()
_SERIALIZABLE = [entry for entry in registered_frozen_types() if entry.schema_version is not None]
_IDS = [entry.cls.__qualname__ for entry in _SERIALIZABLE]


def _instances(entry: FrozenTypeEntry) -> st.SearchStrategy[object]:
    strategy = _STRATEGIES.get(entry.cls)
    if strategy is not None:
        return strategy
    return st.sampled_from(entry.fixture_factory())


def test_every_task_002_type_has_a_generative_strategy() -> None:
    covered = set(_STRATEGIES)
    serializable = {entry.cls for entry in _SERIALIZABLE}
    uncovered = {cls.__qualname__ for cls in serializable - covered}
    # Fixture-only fallback is permitted for types registered by later tasks, but
    # everything serializable as of Task 002 must be generatively covered — except
    # deliberate test-local scaffolding (the migration fixture lives in
    # test_serialize.py and may not even be registered yet at collection time).
    assert uncovered <= {"MigratingRecord"}


@pytest.mark.parametrize("entry", _SERIALIZABLE, ids=_IDS)
@given(data=st.data())
def test_json_roundtrip_equality_and_hash(entry: FrozenTypeEntry, data: st.DataObject) -> None:
    x = data.draw(_instances(entry))
    doc = to_json(x)
    decoded = from_json(doc, entry.cls)
    assert type(decoded) is entry.cls
    assert decoded == x
    assert hash(decoded) == hash(x)
    assert from_json(doc) == x  # the literal round-trip law, untyped form
    assert canonical_json_bytes(decoded) == canonical_json_bytes(x)
