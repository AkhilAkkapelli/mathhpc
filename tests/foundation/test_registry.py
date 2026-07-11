"""Registry semantics, discovery of unregistered frozen types, and the generic
per-type invariant suite (Erratum E-3)."""

import dataclasses
import importlib
import pkgutil
from dataclasses import dataclass

import pytest

import mathhpc
from mathhpc.foundation import (
    ClaimId,
    EvidenceId,
    assert_deeply_immutable,
    register_frozen_type,
    registered_frozen_types,
)
from mathhpc.foundation.registry import FrozenTypeEntry

# Importing the fixture names below also executes the module's registrations.
from tests.foundation.fixtures import ExampleNested, ExampleRecord

# Frozen dataclasses under src/ that are deliberately not foundation records.
# Task 001 needs none; the constant exists so an exemption is an explicit,
# reviewable decision rather than a scan tweak.
_DISCOVERY_EXEMPT: frozenset[type] = frozenset()


@dataclass
class _NotFrozen:
    x: int


class _NotADataclass:
    pass


def test_register_requires_frozen_dataclass() -> None:
    with pytest.raises(TypeError, match="not a frozen dataclass"):
        register_frozen_type(_NotFrozen, fixture=lambda: (_NotFrozen(1),))
    with pytest.raises(TypeError, match="not a frozen dataclass"):
        register_frozen_type(_NotADataclass, fixture=lambda: (_NotADataclass(),))


def test_duplicate_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_frozen_type(ClaimId, fixture=lambda: (ClaimId(0),))


def test_registered_frozen_types_returns_entries() -> None:
    classes = {entry.cls for entry in registered_frozen_types()}
    assert {ClaimId, EvidenceId, ExampleRecord, ExampleNested} <= classes


def _src_frozen_dataclasses() -> set[type]:
    found: set[type] = set()
    package_paths = list(mathhpc.__path__)
    for module_info in pkgutil.walk_packages(package_paths, prefix="mathhpc."):
        module = importlib.import_module(module_info.name)
        for obj in vars(module).values():
            if not (isinstance(obj, type) and obj.__module__ == module.__name__):
                continue
            if not dataclasses.is_dataclass(obj):
                continue
            params = getattr(obj, "__dataclass_params__", None)
            if bool(getattr(params, "frozen", False)):
                found.add(obj)
    return found


def test_no_unregistered_frozen_types_in_src() -> None:
    registered = {entry.cls for entry in registered_frozen_types()}
    missing = _src_frozen_dataclasses() - registered - _DISCOVERY_EXEMPT
    assert not missing, (
        f"unregistered frozen foundation types: {sorted(t.__qualname__ for t in missing)}"
    )


_ENTRIES = registered_frozen_types()
_ENTRY_IDS = [entry.cls.__qualname__ for entry in _ENTRIES]


@pytest.mark.parametrize("entry", _ENTRIES, ids=_ENTRY_IDS)
def test_all_registered_types_have_fixture_factory(entry: FrozenTypeEntry) -> None:
    instances = entry.fixture_factory()
    assert isinstance(instances, tuple)
    assert instances, f"{entry.cls.__qualname__} fixture factory returned no instances"
    for instance in instances:
        assert type(instance) is entry.cls


@pytest.mark.parametrize("entry", _ENTRIES, ids=_ENTRY_IDS)
def test_generic_suite_immutability(entry: FrozenTypeEntry) -> None:
    for instance in entry.fixture_factory():
        assert_deeply_immutable(instance)


@pytest.mark.parametrize("entry", _ENTRIES, ids=_ENTRY_IDS)
def test_generic_suite_field_mutation_raises(entry: FrozenTypeEntry) -> None:
    for instance in entry.fixture_factory():
        assert dataclasses.is_dataclass(instance) and not isinstance(instance, type)
        for field in dataclasses.fields(instance):
            with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
                setattr(instance, field.name, None)


@pytest.mark.parametrize("entry", _ENTRIES, ids=_ENTRY_IDS)
def test_generic_suite_hash_eq_law(entry: FrozenTypeEntry) -> None:
    instances = entry.fixture_factory()
    for left in instances:
        for right in instances:
            if left == right:
                assert hash(left) == hash(right)
