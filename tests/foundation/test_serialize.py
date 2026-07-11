"""Unit tests for the canonical JSON serializer, its strictness, and the
schema-version/migration scaffold."""

import json
from dataclasses import dataclass
from typing import cast

import pytest

from mathhpc.foundation import (
    ClaimId,
    ContractId,
    ContractNumeric,
    EvidenceId,
    FrozenDict,
    MathematicalReal,
    RoundingMode,
    SerializationError,
    canonical_json_bytes,
    from_json,
    register_frozen_type,
    register_schema_migration,
    to_json,
    validate_frozen_instance,
)
from tests.foundation.fixtures import Box, ExampleNested, ExampleRecord, SerializationProbe

# --------------------------------------------------------------------------------------
# Migration scaffold fixture: current schema_version 2, with a v1 -> v2 migration
# that renames the field "name" to "label".
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MigratingRecord:
    claim: ClaimId
    label: str

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.claim) is not ClaimId:
            raise TypeError("claim must be ClaimId")


def _rename_name_to_label(fields: dict[str, object]) -> dict[str, object]:
    migrated = dict(fields)
    migrated["label"] = migrated.pop("name")
    return migrated


def _doc_fields(doc: dict[str, object]) -> dict[str, object]:
    fields = doc["fields"]
    assert isinstance(fields, dict)
    return cast("dict[str, object]", fields)


register_frozen_type(
    MigratingRecord,
    fixture=lambda: (
        MigratingRecord(ClaimId(1), "a"),
        MigratingRecord(ClaimId(1), "a"),
        MigratingRecord(ClaimId(2), "b"),
    ),
    schema_version=2,
)
register_schema_migration(MigratingRecord, from_version=1, migrate=_rename_name_to_label)


# --------------------------------------------------------------------------------------
# Document shape
# --------------------------------------------------------------------------------------


def test_to_json_document_shape() -> None:
    assert to_json(ClaimId(5)) == {
        "$type": "ClaimId",
        "schema_version": 1,
        "fields": {"value": 5},
    }


def test_from_json_untagged_resolves_type_from_tag() -> None:
    x = ContractNumeric(ContractId(9))
    assert from_json(to_json(x)) == x


def test_canonical_bytes_are_valid_compact_json() -> None:
    x = ClaimId(5)
    assert json.loads(canonical_json_bytes(x).decode("utf-8")) == to_json(x)


def test_to_json_rejects_unregistered_frozen_type() -> None:
    with pytest.raises(SerializationError, match="not a registered"):
        to_json(Box(item=1))


def test_to_json_rejects_non_dataclass() -> None:
    with pytest.raises(SerializationError, match="expects a frozen foundation instance"):
        to_json({"just": "a dict"})


# --------------------------------------------------------------------------------------
# Strictness: internal formats reject drift
# --------------------------------------------------------------------------------------


def test_unknown_top_level_key_rejected() -> None:
    doc = to_json(ClaimId(5))
    doc["extra"] = 1
    with pytest.raises(SerializationError, match="document keys"):
        from_json(doc, ClaimId)


def test_unknown_field_rejected() -> None:
    doc = to_json(ClaimId(5))
    fields = _doc_fields(doc)
    fields["smuggled"] = 1
    with pytest.raises(SerializationError, match="unknown fields"):
        from_json(doc, ClaimId)


def test_missing_field_rejected() -> None:
    doc = to_json(ClaimId(5))
    fields = _doc_fields(doc)
    del fields["value"]
    with pytest.raises(SerializationError, match="missing fields"):
        from_json(doc, ClaimId)


def test_wrong_type_tag_rejected() -> None:
    with pytest.raises(SerializationError, match=r"\$type"):
        from_json(to_json(EvidenceId(5)), ClaimId)


def test_int_field_rejects_float_and_bool_json_values() -> None:
    doc = to_json(ClaimId(5))
    fields = _doc_fields(doc)
    fields["value"] = 5.0
    with pytest.raises(SerializationError, match="expected int"):
        from_json(doc, ClaimId)
    fields["value"] = True
    with pytest.raises(SerializationError, match="expected int"):
        from_json(doc, ClaimId)


def test_decoded_record_reruns_constructor_validation() -> None:
    doc = to_json(ClaimId(5))
    fields = _doc_fields(doc)
    fields["value"] = 2**70
    with pytest.raises(ValueError, match="u64"):
        from_json(doc, ClaimId)


# --------------------------------------------------------------------------------------
# Schema versions and migrations
# --------------------------------------------------------------------------------------


def test_newer_document_version_rejected() -> None:
    doc = to_json(ClaimId(5))
    doc["schema_version"] = 2
    with pytest.raises(SerializationError, match="newer than"):
        from_json(doc, ClaimId)


def test_version_gap_without_migration_rejected() -> None:
    doc = to_json(ClaimId(5))
    doc["schema_version"] = 0
    with pytest.raises(SerializationError, match="no migration"):
        from_json(doc, ClaimId)


def test_migration_applies_to_old_document() -> None:
    old_doc: dict[str, object] = {
        "$type": "MigratingRecord",
        "schema_version": 1,
        "fields": {"claim": to_json(ClaimId(4)), "name": "renamed"},
    }
    assert from_json(old_doc, MigratingRecord) == MigratingRecord(ClaimId(4), "renamed")


def test_current_document_skips_migration() -> None:
    x = MigratingRecord(ClaimId(4), "direct")
    assert from_json(to_json(x), MigratingRecord) == x


def test_register_migration_rejects_bad_from_version_and_duplicates() -> None:
    with pytest.raises(ValueError, match="from_version"):
        register_schema_migration(MigratingRecord, from_version=2, migrate=lambda f: f)
    with pytest.raises(ValueError, match="already registered"):
        register_schema_migration(MigratingRecord, from_version=1, migrate=lambda f: f)
    with pytest.raises(SerializationError, match="not a registered"):
        register_schema_migration(Box, from_version=1, migrate=lambda f: f)


def test_register_frozen_type_validates_schema_version() -> None:
    @dataclass(frozen=True, slots=True)
    class _Bad:
        x: int

    with pytest.raises(ValueError, match="schema_version"):
        register_frozen_type(_Bad, fixture=lambda: (_Bad(1),), schema_version=0)


# --------------------------------------------------------------------------------------
# Scalar edge encodings
# --------------------------------------------------------------------------------------


def _probe(**overrides: object) -> SerializationProbe:
    base: dict[str, object] = {
        "ratio": 0.5,
        "blob": b"\x00\xff",
        "domain": MathematicalReal(),
        "note": None,
        "rounding": RoundingMode.RNE,
    }
    base.update(overrides)
    return SerializationProbe(**base)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_nan_rejected_at_encode() -> None:
    with pytest.raises(SerializationError, match="NaN"):
        to_json(_probe(ratio=float("nan")))


def test_infinities_roundtrip() -> None:
    for value in (float("inf"), float("-inf")):
        x = _probe(ratio=value)
        assert from_json(to_json(x), SerializationProbe) == x


def test_bytes_encoding_shape_and_bad_base64_rejected() -> None:
    doc = to_json(_probe(blob=b"\x00\xff"))
    fields = _doc_fields(doc)
    assert fields["blob"] == {"$b": "AP8="}
    fields["blob"] = {"$b": "not*base64"}
    with pytest.raises(SerializationError, match="base64"):
        from_json(doc, SerializationProbe)


def test_enum_encoded_by_name_and_unknown_member_rejected() -> None:
    doc = to_json(_probe(rounding=RoundingMode.RTZ))
    fields = _doc_fields(doc)
    assert fields["rounding"] == "RTZ"
    fields["rounding"] = "RXX"
    with pytest.raises(SerializationError, match="not a member"):
        from_json(doc, SerializationProbe)


def test_union_field_dispatches_on_type_tag() -> None:
    for domain in (MathematicalReal(), ContractNumeric(ContractId(3))):
        x = _probe(domain=domain)
        decoded = from_json(to_json(x), SerializationProbe)
        assert decoded.domain == domain
    doc = to_json(_probe())
    fields = _doc_fields(doc)
    domain_doc = fields["domain"]
    assert isinstance(domain_doc, dict)
    domain_doc["$type"] = "NotADomain"
    with pytest.raises(SerializationError, match="matches no union member"):
        from_json(doc, SerializationProbe)


def test_optional_field_roundtrips_both_arms() -> None:
    assert from_json(to_json(_probe(note=None)), SerializationProbe).note is None
    assert from_json(to_json(_probe(note=41)), SerializationProbe).note == 41


# --------------------------------------------------------------------------------------
# Canonical collection order (Erratum E-4 concept 3)
# --------------------------------------------------------------------------------------


def test_frozendict_serialization_order_is_canonical_not_insertion() -> None:
    forward = ExampleRecord(ClaimId(1), "r", FrozenDict({"a": 1, "b": 2}), ())
    backward = ExampleRecord(ClaimId(1), "r", FrozenDict({"b": 2, "a": 1}), ())
    assert canonical_json_bytes(forward) == canonical_json_bytes(backward)
    fields = _doc_fields(to_json(forward))
    assert fields["meta"] == [["a", 1], ["b", 2]]


def test_frozenset_serialization_order_is_canonical() -> None:
    a = ExampleNested(ExampleRecord(ClaimId(1), "r", FrozenDict({}), ()), frozenset({3, 1, 2}))
    b = ExampleNested(ExampleRecord(ClaimId(1), "r", FrozenDict({}), ()), frozenset({2, 3, 1}))
    assert canonical_json_bytes(a) == canonical_json_bytes(b)


def test_frozendict_duplicate_pairs_in_document_rejected() -> None:
    record = ExampleRecord(ClaimId(1), "r", FrozenDict({"a": 1}), ())
    doc = to_json(record)
    fields = _doc_fields(doc)
    fields["meta"] = [["a", 1], ["a", 2]]
    with pytest.raises(SerializationError, match="duplicate"):
        from_json(doc, ExampleRecord)
