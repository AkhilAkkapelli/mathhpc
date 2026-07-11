"""Canonical JSON serialization for registered frozen foundation types (Task 002).

Canonical serialization order — concept (3) of Erratum E-4 — is defined *here*,
not on the data types: unordered collections are emitted sorted by the canonical
serialized bytes of their elements (for ``frozenset``) or keys (for
``FrozenDict``), independent of Python ordering or insertion order.

Encoding rules (value-driven):

=============================  ====================================================
Python value                   JSON encoding
=============================  ====================================================
``None`` / ``bool`` / ``int``  the JSON native (ints are exact at any magnitude)
``str``                        JSON string
``float`` (finite)             JSON number (shortest-repr round-trip is exact)
``float`` (infinity)           ``{"$f": "inf"}`` / ``{"$f": "-inf"}``
``float`` (NaN)                **rejected** — NaN breaks the round-trip equality law
``bytes``                      ``{"$b": "<base64>"}``
``Enum`` member                the member *name* (class comes from the annotation)
``tuple``                      JSON array of encoded elements, in order
``frozenset``                  JSON array sorted by canonical encoded bytes
``FrozenDict``                 JSON array of ``[key, value]`` pairs sorted by the
                               canonical encoded bytes of the key
registered frozen dataclass    ``{"$type": <class name>, "schema_version": <int>,
                               "fields": {<name>: <encoded>, ...}}``
=============================  ====================================================

Decoding is annotation-driven and strict. These are internal formats (spec §2 of
Addendum A1): unknown fields, missing fields, wrong ``$type`` tags, unsupported
shapes, and version gaps without a migration are all ``SerializationError`` —
never silently tolerated. Every decoded record is built through its constructor,
so ``__post_init__`` validation re-runs; a document carrying a value the
constructor refuses raises that constructor's own ``TypeError``/``ValueError``.

Supported field annotations: the leaf classes (``bool``, ``int``, ``float``,
``str``, ``bytes``, ``None``), ``Enum`` subclasses, registered serializable
frozen dataclasses, ``tuple[T, ...]`` and fixed-arity ``tuple[T1, ..., Tn]``,
``frozenset[T]``, ``FrozenDict[K, V]``, PEP 695 type aliases, and unions of the
above where the value is decidable (``X | None``, or unions of registered frozen
dataclasses dispatched on ``$type`` — e.g. ``SemanticDomain``). Anything else is
rejected at decode time.

Schema-version scaffold: each serializable registry entry carries the *current*
``schema_version``; ``register_schema_migration`` installs stepwise
``from_version -> from_version + 1`` migrations over the encoded ``fields``
document, applied when an older document is decoded. The golden-archive CI gate
over all past versions (invariant 5) becomes merge-blocking at Task 015.
"""

import base64
import dataclasses
import enum
import json
import math
import types
import typing
from collections.abc import Callable, Mapping
from typing import TypeAliasType, cast, get_args, get_origin, overload

from mathhpc.foundation.immutable import FrozenDict
from mathhpc.foundation.registry import (
    FrozenTypeEntry,
    registered_frozen_types,
)

__all__ = [
    "SerializationError",
    "canonical_json_bytes",
    "from_json",
    "register_schema_migration",
    "to_json",
]


class SerializationError(Exception):
    """A value or document violates the canonical serialization format."""


_TYPE_KEY = "$type"
_VERSION_KEY = "schema_version"
_FIELDS_KEY = "fields"
_DOC_KEYS = frozenset({_TYPE_KEY, _VERSION_KEY, _FIELDS_KEY})

_FieldsDoc = dict[str, object]
_Migration = Callable[[_FieldsDoc], _FieldsDoc]

_MIGRATIONS: dict[tuple[type, int], _Migration] = {}


# --------------------------------------------------------------------------------------
# Registry access
# --------------------------------------------------------------------------------------


def _serializable_entry(cls: type[object]) -> FrozenTypeEntry:
    for entry in registered_frozen_types():
        if entry.cls is cls:
            if entry.schema_version is None:
                raise SerializationError(
                    f"{cls.__qualname__} is registered without a schema_version (not serializable)"
                )
            return entry
    raise SerializationError(f"{cls.__qualname__} is not a registered frozen foundation type")


def _serializable_class_named(name: str) -> type[object]:
    matches = [
        entry.cls
        for entry in registered_frozen_types()
        if entry.schema_version is not None and entry.cls.__name__ == name
    ]
    if not matches:
        raise SerializationError(f"unknown $type tag: {name!r}")
    if len(matches) > 1:
        raise SerializationError(f"$type tag {name!r} is ambiguous across registered types")
    return matches[0]


def register_schema_migration(cls: type[object], *, from_version: int, migrate: _Migration) -> None:
    """Install the ``from_version -> from_version + 1`` fields-document migration.

    ``cls`` must be registered as serializable and ``from_version`` must be an
    already-superseded version (``1 <= from_version < current``). Duplicate
    installation is an error, mirroring the frozen-type registry's rule.
    """
    entry = _serializable_entry(cls)
    current = entry.schema_version
    assert current is not None
    if type(from_version) is not int or not 1 <= from_version < current:
        raise ValueError(
            f"from_version must satisfy 1 <= from_version < {current}, got {from_version!r}"
        )
    key = (cls, from_version)
    if key in _MIGRATIONS:
        raise ValueError(f"migration for {cls.__qualname__} v{from_version} is already registered")
    _MIGRATIONS[key] = migrate


# --------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------


def _fragment_bytes(fragment: object) -> bytes:
    return json.dumps(
        fragment, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _encode(value: object) -> object:
    t = type(value)
    if value is None or t is bool or t is str or t is int:
        return value
    if t is float:
        f = cast(float, value)
        if math.isnan(f):
            raise SerializationError("NaN is not serializable (it breaks the equality law)")
        if math.isinf(f):
            return {"$f": "inf" if f > 0 else "-inf"}
        return f
    if t is bytes:
        return {"$b": base64.b64encode(cast(bytes, value)).decode("ascii")}
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, tuple):
        return [_encode(item) for item in cast("tuple[object, ...]", value)]
    if isinstance(value, frozenset):
        encoded = [_encode(item) for item in cast("frozenset[object]", value)]
        return sorted(encoded, key=_fragment_bytes)
    if isinstance(value, FrozenDict):
        mapping = cast("FrozenDict[object, object]", value)
        pairs = [[_encode(k), _encode(v)] for k, v in mapping.items()]
        return sorted(pairs, key=lambda pair: _fragment_bytes(pair[0]))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _encode_record(value)
    raise SerializationError(f"type {t.__qualname__} is not serializable")


def _encode_record(obj: object) -> dict[str, object]:
    cls = type(obj)
    entry = _serializable_entry(cls)
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise SerializationError(f"{cls.__qualname__} is not a dataclass instance")
    fields = {field.name: _encode(getattr(obj, field.name)) for field in dataclasses.fields(obj)}
    return {_TYPE_KEY: cls.__name__, _VERSION_KEY: entry.schema_version, _FIELDS_KEY: fields}


def to_json(obj: object) -> dict[str, object]:
    """Encode a registered serializable frozen instance as a tagged JSON document."""
    if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
        raise SerializationError(
            f"to_json expects a frozen foundation instance, got {type(obj).__qualname__}"
        )
    return _encode_record(obj)


def canonical_json_bytes(obj: object) -> bytes:
    """The canonical byte serialization: compact JSON, sorted object keys, UTF-8."""
    return _fragment_bytes(to_json(obj))


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------


def _field_annotations(cls: type[object]) -> dict[str, object]:
    cached = _ANNOTATION_CACHE.get(cls)
    if cached is None:
        cached = dict(typing.get_type_hints(cls))
        _ANNOTATION_CACHE[cls] = cached
    return cached


_ANNOTATION_CACHE: dict[type, dict[str, object]] = {}


def _unalias(ann: object) -> object:
    while isinstance(ann, TypeAliasType):
        ann = ann.__value__
    return ann


def _decode(value: object, ann: object, where: str) -> object:
    ann = _unalias(ann)
    origin = get_origin(ann)
    if origin is types.UnionType or origin is typing.Union:
        return _decode_union(value, tuple(cast("tuple[object, ...]", get_args(ann))), where)
    if origin is tuple:
        return _decode_tuple(value, tuple(cast("tuple[object, ...]", get_args(ann))), where)
    if origin is frozenset:
        (elem_ann,) = cast("tuple[object]", get_args(ann))
        if not isinstance(value, list):
            raise SerializationError(f"{where}: expected array for frozenset")
        items = [
            _decode(item, elem_ann, f"{where}[{i}]")
            for i, item in enumerate(cast("list[object]", value))
        ]
        result = frozenset(items)
        if len(result) != len(items):
            raise SerializationError(f"{where}: duplicate frozenset elements (non-canonical)")
        return result
    if origin is FrozenDict:
        key_ann, value_ann = cast("tuple[object, object]", get_args(ann))
        if not isinstance(value, list):
            raise SerializationError(f"{where}: expected array of pairs for FrozenDict")
        pairs: list[tuple[object, object]] = []
        for i, item in enumerate(cast("list[object]", value)):
            if not isinstance(item, list) or len(cast("list[object]", item)) != 2:
                raise SerializationError(f"{where}[{i}]: expected a [key, value] pair")
            raw_key, raw_value = cast("list[object]", item)
            pairs.append(
                (
                    _decode(raw_key, key_ann, f"{where}[{i}].key"),
                    _decode(raw_value, value_ann, f"{where}[{i}].value"),
                )
            )
        try:
            return FrozenDict(pairs)
        except ValueError as exc:
            raise SerializationError(f"{where}: {exc}") from None
    if ann is type(None):
        if value is not None:
            raise SerializationError(f"{where}: expected null")
        return None
    if isinstance(ann, type):
        return _decode_plain(value, ann, where)
    raise SerializationError(f"{where}: unsupported annotation {ann!r}")


def _decode_union(value: object, args: tuple[object, ...], where: str) -> object:
    members = tuple(_unalias(a) for a in args)
    if type(None) in members:
        if value is None:
            return None
        members = tuple(m for m in members if m is not type(None))
    if len(members) == 1:
        return _decode(value, members[0], where)
    if isinstance(value, Mapping):
        doc = cast("Mapping[str, object]", value)
        tag = doc.get(_TYPE_KEY)
        for member in members:
            if isinstance(member, type) and member.__name__ == tag:
                return _decode_record_doc(doc, member)
        raise SerializationError(f"{where}: $type {tag!r} matches no union member")
    raise SerializationError(f"{where}: cannot decode union without a $type tag")


def _decode_tuple(value: object, args: tuple[object, ...], where: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise SerializationError(f"{where}: expected array for tuple")
    items = cast("list[object]", value)
    if len(args) == 2 and args[1] is Ellipsis:
        elem_ann = args[0]
        return tuple(_decode(item, elem_ann, f"{where}[{i}]") for i, item in enumerate(items))
    if len(items) != len(args):
        raise SerializationError(f"{where}: expected {len(args)} tuple elements, got {len(items)}")
    return tuple(
        _decode(item, ann, f"{where}[{i}]")
        for i, (item, ann) in enumerate(zip(items, args, strict=True))
    )


def _decode_plain(value: object, ann: type[object], where: str) -> object:
    if ann is bool:
        if type(value) is not bool:
            raise SerializationError(f"{where}: expected bool")
        return value
    if ann is int:
        if type(value) is not int:
            raise SerializationError(f"{where}: expected int")
        return value
    if ann is float:
        if type(value) is float:
            return value
        if isinstance(value, Mapping):
            doc = cast("Mapping[str, object]", value)
            if set(doc.keys()) == {"$f"}:
                marker = doc["$f"]
                if marker == "inf":
                    return math.inf
                if marker == "-inf":
                    return -math.inf
        raise SerializationError(f"{where}: expected float")
    if ann is str:
        if type(value) is not str:
            raise SerializationError(f"{where}: expected str")
        return value
    if ann is bytes:
        if isinstance(value, Mapping):
            doc = cast("Mapping[str, object]", value)
            payload = doc.get("$b")
            if set(doc.keys()) == {"$b"} and type(payload) is str:
                try:
                    return base64.b64decode(payload, validate=True)
                except ValueError as exc:
                    raise SerializationError(f"{where}: invalid base64: {exc}") from None
        raise SerializationError(f"{where}: expected {{'$b': <base64>}} for bytes")
    if issubclass(ann, enum.Enum):
        if type(value) is not str:
            raise SerializationError(f"{where}: expected enum member name string")
        try:
            return ann[value]
        except KeyError:
            raise SerializationError(
                f"{where}: {value!r} is not a member of {ann.__qualname__}"
            ) from None
    if dataclasses.is_dataclass(ann):
        return _decode_record_doc(value, ann)
    raise SerializationError(f"{where}: unsupported annotation {ann.__qualname__}")


def _decode_record_doc(doc: object, cls: type[object]) -> object:
    entry = _serializable_entry(cls)
    current = entry.schema_version
    assert current is not None
    if not isinstance(doc, Mapping):
        raise SerializationError(f"{cls.__qualname__}: document must be a JSON object")
    mapping = cast("Mapping[str, object]", doc)
    keys = set(mapping.keys())
    if keys != set(_DOC_KEYS):
        raise SerializationError(
            f"{cls.__qualname__}: document keys must be exactly "
            f"{sorted(_DOC_KEYS)}, got {sorted(keys)}"
        )
    if mapping[_TYPE_KEY] != cls.__name__:
        raise SerializationError(
            f"{cls.__qualname__}: $type is {mapping[_TYPE_KEY]!r}, expected {cls.__name__!r}"
        )
    version = mapping[_VERSION_KEY]
    if type(version) is not int:
        raise SerializationError(f"{cls.__qualname__}: schema_version must be int")
    raw_fields = mapping[_FIELDS_KEY]
    if not isinstance(raw_fields, Mapping):
        raise SerializationError(f"{cls.__qualname__}: 'fields' must be a JSON object")
    fields: _FieldsDoc = dict(cast("Mapping[str, object]", raw_fields))
    if version > current:
        raise SerializationError(
            f"{cls.__qualname__}: document schema_version {version} is newer than "
            f"the supported version {current}"
        )
    while version < current:
        migration = _MIGRATIONS.get((cls, version))
        if migration is None:
            raise SerializationError(
                f"{cls.__qualname__}: no migration from schema_version {version} to {version + 1}"
            )
        fields = migration(fields)
        version += 1
    if not dataclasses.is_dataclass(cls):
        raise SerializationError(f"{cls.__qualname__} is not a dataclass")
    names = tuple(field.name for field in dataclasses.fields(cls))
    unknown = set(fields) - set(names)
    if unknown:
        raise SerializationError(
            f"{cls.__qualname__}: unknown fields {sorted(unknown)} (internal formats "
            "reject unknown fields)"
        )
    missing = set(names) - set(fields)
    if missing:
        raise SerializationError(f"{cls.__qualname__}: missing fields {sorted(missing)}")
    annotations = _field_annotations(cls)
    kwargs = {
        name: _decode(fields[name], annotations[name], f"{cls.__qualname__}.{name}")
        for name in names
    }
    return cast("Callable[..., object]", cls)(**kwargs)


@overload
def from_json(doc: object) -> object: ...
@overload
def from_json[T](doc: object, expected: type[T]) -> T: ...
def from_json(doc: object, expected: type[object] | None = None) -> object:
    """Decode a tagged JSON document back into its frozen foundation instance.

    With ``expected`` given, the document's ``$type`` must name exactly that class
    and the result is typed accordingly; without it, the class is resolved from
    the ``$type`` tag among registered serializable types.
    """
    if expected is None:
        if not isinstance(doc, Mapping):
            raise SerializationError("document must be a JSON object")
        mapping = cast("Mapping[str, object]", doc)
        tag = mapping.get(_TYPE_KEY)
        if type(tag) is not str:
            raise SerializationError("document has no $type tag")
        expected = _serializable_class_named(tag)
    return _decode_record_doc(cast(object, doc), expected)
