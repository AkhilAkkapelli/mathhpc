"""Deep-immutability substrate: ``FrozenDict``, the recursive validator, and path-precise errors.

Distinct concepts, deliberately kept separate (Addendum A1.1, Erratum E-4):

- Iteration order of ``FrozenDict`` is insertion order — a presentation convenience only.
- Semantic equality of ``FrozenDict`` is order-independent agreement with any ``Mapping``.
- Hash computation is order-independent and *lazy*: immutability does not imply
  hashability; hashability is checked only when ``hash()`` is requested (Decision A).
- Canonical serialization order is NOT defined here; it belongs to the Task 002
  serializer (sort over canonical serialized key bytes).
- Human-readable presentation order is a renderer choice.
"""

import dataclasses
import enum
from collections.abc import Iterable, Iterator, Mapping
from typing import Final, NamedTuple, NoReturn, assert_never, cast

__all__ = [
    "FrozenDict",
    "MutableFieldError",
    "assert_deeply_immutable",
    "validate_frozen_instance",
]


# --------------------------------------------------------------------------------------
# Path machinery (private): structured segments plus a precise human-readable rendering.
# NamedTuples, not dataclasses, so the frozen-type discovery scan has nothing to exempt.
# --------------------------------------------------------------------------------------


class _Attr(NamedTuple):
    name: str


class _Index(NamedTuple):
    i: int


class _Key(NamedTuple):
    key_repr: str


class _KeyObj(NamedTuple):
    key_repr: str


class _Elem(NamedTuple):
    elem_repr: str


_PathSeg = _Attr | _Index | _Key | _KeyObj | _Elem


def _render_segment(seg: _PathSeg) -> str:
    match seg:
        case _Attr(name):
            return f".{name}"
        case _Index(i):
            return f"[{i}]"
        case _Key(key_repr):
            return f"[{key_repr}]"
        case _KeyObj(key_repr):
            return f"[{key_repr}]#key"
        case _Elem(elem_repr):
            return f"{{{elem_repr}}}"
        case _:
            assert_never(seg)


class MutableFieldError(TypeError):
    """A mutable (or unrecognized) value was found inside supposedly immutable data.

    Carries the structured offending path (``path``) and a precise rendered form
    (``rendered``), e.g. ``root.payload.metadata['shape'][2]``.
    """

    def __init__(self, root: str, path: tuple[_PathSeg, ...], reason: str) -> None:
        self.path: tuple[_PathSeg, ...] = path
        self.reason: str = reason
        self.rendered: str = root + "".join(_render_segment(s) for s in path)
        super().__init__(f"{self.rendered}: {reason}")


# --------------------------------------------------------------------------------------
# FrozenDict
# --------------------------------------------------------------------------------------


class FrozenDict[K, V](Mapping[K, V]):
    """Immutable mapping with order-independent equality and lazy, cached hashing.

    - Construction accepts a ``Mapping`` or an iterable of ``(key, value)`` pairs;
      duplicate keys raise ``ValueError``.
    - Iteration order is insertion order (presentation only; never semantic).
    - ``fd == other`` is semantic mapping equality against any ``Mapping``.
    - ``hash(fd)`` is computed on first request over the *set* of entries and cached;
      an unhashable value raises ``TypeError`` at that point (immutability and
      hashability are distinct properties).
    - No mutating API exists; instance attributes are sealed after construction.
    """

    __slots__ = ("_index", "_items", "_hash")

    _index: dict[K, V]
    _items: tuple[tuple[K, V], ...]
    _hash: int | None

    def __init__(self, source: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> None:
        if isinstance(source, Mapping):
            pairs: list[tuple[K, V]] = list(cast("Mapping[K, V]", source).items())
        else:
            pairs = list(source)
        index: dict[K, V] = {}
        for key, value in pairs:
            if key in index:
                raise ValueError(f"duplicate key: {key!r}")
            index[key] = value
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_items", tuple(index.items()))
        object.__setattr__(self, "_hash", None)

    def __getitem__(self, key: K) -> V:
        return self._index[key]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[K]:
        return iter(k for k, _ in self._items)

    def __hash__(self) -> int:
        cached = self._hash
        if cached is None:
            try:
                cached = hash(frozenset(self._items))
            except TypeError as exc:
                raise TypeError(
                    f"FrozenDict is unhashable because an entry is unhashable: {exc}"
                ) from None
            object.__setattr__(self, "_hash", cached)
        return cached

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            mapping = cast("Mapping[object, object]", other)
            if len(self._items) != len(mapping):
                return False
            try:
                return all(mapping[key] == value for key, value in self._items)
            except KeyError:
                return False
        return NotImplemented  # pyright: ignore[reportReturnType]

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("FrozenDict is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("FrozenDict is immutable")

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._items)!r})"

    def __copy__(self) -> "FrozenDict[K, V]":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "FrozenDict[K, V]":
        return self

    def __reduce__(self) -> tuple[object, ...]:
        return (FrozenDict, (self._items,))


# --------------------------------------------------------------------------------------
# Deep-immutability validator
# --------------------------------------------------------------------------------------

_LEAF_TYPES: Final[frozenset[type]] = frozenset({type(None), bool, int, float, str, bytes})

_NAMED_MUTABLE: Final[dict[type, str]] = {
    list: "list",
    dict: "dict",
    set: "set",
    bytearray: "bytearray",
    memoryview: "memoryview",
}


def _is_frozen_dataclass_instance(value: object) -> bool:
    if isinstance(value, type) or not dataclasses.is_dataclass(value):
        return False
    params = getattr(type(value), "__dataclass_params__", None)
    return bool(getattr(params, "frozen", False))


def assert_deeply_immutable(value: object, *, root: str = "root") -> None:
    """Raise ``MutableFieldError`` if ``value`` contains anything mutable, at any depth.

    Accepted: exact-type leaves (``None``, ``bool``, ``int``, ``float``, ``str``,
    ``bytes``), ``Enum`` members, ``tuple``, ``frozenset``, ``FrozenDict`` and frozen
    dataclass instances (all recursed). Everything else — including non-frozen
    dataclasses and unknown custom classes — is rejected. Cycles are rejected;
    shared acyclic substructure is accepted.
    """
    _validate(value, root, (), set(), set())


def _validate(
    value: object,
    root: str,
    path: tuple[_PathSeg, ...],
    on_path: set[int],
    seen_ok: set[int],
) -> None:
    t = type(value)
    if t in _LEAF_TYPES or isinstance(value, enum.Enum):
        return
    vid = id(value)
    if vid in seen_ok:
        return
    if vid in on_path:
        raise MutableFieldError(root, path, "cycle detected")
    named = _NAMED_MUTABLE.get(t)
    if named is not None:
        raise MutableFieldError(root, path, f"mutable container: {named}")
    on_path.add(vid)
    try:
        if isinstance(value, FrozenDict):
            mapping = cast("FrozenDict[object, object]", value)
            for key, val in mapping.items():
                _validate(key, root, (*path, _KeyObj(repr(key))), on_path, seen_ok)
                _validate(val, root, (*path, _Key(repr(key))), on_path, seen_ok)
        elif isinstance(value, tuple):
            elements = cast("tuple[object, ...]", value)
            for i, elem in enumerate(elements):
                _validate(elem, root, (*path, _Index(i)), on_path, seen_ok)
        elif isinstance(value, frozenset):
            members = cast("frozenset[object]", value)
            for elem in members:
                _validate(elem, root, (*path, _Elem(repr(elem))), on_path, seen_ok)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            if not _is_frozen_dataclass_instance(value):
                raise MutableFieldError(root, path, "mutable (non-frozen) dataclass")
            for field in dataclasses.fields(value):
                _validate(
                    getattr(value, field.name), root, (*path, _Attr(field.name)), on_path, seen_ok
                )
        else:
            raise MutableFieldError(
                root, path, f"type {t.__qualname__} is not registered as immutable"
            )
    finally:
        on_path.discard(vid)
    seen_ok.add(vid)


def validate_frozen_instance(obj: object) -> None:
    """Validate every *field value* of a frozen dataclass instance.

    Deliberately does not validate ``obj`` itself as a node: ``__post_init__`` runs
    once at construction and validation only inspects already-constructed field
    values, so this cannot re-enter any ``__post_init__``. The rendered error root
    is the instance's class name.
    """
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("validate_frozen_instance expects a dataclass instance")
    if not _is_frozen_dataclass_instance(obj):
        raise TypeError(f"{type(obj).__qualname__} is not a frozen dataclass")
    root = type(obj).__qualname__
    for field in dataclasses.fields(obj):
        _validate(getattr(obj, field.name), root, (_Attr(field.name),), set(), set())
