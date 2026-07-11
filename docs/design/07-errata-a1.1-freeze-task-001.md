# Addendum A1.1 Errata · Foundation v1.0 Freeze · Task 001 Implementation Specification

---

# Part I — Addendum A1.1 Errata (exactly four corrections)

**E-1 (amends Addendum A1 §3, claim C1′).** The perturbed-SPD claim is parameterized by the licensing theorem; the generic γ-expression is removed from the claim statement:

```
C1′ = PerturbedSPD {
    original: A@instance,
    bound: BoundRef { specpack: "lapack-reference", theorem: "chol_backward_error_v1" }
}   domain = MathematicalReal (conclusion), licensed under IEEE754{b64} premises
```
The SpecPack theorem entry `chol_backward_error_v1` must state: norm (‖·‖₂ or ‖·‖F as licensed), bound expression, rounding assumptions (RNE, no double rounding), overflow assumption (no overflow occurred — INFO=0 does not by itself certify this; the theorem lists it as a premise), underflow/subnormal assumptions, matrix-size restriction (e.g. `c(n)·u < 1`), implementation applicability (reference LAPACK dpotrf; vendor variants require their own entry or an explicit applicability extension), library/version range, and required semantic domain of the premises. Bridge B1 now produces C1′ *with its `BoundRef` attached*, and the bridge obligation includes discharging (or guarding, or `Assumed`-listing) each theorem premise. **Adopted invariant:** *a successful numerical algorithm may establish only the exact postcondition licensed by its specific SpecPack theorem and satisfied bridge obligations.* Exact mathematical `SPD(A)` is never claimed from finite-precision DPOTRF success. Task-027 acceptance (Addendum A1 §6.6) gains: the provenance rendering of C1′ must display its `BoundRef`.

**E-2 (amends Addendum A1 §4, Failed branch).** The bit-for-bit reproduction claim for the anchor under detected aliasing is retracted. Corrected semantics:

```
Failed alias guard ⇒
  · the optimized GEMM replacement is NOT licensed (the noalias premise of the
    fortran-language-semantics SpecPack claim is violated at this instance);
  · the retained anchor implementation executes ONLY if the configured execution
    policy permits (alias_violation_policy = execute_anchor (default) | fail);
  · no portable semantic guarantee is made beyond the implementation's documented
    behavior for the original, nonconforming case — the report says exactly this;
  · the violated alias precondition (which pair, the overlapping intervals) is
    reported explicitly.
```
Case discipline: *conforming caller* → the language-rule premise is admissible and the optimized arm may be licensed; *nonconforming caller detected* → optimized arm not licensed; *anchor available* → executes per policy; *strict mode* (`alias_violation_policy = fail`) → explicit `RuntimeFailure.Abort` with the precondition violation. The general Fortran-language SpecPack claim is **not** globally demoted by an instance-level conformance violation: the claim quantifies over conforming programs, and a nonconforming instance is outside its scope — recorded as a scope-mismatch event, not a counterexample to the claim.

**E-3 (amends Addendum A1 §2 invariant scheduling and §6.7).** Frozen-type invariant testing is incremental. Task 001 implements the *generic* deep-immutability infrastructure and tests it against representative fixture types only. Every subsequent task that introduces a frozen foundation type must register it in the frozen-type test registry in the same PR; CI parametrizes the invariant suites (construction immutability, nested mutation, wrong-ID-kind, JSON round-trip, hash stability, schema-version compatibility — each applied where applicable per registry-entry capabilities) over the registry. Merge rule: **no frozen foundation type without a registry entry.** The Addendum A1 phrase "every frozen type" in invariants 1/3/4 is read as "every *registered* frozen type", with the discovery test making registration itself unskippable.

**E-4 (amends Addendum A1 §2, `frozendict`).** The immutable mapping does **not** require Python-orderable keys, and its iteration order is not canonical. Four concepts are documented as distinct and are never conflated: (1) **semantic equality** — same key/value pairs, order-insensitive; (2) **canonical hashing** — order-insensitive, derived from the item *set*; (3) **canonical serialization order** — defined by the serializer (Task 002) as ordering on *canonical serialized key bytes*, independent of Python key ordering or insertion order; (4) **presentation order** — whatever a renderer chooses for humans (repr uses insertion order). `FrozenDict` provides (1), deep-immutability participation, and (2) where values are hashable; it deliberately provides neither (3) nor (4).

---

# Part II — Freeze declaration

**Foundation v1.0 status: FROZEN.**

The frozen corpus is: the seventeen foundational principles, the Plan↔Runtime boundary as corrected by Addendum A1 §1, the frozen types and invariants of the implementation specification as corrected by Addendum A1 and this errata. Any future foundation modification proceeds exclusively through the Foundation Change Request process and must be justified by concrete implementation evidence of a blocking flaw not solvable by adapter, extension point, implementation-local mechanism, or compatible refinement. The project is in implementation mode.

---

# Part III — Task 001 Implementation Specification

## 1. Exact Task 001 objective

Task 001 creates **the minimal trustworthy immutable substrate on which frozen Foundation v1.0 records can later be constructed**: an immutable mapping (`FrozenDict`), a deep-immutability validator with exact path reporting (`assert_deeply_immutable`, `MutableFieldError`), the canonical frozen-dataclass validation pattern (`validate_frozen_instance`), strongly distinct ID dataclasses (`ClaimId`, `EvidenceId` as real foundation types; the pattern for all later IDs), and the frozen-type registry with its first parametrized invariant tests over representative fixtures.

It deliberately does **not** establish: any semantic content (no `SemanticDomain`, `Claim`, `Evidence`, …), serialization/canonical ordering (Task 002), schema versions/migrations (Task 002), or any compiler behavior.

**Completion criteria:** every item of §15's checklist is yes; §14's CI gates pass from a clean checkout; the public API is exactly §13.

## 2. Exact repository files

Project/package name: **`mathhpc`** (as in the accepted spec §15).

```
pyproject.toml                      # project metadata, deps, pytest config, ruff config
pyrightconfig.json                  # strict mode; excludes tests/typecheck from the normal run
src/mathhpc/__init__.py             # package marker; version string only
src/mathhpc/py.typed                # PEP 561 marker so pyright treats the package as typed
src/mathhpc/foundation/__init__.py  # the public API re-exports of §13 — nothing else
src/mathhpc/foundation/immutable.py # FrozenDict, MutableFieldError, FieldTypeError,
                                    # assert_deeply_immutable, validate_frozen_instance,
                                    # register_immutable_leaf, path machinery (private)
src/mathhpc/foundation/ids.py       # ClaimId, EvidenceId, _check_u64 (private helper)
src/mathhpc/foundation/registry.py  # RegistryEntry, register_frozen_type,
                                    # registered_frozen_types, RegistrationError,
                                    # _IMPLEMENTATION_LOCAL exemption set, discovery helper
tests/foundation/__init__.py
tests/foundation/fixtures.py        # representative frozen fixture dataclasses, registered here
tests/foundation/test_frozendict.py # unit tests for FrozenDict
tests/foundation/test_immutable.py  # unit tests for validator, pattern, paths, cycles
tests/foundation/test_ids.py        # unit tests for ID semantics
tests/foundation/test_registry.py   # registration rules + discovery gate
tests/foundation/test_properties.py # all Hypothesis property tests (§9)
tests/typecheck/invalid_id_assignment.py   # deliberate static errors + expect-error markers
tests/typecheck/test_pyright_negatives.py  # harness: runs pyright --outputjson, matches markers
.github/workflows/ci.yml            # the §14 gates
```

No other files. No empty packages for later tasks.

## 3. Exact Python dependencies

| Dependency | Kind | Min version | Why | Avoidable? |
|---|---|---|---|---|
| Python | runtime | 3.12 | `slots=True` dataclasses mature; modern `match`/typing | no |
| *(none)* | runtime | — | Task 001 is stdlib-only at runtime | — |
| pytest | test | 8.0 | test runner, parametrization over the registry | only by writing a runner |
| hypothesis | test | 6.100 | recursive structure generation + path-tracked mutant splicing (§9); hand-rolled generators would be weaker and larger | technically, at real cost |
| pyright | dev | 1.1.390 | strict static gate + the negative-test harness (JSON output) | no — it is the exhaustiveness/ID-misuse mechanism |
| ruff | dev | 0.6 | format + lint, one tool | yes; kept for hygiene at near-zero cost |

**External `frozendict` package: not used.** Decision: implement internally (~90 lines). Reasons: we require a specific hash policy (order-insensitive, lazy, tuple-like raise-on-unhashable-value), a specific equality policy versus arbitrary `Mapping`s, `__setattr__` hardening, pickle behavior under our control, first-class participation in the deep validator, and precise typing under pyright strict — auditing and pinning a third-party package for this costs more than the file.

## 4. Exact `FrozenDict` API

```python
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

class FrozenDict(Mapping[K, V]):
    def __init__(self, data: Mapping[K, V] | Iterable[tuple[K, V]] = (), /, **kwargs: V) -> None
    # Supported (Mapping ABC + explicit):
    #   fd[key] · len(fd) · iter(fd) · key in fd · fd.get/keys/values/items
    #   fd == other · hash(fd) · repr(fd) · copy.copy(fd) · pickle
```

- **Generic typing:** `FrozenDict[K, V]`, `K` bound `Hashable`. `**kwargs` construction implies `K = str` for those entries; pyright enforces this at call sites.
- **Construction:** materializes a private `dict`; keys must be hashable (a `TypeError` from `dict()` propagates). **Duplicate keys: last wins** (plain `dict` semantics; positional data first, then `kwargs`), documented — construction is not the place to police duplicates.
- **Nested values:** *not* validated at construction. `FrozenDict` is a container, not a policy; deep immutability is `assert_deeply_immutable`'s job at the frozen-record boundary. (Consequence: a `FrozenDict` holding a `list` is constructible, unhashable, and will be rejected by the validator wherever it matters.)
- **Equality (semantic):** order-insensitive. Equal to another `FrozenDict` or any `Mapping` with equal item sets; `NotImplemented` for non-mappings.
- **Hash (canonical):** `hash(frozenset(items))`, computed lazily, cached; order-insensitive by construction; **raises `TypeError` if any value is unhashable** — exactly the `tuple`-containing-`list` behavior, so the hash-equality law holds wherever `hash` is defined.
- **Iteration:** insertion order, stable, and explicitly **neither semantic nor canonical** (E-4). Canonical serialization order (canonical serialized key bytes) is the Task-002 serializer's concern; `FrozenDict` exposes no ordering hook.
- **repr:** `FrozenDict({'a': 1, 'b': 2})` in insertion order — presentation only.
- **Copy:** `__copy__` returns `self` (immutable). `__reduce__` returns `(FrozenDict, (dict(self),))`, giving correct **pickle** and a working `deepcopy` (which then deep-copies values via the reduce path).
- **Mutation hardening:** no mutating methods exist (`fd[k] = v` → `TypeError`: no `__setitem__`); `__slots__` plus an overriding `__setattr__`/`__delattr__` that always raise `AttributeError` (construction uses `object.__setattr__`).
- **JSON:** not implemented in Task 001 (serializer is Task 002); `json.dumps(fd)` fails today, by design.

## 5. Exact deep-immutability validator

```python
def assert_deeply_immutable(value: object) -> None   # raises MutableFieldError
class MutableFieldError(TypeError):
    path: str          # e.g. "root.payload.metadata['shape'][2]"
    offender: type
    reason: str        # "mutable builtin" | "non-frozen dataclass" |
                       # "type not registered as immutable" | "reference cycle detected"
```

**Recursive rules (in check order):**

| Value | Behavior |
|---|---|
| `None`, `bool`, `int`, `float`, `str`, `bytes` | accept (leaf) |
| `Enum` member | accept the member, **then recurse into `member.value`** (Enum payloads must be immutable too) |
| type in the extension registry (`register_immutable_leaf(tp)`) | accept (leaf) — the explicit hook for future `ArtifactRef`-like opaque immutables |
| `list`, `dict`, `set`, `bytearray`, `memoryview` | **reject**, reason "mutable builtin" |
| `tuple` | recurse per element, path `[i]` |
| `frozenset` | recurse per element, path `{elem!r}` (no index exists) |
| `FrozenDict` | recurse into every key (path `.key(K!r)`) and value (path `[K!r]`) |
| frozen dataclass instance | recurse per field, path `.fieldname` |
| non-frozen dataclass instance | **reject**, reason "non-frozen dataclass" |
| anything else (incl. NumPy arrays — NumPy is not a dependency and needs no special case) | **reject**, reason "type not registered as immutable" — default-deny |

**Cycles: rejected.** Two `id()` sets: `on_stack` (recursion stack — revisit ⇒ `MutableFieldError` with reason "reference cycle detected"; a cycle in allegedly immutable data implies post-construction mutation via `object.__setattr__` or C-level tricks, which is exactly what this substrate exists to refuse) and `done` (already-validated — shared sub-objects in a DAG are validated once, keeping the walk linear).

**Path representation:** internally a tuple of segment strings starting at `"root"`; rendered by concatenation. Segment grammar: attribute `.name`; sequence index `[2]`; mapping value `['shape']` (key `repr`); mapping key `.key('shape')`; frozenset element `{3}`. Example rendering matches the required form: `root.payload.metadata['shape'][2]`.

## 6. Exact frozen-dataclass validation pattern

```python
@dataclass(frozen=True, slots=True)
class Example:
    claim: ClaimId
    tags: tuple[str, ...]
    def __post_init__(self) -> None:
        validate_frozen_instance(self)
```

`validate_frozen_instance(obj)` is provided by Task 001 and differs from `assert_deeply_immutable` in two ways, both required for constructor use:

1. **No self-recursion hazard, no re-walk blowup.** It walks *fields*, never re-entering `__post_init__` (the walk is `getattr`-based and constructs nothing). Instances of **registered** frozen foundation types encountered as field values are treated as already-validated leaves — they ran their own `__post_init__` at construction — so total validation cost across a program is linear, while `assert_deeply_immutable` (used by tests and the registry suite) remains the full-depth belt-and-braces walk.
2. **Runtime field-type check for plain-class annotations.** For each field whose resolved annotation is a plain class (`ClaimId`, `FrozenDict`, …), it checks `isinstance` and raises `FieldTypeError(TypeError)` (path, expected, actual) on mismatch — this is the runtime half of wrong-ID-kind rejection. Generic/union annotations are skipped at runtime (pyright strict owns those statically); the limitation is documented. Type hints are resolved once per class and cached.

No decorators, no metaclasses: the pattern is the two lines above, enforced socially by the registry suite (every registered type's fixture is constructed, so a type that forgot the `__post_init__` call fails the nested-mutation invariant test).

## 7. Exact ID type strategy

Task 001 ships **two real foundation ID types** — `ClaimId` and `EvidenceId` — because the wrong-kind tests need two genuinely distinct kinds and both are guaranteed stable consumers from Task 003 onward. No other IDs are defined; later tasks add each ID in the PR introducing its record, following this exact pattern (explicit classes, no dynamic factory — pyright must see each class):

```python
@dataclass(frozen=True, slots=True)
class ClaimId:
    value: int
    def __post_init__(self) -> None:
        _check_u64(self.value, "ClaimId")     # rejects bool (explicitly), non-int, <0, >= 2**64
```

- **Representation:** one `int` field with u64 range semantics; `bool` rejected explicitly (it is an `int` subclass).
- **Equality:** dataclass-generated; returns `NotImplemented` across classes, so `ClaimId(1) == EvidenceId(1)` is `False` (tested). No inheritance between ID types, ever.
- **Hashing:** dataclass-generated from fields. `hash(ClaimId(1)) == hash(EvidenceId(1))` is permitted — hash collision across kinds is legal because equality already separates them; the hash-equality law is unaffected. Documented, tested as documentation.
- **Ordering: none.** `order=False`; `ClaimId(1) < ClaimId(2)` raises `TypeError` (tested). IDs are opaque names, not numbers.
- **Serialization:** Task 002's responsibility; no `to_json` here.
- **repr:** dataclass default, `ClaimId(value=17)`.
- **Static misuse** (`consume_claim(EvidenceId(...))`) is a pyright `reportArgumentType` error — enforced by the §10 negative harness. **Runtime misuse** (wrong ID kind stored in a frozen dataclass field annotated with a plain ID class) raises `FieldTypeError` from `validate_frozen_instance` (§6.2).

## 8. Frozen-type registry

```python
@dataclass(frozen=True, slots=True)
class RegistryEntry:
    tp: type
    fixture_factory: Callable[[], object]     # returns one valid representative instance
    # capability flags (serializable, schema_version, …) are added by Task 002+;
    # RegistryEntry is implementation-local, so extending it later is free.

class RegistrationError(Exception): ...

def register_frozen_type(tp: type, *, fixture_factory: Callable[[], object]) -> type
def registered_frozen_types() -> Mapping[type, RegistryEntry]     # read-only view
```

- **Who registers, when:** the module that defines a frozen foundation type calls `register_frozen_type` immediately after the class definition (plain explicit call at module top level — visible in the diff, no import hooks, no metaclass magic). Task 001's fixture types register in `tests/foundation/fixtures.py`; `ClaimId`/`EvidenceId` register in `ids.py`.
- **Registration validation:** `tp` must be a dataclass with `frozen=True` and `__slots__`; `fixture_factory` required. Violations raise `RegistrationError`.
- **Duplicate registration:** always an error (`RegistrationError`) — re-registration is a symptom of import confusion, not a feature.
- **CI discovery of unregistered types:** `test_registry.py` walks the `mathhpc.foundation` package (via `pkgutil` + import), collects every frozen dataclass *defined* there, and asserts each is either registered or listed in the explicit `_IMPLEMENTATION_LOCAL` exemption set in `registry.py` (Task 001 exemptions: `RegistryEntry` itself). Fixtures in `tests/` are covered because the property/invariant suites parametrize over `registered_frozen_types()` — an unregistered fixture simply gets no coverage and the fixture module's own assertion (`assert Fx in registered_frozen_types()`) fails.
- **Fixtures/factories:** associated with entries from day one (the invariant suites need instances); Hypothesis strategies per type may be attached in later tasks as an added optional field.
- **Incremental growth (E-3):** the invariant suites are written once, parametrized over the registry; every later frozen type inherits the full battery by registering.

## 9. Property-based testing strategy (Hypothesis — justified by §3)

Strategies (in `test_properties.py`):

```python
leaves       = st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text() | st.binary()
hashables    = leaves | st.tuples(leaves, ...)                     # for frozenset elements / dict keys
immutables   = st.recursive(
                 leaves,
                 lambda ch: st.lists(ch, max_size=4).map(tuple)
                          | st.frozensets(hashables, max_size=4)
                          | frozendicts(keys=st.text() | st.integers(), values=ch),
                 max_leaves=25)
mutants      = st.sampled_from([[1], {"k": 1}, {1, 2}, bytearray(b"x"), MutableFixture(0)])
infected     = composite: recursively build an immutable skeleton, splice one drawn mutant
               at one drawn position, RETURN (value, expected_path_segments)   # path built top-down
```

Exact properties:

| # | Property |
|---|---|
| P1 | For any generated `FrozenDict`: item assignment raises `TypeError`; `pop/update/setdefault/clear/__delitem__` do not exist (`AttributeError`); `fd._d = {}` and any attribute set/delete raise `AttributeError`. |
| P2 | For any item list and any two permutations of it: the two `FrozenDict`s are `==` and their hashes are equal (order-insensitivity of both equality and hash). |
| P3 | Hash-equality law: for generated pairs `(a, b)` of hashable foundation values (FrozenDicts, IDs, fixtures): `a == b` implies `hash(a) == hash(b)`. |
| P4 | `assert_deeply_immutable` accepts every value drawn from `immutables`. |
| P5 | For every `(value, path)` from `infected`: the validator raises `MutableFieldError`, and `err.path == render(path)` exactly — at arbitrary generated depth (subsumes the "finds mutants at any depth" requirement). |
| P6 | For every registered frozen type's fixture: `assert_deeply_immutable(fixture)` passes; every field-mutation attempt (`setattr`) raises `FrozenInstanceError`/`AttributeError`; constructing the fixture type with one field replaced by a drawn mutant raises `MutableFieldError` (or `FieldTypeError` where the annotation is a plain class). |
| P7 | Wrong-ID-kind: for the fixture type with a `ClaimId`-annotated field, constructing it with an `EvidenceId` of any value raises `FieldTypeError`; and `ClaimId(n) != EvidenceId(n)` for all generated `n`. |
| P8 | Cycle policy: a frozen fixture forced into a self-cycle via `object.__setattr__` makes the validator raise `MutableFieldError` with reason "reference cycle detected"; DAG sharing (same immutable sub-object reachable twice) passes. |

## 10. Static typing tests

`pyrightconfig.json`: `"typeCheckingMode": "strict"`, `include: ["src", "tests"]`, `exclude: ["tests/typecheck"]` (those files contain deliberate errors), `pythonVersion: "3.12"`, `reportMissingTypeStubs: true`.

**Negative tests — expected-error comments matched by diagnostic rule (chosen as least brittle):** golden full-output matching breaks on every pyright upgrade; bare rule-code matching anywhere in the file is too loose. The harness (`test_pyright_negatives.py`) runs `pyright --outputjson tests/typecheck/invalid_id_assignment.py`, and asserts a **one-to-one match between diagnostics and marker comments by (line, rule)**: every line carrying `# expect-error: reportArgumentType` produces exactly one error with that rule, and no unmarked line produces any error.

```python
# tests/typecheck/invalid_id_assignment.py
def consume_claim(x: ClaimId) -> None: ...
consume_claim(EvidenceId(7))          # expect-error: reportArgumentType
c: ClaimId = EvidenceId(7)            # expect-error: reportAssignmentType
```

## 11. Exact unit tests (complete list)

`test_frozendict.py`: `test_frozendict_lookup_and_contains` · `test_frozendict_len_and_iter_insertion_order` · `test_frozendict_kwargs_and_pairs_construction` · `test_frozendict_duplicate_keys_last_wins` · `test_frozendict_equality_ignores_insertion_order` · `test_frozendict_equality_with_plain_mapping` · `test_frozendict_not_equal_to_non_mapping` · `test_frozendict_hash_matches_equality` · `test_frozendict_hash_order_insensitive` · `test_frozendict_hash_raises_on_unhashable_value` · `test_frozendict_has_no_mutating_api` · `test_frozendict_setattr_and_delattr_blocked` · `test_frozendict_repr_presentation_order` · `test_frozendict_copy_returns_self` · `test_frozendict_pickle_roundtrip_equal`

`test_immutable.py`: `test_validator_accepts_allowed_leaves` · `test_validator_accepts_nested_immutable_composites` · `test_validator_rejects_mutable_builtins` (parametrized: list/dict/set/bytearray/memoryview) · `test_validator_rejects_mutable_dataclass` · `test_validator_rejects_unregistered_class` · `test_validator_accepts_registered_leaf_type` · `test_validator_recurses_into_enum_value` · `test_validator_reports_nested_path_exactly` · `test_validator_detects_cycle` · `test_validator_accepts_shared_dag_subobjects` · `test_post_init_pattern_rejects_mutable_field_at_construction` · `test_validate_frozen_instance_trusts_registered_children` · `test_field_type_error_on_plain_class_annotation_mismatch`

`test_ids.py`: `test_claim_id_equality_same_kind` · `test_claim_id_not_equal_to_evidence_id` · `test_cross_kind_hash_collision_permitted_documented` · `test_id_rejects_bool_negative_and_overflow` · `test_id_has_no_ordering` · `test_id_repr`

`test_registry.py`: `test_register_and_lookup_roundtrip` · `test_duplicate_registration_rejected` · `test_non_frozen_or_slotless_type_registration_rejected` · `test_missing_fixture_factory_rejected` · `test_all_registered_types_have_working_fixture_factory` · `test_registered_fixture_instances_pass_deep_validation` · `test_registry_view_is_read_only` · `test_no_unregistered_frozen_types_in_foundation_package`

`test_properties.py`: P1–P8 of §9. `test_pyright_negatives.py`: `test_negative_diagnostics_match_markers_one_to_one`.
