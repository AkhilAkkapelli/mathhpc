# Implementation Spec Addendum A1 — Pre-Task-001 Corrections

**Amends `hpc-mvp-implementation-spec.md`. Contains exactly: (1) the corrected Plan↔Runtime boundary, (2) the immutable-collection/type-safety strategy, (3) the DPOTRF claim chain, (4) the alias-guard semantics, (5) the reordered 30-task sequence with the vertical slice at Task 012, (6) changed acceptance criteria. Two pre-freeze interface amendments are made under the final-boundary-review window: `select_branch` is removed from `PlanRuntime`, and `InconclusiveReason` gains `UNSUPPORTED_LAYOUT`. Foundation v1.0 freezes on adoption of this addendum. Nothing else in the spec changes.**

---

## 1. Corrected Plan↔Runtime boundary `[FROZEN]`

**Decision: `select_branch` moves off `PlanRuntime` into a `BranchTable` owned by the immutable `Plan`.** Your invariant is adopted verbatim as **I-RT**: *guard evaluation belongs to the runtime; branch semantics belong to the immutable plan.* Retaining `select_branch` on `PlanRuntime` was rejected because any runtime implementation could then deviate from the compiled table — adaptive dispatch smuggled in through the trait — silently violating both plan immutability and D2. There is no offsetting benefit: when adaptivity is ever wanted, it must be expressed as *plan structure* (more guards, richer tables) so it stays compiled, verified, and auditable. "Adaptive dispatch" in the plan-content list is hereby defined as exactly that and nothing more.

```python
# ---- plan-owned branch semantics (pure data + pure function) ---------------
Outcome = Literal["P", "F", "I"]          # Passed / Failed / Inconclusive
Pat     = Literal["P", "F", "I", "*"]

@frozen class BranchTable:                                     # lives in Plan IR
    fork:   RuntimeDecisionId
    guards: tuple[GuardId, ...]            # fixed evaluation order
    rows:   tuple[tuple[tuple[Pat, ...], BranchId], ...]      # first-match wins
    def select(self, outcomes: tuple[Outcome, ...]) -> BranchId:
        # pure: no state, no clock, no data — a total function of `outcomes`
        for pat, arm in self.rows:
            if all(p == "*" or p == o for p, o in zip(pat, outcomes, strict=True)):
                return arm
        raise UnreachableRow(...)          # provably impossible, see construction checks

# ---- the runtime interface, now three methods ------------------------------
class PlanRuntime(Protocol):
    def eval_guard(self, g: GuardId, ctx: RuntimeContext) -> GuardResult: ...
    def emit(self, e: RuntimeEvent) -> None: ...
    def fail(self, f: RuntimeFailure) -> NoReturn: ...
```

**Construction-time checks on every `BranchTable` (plan verifier, compile time):** totality over all outcome vectors reachable under each guard's decisiveness typing (a refutation-only guard contributes only {F, I}); no arm whose legality depends on a guarded claim being Established is reachable through any row matching that guard's outcome as `I` or `*`-covering-`I`; unreachable rows are warnings; a fork with no fallback arm must carry `on_no_arm: RuntimeFailure.Abort{...}` **and** the compile-time note obligation (§2.3 of the spec) is checked here.

**The executor** (part of the generated driver / reference runtime harness, not of the `PlanRuntime` trait): for each fork, call `eval_guard` per the table's guard order, compute `arm = table.select(outcomes)`, `emit(BranchTaken(fork, arm, outcomes))`, proceed. `InconclusiveReason` (unchanged variants plus one): `BUDGET_EXHAUSTED | REFUTATION_ONLY_PROBE | NUMERICALLY_MARGINAL | DATA_UNAVAILABLE | UNSUPPORTED_LAYOUT`.

---

## 2. Immutable-collection and type-safety strategy `[FROZEN invariants; LOCAL mechanics]`

Python's `@dataclass(frozen=True)` is shallow and its unions are unsealed; the strategy makes deep immutability and exhaustiveness *tested invariants* rather than assumptions.

**Collections at frozen-type public boundaries.** Allowed field types are exactly: `int, float, bool, str, bytes, None`, `Enum`, other frozen foundation dataclasses, `tuple[T, ...]`, `frozenset[T]`, and `frozendict[K, V]` — a small hashable `Mapping` in `foundation/immutable.py` with canonical (key-sorted) iteration, so serialization and hashing are order-stable. `list`, `dict`, `set`, and NumPy arrays are **banned** in frozen types; bulk payloads live behind `ArtifactRef` (content hash + immutable `bytes`). Every frozen dataclass uses `frozen=True, slots=True` and runs `assert_deeply_immutable(self)` in `__post_init__` — a recursive whitelist validator, always on (foundation objects are small), raising `MutableFieldError` with the offending path.

**Tagged unions.** Unions of frozen dataclasses, consumed only through `match` with `typing.assert_never` in the final case; pyright `--strict` in CI makes non-exhaustive matches build failures. Because Python cannot seal a union, a reflection **registry test** pairs every published union (`EvidenceStatus`, `ClaimState`, `GuardResult`, `Feedback`, …) with every published fold/visitor over it and fails if a variant is unhandled — catching the "added a variant, missed a consumer" hole that `assert_never` alone misses when a `case _` was written.

**ID types.** Each ID (`ClaimId`, `EvidenceId`, `DecisionId`, `PlanId`, `GuardId`, …) is a distinct one-field frozen dataclass — not `NewType`, which erases at runtime. Static misuse is a pyright error; runtime misuse raises in `__post_init__` field-type validation. No implicit `int` conversion exists.

**Serialization and schema versioning.** Every frozen type: `to_json`/`from_json` with `schema_version: int` per schema; round-trip law `from_json(to_json(x)) == x` and `hash(from_json(to_json(x))) == hash(x)`. Unknown fields in foundation payloads are **rejected** (these are internal formats; silent tolerance hides drift); version bumps ship an explicit migration function, and a **golden archive** (`tests/golden/schemas/v*/`) of serialized instances from every past version must load through migrations in CI forever.

**The five tested invariants (CI-blocking from Task 001):**
1. *Nested-mutation rejection* — Hypothesis generates instances of every frozen type, attempts mutation along random attribute/element paths, asserts an exception on every path; plus construction with a smuggled `list`/`dict`/array raises `MutableFieldError`.
2. *Exhaustiveness* — pyright `--strict` clean (no `reportMatchNotExhaustive`), plus the union/consumer registry test.
3. *ID misuse* — constructing any foundation record with a wrong-kind ID raises; a pyright golden test asserts the static error is also reported.
4. *JSON round-trip* — equality + hash stability for Hypothesis-generated instances of every frozen type.
5. *Schema-version compatibility* — all archived goldens load via migrations; a new schema version without a migration function fails CI.

---

## 3. The exact DPOTRF claim chain `[FROZEN modeling]`

Three claims that the earlier spec's `P_spd` conflated, now separated by semantic domain, plus the perturbed forms the bridges actually deliver:

```
C_sym  : Property(A, symmetric)                          domain = MathematicalReal (of the matrix the
                                                          fp64 array denotes)   — from `assume`, Assumed
C1     : Property(A, positive_definite)                  domain = MathematicalReal        ← `infer` target
C1'    : ∃ΔA, ‖ΔA‖ ≤ γ_chol(n)·u·‖A‖ : spd_ℝ(A + ΔA)     domain = MathematicalReal (perturbed SPD)
C1''   : λ_min(A) > γ_chol(n)·u·‖A‖                       domain = MathematicalReal ("safely SPD")
C2     : dpotrf(A) completes with INFO = 0               domain = IEEE754{b64, RNE, E2}
                                                          (a fact about the FP algorithm's execution)
C3     : ContractSat(solve op, relative_residual ≤ 1e-12) domain = ContractNumeric(active contract)
```

Bridge obligations (all instantiated from the `lapack-reference` SpecPack's `Specified` backward-stability claims; the constants are the classical Cholesky bounds):

- **B1 (success bridge):** `Established(C2) ⇒ Established(C1′)`. Successful FP Cholesky proves the *perturbed* matrix is SPD — **it does not establish C1**, because a slightly indefinite A within the perturbation ball can factor.
- **B2 (failure bridge):** `Refuted-instance(C2) ⇒ Refuted(C1″)`. INFO>0 proves A is not *safely* SPD — **it does not refute C1**, because a barely-SPD A (λ_min below the rounding threshold) can fail FP factorization.
- **B3 (arm-legality bridge):** what the Cholesky *solve arm* actually needs is never C1. It is: `C2` (the computed factor L̂ exists) + SpecPack claims `L̂L̂ᵀ = A + ΔA₁` (dpotrf) and dpotrs backward stability (`x̂` solves `(A+ΔA₂)x̂ = b`) ⇒ residual guard `G_res` is well-posed ⇒ `C3` is decided **directly and per-instance** by `G_res`. The arm is gated on C2's guard outcome and C3's guard, with C1 nowhere in the legality chain.
- All of C2, C1′-via-B1, and refutations are scoped `Instance × Window(kill = first write to A)`; `C_sym` is consumed as a SpecPack *precondition* — if it is false, C2's statement silently concerns the symmetrized matrix that dpotrf actually read (tril), which is why `assume symmetric(A)` stays in the "unverified trust" section of the report.

**Success path (INFO = 0, residual passes):**
```
PotrfProbe → Passed
  E1: Evidence{C2, RuntimeChecked{PotrfProbe}, Instance×Window, IEEE754}      → C2: Established
  E2: Evidence{C1', StaticallyDerived{bridge B1}, provenance=[E1, specpack]}  → C1': Established
  C1: remains Unknown  (honestly — no admissible evidence for exact ℝ-SPD exists)
G_res → Passed
  E3: Evidence{C3, RuntimeChecked{G_res}, Instance×Window, ContractNumeric}   → C3: Established@instance
Provenance: positive_definite(A) [MathematicalReal]: Unknown ·
            perturbed-SPD (C1'): Established ← B1 ← dpotrf INFO=0 ·
            contract residual: Established (measured 3.1e-13)
```

**Failure path (INFO = k > 0):**
```
PotrfProbe → Failed(cex = {leading minor k})
  E1: Evidence{C2, RuntimeChecked counterexample}                             → C2: Refuted@instance
  E2: Evidence{C1'', StaticallyDerived{bridge B2}, provenance=[E1]}           → C1'': Refuted@instance
  C1: remains Unknown  (not refuted — B2 does not reach it)
BranchTable row (F, ·) → dsytrf/dsytrs arm (Auto/Prefer) or on_no_arm = Abort{cex} (Require/Use pin)
Provenance: notes explicitly: "exact SPD over ℝ neither established nor refuted;
            numerically-safe SPD refuted at this instance."
```

The `infer positive_definite(A)` campaign's honest lifetime answer is therefore: compile-time `Unknown`; runtime `Unknown` with `C1′` Established (success) or `C1″` Refuted (failure). The system never asserts mathematical SPD from a factorization — that sentence is now enforced by types, since B1's output claim is C1′, not C1.

---

## 4. Exact alias-guard semantics `[FROZEN semantics; LOCAL probe code]`

**Purpose.** Protect the optimized GEMM arm (dgemm on C, A, B) against aliasing that the Fortran-rule SpecPack claim assumes away. Only write-vs-read pairs matter: **(C,A) and (C,B) are checked; A–B overlap is harmless** (both read-only, dgemm-safe) and is not checked.

**Two overlap notions, kept distinct.** *Exact element overlap*: some array element's storage is shared. *Bounding-interval overlap*: the byte intervals `[base, base + span)` intersect. Interval disjointness ⇒ element disjointness (always sound for `Passed`). Interval intersection ⇒ element overlap **only for contiguous arrays**: two dense fp64 sequences whose byte ranges intersect necessarily share storage bytes (whole or partial elements — partial is aliasing too, and worse). For strided/non-contiguous sections, interval intersection proves nothing (interleaving), and the exact test is number-theoretic — out of MVP scope.

**Guard algorithm (decisive within its soundness domain):**
```
for X in {C, A, B}: if not IS_CONTIGUOUS(X)      → Inconclusive(UNSUPPORTED_LAYOUT)
for X:              base(X) via c_loc(X) — generated wrapper declares dummies
                    real(c_double), target, contiguous;  if unobtainable
                                                  → Inconclusive(DATA_UNAVAILABLE)
intervals: I(X) = [base(X), base(X) + 8·size(X))
if I(C)∩I(A) = ∅ and I(C)∩I(B) = ∅               → Passed(evidence: noalias(C,A)@inst, noalias(C,B)@inst)
else                                              → Failed(counterexample: the overlapping interval pair)
```
**Soundness domain, stated precisely:** the guard may return `Passed` **only** when all three arrays are verified contiguous, all base addresses were obtained, and both write-read interval pairs are disjoint — under which conditions interval disjointness is exact, so `Passed` is never issued without established non-overlap. `Failed` is likewise decisive *within this domain* (contiguous + intervals intersect ⇒ real storage sharing). Everything else — non-contiguous descriptors, unavailable addresses, any doubt — is `Inconclusive` with the specific reason. The MVP recovery subset (whole explicit/assumed-shape dummies) makes the contiguous case the common path; the `contiguous` dummy attribute means a non-contiguous actual triggers compiler copy-in/out, and since copy semantics interacting with aliasing are their own hazard, the wrapper *additionally* passes the caller's `IS_CONTIGUOUS` verdict computed on the actual arguments — belt and braces.

**Branch effects (the fork's table):**
```
(P) Passed        → dgemm arm.  RuntimeEvidence attaches to noalias claims,
                    Instance × Window(kill = wrapper return); corroborates the
                    Specified fortran-language-semantics claim.
(F) Failed        → anchor-loop arm + emit CounterexampleFound. The caller violated the
                    Fortran standard; the anchor reproduces the original program's behavior
                    bit-for-bit (it IS the original loop), which is the only honest semantics
                    for a non-conforming call. The Specified language-rule claim is NOT demoted
                    (it quantifies over conforming programs); the event is reported.
(I) Inconclusive  → anchor-loop arm (Unknown never enables the optimistic arm) + telemetry
                    with the reason; provenance distinguishes I from F explicitly.
```
Policy knob (recovery time): `alias_guard = allow_inconclusive_fallback` (default, as above) `| require_decisive` (reject recovery of kernels whose call sites cannot guarantee the decisive domain — for users who refuse silent fallback rates).

---

## 5. Reordered 30-task sequence — vertical slice at Task 012

Thin versions in Wave 1 respect every frozen invariant (evidence immutable and domain-tagged, decisions carry `depends_on` + epoch, provenance rendered from real records, contracts present); they defer *coverage*, never *discipline*.

**Wave 1 — the slice path (001–012):**

| ID | Title (deps) | Acceptance |
|---|---|---|
| 001 | **Immutability substrate**: `frozendict`, `assert_deeply_immutable`, ID dataclasses (—) | invariants 1 & 3 of §2 green |
| 002 | `SemanticDomain`, `SourceSpan`, IDs, JSON round-trip harness, `schema_version` scaffold (001) | invariant 4 green for these types |
| 003 | Minimal `Claim`/`ClaimKey`/`Evidence` — full `EvidenceStatus` union defined; producers for `StaticallyDerived`, `Specified` only (002) | equivalence-without-domain negative test |
| 004 | Minimal `FoundationStore`: append/state/snapshot; fold v0 = {Established, Refuted, Unknown} (003) | permutation-invariance property test green |
| 005 | `Contract` (D/A/E) + profiles `strict/faithful/fast`; profile-compatibility gate only (002) | meet-lattice unit tests |
| 006 | Minimal `Decision` + full `DecisionControl` union (Auto exercised) (004) | epoch + `depends_on` recorded and serialized |
| 007 | Math lexer + parser, slice subset: tensor decls + `C = matmul(A, B)` (002) | golden trees; span-precise errors |
| 008 | HIR minimal: `TensorType`, dim checks, `MatMul` node, printer (007) | shape negatives; snapshot |
| 009 | SpecPack schema + loader + `blas-reference/dgemm` (003, 005) | schema-valid load; contract profile parsed |
| 010 | Planner v0 (Auto): dgemm vs anchor candidates, profile gate, cost=flops, Decision (006, 008, 009) | dgemm chosen; decision cites claim states |
| 011 | Fortran backend + compile/run harness + provenance renderer v0 (010) | generated unit compiles & runs (gfortran+openblas) |
| 012 | **VS-1 — Test A end to end** (011) | parse→HIR→claims→decision→dgemm→execute→NumPy diff within the A2 bound→schema-valid report; e2e asserts the invariant checklist: every evidence object immutable & domain-tagged, decision has epoch, report reproduces the decision chain |

**Wave 2 — deepen (013–030):** 013 full `EvidenceStatus` producers + `Scope`/`Validity` + fold liveness · 014 five-state fold (Conditional, Contested, scope-intersection) + full Hypothesis suite · 015 remaining §2 invariant suites: nested-mutation fuzz across all types, union/consumer registry, schema-golden archive (invariants 2 & 5) · 016 **`BranchTable`/`Fork` in Plan IR + 3-method `PlanRuntime` + reference runtime** (§1) · 017 `Feedback` + `ExclusionCertificate` + point-exclusion replan loop · 018 directive lowering (assume/infer/require/prefer(@N)/use) · 019 machine probe + measured cost ordering · 020 dot/matvec/solve HIR ops + metadata · 021 lapack + fortran-language-semantics + vendor packs + install diff-check (adversarial FP) · 022 Julia backend + cross-target comparison · 023 control levels + Pareto prefs + canonical tie-break (Tests E-compile, G) · 024 property claims + effects/kill + frame rule (Test F) · 025 `infer` orchestrator + priced unlock · 026 guard library: **alias guard per §4**, QuadFormProbe, **PotrfProbe per §3**, residual; decisiveness typing · 027 SPD solve e2e with the §3 claim chain (Tests D, E-runtime, H) · 028 ANTLR F2018 + subset gate + resolution · 029 Loop IR + islpy domains/accesses/deps + arith-order goldens · 030 GEMM recognizer + obligations + γ/E bridges + strict rejection (Tests B, C).

Dependency shape: 001→…→012 linear; 013–017 fan out from 003/004/010; 018–027 need 012's running system; 028–030 need 012 + 013–017.

---

## 6. Changed acceptance criteria (delta only)

1. **New Task 001** (immutability substrate) with §2 invariants 1 & 3 as its acceptance; former task numbering shifted per §5.
2. **VS-1 moves from Task 021 → Task 012**, with the added *no-bypass checklist* assertion inside the e2e test (immutable + domain-tagged evidence; decision epoch; report/decision-chain reproducibility).
3. **Fold acceptance split**: Task 004 = three-state fold + permutation invariance; Task 014 = five states + scope-intersection + no-false-Contested + failed-proof⇒Unknown.
4. **Task 016 (was 010)**: `PlanRuntime` has exactly `eval_guard/emit/fail`; `BranchTable.select` is a method on frozen plan data; **new test**: a hostile `PlanRuntime` implementation cannot change the selected arm for fixed guard outcomes (arm identity asserted independent of the runtime object). Construction checks: totality under decisiveness typing; optimistic-arm-unreachable-via-`I`; `on_no_arm` + compile-time-note obligation.
5. **Task 026 alias guard**: property tests generate contiguous-disjoint / contiguous-overlapping (incl. partial-element offsets) / strided-interleaved cases; assertions: *never* `Passed` on any overlapping case (soundness), `Failed` exactly on contiguous intersection, `Inconclusive(UNSUPPORTED_LAYOUT|DATA_UNAVAILABLE)` otherwise; `InconclusiveReason.UNSUPPORTED_LAYOUT` added pre-freeze.
6. **Task 027 provenance**: the report must render C_sym, C1, C1′, C1″, C2, C3 as distinct claims with domains; acceptance includes: after a fully successful solve, `positive_definite(A) [MathematicalReal]` renders as **Unknown** and `C1′` as Established-via-B1 — the honest-Unknown assertion is a hard test, not prose.
7. **CI gate from Task 001**: pyright `--strict` and §2 invariants 1, 3, 4 are merge-blocking; invariants 2 and 5 become merge-blocking at Task 015.
