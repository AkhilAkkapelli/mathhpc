# MVP Implementation Specification — Foundation v1.0

**Status: Foundation v1.0 is FROZEN upon adoption of §2 (the typed Plan↔Runtime interface with `Inconclusive`). All 17 frozen principles of the freeze prompt are adopted verbatim as `docs/foundation-v1.0.md`. This document is the implementation specification: every decision below is either an application of the frozen foundation or an implementation-local choice, and is labeled as such. Foundation changes from here require a formal Foundation Change Request (FCR) per the governing principle.**

Output map to the 30 required items: §1 freeze (1) · §2 PlanRuntime (2, 15) · §3 scope + non-goals (3, 4) · §4 language (5) · §5 frontends (6, 7) · §6 IRs (8) · §7 core types (9, 10, 11) · §8 property system + infer (12) · §9 planner + control (13, 14) · §10 SpecPacks (16) · §11 verification + FP contracts (17, 18) · §12 Semantic Recovery Pipeline + demos (19) · §13 codegen (20) · §14 provenance (21) · §15 repository (22) · §16 testing + end-to-end tests (23, 24) · §17 milestones (25) · §18 first 30 tasks (26) · §19 vertical slice (27) · §20 risks (28) · §21 research vs engineering (29) · §22 tomorrow (30).

---

## 1. Foundation v1.0 freeze statement

The foundational architecture — the Claim+Evidence+Decision spine; semantic domains with bridge obligations; five-state claims; temporal/region-sensitive scopes; the AI-proposes/verifiers-decide boundary; three-tier rewrites; four-level user control; typed Pareto preferences with order-free tie-breaking; explicit search domains with evidence-backed region exclusion; evidence-aware canonicalization; three-axis contracts; typed upward feedback; honest replanning termination; kind-indexed placement domains; infer/discover campaigns; append-only evidence with commutative folds; semantic recovery as a first-class objective — is frozen as **Foundation v1.0**, completed by the Plan↔Runtime interface of §2.

Frozen means: the *types, invariants, and interfaces* in §2 and §7 marked `[FROZEN]` change only via FCR (scenario, responsible type/invariant, why no adapter suffices, smallest change, compatibility, migration, regression tests). Everything marked `[LOCAL]` is implementation-local and changes freely. The MVP's purpose is to attack the frozen set with real code.

---

## 2. Final typed Plan↔Runtime interface `[FROZEN]`

```python
# ---- identities ----------------------------------------------------------
class GuardId(NamedTuple):            plan: PlanId; idx: int
class BranchId(NamedTuple):           plan: PlanId; fork: int; arm: int
class RuntimeDecisionId(NamedTuple):  plan: PlanId; fork: int

# ---- context: what a guard may look at (nothing else) --------------------
@frozen
class RuntimeContext:
    instance: InstanceId              # hash of the runtime invocation (arg identities + shapes)
    shapes:   dict[str, tuple[int, ...]]
    handles:  dict[str, DataHandle]   # opaque array handles; guards access data ONLY through
                                      # compiler-emitted probe functions bound to the guard
    budget:   GuardBudget             # {max_flops, max_bytes, max_wall}

# ---- results --------------------------------------------------------------
@frozen
class RuntimeEvidence:                # the ONLY way runtime facts enter the evidence system
    claim: ClaimId                    # pre-registered at compile time; a guard cannot invent claims
    status: EvidenceStatus            # RuntimeChecked{...} or Measured{...} — nothing else constructible
    scope: Scope                      # DataExtent.Instance(instance) × ProgramExtent.Window(start_ev, kill_ev)
    domain: SemanticDomain            # inherited from the registered claim; carried, not chosen, at runtime
    payload: Payload                  # residual value, timing, counterexample data ...

class GuardResult:                    # tagged union
    Passed(evidence: RuntimeEvidence)                 # decisive positive
    Failed(counterexample: RuntimeEvidence)           # decisive negative
    Inconclusive(reason: InconclusiveReason)          # NOT evidence. Budget exhausted, probe
                                                      # refutation-only and found nothing, precision
                                                      # insufficient, data inaccessible.
class InconclusiveReason(Enum):
    BUDGET_EXHAUSTED; REFUTATION_ONLY_PROBE; NUMERICALLY_MARGINAL; DATA_UNAVAILABLE

# ---- events / failures -----------------------------------------------------
class RuntimeEvent:                   # buffered; flushed to the FoundationStore after the run
    PropertyDiscovered(ev: RuntimeEvidence)
    CounterexampleFound(ev: RuntimeEvidence)
    NumericalViolation(contract: ContractId, ev: RuntimeEvidence)
    PerformanceMeasurement(ev: RuntimeEvidence)
    BranchTaken(decision: RuntimeDecisionId, arm: BranchId, because: list[GuardId])
class RuntimeFailure:
    Abort { decision: RuntimeDecisionId, counterexample: RuntimeEvidence | None,
            explanation: ProvenanceRef }              # e.g. Require/Use pinned, no fallback
    ContractViolated { contract: ContractId, ev: RuntimeEvidence }

# ---- the interface ----------------------------------------------------------
class PlanRuntime(Protocol):
    def eval_guard(self, g: GuardId, ctx: RuntimeContext) -> GuardResult: ...
    def select_branch(self, d: RuntimeDecisionId,
                      outcomes: Sequence[tuple[GuardId, GuardResult]]) -> BranchId: ...
    def emit(self, e: RuntimeEvent) -> None: ...
    def fail(self, f: RuntimeFailure) -> NoReturn: ...
```

**Frozen semantics.**
1. **Plans are immutable at runtime.** `select_branch` is a *pure table lookup* compiled into the plan: `(fork, ordered guard outcomes) → arm`. Given the same outcomes it returns the same arm — determinism axis D2 is preserved through runtime dispatch by construction.
2. **`Inconclusive` maps to claim-state `Unknown`, never `Refuted`.** The branch table may therefore route `Inconclusive` only to arms whose legality does not depend on the guarded claim: the *fallback arm* if one exists. `Passed` routes to the optimistic arm; `Failed` routes to the fallback arm *and* emits the counterexample.
3. **No fallback + non-`Passed` outcome ⇒ `fail(Abort)`.** This situation exists only when a `require`/`use` pin forbade compiling a fallback; the compiler must have emitted a compile-time note ("failure mode is abort") when it built such a plan — the abort at runtime is the pre-announced consequence, never a surprise. `Inconclusive` and `Failed` both abort here, with different explanations (Unknown-not-established vs. refuted-with-counterexample).
4. **Runtime evidence is append-only and pre-registered.** Guards can only attach evidence to claims the compiler registered, with scope `Instance × Window`; the window's kill event is wired to the effect analysis (first write to the guarded object closes it). The telemetry buffer flushes into the FoundationStore post-run; nothing at runtime reads the store.
5. Every guard declares whether it is **decisive** (can return Passed and Failed: e.g. `dpotrf` info as an SPD decision) or **refutation-only** (can return Failed or Inconclusive, never Passed: e.g. a sampled quadratic-form probe `xᵀAx`). The compiler type-checks branch tables against this: an optimistic arm may not be reachable solely via a refutation-only probe.

---

## 3. Exact MVP scope and non-goals

### 3.1 Scope

- **Operations:** dot, matvec, matmul, SPD solve. Generic reduction `Σ` is included *only* as the einsum-style surface form that already denotes the four ops (`C[i,j] = Σ(k) A[i,k]·B[k,j]`) — no standalone general reduction op.
- **Input paths:** (A) native mathematical source → Typed Math HIR; (B) restricted Fortran → Semantic Recovery Pipeline → verified Math HIR. Both converge in one pipeline.
- **HPC scope (accepting the prompt's default split, with two justified adjustments):** IN: vendor BLAS/LAPACK selection via SpecPacks; flop/byte metadata per op; minimal machine description (cores, sockets, BLAS vendor, measured single/multi-thread dgemm rate + STREAM triad — one 15-minute probe script); plan selection; provenance. **Adjustment 1 (add, nearly free):** threaded-BLAS selection *is* the MVP's "OpenMP" — thread count + `OMP_PLACES/PROC_BIND` in the generated driver, decided from the machine description; generating parallel loops stays out. **Adjustment 2 (add as the single stretch autotune knob):** the crossover size below which the naive loop beats `dgemm` dispatch overhead — one measured, machine-keyed parameter exercising the tuning-DB key format (`CandidateSignature`) end to end. STRETCH: measured benchmark evidence feeding `Measured` status. DEFERRED: MPI, GPU, SUMMA, 2.5D, NUMA placement beyond thread pinning, roofline beyond the two probe numbers, fault tolerance, energy.
- **Contracts:** profiles `strict = ⟨D2, A1, E1⟩` (anchor-order-preserving), `faithful = ⟨D2, A2(classical γ-bounds), E2⟩` (default), `fast = ⟨D3, A3, E3⟩`. All three axes represented; only the gates these profiles exercise are implemented (reassociation-vs-A1, tree/threading-vs-D, E feature checks on rewrites).
- **Verification:** structural + shape checks, islpy affine dependence/access analysis, Z3 for integer/index side conditions, property-based differential testing (Hypothesis), runtime guards, bundled SpecPacks. **Condition estimation: omitted entirely** — the `require relative_residual(...)` contract is enforced by an a-posteriori residual guard, which is cheap, decisive, and instance-certified; nothing in the MVP's plan space needs κ(A). (This applies §9 of the prompt: the estimator-as-algorithm-selection design exists in the foundation docs; the MVP simply has no consumer for it.)
- **AI:** none required. A deterministic GEMM/matvec/dot pattern recognizer is the primary recognizer; an optional `Recognizer` plug-in interface accepts external hypothesis producers (an LLM behind it later) emitting `AiInferred` — the pipeline is exercised by a *mock* recognizer in tests. AI optional, verification mandatory, compiler fully functional offline.

### 3.2 Final non-goals (explicit)

General C++/Python/Julia/LLVM-IR recovery; arbitrary Fortran (incl. pointers, EQUIVALENCE, COMMON, coarrays, non-affine bounds, irreducible control flow, unknown calls); MPI/multi-GPU/SUMMA/2.5D codegen; fault tolerance; energy; Lean in the compilation path; training or fine-tuning any model; distributed autotuning; general mathematical notation; general theorem proving; symbolic algebra; a polyhedral *scheduling* framework (islpy is used for *analysis* only); sparse/tensor/FFT/PDE/AD ops; cryptographic SpecPack signing (schema field reserved); condition estimation; mixed precision; GPU anything.

---

## 4. Implementation language: **Python 3.12+, strictly typed** `[LOCAL]`

| Criterion | Rust | C++ | Julia | **Python (chosen)** |
|---|---|---|---|---|
| ADTs/pattern matching/immutability for the spine | excellent | poor | good (immutable structs, MLStyle) | good: `@dataclass(frozen=True)`, `match`, `assert_never` exhaustiveness under pyright `--strict` |
| Parsing ecosystem incl. **your ANTLR Fortran-2018 grammar** | antlr-rust immature | ANTLR C++ ok, painful | no ANTLR target | **ANTLR Python target is first-class — the grammar is reused directly** |
| SMT | z3 bindings ok | ok | thin | **z3-solver: the best Z3 API that exists** |
| Polyhedral analysis | none good | isl C API | none | **islpy: mature isl bindings** |
| Differential testing vs references | manual | manual | native LA | **NumPy/SciPy reference oracle + Hypothesis property testing in-process** |
| LLVM/MLIR interop | good | native | n/a-ish | adequate — **and the MVP does not need it: codegen emits Fortran/Julia source calling BLAS** |
| Your listed experience | none | C only | yes | **yes** |
| Compiler runtime performance | best | best | good | worst — **and irrelevant: MVP compiles 4 ops; generated-code performance is BLAS's job** |
| Long-term maintainability | best | ok | risky | honest weakness — mitigated below |

**Decision: Python.** The MVP's product is a *verdict on the architecture*, and the dominant risks (§20) are semantic — scope/validity bugs, fold correctness, FP-equivalence handling — which are attacked fastest in the language where the developer and the analysis ecosystem (z3, islpy, Hypothesis, NumPy oracle, ANTLR) are strongest. Mitigations for the honest weakness: (a) pyright `--strict` in CI from task 1, frozen dataclasses everywhere, tagged unions with exhaustive `match`; (b) **every `[FROZEN]` type gets a versioned JSON schema and round-trip serialization from day one** — the foundation is a language-independent data model, so a post-MVP port (Rust is the natural successor if the architecture survives) migrates data and tests, not just intentions; (c) module boundaries mirror the foundation (one package per frozen subsystem, §15). Not chosen: Rust (learning a language while falsifying an architecture doubles the risk being tested), Julia (kept as a *codegen target*, where it shines), C++ (ergonomics tax on exactly the spine code that must be correct).

---

## 5. Frontend decisions

### 5.1 Native mathematical frontend: **hand-written recursive descent + Pratt expressions** `[LOCAL]`

| | ANTLR | tree-sitter | **recursive descent (chosen)** | PEG/combinators |
|---|---|---|---|---|
| Fit for a small DSL (~40 productions) | heavy: grammar + generated visitor + build step | built for editors/incremental, error *recovery* over error *messages* | 2–3 days, zero deps, total control | fine, but adds a library where none is needed |
| Error messages with spans | template-y | weak | **best — hand-crafted, span-precise** | ok |
| Unicode `ℝ`, `Σ`, `∈` + ASCII aliases | fine | fine | trivial in the lexer | fine |
| Grammar evolution | regenerate | regenerate | edit a function | edit |

ANTLR familiarity is real but the payoff is on the *Fortran* side where the grammar already exists; for a small math DSL the recursive-descent parser is less total work than maintaining a second grammar file, and error quality is a user-facing feature of this language. Lexer accepts Unicode with mandatory ASCII equivalents (`ℝ`/`Real`, `Σ`/`sum`, `∈`/`in`, `×`/`*`); the parser is deterministic LL with Pratt precedence for expressions; ambiguity is prevented by construction and locked by golden tests.

### 5.2 Fortran frontend: **your ANTLR Fortran-2018 grammar (Python target) + a strict subset gate** `[LOCAL]`

| | Flang | LFortran | restricted new parser | **ANTLR F2018 grammar (chosen)** |
|---|---|---|---|---|
| Semantic info | full, via FIR/MLIR — enormous integration + C++ boundary + source-map excavation | good ASR, Python API — but a fast-moving external dependency owning our front door | full control, weeks of work, third parser to maintain | full *syntax* for free; we implement semantics only for the subset |
| Source mapping | hard through FIR | decent | trivial | **trivial — token spans native** |
| Effort to MVP | very high | medium, coupling risk | medium | **lowest, reuses your asset** |
| Honest failure on unsupported constructs | must suppress features | must gate ASR | by construction | **by construction: parse everything, gate explicitly** |

Architecture: parse full Fortran 2018 syntax → **subset gate** (whitelist walk; any node outside the subset ⇒ `Unsupported{construct, span}` and the unit is *left untranslated*, never partially guessed) → symbol/type resolution implemented only for the subset. Long-term, LFortran/Flang can slot behind the same `RecoveredLoopIR` interface — the frontend choice is deliberately not load-bearing.

**Exact supported subset (MVP):** one `subroutine` per unit; dummy args of `real(8)` rank-1/2 with explicit or assumed shape, `integer` scalars; locals: `real(8)` scalars, `integer` scalars; `do` loops with affine bounds and step `+1` (constant step ±c accepted, normalized); assignments whose subscripts are affine in loop indices and integer dummies; scalar temporaries; `implicit none` required; **no**: procedure calls, pointers, `equivalence`, `common`, `goto`/`exit`/`cycle`, I/O, `where`, array expressions (MVP: scalarized loops only), functions with side effects, `save`. Aliasing: Fortran's dummy-argument anti-aliasing rule (a dummy that is written must not be associated with any other accessible entity) is modeled as a claim from the bundled **`fortran-language-semantics` SpecPack** — `Specified{Bundled}` evidence, which is exactly what a language standard is; a belt-and-braces O(1) extent-overlap runtime guard is still emitted (it exercises PlanRuntime and defends against non-conforming callers), with the original loop as the fallback arm.

---

## 6. Exact IR design

Five IRs exist in the MVP. The seven long-term layers (Algorithm/Parallel/Distribution/Communication/Memory-Layout/Schedule/Hardware-Mapping IR) are **conceptual seams only**: each is represented by *decision slots and metadata fields on Plan IR nodes* (an `AlgorithmChoice` decision, a `threads` field, a `CandidateSignature`), not by empty layer scaffolding. When a future feature needs a real layer, the slot is where it grows.

### 6.1 Surface AST `[LOCAL]`
Nodes: `Program(decls, stmts)`, `TensorDecl(name, space: ℝ^shape, span)`, `Directive(kind ∈ {assume, infer, require, prefer, prefer@N, use, optimize}, payload, span)`, `Assign(lhs, expr, span)`, `SumExpr(index, range, body)`, `Call(solve|dot|…)`, `BinOp`, `Index(base, subscripts)`. Invariants: every node carries a `SourceSpan`; no name resolution; no types. Textual form = the source itself; golden tests pin the parse tree as s-expressions.

### 6.2 Typed Mathematical HIR `[FROZEN interfaces, LOCAL internals]`
```python
@frozen class HirOp:
    id: HirId
    kind: OpKind            # Dot | MatVec | MatMul | Solve | (recognized from SumExpr where applicable)
    operands: tuple[ValueRef, ...]
    result: ValueRef
    result_type: TensorType         # dtype=f64, shape: tuple[Dim,...]  (Dim = int | SymDim)
    domain: SemanticDomain          # the op's ARITHMETIC domain: IEEE754{b64, RNE, E-profile}
    contract: ContractId            # required ⟨D,A,E⟩ at this op (meet over consumers)
    provenance: ProvenanceRef       # → source span(s) or → RecoveredLoopIR region
```
Invariants: SSA (values immutable; surface-level re-assignment introduces new values + kill events for claims on the storage object — §8); shape consistency proven by the dim solver; every op has a `SemanticDomain` and provenance; directives are *not* HIR nodes — they become Claims (`assume/infer`), Contract atoms (`require residual…`), or Decision controls (`prefer/use`), each provenance-linked to the directive's span. Textual form:
```
%x: R^[n] = hir.solve %A: R^[n,n], %b: R^[n]
     {domain = ieee754<b64,rne,E2>, contract = faithful, prov = src(12:5-12:19)}
```

### 6.3 Recovered Loop IR `[LOCAL]`
```python
@frozen class LoopNest:
    domain: IslSet                  # { S[i,j,k] : 1<=i<=m ∧ 1<=j<=n ∧ 1<=k<=p }
    stmts: tuple[StmtInstance,...]  # body statements with IslMap access functions per array ref
    writes/reads: dict[Array, IslMap]
    deps: IslUnionMap               # exact flow/anti/output dependences (islpy)
    arith_order: ArithOrderClass    # per reduction target: the sequential accumulation order as a
                                    # lexicographic order on the domain, e.g. "c(i,j) summed over k ascending"
    spans: SourceMap
```
Invariants: domain and access maps are exact and affine (the subset gate guarantees representability); `deps` recomputed-and-checked, never trusted from construction; `arith_order` present for every reduction (it is what the D/A bridges consume). Textual form: isl notation + per-statement span table.

### 6.4 Plan IR `[FROZEN interfaces]`
```python
@frozen class Plan:
    id: PlanId
    ops: tuple[PlanNode, ...]
    forks: tuple[Fork, ...]         # runtime decision points
    guards: tuple[Guard, ...]
    decisions: tuple[DecisionId, ...]   # full decision record refs (why this plan)
    contract: ContractId
    signature: CandidateSignature
@frozen class PlanNode:   # one lowered op
    op: HirId
    impl: Impl              # SpecCall{pack, symbol="dgemm", binding} | GeneratedLoop{loop_ir} 
    threads: int; env: tuple[tuple[str,str],...]     # the "OpenMP" of the MVP
@frozen class Fork:
    id: RuntimeDecisionId
    table: tuple[tuple[GuardOutcomePattern, BranchId], ...]   # the pure select_branch table
    arms: tuple[tuple[PlanNodeRef, ...], ...]
    on_no_arm: RuntimeFailure       # Abort spec for pinned cases
@frozen class Guard:
    id: GuardId; checks: ClaimId; decisive: bool; probe: ProbeSpec; budget: GuardBudget
```
Invariants: every fork's table is total over reachable outcome vectors and type-checked against guard decisiveness (§2.5); every arm's legality gates were passed with the guard's claim assumed in the corresponding state; `signature` computed per Foundation §2.10 (skeleton, arith-order class, placement — trivial in MVP — comm — empty — launch — threads/env).

### 6.5 Target IR `[LOCAL]`
A thin typed template model per backend: `TargetUnit(lang ∈ {Fortran, Julia}, decls, calls, driver, provenance_comments)`. Invariant: every emitted call site is traceable to a PlanNode; generated source embeds provenance IDs as comments (`! prov: plan P7 node 3 ← hir %C ← src 14:1`). Example (Fortran): a module with `solve_spd(A, b, x, info)` calling `dpotrf/dpotrs`, plus a driver with residual guard and branch logic realized as plain `if` on `info` — the PlanRuntime reference implementation *is* the generated control flow plus a telemetry file writer.

---

## 7. Exact core data types

Legend: **F** = frozen foundation (JSON-schema'd, FCR to change), **L** = implementation-local. "Exercised by" names the test/scenario (§16).

```python
# ---------- identity & scope ------------------------------------------------
SourceSpan(file, line0, col0, line1, col1)                       # L — every AST/HIR node; Tests A,B
ProvenanceRef = SourceSpan | HirId | LoopRegionId | DirectiveId  # F — provenance reports; Test A
ClaimId, EvidenceId, DecisionId, PlanId, ContractId: opaque u64  # F

@frozen class ClaimKey:                                          # F — dedup/aggregation identity
    stmt: Statement          # Property(obj, prop) | Equivalence(a, b, domain) | Bound(...) | ContractSat(...)
    domain: SemanticDomain | None    # REQUIRED for Equivalence (Foundation 2.2)
@frozen class Claim:  id: ClaimId; key: ClaimKey; scope: Scope   # F — everywhere

class ClaimState:                                                # F — fold output; Tests D,E,F,H
    Established(strongest: EvidenceStatus) | Refuted(by: EvidenceId)
    | Conditional(on: tuple[ClaimId,...]) | Contested(pair) | Unknown(hypothesized: EvidenceId|None)

@frozen class Scope:    extent: DataExtent; at: ProgramExtent    # F — Test F (mutation)
DataExtent    = Universal | Shaped(shape_class) | Instance(id) | Machine(topo_hash)
ProgramExtent = Value(ValueRef) | Region(IrRegionRef) | Window(start: EventId, kill: EventId|OPEN)
@frozen class Validity: topo: Hash|ANY; module: Hash|ANY; killed_by: tuple[EffectPattern,...]  # F

# ---------- evidence ----------------------------------------------------------
class EvidenceStatus:                                            # F — the sum type; no scores on proof classes
    Assumed(who) | Specified(pack, provenance) | StaticallyDerived(rule)
    | SmtProved(solver, artifact) | FormallyProved(system, artifact)      # FormallyProved: unused in MVP, frozen anyway
    | RuntimeChecked(guard, when) | Measured(machine, n, ci) | Predicted(model, calib)
    | AiInferred(model, score)
@frozen class Evidence:                                          # F — immutable event
    id: EvidenceId; claim: ClaimId; status: EvidenceStatus
    provenance: tuple[EvidenceId,...]; scope: Scope; validity: Validity; artifact: ArtifactRef
# Refutation is Evidence whose Statement is the negation-with-witness (Counterexample payload).

# ---------- store --------------------------------------------------------------
Epoch = int                                                      # F
class FoundationStore(Protocol):                                 # F — the concurrency contract (v3.1 §7)
    def append(self, ev: Evidence | FeedbackEvent | SuppressionEvent) -> Epoch: ...
    def state(self, key: ClaimKey, scope: Scope, at: Epoch) -> ClaimState: ...   # commutative fold
    def snapshot(self) -> Epoch: ...
# MVP impl: in-memory grow-only list + one counter.               L for the impl, F for the protocol

# ---------- decisions, control, feedback, search --------------------------------
class DecisionControl: Auto | Prefer(value, cls: PriorityClass, rank: int|None) | Require(value) | Use(value, params)  # F — Test E
@frozen class Decision:                                          # F — provenance; Tests A,D,E,G
    id: DecisionId; layer: str; subject: IrRef; control: DecisionControl
    chosen: AltRef; considered: tuple[AltRef,...]; excluded: tuple[ExclusionCertificate,...]
    depends_on: tuple[tuple[ClaimId, ClaimState],...]; epoch: Epoch
class Feedback:                                                  # F — LIR-less MVP still uses 4 kinds
    Infeasible | ResourceViolation | NumericalViolation | Counterexample
    | PropertyDiscovered | CostEstimate | PerformanceMeasurement   # (full set frozen; MVP emits a subset)
class SearchDomain:  Enumerated(alts) | BoundedDiscrete(params) | Symbolic(vars,cons) | Generative(gen)  # F
@frozen class ExclusionCertificate: region: Predicate; reason: FeedbackRef; justification: EvidenceId    # F

# ---------- contracts ------------------------------------------------------------
@frozen class Contract:  det: Determinism; acc: Accuracy; exc: ExceptionalBehavior     # F — Test C
Determinism  = D1_BitwisePortable | D2_BitwiseConfig | D3_Deterministic | D4_AnyOrder
Accuracy     = A1_IeeeStrict | A2_BoundedError(eps, norm) | A3_BackwardStable(cls) | A4_Statistical | A5_BestEffort
ExceptionalBehavior = frozenset drawn from {NAN_PROP, INF, SIGNED_ZERO, SUBNORMAL, FP_FLAGS}   # E1..E4 named profiles
SemanticDomain = MathematicalReal | MathematicalComplex | IntegerExact(width, ovf)
               | IEEE754(format, rounding, exc_profile) | ContractNumeric(ContractId)          # F — Tests B,C

# ---------- planning artifacts -----------------------------------------------------
@frozen class CandidateSignature: skeleton: Hash; arith_order: Hash; placement: Hash; comm: Hash; launch: Hash  # F — tuning-DB key; stretch autotune task
Plan, PlanNode, Fork, Guard, GuardResult, RuntimeEvidence, RuntimeEvent, RuntimeFailure   # F — §2, §6.4; Tests D,E,H
```

Deliberately **not** built (no MVP scenario): PlacementDomain kinds beyond a stub, CIR/LIR types, rewrite-tier database (the MVP has no rewrite engine — SpecCall-vs-loop selection is algorithm choice, not rewriting; the tier enum is frozen on paper only), attestation types beyond the reserved `SpecProvenance.Signed` variant.

---

## 8. Property system, mutation, and `infer`

**Properties:** `symmetric`, `positive_definite`, `shape(...)` only. Property algebra rules (all `StaticallyDerived`): `GᵀG ⇒ symmetric ∧ (full-rank ⇒ spd)` — present but unexercised unless a demo constructs A; `spd ⇒ symmetric`; shape rules from the dim solver. **Mutation kill (Test F):** surface re-assignment or a recovered loop writing an array generates a `KillEvent(object, region)`; the store's liveness check closes every `Window`-scoped claim on intersecting objects; transfer rules are out of MVP scope except the trivial frame rule (disjoint writes preserve — needed because the SPD solve writes `x`, not `A`). **`infer positive_definite(A)` campaign (Test D/E/H path):** cascade = property algebra → static (nothing for input data) → *runtime-check synthesis*, which offers two probes from the guard library: `QuadFormProbe(k samples)` — refutation-only, cheap, returns `Failed(cex=x)` or `Inconclusive(REFUTATION_ONLY_PROBE)`; `PotrfProbe` — decisive, cost ≈ the factorization itself, so the planner fuses it *into* the Cholesky arm (`dpotrf` info return **is** the guard). Compile-time outcome of the campaign is honestly `Unknown` + a conditional plan; the artifact prices the paths. AI/SMT stages are stubs returning nothing (interface exercised by a mock in tests).

## 9. Planner and user control

Pipeline per HIR op: enumerate candidates (SpecCall impls whose SpecPack pre/postconditions' claims are Established/Conditional-guardable + the GeneratedLoop baseline) → legality/contract gates (per-candidate: SpecPack `contract_profile` must meet the op's required contract — e.g. threaded OpenBLAS dgemm carries `D3` in its pack; under a required `D2` the planner selects the single-thread binding and *reports the price*) → control filtering (`Require/Use` restrict; `Use` verifies-not-searches; pinned-and-illegal ⇒ compile error quoting the pin) → cost order (flop metadata ÷ measured machine rates; below the dgemm-crossover knob the naive loop wins) → preference resolution (class-lex Pareto; `prefer@N` ranks; canonical-hash tie-break + conflict certificate; **Test G**: permutation invariance) → Decision records with `depends_on` states at the snapshot epoch. Feedback in MVP: `Counterexample` (guards), `NumericalViolation` (residual), `PerformanceMeasurement` (stretch); the monotone-exclusion machinery runs with point exclusions only (region certificates: the one BoundedDiscrete domain — crossover knob — uses monotonicity lifting as its single showcase).

## 10. SpecPacks (MVP schema)

```yaml
pack: blas-reference          # bundled packs: blas-reference, lapack-reference,
version_range: ">=3.8"        #                fortran-language-semantics, vendor-openblas
provenance: Bundled
entries:
  - symbol: dgemm
    semantics: "C <- alpha*op(A)*op(B) + beta*C"     # bound to a HIR MatMul template
    domain: IEEE754{b64, rne, E2}
    numerical_contract: {acc: A2, bound: "classical gamma_k forward model", det: D2@single_thread | D3@threaded}
    preconditions:  [shape(A)=(m,k), shape(B)=(k,n), shape(C)=(m,n), noalias(C,A), noalias(C,B), ld constraints]
    postconditions: [defines(C)]
    effects: {writes: [C], reads: [A,B]}
    layout: column_major
    abi: {lib: openblas|reference, link: "-lopenblas"}
    validation: {diff_check: vs numpy, samples: 200, adversarial_fp: [nan, inf, -0.0, subnormal]}
```
Trust: `Bundled` only; `LocalFile` behind a flag with sha256 pin; `Signed` reserved. `dpotrf` additionally declares its dual role: postcondition `info=0 ⇒ positive_definite(A)@instance` — the SpecPack is what licenses the potrf-as-guard pattern. Install-time diff-check runs the validation recipes and appends `Measured` corroboration (or a `Counterexample`, demoting the entry and failing installation loudly).

## 11. Verification & floating-point contract strategy

Accepting the prompt's defaults, sharpened: **Lean — out** (nothing in the MVP needs symbolic-dimension formal proofs; the `FormallyProved` status exists, unpopulated). **SMT (Z3) — in, narrowly:** integer side conditions from recovery (bound relations, divisibility, subscript injectivity where isl leaves a residue) and nothing else; every obligation cached by module hash. **islpy — the workhorse:** exact dependence analysis is `StaticallyDerived` and covers the whole restricted subset. **Property-based differential testing (Hypothesis) — heavy:** every codegen path and every SpecPack entry diff-tested against NumPy/SciPy oracles, with adversarial FP inputs (NaN/±Inf/−0.0/subnormals) checking the *E-axis* claims specifically. **Runtime guards — cheap and semantically load-bearing:** overlap check, `dpotrf` info, residual check, QuadForm probe. **FP contracts:** the three profiles of §3.1; the two implemented bridges are (i) the classical-error-model value bridge for "loop order ≡ blocked/BLAS order" (`γ`-bound algebra, `StaticallyDerived`), and (ii) the E-feature check on candidate substitutions (SpecPack E-profile ⊇ required E-profile). `strict`'s A1 forbids reassociation ⇒ recovered loops are *retargeted but never re-ordered* under it (Test C); no LLVM fast-math flags exist in the MVP because no LLVM path exists — the frozen derivation table lives in `docs/` awaiting the codegen that needs it.

---

## 12. Semantic Recovery Pipeline (MVP instantiation) and the two demonstrations

### 12.1 Pipeline stages (Fortran subset)

```
S0 Parse (ANTLR F2018, Python target)          → full parse tree + token spans
S1 Subset gate                                  → in-subset unit | Unsupported{construct, span} (untranslated)
S2 Symbol/type/shape resolution (subset only)   → typed arrays, loop variables, scalars
S3 Loop normalization + iteration domains       → islpy sets;  non-affine ⇒ Unsupported
S4 Access-function extraction                   → islpy maps per array reference
S5 Reduction recognition                        → accumulation targets + reduction dims + ArithOrderClass
S6 Dependence analysis (exact, islpy)           → deps; recompute-and-check invariant
S7 Alias obligations                            → Fortran-rule claim (Specified{Bundled}) + O(1) runtime overlap guard
S8 Hypothesis: deterministic matcher first, Recognizer plug-ins second (AiInferred) 
S9 Equivalence obligations + bridges            → MathematicalReal equivalence (islpy/structural)
                                                  + ContractNumeric bridge (γ-model) per active contract
S10 HIR construction + versioned lowering       → hir.matmul{accumulate} with guard/fallback fork
S11 Provenance linking                          → HIR op ↔ loop region ↔ source lines
```

### 12.2 Demonstration B: the Fortran GEMM kernel, end to end

```
Source (kernel.f90:14–22)      do j / do k / t = b(k,j) / do i / c(i,j) = c(i,j) + a(i,k)*t
S3 domain                      { S[j,k,i] : 1≤j≤n ∧ 1≤k≤p ∧ 1≤i≤m }
S4 accesses                    c: S[j,k,i]→C[i,j] (rw)   a: →A[i,k] (r)   b: →B[k,j] (r, hoisted into t — S2 proves t loop-invariant per (j,k))
S5 reduction                   target C[i,j], reduction dim k, ArithOrderClass = "k ascending, sequential per (i,j)"
S6 deps                        only the C[i,j] self-accumulation chain over k — matches reduction form; no other deps
S7 alias                       noalias(C,A), noalias(C,B): Specified{fortran-language-semantics} + overlap guard G1
S8 hypothesis                  MatMul-accumulate(C, A, B)  [deterministic matcher; AI not consulted]
S9 claims                      E_R:  loop ≡ C←C+A·B          domain = MathematicalReal        → Established (StaticallyDerived: domain/access/dep match of the contraction template)
                               E_C:  loop ≡ᶜ dgemm-lowering  domain = ContractNumeric(active) → see below
S10 HIR                        %C1 = hir.matmul %A,%B accumulate %C0 {domain=ieee754<b64,rne,E2>}
                               plan fork F1: [G1 Passed → SpecCall dgemm | G1 Failed → GeneratedLoop(anchor)]
```
**The floating-point analysis, explicitly.** The anchor accumulates each C[i,j] strictly in k-ascending scalar order; `dgemm` blocks and reorders the k-sum (and may FMA). Therefore:
- `E_R` (MathematicalReal) — **Established**, and *insufficient by itself* for any IEEE substitution (Foundation 2.2).
- Under **`strict` ⟨D2,A1,E1⟩**: A1 forbids reassociation/contraction ⇒ the ContractNumeric bridge is *unsatisfiable* ⇒ **replacement rejected**; the plan retargets the anchor loop verbatim (Fortran out, same order), and the provenance report states: "dgemm lowering rejected: value bridge cannot meet A1 (accumulation order differs); E_R established but not admissible for IEEE legality." This is **Test C**.
- Under **`faithful` ⟨D2,A2(γ),E2⟩** (default): value bridge — both orders satisfy the classical bound `|Ĉ−C| ≤ γ_p·(|A||B|)ᵢⱼ`; dgemm's SpecPack asserts the same model (`Specified`), the anchor's is derived (`StaticallyDerived`) ⇒ A2 met. E bridge: SpecPack E-profile E2 ⊇ required E2 (diff-checked on adversarial inputs at install). D bridge: single-thread dgemm is D2 per pack; threaded is D3 ⇒ under required D2 the planner binds single-thread **or** reports the D3 price if the user prefers speed. ⇒ `E_C` **Established (Conditional on G1)** ⇒ **replacement permitted**, guarded, anchor retained as fallback arm. No numerical-error *runtime* bridge is needed — the bridge is static; the overlap guard G1 is the only runtime obligation.
- Provenance report carries all of: `E_R: Established`, `IEEE754 bitwise-vs-anchor: not claimed`, `ContractNumeric(faithful): Established/Conditional(G1)`, the rejected-under-strict note, fallback status.

### 12.3 Demonstration D/E/H: native SPD solve

```
assume symmetric(A)         → Claim P_sym: Established{Assumed} (artifact: "unverified trust" section)
infer  positive_definite(A) → campaign ⇒ compile-time Unknown; runtime probes offered (QuadForm, Potrf)
x = solve(A,b); require relative_residual(A*x−b) < 1e-12   → contract atom + residual guard G_res
Candidates: chol(dpotrf+dpotrs)  [needs spd]   |   sym-indefinite(dsytrf+dsytrs) [needs symmetric only]
Plan (Auto): fork F_spd keyed on PotrfProbe (decisive, fused into the chol arm):
   info=0  → RuntimeEvidence{P_spd: RuntimeChecked, Instance×Window(kill = first write A)} → dpotrs → G_res
   info>0  → CounterexampleFound(P_spd) → arm dsytrf/dsytrs → G_res
G_res Failed → NumericalViolation → RuntimeFailure.ContractViolated (MVP: no refinement loop)
QuadFormProbe (optional pre-probe, refutation-only): Failed ⇒ skip chol arm cheaply;
   Inconclusive(REFUTATION_ONLY_PROBE) ⇒ proceed to PotrfProbe            ← Test H's Inconclusive, benign path
Control-level matrix (Test E):  Auto → as above.  Prefer chol → same, override priced if fallback taken.
   Require chol → fallback arm FORBIDDEN; compile-time note emitted; info>0 ⇒ fail(Abort{cex})   ← Test H's hard path
   Use chol{...} → as Require plus pinned params, verified not searched.
```
Condition estimation: absent, by design — `require` is discharged by G_res a-posteriori, which is decisive and instance-certified.

---

## 13. Code-generation strategy

**Targets: Fortran 2018 (+BLAS/LAPACK) primary, Julia (+LinearAlgebra/direct ccall) secondary. C++ deferred** — it demonstrates nothing Fortran+Julia don't, and adds a build-system tax. Rationale: Fortran is the audience-native proof ("your loop came back faster, and here's why"), reuses your expertise, and links BLAS trivially; Julia is near-free to emit, executes instantly (doubles as a cross-check executor in tests), and proves the multi-target claim: **one Plan → two semantically equivalent TargetUnits**, with the equivalence being — precisely — a `ContractNumeric` claim diff-tested by the harness. Backends are string-template emitters over Target IR (no MLIR/LLVM in MVP); every emitted call carries a provenance comment; drivers embed the guards, fork `if`-logic, residual check, telemetry-file writer (the reference PlanRuntime), and the `env` exports (threads, pinning).

## 14. Provenance schema (minimal)

```json
{ "compilation": {"program_hash":"…","machine":"…","contract":"faithful","epoch":412},
  "operations": [{
    "hir": "%C1", "kind": "MatMul+accumulate",
    "source": {"kind":"recovered","file":"kernel.f90","lines":"14-22","loop_region":"L3"},
    "claims": [{"key":"Equivalence(L3, matmul, MathematicalReal)","state":"Established",
                "evidence":[{"status":"StaticallyDerived","rule":"contraction-template"}]},
               {"key":"Equivalence(L3, dgemm-lowering, ContractNumeric(faithful))","state":"Conditional",
                "on":["noalias(C,A)@G1"],"bridge":{"A":"gamma_p model","E":"E2⊇E2","D":"D2@single-thread"}}],
    "decision": {"id":"D17","control":"Auto","chosen":"SpecCall(blas-reference/dgemm)",
                 "considered":["GeneratedLoop(anchor)"],"rejected":[{"alt":"threaded dgemm","why":"D3 < required D2","price":{"time":"-38%"}}],
                 "depends_on":[["E_C","Conditional"]]},
    "guards": [{"id":"G1","kind":"overlap","decisive":true,"fallback":"anchor loop"}],
    "ai": {"involved": false}, "unverified_assumptions": []
  }]}
```
The human-readable report of the freeze prompt's §21 is a renderer over exactly this JSON — one schema, two views. Anti-explosion rule: reports reference evidence by ID with depth-1 inlining; full derivation DAGs render on demand only.

## 15. Repository structure (Python)

```
mathhpc/
├── pyproject.toml                    # pyright strict, ruff, pytest, hypothesis
├── src/mathhpc/
│   ├── foundation/                   # [FROZEN] domains.py span.py claims.py evidence.py
│   │   ├── store.py contracts.py decisions.py feedback.py search.py signatures.py
│   │   └── schemas/                  # versioned JSON schemas for every frozen type
│   ├── plan_runtime/                 # [FROZEN] interface.py  guards.py(lib)  reference_rt.py
│   ├── frontend_math/                # lexer.py parser.py ast.py
│   ├── hir/                          # types.py ops.py dims.py builder.py printer.py
│   ├── recovery/                     # antlr/(generated) gate.py resolve.py loop_ir.py
│   │   └── domains.py accesses.py deps.py arith_order.py recognize.py obligations.py
│   ├── properties/                   # algebra.py effects.py(kill) infer.py
│   ├── planner/                      # candidates.py gates.py cost.py prefs.py plan_ir.py
│   ├── verification/                 # bridges.py smt.py difftest.py
│   ├── specs/                        # loader.py schema.py
│   ├── codegen/                      # target_ir.py fortran.py julia.py drivers.py
│   ├── provenance/                   # model.py render.py
│   └── machine/                      # probe.py model.py
├── specpacks/                        # blas-reference/ lapack-reference/ fortran-language-semantics/ vendor-openblas/
├── tests/                            # unit/ golden/ property/ differential/ negative/ e2e/
├── examples/                         # gemm.mh  spd_solve.mh  kernel.f90
└── docs/                             # foundation-v1.0.md  fcr/  (v1..v3.1 design docs)
```

## 16. Testing architecture

Organized by **trust boundary**, not by module. (1) *Fold correctness*: Hypothesis tests — order-independence of `state()`, scope-intersection (no false Contested), failed-proof⇒Unknown, kill-event liveness. (2) *Gates*: negative tests where the compiler must refuse — the freeze prompt's six refusal cases are all encoded: ℝ-proved but A1 forbids (Test C); recognizer proposes but alias analysis can't establish and guard budget exceeded ⇒ untranslated; SPD then mutated ⇒ stale evidence inadmissible (Test F); Inconclusive + Use pin ⇒ Abort with the compile-time note (Test H); preference permutation invariance (Test G); Predicted-only equivalence must not suppress (unit test on the suppression rule). (3) *Bridges/FP*: differential tests incl. adversarial FP values against SpecPack E-claims. (4) *Recovery*: golden isl domains/accesses/deps per kernel; subset-gate negative corpus (pointers, COMMON, non-affine…) must yield `Unsupported`, never a translation. (5) *Codegen*: compile-and-run both targets, cross-compare, oracle-compare. (6) *Provenance*: schema validation + "every Decision reachable from the report" completeness check. End-to-end: **A** native GEMM→dgemm→run→provenance; **B** Fortran GEMM recovery→dgemm; **C** strict-contract rejection; **D** SPD solve→dpotrf/dpotrs; **E** Unknown-SPD × {Auto, Prefer, Require, Use}; **F** mutation invalidation; **G** preference permutation; **H** runtime Inconclusive with and without fallback.

---

## 17. Milestone plan (dependency-aware; difficulty 1–5, not calendar)

| Phase | Deliverables | Done when | Depends | Key tests | Diff. | Top risk |
|---|---|---|---|---|---|---|
| **P0 Foundation + runtime boundary** | frozen types + JSON schemas; FoundationStore + fold; contracts; decisions/feedback; PlanRuntime + reference RT; guard library skeleton | all frozen types round-trip; fold property suite green | — | fold Hypothesis suite; schema round-trips | 3 | scope/liveness fold bugs |
| **P1 Math frontend** | lexer/parser/AST; typed HIR + dim solver; directives→claims/controls | examples/gemm.mh & spd_solve.mh → HIR snapshots | P0 | parser golden; HIR snapshot; negative parses | 2 | span bookkeeping |
| **P2 Core ops** | Dot/MatVec/MatMul/Solve HIR ops; flop/byte metadata; machine probe | HIR for all four ops with metadata | P1 | metadata unit tests | 1 | — |
| **P3 Property/evidence engine** | symmetric/spd claims; effects+kill; frame rule; `infer` orchestrator (algebra→runtime-synthesis; AI/SMT stubs) | Test F green; infer returns honest Unknown | P0,P2 | mutation-kill; campaign caching | 3 | kill-event precision |
| **P4 Planner** | candidates; contract gates; Auto/Prefer/Require/Use; Pareto+tie-break; Decision records; point-exclusion loop | Test E control matrix (compile-time half); Test G | P2,P5-specs | prefs permutation; escalation | 4 | gate/control entanglement |
| **P5 SpecPacks + codegen** | pack loader + 4 bundled packs + diff-check; Fortran & Julia backends + drivers; provenance renderer | **VS-1 (§19) = Test A green** | P2 (P4-thin: Auto only) | install diff-checks; cross-target; provenance completeness | 3 | ABI/layout details |
| **P6 Fortran recovery** | ANTLR integration; subset gate; Loop IR; islpy domains/accesses/deps; arith-order; GEMM recognizer; obligations | Test B green (faithful contract) | P0,P2,P5 | golden isl artifacts; gate negative corpus | **5** | delinearization/normalization edge cases |
| **P7 FP contracts + bridges** | γ-model value bridge; E-feature check; strict-rejection path | Test C green | P6 | bridge unit tests; adversarial FP difftests | 3 | over/under-claiming the γ model |
| **P8 Runtime guards + fallback** | overlap, PotrfProbe, QuadForm(Inconclusive), residual; fork tables; telemetry→store | Tests D, H green; Test E runtime half | P3,P4,P5 | guard decisiveness typing; abort-with-note | 3 | guard/branch table totality |
| **P9 Provenance + demos** | full reports for A–H; docs; (stretch: crossover autotune knob + Measured evidence) | all eight e2e tests green in CI | all | e2e suite | 2 | report sprawl |

Critical path: P0→P1→P2→P5(VS-1)→P6→P7; P3/P4/P8 branch off P0/P2 and merge at P8/P9.

## 18. First 30 implementation tasks (dependency order; ≈1 PR each)

| ID | Title | Purpose / files | Deps | Acceptance + tests |
|---|---|---|---|---|
| 001 | `SemanticDomain` + serialization | foundation/domains.py, schemas/ | — | round-trip JSON; equality/hash; pyright strict clean |
| 002 | `SourceSpan`, `ProvenanceRef`, IDs | foundation/span.py | — | span arithmetic unit tests |
| 003 | `ClaimKey`/`Claim`/`Statement` mini-AST | foundation/claims.py | 001,002 | Equivalence without domain = type error (negative test) |
| 004 | `EvidenceStatus` + `Evidence` records | foundation/evidence.py | 003 | immutability enforced; round-trip |
| 005 | `Scope` (DataExtent×ProgramExtent) + `Validity` | foundation/claims.py | 002 | intersection unit tests incl. Window overlap |
| 006 | `FoundationStore`: append/state/snapshot + five-state fold | foundation/store.py | 004,005 | fold on hand-built evidence sets → expected states |
| 007 | Fold property suite | tests/property/ | 006 | Hypothesis: permutation-invariance; no-false-Contested; failed-proof⇒Unknown |
| 008 | `Contract` (D/A/E) + meet + named profiles | foundation/contracts.py | 001 | meet lattice laws property-tested |
| 009 | `Decision`/`DecisionControl`/`Feedback`/`ExclusionCertificate` | foundation/decisions.py, feedback.py | 006 | decision `depends_on` epoch recording test |
| 010 | PlanRuntime interface + `GuardResult{P/F/Inconclusive}` + reference RT | plan_runtime/ | 004,009 | branch-table totality checker; decisiveness typing negative test |
| 011 | Math lexer (Unicode+ASCII) | frontend_math/lexer.py | — | token goldens incl. `ℝ Σ ∈ ×` aliases |
| 012 | Recursive-descent parser → AST | frontend_math/parser.py | 011,002 | golden s-expr trees; error-message span tests |
| 013 | Dim solver + `TensorType` + HIR builder (4 ops) | hir/ | 012,001 | shape-mismatch negatives; HIR snapshots |
| 014 | Directive lowering: assume/infer/require/prefer(@N)/use | hir/builder.py | 013,008,009 | directives become claims/contract atoms/controls with provenance |
| 015 | HIR printer + snapshot harness | hir/printer.py | 013 | stable textual form goldens |
| 016 | Machine probe + model | machine/ | — | probe emits json {vendor, threads, dgemm_gflops(1,T), triad} |
| 017 | SpecPack schema + loader + 4 bundled packs | specs/, specpacks/ | 001,008 | schema validation; load; `dpotrf ⇒ spd@instance` postcondition parsed |
| 018 | Planner v0 (Auto): candidates+contract gate+cost | planner/ | 013,016,017 | matmul → dgemm chosen; Decision recorded with claim states |
| 019 | Fortran backend + driver + compile-run harness | codegen/ | 018 | generated code compiles (gfortran+openblas), runs |
| 020 | Provenance renderer v0 | provenance/ | 018 | JSON schema-valid; report names decision + evidence |
| 021 | **VS-1: Test A end-to-end** | tests/e2e/ | 019,020 | matmul source → run → ‖C−C_numpy‖ within A2 bound → report |
| 022 | Julia backend + cross-target test | codegen/julia.py | 021 | both targets agree within contract bound |
| 023 | Control levels + Pareto prefs + tie-break | planner/prefs.py | 018 | Test G; Require/Use escalation negatives; override pricing |
| 024 | Property claims + effects/kill + frame rule | properties/ | 006,013 | **Test F** |
| 025 | `infer` orchestrator + priced-unlock report | properties/infer.py | 024,017 | SPD campaign → Unknown + offered probes; cache hit test |
| 026 | Guard library: overlap, PotrfProbe, QuadForm, residual | plan_runtime/guards.py | 010,017 | decisive/refutation-only typing; Inconclusive path unit tests |
| 027 | SPD solve e2e: forks + fallback + abort-with-note | planner/, codegen/ | 023,025,026 | **Tests D, E, H** |
| 028 | ANTLR F2018 integration + subset gate + resolution | recovery/ | 002 | subset corpus parses; negative corpus ⇒ `Unsupported{span}` |
| 029 | Loop IR + islpy domains/accesses/deps + arith-order | recovery/ | 028 | golden isl sets/maps/deps for the demo kernel |
| 030 | GEMM recognizer + obligations + γ/E bridges + strict rejection | recovery/, verification/ | 029,021,008 | **Tests B and C**; alias-unprovable⇒untranslated negative |

## 19. Earliest executable vertical slice

**VS-1 = task 021, end of thin-P5** — before properties, controls beyond Auto, guards, and all of Fortran recovery: `C = matmul(A,B)` → parse → HIR → claim `MatMul semantics` (StaticallyDerived from the typed op itself) → Auto decision → `blas-reference/dgemm` SpecPack → Fortran codegen → compile, execute, compare against NumPy within the A2 bound → provenance report naming the decision, the SpecPack, the contract, and the evidence. Roughly 21 PRs in; everything after it extends a *running* system.

## 20. Technical risk register (top 15)

| # | Risk | L | I | Detection | Mitigation | Fallback |
|---|---|---|---|---|---|---|
| 1 | Fortran frontend complexity leaks past the gate | M | H | negative corpus in CI | whitelist gate; parse-all/translate-little | shrink subset further |
| 2 | Alias analysis gaps (recovery) | M | H | untranslated-rate metric on corpus | Fortran-rule SpecPack + O(1) guard + anchor fallback | guard-always mode |
| 3 | FP-equivalence over/under-claiming | M | **H** | adversarial difftests; strict-mode tests | bridges only via γ-algebra; A1 path never reorders | reject more (safe direction) |
| 4 | Recovery false positives (wrong semantics accepted) | L | **H** | differential tests per recovered op; anchor cross-run in CI | deterministic matcher primary; equivalence obligations mandatory; versioned lowering | disable recognizer, keep gate |
| 5 | Scope/validity (kill/window) bugs | **H** | H | fold property suite; Test F mutations fuzzed | §3 semantics centralized in store; no ad-hoc liveness checks | conservative: any write kills all claims on object |
| 6 | Evidence-aggregation subtleties (false Contested / stale Established) | M | H | Hypothesis fold suite (007) | per-scope-intersection fold; single implementation | manual state override forbidden — fix the fold |
| 7 | SMT scope creep | M | M | obligation count/time in CI | islpy first; Z3 for integer residues only; cache by module hash | drop SMT, widen `Unsupported` |
| 8 | Cross-language codegen divergence | M | M | cross-target contract-bound comparison in CI | one Target IR, thin emitters; drivers share templates | single target (Fortran) |
| 9 | SpecPack errors (wrong bound/effect/ABI) | M | H | install-time diff-check incl. adversarial FP | packs reviewed in-repo; smallest claims that suffice | pin reference BLAS only |
| 10 | Runtime guard cost distorts results | L | M | guard timing in telemetry | budgets in `GuardBudget`; probes O(1)/O(n) | drop optional probes; keep decisive ones |
| 11 | IR overengineering | M | M | "empty layer" review at each phase gate | seams-as-slots rule (§6); FCR discipline | delete unused nodes ruthlessly |
| 12 | Architecture/code mismatch (docs say, code does) | M | H | provenance completeness test; frozen-schema round-trips | frozen types are *the* code; docs generated from schemas where possible | FCR process |
| 13 | AI-integration distraction | M | M | phase gates; AI absent from critical path | Recognizer plug-in + mock only | ship with deterministic matcher |
| 14 | Cost-model inaccuracy misleads plans | M | L (MVP) | stretch: Measured vs Predicted deltas logged | two-number machine model, no promises | order by flops only |
| 15 | Provenance explosion | M | M | report-size metric | depth-1 inlining, IDs beyond | on-demand DAG rendering |

## 21. Engineering vs research

**Engineering (known work):** everything in §18; restricted-subset recovery for affine GEMM-class kernels (islpy makes S3–S6 standard practice); γ-model bridges for reordering-vs-BLAS; SpecPack diff-checking; two-backend codegen; the fold; provenance. **Research (open questions, tracked in `docs/research/`, never on the critical path):** breadth of semantic recovery beyond affine kernels (pointer-based C, irregular control flow); efficient FP semantic-equivalence verification beyond the classical error-model algebra (per-instance certified bounds, FP-SMT scalability); whether AI recognizers find semantics deterministic matchers miss, and at what false-positive cost; principled fusion of analytical models, learned corrections, and measurements; whether optimization certificates (comm/I/O lower bounds) can be made *decision-grade* rather than diagnostic at real scale. The MVP produces *evidence about* the first three without depending on answers to any.

## 22. What to implement first, tomorrow

Tasks **001–004** in one sitting — `SemanticDomain`, `SourceSpan`/IDs, `Claim`/`ClaimKey` (with the "equivalence requires a domain" negative test), immutable `Evidence` — then start **006** (the store and the five-state fold) and stop only when the first fold property test (permutation invariance on a three-evidence set) is green. That test is the architecture's heart under version control by tomorrow night; everything else in this specification stands on it.
