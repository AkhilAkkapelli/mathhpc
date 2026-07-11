# HPC Architecture v3 — Amendments to v2

**Amends v2 (`hpc-architecture-v2-foundations.md`). The v2 spine is retained: `Claim + Evidence + Decision`, the typed feedback bus, hierarchical placement domains, the contract lattice, optimization certificates, and four-level control. This document resolves eight issues; each section states what it amends. §9 consolidates every changed invariant. Anything not amended here stands as written in v2.**

---

## 1. `Prefer` as typed soft constraints (amends v2 §5)

### 1.1 Why the scalar weight dies

`Prefer{value, weight: Weight}` presumed a common currency across preferences about algorithms, seconds, joules, bytes, and determinism levels. No such currency exists, and inventing one (normalize-everything-to-time) smuggles a modeling decision into user-facing semantics. v3 replaces the weight with **typed preferences, priority classes, and Pareto resolution**; scalarization survives only as an explicit opt-in.

### 1.2 Types

```rust
enum Preference {
    Categorical { subject: DecisionSubject, value: AltRef },          // prefer algorithm = cholesky
    Threshold   { quantity: Quantity, rel: Rel, target: SymOrConcrete<Qty> },  // prefer time <= 10s, memory/rank <= 96GiB
    Minimize    { quantity: Quantity },                               // prefer minimize energy
    ContractPref{ axis: ContractAxis, at_least: AxisLevel },          // prefer determinism >= D2
}
enum Quantity { PredictedTime, Energy, MemoryPeak{space}, CommVolume{level}, Messages, ResourceUse(ResourceKind), ... }

struct SoftConstraint {
    pref: Preference,
    class: PriorityClass,      // totally ordered, program-declared; default order: strong > normal > hint
    scope: IrRegionRef,        // preferences attach to regions like all directives
    id: PrefId,                // stable, for reports and declaration-order tie-breaks
}
```

`require` and `use` are unchanged (hard). Every `Preference` variant has a **satisfaction functional** over candidate plans:
- `Categorical` → boolean;
- `Threshold` → boolean plus signed margin;
- `Minimize` → the quantity's predicted value (with its evidence — `Predicted` or `Measured`);
- `ContractPref` → boolean on the plan's effective contract at the scope.

### 1.3 Resolution: class-lexicographic Pareto

Given the feasible set F (all plans satisfying every `require`/`use` and every verifier), resolution proceeds class by class, strongest first:

1. Within class k, define the satisfaction vector `s_k(p)` over that class's preferences (booleans and minimized quantities). Compute the **Pareto-non-dominated subset** `F_k ⊆ F_{k-1}` (p dominates q iff p is ≥ q componentwise and > on at least one component; for `Minimize`, smaller is better).
2. Proceed to the next class restricted to `F_k`. Lexicographic across classes: a `hint` can never cost a `strong` preference anything.
3. **Deterministic tie-break** on the final surviving set, in order: (a) the *final objective* — `PredictedTime` unless the program declares otherwise; (b) preference declaration order (earlier `PrefId` wins remaining boolean ties); (c) canonical plan hash. Step (c) exists so that compilation itself is a pure function of (program, machine model, tuning DB) — a new invariant, see §9 (I-10): reproducible *compilation* is now guaranteed, not just reproducible execution.

**Conflict semantics.** Two preferences in the same class are *in conflict* iff no plan in the incoming set satisfies both. Conflicts are not errors (these are soft constraints); they produce a **conflict certificate** in the plan artifact: the conflicting set, and for each maximal jointly-satisfiable subset, the frontier point that realizes it. Conflicts *across* classes are not conflicts at all — the stronger class simply wins, and the report says so.

### 1.4 The price of honoring a preference

With no common currency, a price is a **vector delta between frontier points**, not a number:

> For an unhonored preference π: `price(π)` = the componentwise difference between the selected plan's quantity vector (time, energy, peak memory, comm volume, contract levels, and the satisfaction bits of every *other* preference) and the best frontier plan that satisfies π. If no feasible plan satisfies π: `price(π) = Infeasible{feedback}` — the blocking feedback (capacity proof, contract gate, certificate) is attached.

So an override report reads: "`prefer algorithm = cholesky` not honored (spd unproven, guard cost 4.1% > threshold). Honoring it: +0 time, requires `assert A: spd` or accepts +4.1% guard; would also unsatisfy `hint memory/rank ≤ 96GiB` (+7 GiB)." The exchange is explicit; the user, who does own a common currency (their judgment), converts.

### 1.5 Explicit scalarization

```text
with { objective = weighted { predicted_time: 0.7 s⁻¹-normalized, energy: 0.3 kJ⁻¹-normalized } }
```
Legal only when every term is a `Quantity` with declared normalization; replaces step 3(a) of the tie-break (classes and Pareto still apply above it — scalarization is a final objective, not a bypass of priorities). The artifact records the scalarization verbatim.

---

## 2. `SearchDomain` and region exclusion (amends v2 §1.3)

### 2.1 The abstraction

v2's anti-livelock invariant assumed finite enumerated alternative sets. Real planning spaces are mixed. Every `Decision` now owns a `SearchDomain`:

```rust
enum SearchDomain {
    Enumerated { alts: Vec<AltRef> },
    Discrete   { params: Vec<IntParam>, constraints: ConstraintSet },
                 // IntParam = { name, range: [lo, hi], stride|grid, units }
                 // e.g. block_size ∈ {64..2048 step 32}, lookahead ∈ {0..4}, grid_pr·grid_pc = P
    Symbolic   { vars: Vec<SymVar>, constraints: PresburgerOrPoly },
                 // dims-parametric families resolved at specialization time
    Generative { generator: GeneratorRef, canon: CanonicalizerRef, budget: Budget },
                 // schedule generators, transform-script spaces, learned proposal distributions
}

struct ExclusionCertificate {
    domain: DecisionId,
    region: RegionPredicate,       // quantifier-free formula over the domain's params:
                                   //   block_size >= 768 ∧ lookahead >= 2
    reason: FeedbackRef,           // the feedback that induced it
    justification: EvidenceId,     // WHY the whole region (not just the point) is excluded — see 2.2
    validity: Validity,            // expires with topo hash, module hash, etc., like all evidence
}
```

The excluded set of a decision is the **disjunction of its certificates' regions** — a growing predicate. Search operates over `domain ∧ ¬excluded`. The v2 monotonicity invariant is restated (I-1 in §9):

> Every `Revised` response must add an exclusion certificate whose region **contains the failed candidate**, and no candidate inside the excluded predicate is ever proposed again (generators included — see 2.3).

### 2.2 From point failure to region certificate

A feedback message justifies excluding a *region* only when the generalization is itself evidence-backed. Three generalization operators, in order of preference:

1. **Monotonicity lifting.** If the violated quantity `q(params)` carries a claim `Monotone(q, param, ↑)` with `StaticallyDerived` evidence — and the cost-model algebra supplies these mechanically, since footprint/volume formulas are polynomials with nonnegative coefficients in the relevant knobs — then a violation at `param = v` lifts to `param ≥ v` (conditioned on the other params' values or their own monotone directions). Example, the v2 §8.1 HBM violation: `footprint(nb, la) = c₁·nb²·(la+1) + c₂`, monotone ↑ in both ⇒ certificate `nb ≥ 768 ∧ la ≥ 2` from a single failure at (768, 2), not the point alone. The certificate's justification cites the monotonicity claims.
2. **Infeasibility-core extraction.** When feasibility is checked by an LP/ILP/SMT system (memory packing, knob constraints, contract gates), the unsat core is a conjunction of violated constraints; the certificate region is the core's negation projected onto the decision's params. Justification = the core artifact (`SmtProved`-class).
3. **Point exclusion** — always available, always sound; used when neither operator applies (e.g., a `PerformanceMeasurement` that merely disappoints excludes nothing — it *reranks*; only violations exclude).

Certificates are `Claim`s ("region R is infeasible for decision d") and flow through the evidence system: they appear in the artifact, expire with their validity bindings (a topology upgrade un-excludes memory-based regions), and can be *falsified* — a later verified plan inside an excluded region is a `Counterexample` against the certificate, demoting its justification (this catches unsound monotonicity claims, which would be compiler bugs, loudly).

### 2.3 Termination per domain kind

| Kind | Guarantee |
|---|---|
| `Enumerated` | finite; v2 argument unchanged |
| `Discrete` | finite but large; monotonicity/core certificates excise regions of nonzero measure per blocking failure, and the budget bounds sampling of the remainder; the *no-retry* invariant is exact |
| `Symbolic` | exhaustion is not promised; guarantees are (a) no-retry, (b) monotone shrinkage of the constraint system, (c) budget with honest failure (v2 §1.3 carries over verbatim) |
| `Generative` | the generator must be **exclusion-aware**: proposals are canonicalized (`canon` — e.g., schedule-tree hashing modulo commutation) and rejected against the excluded predicate *before* costing; deduplication via canonical hash gives no-retry; termination by budget only. A generator that repeatedly proposes excluded candidates burns its own budget, not the planner's soundness. |

Honest-failure semantics are unchanged: on budget exhaustion, best feasible plan or an error carrying the feedback and certificate trail.

---

## 3. Temporal and region-sensitive claims (amends v2 §7.1)

### 3.1 The problem

`spd(A)` is not a fact about a name; it is a fact about the *contents of a storage object during an interval*. v2's `Scope` covered input-shape and machine extent but not program extent — untenable the moment the language admits in-place updates (blocked in-place Cholesky overwrites A's lower triangle with L; `spd(A)` must die mid-operation, by construction).

### 3.2 Extended types

```rust
struct Scope {
    extent: DataExtent,          // v2: Universal | Shaped{dims,dtype} | Instance{input} | Machine{topo}
    at: ProgramExtent,           // NEW
}
enum ProgramExtent {
    Value(ValueRef),             // property of an SSA value — immutable, never expires (the MIR common case)
    Region { region: IrRegionRef, layer: LayerId },   // holds throughout a static IR region
    PhasePoint(PhaseRef),        // holds at a named plan phase boundary ("post-factorization")
    Window { from: EventRef, until: EventRef },       // runtime interval, closed by an invalidation event
}

struct Validity {               // v2 fields retained (topo_hash, model_version, module_hash, input class)
    ...,
    killed_by: Vec<EffectPattern>,   // NEW: write effects that terminate this evidence
}
```

### 3.3 Semantics

- **MIR stays easy on purpose.** MIR is value-semantic/SSA: properties attach with `ProgramExtent::Value` and are eternal for that value. The temporal machinery activates where mutation exists: surface-language mutable arrays, and every layer from LIR down (buffers).
- **Mutation kills, transfer rules resurrect.** The effect system computes, for each write op, the set of (object, sub-region) pairs written. A claim over a mutable object with `Region`/`Window` extent is killed by any write intersecting its object *unless* a **property-transfer rule** in the (audited) property algebra proves preservation — e.g., `A ← A − v·vᵀ` preserves `symmetric(A)`; a trailing-update `syrk` preserves `spd(trailing block)` given the factorization's standard argument; `scale(A, α>0)` preserves `spd`. Transfer rules produce new evidence (`StaticallyDerived{rule}`) whose provenance links the killed claim and the write — the derivation DAG shows property *lineage across mutation*.
- **Frame rule.** Writes provably disjoint from a claim's object (by the ownership functions at DIR level, by LIR alias facts, or by separation of index sub-regions) do not kill it. Disjointness itself must be claim-backed; unresolvable aliasing kills conservatively.
- **Gate liveness (new verifier obligation, I-6).** Every legality/contract gate application now checks that each consulted claim's `ProgramExtent` *covers the application point* — at compile time by region containment, at runtime by window arithmetic (the event closing a window is emitted by the killing write or phase transition). A claim being "in the database" is no longer sufficient; it must be *live here*.
- **Instance claims compose with windows.** `cond(A) ≤ 3e4 {RuntimeChecked}` from §7 is `extent: Instance{input}, at: Window{from: estimator_done, until: first_write(A)}` — mutate A and the certificate is gone, mechanically.

### 3.4 Worked micro-example (in-place Cholesky)

```
spd(A)        : Window{program_start, potrf_begin}        — user assert, guard at entry
lower_tri(Aₛ) : Window{potrf_end, …}                       — transfer rule: potrf postcondition
spd(A₂₂ - L₂₁L₂₁ᵀ) during the loop : per-step claims from the trailing-update transfer rule;
noalias(panel_buf, A) : Region{cir.epoch_k}                — scoped exactly to the epoch that needs it
```
The v2 SPD example's runtime `Counterexample` on a bad pivot is now, precisely: the *entry guard's* claim was live and false — demotion targets the windowed claim, and the plan artifact shows which window died.

---

## 4. The assertion taxonomy (amends v2 §7 — `UserAsserted` is removed)

Five directives with distinct semantics. `UserAsserted{policy}` disappears as an evidence status; it is replaced by two statuses (`Assumed`, `Promised`) and three directives that *produce* existing statuses.

```text
assume  P        # trust without checking — narrow, flagged, never for equivalence
assert  P        # trust for planning; runtime guard MANDATORY
prove   P        # compile-time obligation; error if not discharged
check   P        # runtime measurement/branch point; no planning trust before it runs
promise P by <id> # expert vouching for foreign-boundary facts; the only user path to equivalence-class claims
```

| Directive | Evidence produced | Admissible at legality gates? | Runtime behavior | On failure |
|---|---|---|---|---|
| `assume` | `Assumed{who, site}` | **Property claims only** (spd, cond ranges, dims, data classes). **Never** for `Equivalence` or memory-safety claims — an assume that could make the compiler emit *wrong code for a well-defined program* is rejected at parse time. Requires the build to enable `unchecked-assumptions`; every assume is listed in the artifact's "unverified trust" section | none — zero cost, that is its point | consequences are the user's; if a later `check`/guard elsewhere falsifies it, normal `Counterexample` demotion applies and the assume site is named |
| `assert` | claim enters planning as `AssertGuarded` (a *pending* status); the compiler **must** synthesize a guard, upgrading it to `RuntimeChecked{guard}`. If no decidable, budget-feasible check exists, `assert` is a **compile error** ("cannot check P at runtime; use `assume` (unchecked) or `prove` (static)") — assert never silently degrades to assume | yes, as `RuntimeChecked` (guard compiled in) | guard runs at the claim's window start | guard failure ⇒ `Counterexample`, precompiled fallback edge or abort per §5.3(v2) control rules |
| `prove` | none from the user; discharging produces `StaticallyDerived`/`SmtProved`/`FormallyProved` | yes (it *is* proof) | none | **compile error** with the failed obligation and the nearest provable weakening as diagnosis |
| `check` | `RuntimeChecked` **after** the check executes; before that, the claim confers **no planning trust** — but the planner may compile *conditional branches* keyed on the outcome (this is the v2 conditional-plan mechanism, now with its own name) | only for plan branches dominated by a passing check (window semantics of §3 enforce this positionally) | evaluated at its program point; emits `PropertyDiscovered` or `Counterexample` | user handler, or default: the else-branch / abort |
| `promise` | `Promised{who, signature, statement}` | the **only** user status admissible for `Equivalence`-class claims, and only for **foreign-boundary** facts the compiler demonstrably cannot analyze (external library semantics: "`dgemm` computes C←αAB+βC"; "this MPI library's allreduce is deterministic per config"). Conditions: per-claim granularity; signed identity recorded; build flag `--accept-promises=<list>`; a dedicated artifact section; and **refutation supremacy** — if the compiler can *disprove* P, the promise is a compile error, never a tie | none by default; the compiler *offers* differential spot-checks where feasible (sampled comparison against a reference) which, if accepted, add `RuntimeChecked` alongside | `Counterexample` demotes the promise **globally for that signer/claim pair** and flags every plan that consumed it |

Conservatism about equivalence, stated as the invariant (I-3): **semantic equivalence is established only by `{StaticallyDerived, SmtProved, FormallyProved, RuntimeChecked-with-guard}`, or by `Promised` restricted to foreign-boundary claims under the conditions above. `Assumed` equivalence does not exist in this architecture.** The asymmetry is deliberate: a wrong property claim usually yields a *detectably* wrong or failing computation (Cholesky breakdown, residual blowup — the system's numerical checks are a second line of defense); a wrong equivalence claim yields a *silently different program*, against which no downstream check is guaranteed to exist.
