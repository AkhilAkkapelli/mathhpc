# HPC Architecture v3.1 — Final Foundation Amendments and Freeze Assessment

**Amends v3 (`hpc-architecture-v3-amendments.md`). All v3 corrections are preserved: typed Pareto preferences, search domains with evidence-backed region exclusion, temporal/region-sensitive claims, the assertion taxonomy, the tiered rewrite lifecycle, the three-axis contract, condition estimation as algorithm selection, and the Semantic Recovery Pipeline. This document resolves seven remaining issues and closes with a freeze assessment (§8). Changed invariants are consolidated in §8.3.**

---

## 1. Preference tie-breaking without declaration order (amends v3 §1.3)

### 1.1 The distinction that dissolves most of the problem

A tie between surviving frontier plans is one of two very different things:

- **Preference-indifferent tie:** the tied plans have *identical satisfaction vectors* — every preference is equally honored by all of them; they differ only in internals the user expressed no view about. Nothing user-visible is at stake; any deterministic pick is fine.
- **Preference-material tie:** the tied plans have *different, incomparable* satisfaction vectors — a genuine unresolved conflict (plan p honors π₁ but not π₂; plan q the reverse). Here the pick decides which of the user's stated wishes loses, and *that* must not hinge on line order.

v3's tie-break chain used declaration order for both. v3.1 removes declaration order from the language semantics entirely.

### 1.2 The three alternatives, compared

**(A) Conflict reporting, no implicit resolution.** Preference-material ties are compile errors demanding disambiguation. *For:* zero silent surprises. *Against:* ties are a function of the machine model and tuning database, not just the program — a benign machine-model update can newly tie two plans and break a previously-green build; conflicts among *soft* constraints becoming hard errors inverts the meaning of "prefer"; and in practice large preference sets on real frontiers would make this a constant irritant.

**(B) Explicit intra-class ranks.** Syntax makes ordering deliberate: `prefer@1 algorithm = cholesky; prefer@2 memory/rank <= 96GiB` — rank 1 beats rank 2 *within* a class (lexicographic sub-ordering inside the Pareto step). *For:* full expressiveness; ordering is written, not positional. *Against:* does not by itself answer what happens between equally-ranked (or unranked) preferences — it needs a base case.

**(C) Canonical tie-breaking independent of declaration order.** The final pick is a pure, permutation-invariant function of plan content (canonical plan hash). *For:* deterministic, reproducible, order-insensitive; and because §1.4(v3) already prices every unhonored preference in the artifact, the pick is arbitrary but never *silent*. *Against:* on preference-material ties the compiler is still deciding which wish loses, just impartially.

### 1.3 Recommendation: (C) as the base case, (B) as the control, (A) as an opt-in mode

Normative resolution chain within a priority class, replacing v3 §1.3 step 3:

1. Pareto within class (unchanged), refined lexicographically by **explicit ranks** where given — unranked preferences share the lowest rank tier.
2. Final objective (predicted time unless declared) — unchanged.
3. Remaining preference-indifferent ties: **canonical plan hash**. Silent, and rightly so.
4. Remaining preference-material ties: canonical plan hash **plus a mandatory prominent conflict certificate**: the tied satisfaction vectors, the vector-deltas between them, and the line "resolved by canonical order; add `prefer@N` ranks to control this choice." Under build mode `--strict-preferences`, step 4 is instead a compile error carrying the same certificate (alternative A, opt-in — appropriate for locked-down production builds).

Declaration order now has **no semantic effect anywhere**. This strengthens the compilation-reproducibility invariant (v3 I-10): the selected plan is invariant not only across reruns but across permutations of the preference list — restated as **I-10′** in §8.3.

### 1.4 Example

```text
with { prefer algorithm = cholesky,            # strong (unranked)
       prefer memory/rank <= 96GiB }           # strong (unranked)
```
Guard-cost feedback makes these conflict: plan P (cholesky, 103 GiB) vs plan Q (ldlt, 91 GiB), equal predicted time. v3 would have picked P because its line came first — and swapping the lines would have flipped the factorization algorithm of a production solver. v3.1: canonical hash picks (say) Q; the artifact's conflict certificate shows both vectors and the deltas (`P: +12 GiB, algorithm honored; Q: memory honored, algorithm not — Δtime 0`). The user who cares writes `prefer@1 algorithm = cholesky` and the choice becomes theirs, in writing.

---

## 2. Evidence-aware canonicalization for generative search (amends v3 §2.3)

### 2.1 The bug being fixed

v3 required generative proposals to be "canonicalized (schedule-tree hashing modulo commutation)" for deduplication. As stated, that conflates *looking alike* with *being interchangeable*: two schedules with identical loop skeletons can differ in reduction-tree shape (different bits under D2), buffer placement (HBM vs DDR), communication structure, or machine behavior — suppressing one as a "duplicate" of the other is unsound for search completeness and can even hide the only contract-legal candidate.

### 2.2 Equivalence strata

```rust
enum EquivKind {
    Syntactic,                                   // identity of canonical signatures (§2.3)
    Semantic   { domain: SemanticDomain },       // same function in the stated domain (§6)
    ContractRel{ contract: Contract },           // observationally indistinguishable under the
                                                 // active ⟨D, A, E⟩ contract at this scope
    PerfEquiv  { machine: TopoHash, tol: Ratio },// cost vectors within tol on this machine
}
```
These are strictly ordered in *strength of what they license*, and they are **claims**, not booleans: `equiv(a, b, kind)` returns a `ClaimId` whose evidence carries a status like any other claim (`StaticallyDerived` for structural identity, `SmtProved` for semantic equivalence on a fragment, `Predicted`/`Measured` for performance equivalence).

### 2.3 Canonical signatures

The syntactic canonical key is computed over the **full candidate signature**, not the loop skeleton:

```rust
struct CandidateSignature {
    skeleton:  ScheduleTreeHash,      // structure modulo proven-commutative reorderings ONLY
    arith_ord: ArithOrderClass,       // reduction tree specs, accumulation orders, FMA contraction sites
    placement: PlacementMapHash,      // space + layout assignments per buffer
    comm:      CommPatternHash,       // collective algs, epochs, message structure
    launch:    LaunchGeometryHash,    // GPU geometry, stream assignment, affinity-relevant binding
}
```
Two candidates differing in any component have different keys — the v3 bug is unrepresentable. "Modulo commutation" in the skeleton is itself evidence-gated: a reordering is quotiented out only if its commutativity is a `Trusted`-tier fact *and* it is arithmetic-order-neutral (else it shows up in `arith_ord`).

### 2.4 Suppression rules (normative)

A generative proposal `b` may be handled as follows relative to an already-costed `a`:

| Condition | Action |
|---|---|
| `Syntactic` identity of signatures | suppress `b` (it *is* `a`) — the only unconditional dedup |
| `ContractRel{active}` **Established** (proof-class) AND `PerfEquiv{this machine}` **Established by `Measured`** | suppress `b` for this machine; the suppression records both claim IDs; validity bound to the topo hash |
| `ContractRel` Established, `PerfEquiv` only `Predicted` | **deprioritize** `b` (search-order effect only); never suppress on predicted evidence |
| `ContractRel` not established (e.g., different reduction trees under D2; different E-relevant op sets) | keep both, always — they are different programs as far as the contract is concerned |
| any suppression whose justifying claim is later demoted (counterexample, expiry) | the suppression is void; `b` re-enters the candidate pool (the no-retry ledger stores suppressions as revocable events, consistent with §7) |

This preserves the v3 no-retry/termination guarantees — suppression and exclusion both operate on signature-level identity — while making "duplicate" a *proved* judgment rather than a structural impression.

---

## 3. `infer P` / `discover` — user-level evidence campaigns (extends v3 §4; no new evidence status)

### 3.1 Semantics

`infer P` is a **directive to the evidence-production system**, orthogonal to the assertion taxonomy: it asserts nothing, trusts nothing, and is not an evidence status. It requests a budgeted, multi-method attempt to move claim P out of the `Unknown` state (§5):

```text
infer P with {
    budget  = { compile: 30s solver + 8 ai-calls, runtime_guard: <= 0.5% of region cost },
    methods = { static, smt, formal?, ai, runtime },      # subset selection; default: all but formal
    on = { established: proceed, refuted: error | branch, unknown: proceed | warn }
}
discover properties(A) with { budget = ... }               # open-ended hypothesis generation + auto-infer
```

**Campaign structure.** An `infer` runs the producer cascade in cost order, short-circuiting on resolution: property-algebra closure → static/polyhedral analysis → SMT on the decidable fragment → AI hypothesis generation *feeding back into* the deterministic verifiers (an AI-proposed lemma or decomposition is only ever a hint to the provers) → runtime-check synthesis (a guard proposal, priced against the runtime budget). Possible outcomes, in claim-state terms (§5): **Established**, **Refuted** (with the counterexample — `infer` can disprove, which `prove` cannot express), **Conditional** ("P holds if Q"; the campaign returns Q as a new claim the planner may guard or recursively infer), or **Unknown** with an honest account of what was tried and what establishing it would cost.

**Distinctions from the v3 taxonomy, exactly:** `prove` demands static success or errors; `assert` trusts now and pays a mandatory guard; `check` runs at execution time with no static attempt; `infer` is best-effort across *all* methods, budgeted, and failure is not an error unless `on.unknown = error`.

### 3.2 Budgets, caching, and planner interaction

Compile-time and runtime budgets are separate ledgers (solver seconds/AI calls vs. guard-cost ceiling as a fraction of the enclosing region's predicted cost). Campaign results are cached keyed by `(claim statement, scope, module hash)` — rebuilds do not re-pay. The planner integrates through §5's state machine: an `Unknown` claim that gates a valuable transformation generates a *priced unlock report* ("establishing `symmetric(A)` would enable syrk: −38% flops; cheapest known path: runtime check at 0.4%") — and an explicit `infer` is the user accepting that price. `discover` is the same machinery in generative mode: producers enumerate candidate claims (property-algebra closure over the region, SRP-S5 pattern proposals, AI Level 1), each auto-inferred within the shared budget; the Semantic Recovery Pipeline's property-inference stage (v3 §8, S5) is hereby *defined as* an internal `discover` invocation — one mechanism, two entry points.

---

## 4. Trusted external specifications: SpecPacks (amends v3 §4 `promise`)

### 4.1 Separation of concerns

v3's `promise P by <id>` bundled three things that must be independent: (a) the *specification content* (what a foreign interface computes), (b) the *trust policy* (which specifications this build accepts), and (c) the *identity/attestation infrastructure* (how we know who vouched). The MVP needs (a) and (b) immediately; (c) is deferred behind a stable interface.

### 4.2 Types

```rust
struct SpecPack {
    name: String,                        // "reference-blas", "openmpi", "vendor-rocblas"
    applies_to: { lib: LibId, versions: VersionRange, abi: AbiId },
    claims: Vec<SpecClaim>,
    provenance: SpecProvenance,
}
struct SpecClaim {
    stmt: Statement,                     // Equivalence(dgemm(args), math.gemm semantics) | Property(...)
    domain: SemanticDomain,              // §6 — "dgemm ≡ C←αAB+βC" is claimed in IeeeFp-with-classical-
                                         // error-bound terms, never in RealField terms
    contract_profile: Contract,          // what ⟨D,A,E⟩ the routine itself provides
                                         //   (e.g. vendor gemm: D3 not D2 across library versions;
                                         //    MPI allreduce: D2 only for documented op/alg settings)
    validation: Option<DiffCheckSpec>,   // opt-in differential spot-check recipe
}
enum SpecProvenance {
    Bundled,                             // shipped in-repo with the compiler, code-reviewed — the MVP trust root
    LocalFile { path, sha256 },          // site-provided; enabled per-hash by build configuration
    Signed { attestation: AttestationRef },  // FUTURE: signed statement binding pack-hash to signer;
                                             // interface reserved, verification infra deferred
}
```

Evidence status `Promised{who, signature}` is **replaced** by `Specified{pack, provenance}` (invariant change I-11, §8.3). Admissibility is unchanged in shape — `Specified` is the only user-supplied status admissible for `Equivalence` claims — but the *trust policy* is now a build-level allowlist over provenance classes: default = `Bundled` only; `LocalFile` requires an explicit flag naming the hash; `Signed` reserved. Refutation supremacy carries over verbatim: a disprovable SpecClaim is a compile error regardless of provenance.

### 4.3 What ships bundled, and validation

MVP bundles four packs: reference BLAS/LAPACK semantics (the mathematical contracts and classical error bounds of `gemm/potrf/trsm/...`), an MPI pack (collective semantics; determinism profiles *per implementation family and settings* — deliberately conservative defaults), and thin vendor packs (cuBLAS/rocBLAS as "BLAS semantics with contract profile D3, plus documented exceptions"). `DiffCheckSpec` enables install-time and version-bump canaries: sampled comparison against the reference implementation (including adversarial FP inputs: NaN, ±Inf, ±0, subnormals — the E-axis features are exactly what vendor libraries most often treat loosely), attaching corroborating `Measured` evidence or a `Counterexample` that demotes the pack claim and flags every consuming plan. The inline `promise` directive survives only as sugar for a single-claim `LocalFile`-class SpecPack embedded in source — same admissibility, same artifact prominence, recorded author string, no cryptography pretended.

---

## 5. Multi-valued claim state (amends v3 §7 / v2 §7)

### 5.1 States and the aggregation fold

```rust
enum ClaimState {
    Established { strongest: EvidenceStatus },   // live admissible positive evidence, no live refutation
    Refuted     { by: EvidenceId },              // live refutation-class evidence, no live positive proof-class
    Conditional { on: Vec<ClaimId> },            // positive evidence whose side conditions are not Established
    Contested   { pair: (EvidenceId, EvidenceId) },  // live proof-class positive AND live refutation — alarm
    Unknown     { hypothesized: Option<EvidenceId> },// nothing above; AI/Predicted support noted, not counted
}
```

`state(claim, scope, epoch)` is a **deterministic, order-independent fold** over the claim's *live* evidence set (liveness = temporal scope coverage per v3 §3 ∧ validity bindings ∧ not superseded by demotion events):

1. Restrict to evidence whose scope **intersects** the queried scope; all comparisons below are per-intersection. (A proof for shape class S and a counterexample for an instance outside S never meet — no false Contested.)
2. Refutation-class present (live `Counterexample`, proved negation) and no live proof-class positive → **Refuted**.
3. Proof-class positive present (`StaticallyDerived | SmtProved | FormallyProved | RuntimeChecked`-in-window `| Specified | Assumed`) and no live refutation → **Established**, recording the strongest status (per-gate admissibility still applies on top: Established-by-`Assumed` passes only gates that admit `Assumed`).
4. Both present on intersecting scopes → **Contested**.
5. Positive evidence carrying undischarged side conditions → **Conditional{on}**.
6. Otherwise → **Unknown**; `AiInferred`/`Predicted` support is annotated but contributes nothing to steps 2–5.

Two principles the fold encodes, per your framing: **absence of proof is not refutation** — a failed `prove`, an SMT timeout, an exhausted `infer` produce *no evidence object at all*, so the state remains Unknown; and **a hypothesis cannot contest anything** — Contested requires refutation-class vs proof-class conflict; `AiInferred` vs Unknown is just Unknown-with-a-hunch.

### 5.2 Gate and planner interaction

| State | Legality/contract gates | Planning |
|---|---|---|
| Established | pass, subject to per-gate status admissibility | normal |
| Conditional | pass **iff** every condition is Established or the gate lifts it into a compiled guard (auto-`assert` of the condition, priced) | conditions appear as unlock prices |
| Unknown | barred | eligible for `infer`; priced unlock reports (§3.2); AI-support may order `infer` spending — never substitute for it |
| Refuted | barred; every `Decision` depending on the claim is invalidated (standard feedback path) | alternatives excluding the claim are searched |
| Contested | **compile error in the affected scope.** Contested means the system caught itself: an unsound transfer rule, a stale window, a bad SpecPack, a verifier bug. Picking a side would be exactly the "confidence over proof" failure this architecture exists to prevent. The error carries the contradiction pair and both derivation DAGs | none until resolved |

---

## 6. Semantic domains and bridge obligations (amends v3 §5 rewrite DB and all Equivalence claims)

### 6.1 Domains

```rust
enum SemanticDomain {
    RealField, ComplexField,                      // exact mathematics
    ExactInt { width: Option<u32>, overflow: Wrap|Trap|Undef },
    IeeeFp   { format: FpFormat, rounding: RMode, exceptional: EProfile },  // the E-axis feature set
    ContractNumeric { contract: Contract },        // equivalence "up to ⟨D,A,E⟩": identical bits where D
                                                   // demands, within the A budget, preserving E features
}
```
Every `Equivalence` claim (and every rewrite-rule proof, SpecClaim, translation-validation result, and `ContractRel` canonicalization judgment) now carries its domain. `Equivalence{RealField}` and `Equivalence{IeeeFp}` are *different claims about the same syntax*; conflating them was a latent soundness hole in v2/v3 — a Lean proof of `(AB)C = A(BC)` over ℝ was formally admissible evidence for an FP rewrite it does not in fact justify.

### 6.2 Bridge obligations (normative)

A theorem in domain X authorizes a transformation on IR in domain Y only through a **bridge** producing a `ContractNumeric` equivalence at the enabling contract point:

1. **Value bridge (A axis).** Rounding-error analysis of both sides yields a bound on the FP-value divergence; it must fit the active accuracy budget. Evidence: `SmtProved` (FP theory) for fixed shapes; `StaticallyDerived` via the classical error-model algebra (the `γₙ = nu/(1−nu)` calculus) for parametric shapes. This is where "exact in ℝ, different bits in FP" is *charged*, per rewrite, per shape class.
2. **Exceptional bridge (E axis).** Case analysis over the protected feature set: NaN propagation, ±Inf, signed zero, subnormals, flag effects. Most classic ℝ-identities fail some case (`x·0→0` vs NaN/Inf; `x−x→0` vs NaN; `x/x→1`); each rule's exceptional analysis is a mandatory database field, and the rule is enabled only at E-levels (or under `Established` finiteness claims) that make the failing cases unobservable.
3. **Determinism bridge (D axis).** Classify the rewrite's arithmetic-order effect — order-preserving / config-pure reorder / data- or timing-dependent reorder — and check against the D level, exactly the v2 §6.3 gating discipline, now derived from the rule's own metadata rather than a hand-maintained table row.
4. **Exact-domain analog.** `ExactInt` theorems used on index/layout arithmetic bridge via range analysis discharging overflow side conditions — same shape, cheaper obligations.

The v3 §5 rewrite-database entry is amended to `{pattern, side_conditions, proof_domain, bridges: {A, E, D}, tier, evidence}`, and the tier invariant is strengthened (**I-12**): *`Trusted` requires the proof in its stated domain **and** discharged bridges for every contract point at which the rule is enabled; the contract gate consults the bridge, never the raw theorem. No ℝ-only theorem fires on FP IR, ever.* Convergence bonus: `ContractNumeric` is precisely what §2's `ContractRel` canonicalization and the SRP's translation validation consume — three mechanisms, one judgment.

---

## 7. Concurrency of the `Claim + Evidence + Decision` spine

### 7.1 The model: append-only events, deterministic folds, versioned decisions

The v2/v3 spine was written as if a single sequential pass populates it. The fix is small because the design was already nearly immutable:

- **Evidence and feedback are immutable, append-only events.** Nothing is ever mutated in place; demotions, expiries, suppression revocations (§2.4) are *new* events referencing their targets. The event log with sequence numbers **is** the artifact's derivation DAG and audit trail — event-sourcing falls out rather than being built.
- **Claim state is a deterministic fold over an event set** (§5.1) — commutative and associative by construction, since the fold consumes the *set* of live evidence, not an arrival order. Concurrent producers (analyses, SMT workers, AI proposers, autotuning measurements, runtime telemetry) therefore need no coordination beyond set union: a grow-only event table with atomic append is the entire concurrency story at the claim level.
- **Decisions are epoch-versioned with optimistic validation.** A planner transaction reads a snapshot epoch; each `Decision` records the epoch and the states of its `depends_on` claims; at commit, if any depended-on claim's state changed across the interval, the decision re-plans — which is not new machinery, it is the §1(v2) feedback loop re-expressed as concurrency control. Exclusion certificates and suppressions join `depends_on` like any claim.

### 7.2 What the MVP builds, and what it deliberately does not

MVP: one process, one global epoch counter, an in-memory grow-only table — the interfaces (`append(evidence) → seq`, `state(claim, scope, epoch)`, `snapshot() → epoch`) are what gets frozen, and single-threaded code pays nothing for them. Explicitly *not* built now, and provably not needed for correctness of the model: distributed consensus, cross-claim transactional atomicity (per-claim folds are independent; cross-claim consistency is exactly what epochs give), or a persistent event-store product. Distributed compilation later = partitioned logs + epoch exchange at merge points; the fold's order-independence is the property that makes that a plumbing exercise rather than a redesign. (Readers who want a name: the evidence store is a grow-only set CRDT; we simply refrain from needing the rest of that literature.)

---

## 8. Freeze assessment

### 8.1 Critical scan: what could still be expensive to retrofit

Applying the test *"would deferring this force a redesign of frozen types, or only new implementations behind frozen seams?"* to everything not yet designed:

1. **Plan↔Runtime interface — the one genuine gap. Must be typed before the freeze.** Plans now contain guards, conditional branches keyed on claim events, fallback edges, telemetry emission, and abort semantics — but no document types the boundary the *runtime* sees. Left implicit, every layer would grow ad-hoc runtime hooks, and that is expensive to unwind. The freeze therefore includes this minimal interface, which is deliberately dumb:

```rust
trait PlanRuntime {
    fn eval_guard(&mut self, g: GuardRef, ctx: &Instance) -> GuardOutcome;   // Pass | Fail{cex: ConcreteInstance}
    fn select_branch(&mut self, fork: ForkRef) -> BranchRef;                 // pure function of guard/check
                                                                             // outcomes recorded this run
    fn emit(&mut self, ev: RuntimeEvent);        // PropertyDiscovered | Counterexample | NumericalViolation
                                                 //   | PerformanceMeasurement — the §1(v2) runtime feedback,
                                                 //   buffered, never blocking the critical path
    fn fail(&mut self, mode: FailMode);          // Abort{cex} | TakeFallback{edge} — per §5.3(v2) control rules
}
```
    Plans are immutable at runtime (unchanged); `select_branch` is deterministic given the run's guard outcomes, preserving D2. This is ~a week of design already done above and days of MVP implementation; it blocks the freeze until adopted, then unblocks it.

2. **Non-affine ownership (sparse/irregular/adaptive data) — a named seam, not a blocker.** DIR's coverage/disjointness *proofs* are affine-only; but ownership was already abstracted as a first-class function (`owner(fn)`), and the evidence system supplies the extension path: an `Opaque` ownership function whose coverage claim is `RuntimeChecked` rather than `StaticallyDerived`, with CIR volume conservation checked against enumerated (not symbolically derived) transfer sets. Frozen types don't change; verifier implementations grow. Post-freeze.
3. **Plan linking / separate compilation** — composing independently compiled plans under contracts. The pieces exist (contract meet at interfaces; SpecPacks are exactly "a compiled unit's exported claims"); a linked plan is a plan whose foreign calls carry SpecPack-shaped claim sets. Seam named; post-freeze.
4. **Fault tolerance in plans** — `FailureModel` lives in placement domains; plan-level recovery (checkpoint placement as a certificate-priced decision) is additive: new decisions and feedback kinds, no type changes. Post-freeze.
5. **Runtime specialization of symbolic dims** — already implied by `Shaped` scopes, `Symbolic` search domains, and shape-keyed tuning; a specializer is an implementation, not a foundation. Post-freeze.

Nothing else surfaced that fails the retrofit test. Notably, the three latent soundness issues that *would* have been brutal to retrofit — semantic-domain confusion on equivalence (§6), structural canonicalization suppressing contract-distinct candidates (§2), and binary claim truth (§5) — are the ones fixed in this document; that is why v3.1 exists.

### 8.2 Verdict

**Freeze, after adopting v3.1 including the `PlanRuntime` interface of §8.1.** The foundation now has: a single spine with defined concurrency semantics; sound treatment of truth (five-state claims), of trust (assertion taxonomy + SpecPacks), of meaning (semantic domains + bridges), of preference (order-free typed Pareto), and of search (signature-canonical, region-excluding, budget-terminating) — with every extension pressure identified above landing behind a frozen seam. Further foundational iteration has passed the point of diminishing returns; the remaining risks are implementation-shaped, and only contact with real code retires those.

### 8.3 Changed invariants (v3 → v3.1)

| # | Was (v3) | Now (v3.1) |
|---|---|---|
| I-10′ | Compilation reproducible given (program, machine model, tuning DB); tie-breaks used declaration order | Plan selection additionally **invariant under permutation of preference declarations**; declaration order has no semantic effect; preference-material ties resolve by canonical hash + mandatory conflict certificate, or error under `--strict-preferences`; explicit `prefer@N` ranks are the only ordering semantics (§1) |
| I-13 | Generative dedup by structural canonical hashing | Suppression requires **Established contract-relative equivalence + Measured performance equivalence** (or syntactic signature identity); signatures include arithmetic order, placement, comm, launch; suppressions are revocable events (§2) |
| I-3′ | `Promised{who, signature}` admissible for foreign-boundary equivalence | Status replaced by `Specified{pack, provenance}`; trust = build allowlist over provenance classes (`Bundled` default); attestation infra deferred behind `SpecProvenance::Signed`; refutation supremacy unchanged (§4, I-11) |
| I-14 | Claims implicitly two-state (evidence present/absent) | Five-state fold {Established, Refuted, Conditional, Contested, Unknown}; failed proof ⇒ no evidence ⇒ Unknown; hypotheses cannot contest; **Contested is a compile error**, never resolved by picking a side (§5) |
| I-12 | Trusted tier = "proof exists" | Trusted = proof **in its stated semantic domain** + discharged A/E/D bridges per enabling contract point; no ℝ-only theorem fires on FP IR (§6) |
| I-15 | (implicit) single sequential analysis pass | Evidence/feedback append-only immutable events; claim state a commutative fold; decisions epoch-versioned with optimistic revalidation; frozen interfaces `append/state/snapshot` (§7) |
| I-16 | (absent) | Plans interact with execution only through the four-method `PlanRuntime` interface; plans immutable at runtime; branch selection deterministic given guard outcomes (§8.1) |

### 8.4 MVP deltas from this document (small, and two are simplifications)

(1) Claim-state fold + event table: days, and it *replaces* ad-hoc evidence bookkeeping. (2) SpecPacks with `Bundled` provenance only: replaces the v3 promise/signature plumbing — a net simplification, and the MVP already lives on BLAS/LAPACK/MPI specs. (3) Tag every equivalence claim with its semantic domain from day one — nearly free now, brutal later; MVP bridges are the classical-error-model algebra only. (4) Tie-break change: trivial. (5) Candidate signatures: schedule search is post-MVP, but the signature struct ships with the tuning DB key format now so measurements are never keyed ambiguously. (6) `infer`: a thin orchestrator over producers the MVP has anyway (property algebra, dimension solver, guard synthesis); AI and SMT stages are stubs that return Unknown. (7) `PlanRuntime`: implemented trivially (guards + branch select + a log file) in Milestone 3 where the first conditional plan appears.
