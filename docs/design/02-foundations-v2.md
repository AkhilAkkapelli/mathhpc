# HPC Architecture v2 — Foundational Revisions

**Supersedes and amends v1 (`hpc-architecture.md`). v1's IR layer definitions (§3) and machine model (§1) stand except where amended here. This document fixes the seven foundations: typed upward feedback, placement domains, expanded AI roles, generalized optimization certificates, four-level user control, the reproducibility/correctness lattice, and the evidence system — then shows all seven interacting in two end-to-end examples.**

Type/interface definitions are given in a Rust-flavored ADT notation for compiler-internal structures, and in surface/IR syntax where the construct is user- or IR-visible. The notation is normative for *structure*, not for implementation language.

---

## Amendment summary (what changes in v1 and why)

| v1 principle | v2 revision |
|---|---|
| "Information flows down as refinement, up only as *cost feedback*" | Semantics still flow down only as refinement. Upward flow is a **typed feedback bus** carrying seven kinds of messages, including semantic discoveries and counterexamples — never direct mutation of upper IR. (§1) |
| "NUMA, GPUs, MPI ranks are all just grid levels" | Retracted as a *semantic* identity. All three participate in one hierarchical **PlacementDomain** framework, but with kind-indexed accessibility/ownership/sync/consistency/transfer/cost/failure semantics. "SUMMA at node scope" survives as a statement about *plan shape*, not about semantics. (§2) |
| AI limited to tuning seeds, rank embedding, calibration | AI operates at **three levels** — semantic recognition, planning (including proposing new rewrite rules), prediction/search — behind an unchanged authority boundary. (§3) |
| Communication lower bounds as stopping criteria | Generalized to an **optimization-certificate framework** over seven resource classes. (§4) |
| "User constrains, never specifies" | Retracted as an absolute. **Four control levels**: `auto` / `prefer` / `require` / `use`. Hard requirements and expert specifications are never silently overridden. (§5) |
| Reproducibility contract as a flat enum | A **two-axis contract lattice** (determinism × accuracy) with per-transformation gates at every IR layer. (§6) |
| Property annotations "user-asserted or derived with certificate" | Generalized to a first-class **evidence system**: every claim in the compiler carries structured evidence with one of eight statuses; legality-relevant decisions require proof-class evidence. (§7) |

The seven systems are not independent features; they share one spine: **every fact is a `Claim` with `Evidence`; every decision is a `Decision` with a control level and implicating claims; feedback messages reference decisions and carry evidence; contracts and certificates are particular claim families.** That spine is defined first where needed and reused throughout.

Common identifiers used below:

```rust
struct ClaimId(u64);   struct EvidenceId(u64);   struct DecisionId(u64);
enum LayerId { MIR, AIR, PIR, DIR, CIR, LIR, SIR, HIR, RT }   // RT = runtime

struct Decision {
    id: DecisionId,
    layer: LayerId,
    subject: IrRef,                  // the op/value/attribute decided
    control: ControlLevel,           // §5 — who fixed this and how hard
    alternatives: AltSetRef,         // the proven-legal alternative set searched (empty if `use`-pinned)
    depends_on: Vec<ClaimId>,        // claims whose falsification invalidates this decision
    excluded: Vec<AltRef>,           // alternatives ruled out by prior feedback (anti-livelock, §1.3)
}
```

---

## 1. Typed upward feedback

### 1.1 The type

```rust
enum FeedbackKind {
    /// No legal plan exists at `origin` under the decisions above it.
    Infeasible {
        violated: InvariantRef,          // which layer invariant cannot be satisfied
        witness:  Witness,               // e.g. the unsatisfiable constraint system core
    },
    /// A plan exists but exceeds a physical/logical resource.
    ResourceViolation {
        resource: ResourceKind,          // Memory(space), PinnedBudget, Cores, Streams, Regs, Smem, NICRails, ...
        required: SymOrConcrete<Qty>,
        available: Qty,
        at: TopoRef,                     // where in the machine model
    },
    /// An accuracy/stability contract cannot be met, or was observed violated.
    NumericalViolation {
        contract: ContractRef,           // §6
        margin:   ErrorTerm,             // bound shortfall (static) or observed residual (runtime)
        phase:    Static | Runtime { instance: InstanceRef },
    },
    /// A concrete input falsifying a claim (property, equivalence, dependence summary).
    Counterexample {
        claim: ClaimId,
        instance: ConcreteInstance,      // minimized where possible
    },
    /// New knowledge, flowing UP: a property proven/observed below.
    PropertyDiscovered {
        subject: IrRef,
        property: Prop,                  // cond ≤ 3e4; effective sparsity; achieved overlap capability; ...
        evidence: EvidenceId,            // status decides what it may gate (§7)
    },
    /// Model-based cost of a candidate plan node (the v1 upward channel, now one variant among seven).
    CostEstimate       { plan: PlanNodeRef, cost: CostVector, evidence: EvidenceId },  // status = Predicted
    /// Ground truth from the harness or from production runs.
    PerformanceMeasurement { plan: PlanNodeRef, measured: CostVector, machine: TopoHash, evidence: EvidenceId },  // status = Measured
}

struct Feedback {
    kind: FeedbackKind,
    origin: LayerId,
    implicates: Vec<DecisionId>,     // upstream decisions this message bears on (computed, not guessed:
                                     // every invariant/contract instance records the decisions it depends on)
    severity: Severity,              // Blocking | Advisory
    evidence: EvidenceId,
}
enum Severity { Blocking, Advisory }
```

### 1.2 Routing and handling

Feedback is posted to a per-compilation **feedback bus**; it never edits IR. The routing rule is mechanical:

> **Target = the lowest (most refined) layer that owns an implicated decision whose control level permits revision.**

Each layer's planner implements one interface:

```rust
trait FeedbackHandler {
    /// Must be total. Returns what the layer DID about it.
    fn handle(&mut self, fb: &Feedback) -> Response;
}
enum Response {
    Revised { decision: DecisionId, new_alt: AltRef },   // decision re-made; lowering below it re-runs
    Delegated { to: LayerId },                           // no owned decision can absorb it; pass up
    Waived { justification: EvidenceId },                // Advisory only — recorded in the plan artifact
    Escalated(UserError),                                // Blocking + implicates a require/use decision (§5.3)
}
```

Handling protocol:

1. **Blocking** feedback must reach a `Revised` or `Escalated` response before plan finalization (compile-time) or before the runtime fallback edge is taken (runtime feedback — see the SPD example, §8). `Waived` is a type error for blocking feedback.
2. A layer may only revise **its own** decisions. `PropertyDiscovered` never directly enables a transformation in an upper layer: it is delivered as a new `Claim` + `Evidence`, and the upper layer's own planner/verifier decides what it now permits, under the evidence rules of §7. Upward flow is *information*, downward flow remains the only channel for *semantics*.
3. Runtime-origin feedback (`Counterexample`, runtime `NumericalViolation`, `PerformanceMeasurement`) is double-buffered: it affects (a) the *current execution* only through pre-compiled fallback edges (plans are immutable at runtime), and (b) *future compilations* through the claim/tuning databases.

### 1.3 Termination: monotone replanning

The replanning loop is a fixpoint with two guarantees:

- **Monotonicity (anti-livelock).** A `Revised` response must add the rejected alternative to `Decision.excluded` with the feedback as justification. The alternative sets are finite (or finitely abstracted, e.g. block sizes range over a declared grid); excluded sets grow strictly; therefore each decision can be revised only finitely often.
- **Budget + honest failure.** A global replanning budget (rounds and wall time) bounds the loop; on exhaustion the compiler emits the best *feasible* plan found, or — if no feasible plan was found — an error carrying the full feedback trail. It never emits a plan with unresolved blocking feedback.

### 1.4 Worked micro-examples of routing

- LIR finds double-buffered lookahead-2 panels exceed HBM: `ResourceViolation{Memory(hbm), …}` implicates SIR's `lookahead=2` and DIR's `nb`. SIR is lower and owns `lookahead` (control = `auto`) → SIR revises to `lookahead=1`. DIR never wakes up. 
- Same violation, but the user wrote `use lookahead=2` and `use block_size=1024`: both implicated decisions are pinned → `Escalated`: compile error quoting the capacity proof and both pins (§5.3). No silent override.
- Runtime `potrf` hits a non-positive pivot: `Counterexample{claim: spd(A)}` → the spd claim (UserAsserted) is *demoted* (§7.4), the pre-compiled fallback edge to the pivoted-factorization branch fires, and the counterexample instance is stored against the claim for future compilations of this program.

---

## 2. Placement domains

### 2.1 The abstraction

One hierarchical placement framework; three (extensible) domain kinds with distinct semantics. Distribution IR's grids, ownership functions, block-cyclic machinery, and load-balance math are *shared*; what an edge in the hierarchy **means** is kind-indexed.

```rust
trait PlacementDomain {
    fn kind(&self) -> DomainKind;
    fn children(&self) -> &[DomainRef];
    fn topo(&self) -> TopoRef;                          // binding into the §1(v1) machine model

    // ---- the seven kind-indexed semantic dimensions ----
    fn accessibility(&self, from: DomainRef) -> Access;
    fn ownership(&self)   -> OwnershipModel;
    fn sync(&self)        -> SyncModel;
    fn consistency(&self) -> ConsistencyModel;
    fn transfer(&self, to: DomainRef) -> TransferMechanism;
    fn cost(&self)        -> CostTables;                 // α–β(–γ) tables at this level
    fn failure(&self)     -> FailureModel;
}

enum DomainKind { Distributed, SharedMemory, Accelerator }

enum Access {
    DirectLoadStore { relative_cost: CostRef },   // legal, possibly slow (remote NUMA)
    ExplicitTransferOnly,                          // must appear as comm.* / mem.copy in IR
    Unreachable,
}
enum OwnershipModel {
    PartitionedAddressSpaces,          // ownership = the only access (Distributed)
    SharedCoherent { affinity: Placement },  // ownership = performance home, not access right (SharedMemory)
    DeviceResident { host_visible: bool },   // ownership = residency; outside access illegal or paged (Accelerator)
}
enum SyncModel {
    Messages { collectives: bool, one_sided: bool },     // sync is communication (Distributed)
    SharedPrimitives { atomics: AtomicClass, barriers: bool, locks: bool },
    StreamsEvents { scopes: DeviceScopeSet, host_sync: bool },
}
enum ConsistencyModel {
    MessagePassing,                        // consistency points = message completion
    HwCoherent { model: AcqRelSC },        // hardware cache coherence, acquire/release
    Scoped { levels: [Thread, Block, Device, System] },  // GPU scoped memory model
}
enum TransferMechanism {
    Network { alpha: Time, beta: TimePerByte, rails: u8, gpu_direct: bool },
    CoherentMigration { granularity: CacheLine, page_migration: bool },
    Dma { engines: u8, requires_pinned: bool, peer: bool },
}
enum FailureModel {
    PartialLoss { unit: DomainRef, detect: FailureDetect, recover: RecoveryClass }, // rank/node loss
    FailTogether,                                                   // shared-memory: one fault kills the domain
    DeviceReset { recover: ReinitCost },                            // GPU: resettable, state lost
}
```

### 2.2 Distribution IR over domains (amends v1 §3.4)

`dist.tensor` and `dist.grid` are now parameterized by a domain path, and the *type checker* enforces kind-appropriate obligations:

```mlir
%cluster = dom.distributed  <ranks, topo=cluster>
%node    = dom.shared       <numa x 4, parent=%cluster.rank>
%gpus    = dom.accelerator  <gpu x 8, parent=%node>

%g   = dist.grid grid<16x16> on %cluster            // Distributed level
%gN  = dist.grid grid<2x4>   on %gpus               // Accelerator level, nested per rank
%Ad  = dist.tensor %A on %g  by [bc(%nb), bc(%nb)]
       nested on %gN by [block, block]              // hierarchical distribution, kind-aware
```

What kind-awareness changes, dimension by dimension:

| Dimension | Distributed (ranks) | SharedMemory (NUMA/cores) | Accelerator (GPUs) |
|---|---|---|---|
| **Accessibility** | `ExplicitTransferOnly`: a rank reading a non-owned block *must* lower to `comm.*`; anything else is a type error | `DirectLoadStore{remote-cost}`: "distribution" is a placement/affinity plan; remote access is legal, costed, and flagged when it exceeds a traffic threshold | `ExplicitTransferOnly` from host/peer unless topology declares coherent links; then `DirectLoadStore` with steep cost and an opt-in |
| **Ownership** | Partition = correctness (coverage/disjointness proofs are *correctness* obligations) | Affinity = performance (coverage proof still runs, but violations are performance diagnostics, not errors) | Residency = correctness for kernel operands (LIR space check) |
| **Sync** | CIR ops only; no atomics across ranks | atomics/barriers legal; PIR reductions may lower to shared-memory trees | streams/events (SIR); device-scoped atomics gated by contract (§6) |
| **Consistency** | message completion is the only ordering | acq/rel discipline inserted by SIR where PIR deps cross threads | scoped fences; host sync points explicit |
| **Transfer** | `Network{α,β}` — CIR collectives/p2p | `CoherentMigration` — first-touch, page migration, per-NUMA replication as `mem.copy`-like ops with cache-line cost model | `Dma{engines, pinned}` — `mem.copy` on copy engines; peer DMA on declared links |
| **Verifier obligations** | deadlock-freedom, matching totality, volume conservation | race-freedom (from PIR deps), first-touch/affinity consistency | stream/event affine discipline, space reachability, pinned budget |
| **Failure** | `PartialLoss{rank}` → optional plan-level recovery hooks (checkpoint interval decided against the certificate framework's cost) | `FailTogether` — nothing to plan | `DeviceReset` → re-init cost enters the plan's risk annotation |

The v1 claim "multi-GPU GEMM is SUMMA at node scope" is thus restated precisely: the **ownership functions and the pipelined-broadcast plan shape are shared** (same DIR/CIR planning code), while the lowering of each `comm.bcast` differs by domain kind — MPI/RCCL collective at the Distributed level, peer-DMA ring at the Accelerator level, per-NUMA replication copy at the SharedMemory level — and the verifier runs different obligation sets on each. One planner, three semantics, zero pretense that a NUMA hop is a message.

---

## 3. AI at three levels

The authority boundary is unchanged and is now enforceable *by type*: every AI output is a `Claim` whose evidence status is `AiInferred` (§7), and legality gates statically reject that status. What expands is where AI may *propose*.

### Level 1 — Semantic recognition (of imported/legacy code)

Input: LLVM IR / Fortran / C loop nests, library call sites, whole legacy kernels. Output: **MIR hypotheses** — "this region computes `math.matmul(A, Bᵀ)`", "this array is symmetric here", "this loop is a stencil of radius 2", "this is a Sylvester solve".

```rust
struct SemanticHypothesis {
    region: LowIrRef,
    proposed: MirFragment,               // the mathematical meaning, as real MIR
    preconditions: Vec<Claim>,           // no-alias(A,C), n==m, contiguous strides, ...
    evidence: EvidenceId,                // AiInferred{model, score}
}
```

Every hypothesis enters a fixed **verification pipeline** (deterministic, §8.2 shows it running): polyhedral raising for affine equivalence → SMT for side conditions → runtime-guarded code versioning for the residue → the original code retained as the semantic anchor and fallback. A hypothesis that survives is *re-stated* with proof-class evidence; one that doesn't is either guarded or dropped. AI here changes what the compiler can *see*, never what it may *assume*.

### Level 2 — Planning

AI may propose, at any layer: algorithm choices (`algo.alt` priors), transformation sequences (fusion groupings, tiling structures), distributions and grid shapes, schedules, hardware mappings — and, new in v2, **rewrite rules themselves**:

```rust
struct ProposedRewrite {
    lhs: MirPattern, rhs: MirPattern,
    side_conditions: Vec<Claim>,
    obligation: ProofObligation,     // the equivalence theorem the rule asserts
    evidence: EvidenceId,            // AiInferred until the obligation is discharged
}
```

A proposed rewrite is admitted to the (audited) rule database only when its obligation is discharged: `SmtProved` for bounded/decidable fragments (fixed small dims + symbolic coefficients often suffices for linear-algebra identities), `FormallyProved` (machine-checked, Lean/Coq-class) for general symbolic-dimension rules. Until then it may run only in *versioned* form — rewritten code guarded by a runtime equivalence check against the original — which is usually pointless for performance rules, so in practice: **no proof, no rule**. This is the one place v2 deliberately accepts slower AI payoff for an uncorrupted rule set.

### Level 3 — Prediction and search

Surrogate cost models (proposal distributions for autotuning — the measured harness remains ground truth), learned calibration residuals on the analytical α–β/roofline models, search policies for the quadratic-assignment rank-embedding problem and for the joint (grid, nb, precision, replication) planning space, and diagnostics (trace + plan artifact → natural-language performance explanation with suggested `prefer`-level experiments).

### The boundary, restated as a checklist

AI may propose: mathematical meaning, properties, algorithms, transformations, schedules, mappings, rewrite rules, costs, search moves, explanations. AI may never be the establishing authority for: semantic equivalence, transformation legality, numerical accuracy/stability satisfaction, memory safety, communication correctness (matching/deadlock), contract satisfaction (§6). Mechanically: those five gates accept only evidence in `{UserAsserted(policy-gated), StaticallyDerived, SmtProved, FormallyProved, RuntimeChecked(guard-in-place)}` — `AiInferred` and `Predicted` are not in the set, and there is no score threshold that promotes them (§7.3).

---

## 4. Optimization certificates

The v1 communication-lower-bound stopping rule generalizes to one framework over seven resource classes.

```rust
enum ResourceClass {
    Flops,                       // e.g. 2n²k exact for classical gemm; Ω(n^2.37..) if algorithm free
    CommVolume { level: NetLevel },      // Ω(n³/(P·√M)) memory-dependent; Ω(n²/√(cP)) with replication c
    Messages,                    // latency bound: Ω(√P) for 2D gemm; Ω(c^{3/2}-reduced) for 2.5D
    MemIo { level: MemLevel },   // red–blue pebbling: gemm Ω(n³/√M_cache) per level of the hierarchy
    CriticalPath,                // DAG depth: blocked Cholesky Θ(n/nb) — the strong-scaling wall
    SyncPoints,                  // e.g. Ω(#panels) synchronizations for right-looking factorizations w/o lookahead depth
    Energy,                      // derived: flops·ε_flop^min + Σ_level MemIo_lb(level)·ε_byte(level) + Vol_lb·ε_net
}

struct Certificate {
    quantity: ResourceClass,
    bound: Bound,
    candidate: SymOrMeasured<Qty>,       // the plan's value for this quantity
    gap: Ratio,                           // candidate / bound
}
enum Bound {
    Known { expr: SymExpr, assumptions: Vec<Claim>, provenance: BoundProvenance },
    Unknown,                              // explicit — the planner must never fabricate a bound
}
enum BoundProvenance { Theorem(CitationId), DerivedComposition, TrivialCounting }
```

**Planner semantics.**
- The plan artifact reports the full certificate vector for every major phase. `Unknown` is printed as such.
- **Stopping rule:** stop refining a phase when `gap ≤ θ` *for the resource class that dominates the phase's predicted time* (dominance from the cost vector at the target scale — a phase that is 90% network-bandwidth-bound stops on the `CommVolume` gap; the same phase at small P may be `MemIo`-dominated and keep tuning tiles).
- **Assumption discipline:** bounds hold under `assumptions` (e.g., "classical Θ(n³) algorithm", "per-rank memory M"); if a later decision violates an assumption (Strassen chosen → the classical comm bound no longer applies), the certificate is invalidated automatically because assumptions are `Claim`s wired into the dependency graph like everything else.
- **Composition:** phase bounds sum for sequential phases and max for provably-overlapped ones; the energy bound is always derived (weak but honest) from the flop and I/O bounds — reported as a floor, used only for ranking under `optimize: energy`.
- **Diagnostics dividend:** "trsm phase: CriticalPath gap = 1.05 — near its theoretical wall; further tuning of this phase cannot help; consider batching right-hand sides (changes the bound's assumptions)" — the certificate framework is what lets the compiler say *provably stop trying* instead of *gave up*.

---

## 5. Four levels of user control

### 5.1 Types and surface syntax

```rust
enum ControlLevel<T> {
    Auto,                                     // planner-owned
    Prefer { value: T, weight: Weight },      // soft objective term
    Require { value: T },                     // hard constraint on the decision
    Use { value: T, params: ParamSet },       // expert pin of the decision AND its parameters at a named layer
}
```

```text
x = solve(A, b)                                      # algorithm = auto (implicit)
x = solve(A, b) with { prefer algorithm = cholesky }
x = solve(A, b) with { require algorithm = cholesky }
x = solve(A, b) with { use algorithm = cholesky,
                       process_grid = (4, 8), block_size = 256 }   # pins at AIR and DIR
```

Directives attach to expressions, lexical regions, or whole programs, and each names (implicitly by its subject) the IR layer whose decision it fixes: `algorithm` → AIR, `process_grid`/`block_size`/`distribution` → DIR, `collective.bcast = hierarchical` → CIR, `layout` → LIR, `lookahead`/`tile` → SIR, `rank_map`/`affinity` → HIR. Layers *below* a pin remain `auto` unless separately pinned — `use algorithm = cholesky` fixes nothing about its distribution.

### 5.2 Semantics per level

| Level | Planner freedom | Override rules | Artifact obligations |
|---|---|---|---|
| `auto` | full search over the proven-legal set | revisable by any feedback | decision + evidence recorded |
| `prefer` | search all; preference is a weighted objective term | may be overridden by cost or feedback — **every override is reported** with the reason and the *price of honoring it* (the cost delta of the best preference-satisfying plan, or the specific infeasibility) | override report mandatory |
| `require` | search restricted to plans satisfying the constraint | **never overridden.** If unsatisfiable or illegal, compilation fails with the feedback trail and (as diagnosis only) what relaxing it would yield | constraint + satisfaction proof recorded |
| `use` | none for the pinned decision: the compiler **verifies** the specification (all layer invariants, contracts, evidence gates) but does not search it; downstream layers plan around it | **never overridden.** Verification failure or implicating blocking feedback ⇒ error (`Escalated`), quoting the pin | verification record; "expert pin" flag in the artifact |

Two deliberate asymmetries: (a) `use` is *stronger than* `require` — it also fixes parameters and suppresses search — but it is **not** a licence to skip verification: an expert can pin a bad plan, not an illegal one (an illegal pin — say `use block_size=100` with a microkernel requiring `nb mod 8 == 0` — is a compile error naming the violated knob constraint, never a silent round-up); (b) `prefer` overrides must be *priced*, because an unexplained ignored preference is indistinguishable from a bug to the user.

### 5.3 Interaction with feedback (closing the loop from §1)

Routing skips decisions whose control level forbids revision. Consequences:

- Blocking feedback whose *every* implicated decision is `require`/`use` ⇒ `Escalated(UserError)`: the error message contains the feedback (e.g. the HBM capacity proof), the implicated pins, and — diagnosis only — the nearest feasible relaxation found. The compiler never "helps" by quietly loosening a pin.
- Blocking feedback implicating a mix ⇒ the revisable decisions absorb it (lowest first); pins are planned *around*.
- `prefer` decisions absorb feedback like `auto`, but the override report is emitted.
- Runtime feedback obeys the same rule: a pre-compiled fallback edge is itself a decision; under `require algorithm = cholesky`, the planner is *forbidden* from compiling an LU fallback edge — a runtime spd counterexample then terminates the computation with the counterexample instance rather than silently solving a different problem. This is surfaced at compile time: "note: `require cholesky` + unproven spd ⇒ no fallback exists; failure mode is abort."

---

## 6. The reproducibility/correctness contract lattice

### 6.1 Two axes, one contract

A single chain conflates two independent questions — *is the answer the same?* and *is the answer good?* v2 uses a product lattice:

```rust
struct Contract { det: Determinism, acc: Accuracy }   // ⊑ is componentwise; meet = componentwise strongest

enum Determinism {                       // D1 strongest … D4 weakest
    BitwisePortable,                     // D1: identical bits across machines, P, grids, builds.
                                         //     Pins a reference evaluation order; excludes nearly all of §3–§8 of v1.
    BitwiseConfig,                       // D2: bits are a pure function of (input, machine, P, grid, build).
                                         //     Reduction trees may depend on config but on nothing else —
                                         //     not on tuning-database state, not on timing.
    Deterministic,                       // D3: bits reproducible across reruns given frozen artifacts
                                         //     (plan + tuning DB + module versions). Autotuned order changes
                                         //     are legal between compilations, races/atomics still are not.
    AnyOrder,                            // D4: run-to-run variation permitted within the accuracy contract.
}

enum Accuracy {                          // A1 strongest … A5 weakest
    IeeeStrict,                          // A1: the specified op sequence, correctly rounded; no reassociation,
                                         //     no FMA contraction, no flush-to-zero, no precision changes.
    BoundedError { eps: Sym, norm: NormKind },   // A2: certified forward/normwise bound
    BackwardStable { class: StabilityClass },    // A3: algorithm-level backward-error guarantee
    Statistical { test: TestSpec, tol: Sym },    // A4: distributional equivalence (stochastic algorithms)
    BestEffort,                          // A5: aggressive; sanity checks only
}
```

Defaults: `⟨D3, A3⟩` for factorizations/solves, `⟨D3, A2(classical gemm bound)⟩` for products — reproducible science without banning tuning.

### 6.2 Propagation through the stack

- **MIR:** contracts attach to results; the required contract of a producer = componentwise **meet** (strongest) over its consumers' requirements. MIR rewrites carry contract gates too (e.g. `trace(AB)→Σ∘hadamard` changes the summation order class: requires `det ⊑ D2` be *not* demanded across configs... it preserves D2 — gate: `acc ⊒ A2`, `det` unaffected since the new order is still config-pure).
- **AIR:** algorithm error bounds are checked against `acc` (Strassen needs A2-with-slack or weaker; mixed precision needs A2/A3 plus a convergence certificate). `det` restricts algorithm families whose result depends on data-dependent races (none at AIR normally).
- **PIR:** every `par.reduce` carries `order ∈ {sequential, fixed_tree(spec), config_tree, any}` and the verifier checks it against `det` (table below).
- **DIR:** replication (2.5D) adds a cross-plane reduction ⇒ gated by `det`; distribution itself never changes values.
- **CIR:** collective-algorithm freedom is gated: under D2 the chosen algorithm+tree per collective is pinned as a pure function of config and recorded; under D1, reducing collectives must use the reference order (i.e., mostly: don't); under D3 the choice may come from the tuning DB (frozen artifact); under D4, free (including in-network reduction offload, whose order is switch-firmware-defined).
- **LIR:** wire-compression (fp64→fp32 panels) gated by `acc` budget; layout changes never gated (value-preserving).
- **SIR/codegen:** FMA contraction and vector-reassociation flags derive from `acc = A1` vs weaker; GPU atomics from `det`; LLVM fast-math flag set is *computed from the contract*, never hand-set.

### 6.3 The gating table (normative)

| Transformation | Requires (weakest contract under which it is legal) | Notes |
|---|---|---|
| Reduction reassociation, **fixed spec'd tree** | `det ⊑ D2`? — legal at **D2** if tree is a pure function of config; breaks **D1** | the tree spec goes in the plan artifact |
| Reduction tree **from tuning DB** | legal at **D3**; breaks D2 (bits now depend on DB state) | |
| **Atomics-based** float reduction (GPU/CPU) | **D4** only | deterministic segmented-scan reduction is the D2/D3 alternative, costed |
| **FMA contraction**, vector reassoc within a lane | breaks **A1**; legal at A2↓ (error model includes it) | orthogonal to `det` |
| **Mixed precision + iterative refinement** | breaks A1; legal at **A2/A3** with `SmtProved` convergence certificate (`cond·u_low < 1` claim) | `det` preserved if refinement count is config-pure: iterate-to-tolerance is data-dependent but bit-pure per input — D2-safe, D1-unsafe across machines |
| **Strassen** | **A2** with slack vs the weaker bound, or A3-of-Strassen-class where accepted | flop certificate assumptions change (§4) |
| **2.5D / replication-c** | `det ⊑ D2` with fixed cross-plane tree; **D3** if c or tree comes from tuning | volume certificate improves √c |
| **Collective algorithm switch** (ring↔tree allreduce) | non-reducing collectives (bcast/gather): always legal; **reducing** ones: as reduction-tree rows above | |
| **In-network reduction offload (SHARP-class)** | **D4** (order not spec'd by us) | big α win; priced in the D2/D3 override report |
| **Lookahead / DAG dynamic scheduling** | legal at **D2** for factorizations *if* every reduction in the DAG has fixed trees (execution order varies, arithmetic order doesn't); D4 if work-stealing changes accumulation orders | this is why v1's task runtime is contract-compatible |
| **Wire compression fp64→fp32** | **A2** with budgeted term | contract-gated LIR rewrite |

The table is the *specification*; each row is implemented as a gate predicate `legal(t, Contract) → bool` plus, where accuracy is involved, a proof obligation added to the transformation's evidence requirements.

---

## 7. The evidence system

### 7.1 Types

```rust
struct Claim {
    id: ClaimId,
    stmt: Statement,        // typed AST: Property(subject, prop) | Equivalence(a, b) | ContractSat(op, contract)
                            //           | Bound(resource, expr) | CostClaim(plan, vector) | Precondition(pred)
    scope: Scope,           // Universal | Shaped{dims, dtype} | Instance{input_hash} | Machine{topo_hash}
    evidence: Vec<EvidenceId>,   // possibly several, of different statuses (§7.3: the claim's *effective*
                                 // status for any gate is the strongest admissible one it holds)
}

struct Evidence {
    id: EvidenceId,
    claim: ClaimId,
    status: EvidenceStatus,
    provenance: Vec<EvidenceId>,     // derivation DAG: what this was concluded FROM
    artifact: ArtifactRef,           // the proof term, SMT core, measurement record, model+score, ...
    validity: Validity,              // bindings that expire it: topo_hash, model_version, module_hash, input class
}

enum EvidenceStatus {
    UserAsserted { policy: TrustPolicy },   // trusted by declared policy; may auto-generate a runtime guard
    StaticallyDerived { rule: RuleId },     // audited rule engine: property algebra, affine/polyhedral solver
    SmtProved { solver: Id, core: ArtifactRef },
    FormallyProved { system: Id, cert: ArtifactRef },   // machine-checked (rewrite rules, key algorithms)
    RuntimeChecked { guard: GuardRef, when: CheckTime },// valid for guarded executions only
    AiInferred { model: ModelVersion, score: f64 },     // hypothesis. score is a search prior, nothing more.
    Measured { machine: TopoHash, n: u32, ci: Interval },
    Predicted { model: ModelRef, calib: CalibRecord },
}
```

### 7.2 Confidence is not proof — by construction

`EvidenceStatus` is a **sum type, not a scalar**. There is no numeric confidence field on the proof-class variants, no total order embedding `AiInferred{score: 0.999}` anywhere near `SmtProved`, and no promotion rule keyed on score. The only way an `AiInferred` claim gains standing is that a *new* piece of evidence with a different status is attached to the same claim by the verification machinery. Scores route search effort; they never touch a gate.

### 7.3 Admissibility: which status may gate what

```rust
fn admissible(gate: GateClass, s: &EvidenceStatus) -> bool
```

| Gate class | Admissible statuses |
|---|---|
| **Legality** (semantic equivalence, transformation legality, memory safety, comm correctness) | `StaticallyDerived`, `SmtProved`, `FormallyProved`, `RuntimeChecked` *(guard compiled in)*, `UserAsserted` *(only where policy explicitly extends trust, and a guard is offered)* |
| **Contract satisfaction** (§6 accuracy/determinism) | same set; `UserAsserted` accepted only for input-data claims (e.g. `cond ≤ 1e4`) with guard generation on by default |
| **Search / planning objectives** | all statuses; `AiInferred`/`Predicted` explicitly welcome |
| **Reporting** (plan artifact) | all — printed *with* status; the artifact renders proof-class and hypothesis-class claims in visibly different registers |

### 7.4 Lifecycle: derivation, demotion, expiry

- **Derivation DAG.** The effective strength of a derived claim is the **weakest link** on each derivation path (a `StaticallyDerived` conclusion from an `AiInferred` premise is `AiInferred` for admissibility). The DAG is stored; the plan artifact can render "why does the compiler believe X" to arbitrary depth.
- **Demotion.** A `Counterexample` feedback (§1) attaches falsifying evidence to the claim: the claim's admissible statuses are revoked for the counterexample's scope (and entirely, for `Universal`-scope claims), every `Decision` listing it in `depends_on` is invalidated, and replanning proceeds under §1.3. `UserAsserted` claims demote like any other — the user is *told*, with the instance.
- **Expiry.** `validity` bindings expire evidence mechanically: `Measured` dies with the topology hash, `AiInferred` with the model version, `SmtProved` side-condition proofs with the module hash of the code they were proved about. Expired ≠ falsified: the claim reverts to whatever other evidence it holds.
- **Runtime guards.** `RuntimeChecked` is *conditional* evidence: the guarded plan is legal because the guard aborts/falls back when the check fails. Guards are real IR (a `Claim` → predicate compilation), costed like everything else, and a guard that would cost more than the optimization it enables is reported as such (the offer to `assert` appears in the artifact instead).

---

## 8. The systems interacting

### 8.1 End-to-end: distributed SPD solve

Program:

```text
x = solve(A, b) with {
    require accuracy    = residual(1e-10),      # Contract.acc lower bound, hard
    determinism         = require D2,           # bitwise per config
    prefer  algorithm   = cholesky (weight = strong),
    use     process_grid = (4, 8), block_size = 256,   # expert pins at DIR
    assert  A: spd,                              # UserAsserted claim C1
}
```

**Compile time.**

1. **MIR.** Claim `C1 = Property(A, spd){UserAsserted, guard: on}`. Contract resolved: `⟨D2, A2(residual ≤ 1e-10)⟩`. Both recorded as claims with the directive as provenance.
2. **AIR.** `spd` is legality-relevant for Cholesky (existence). `C1`'s status is admissible (UserAsserted + guard) → `algo.chol` chosen; `prefer cholesky` satisfied, no override report needed. AI (Level 3) proposes mixed-precision fp32+IR as a prior; its convergence needs `cond(A)·u_32 < 1` — claim `C2 = Property(A, cond ≤ 3e4)` exists only as `AiInferred{score .91}` (pattern: A came from a regularized normal-equations pipeline). **Not admissible** for the contract gate. The planner compiles a *conditional plan*: cheap condition estimator (O(n²) sampled Hutchinson-style) at runtime → if it certifies, `C2` gains `RuntimeChecked` and the fp32 branch runs; else fp64 branch. Both branches exist in the plan; the artifact shows the fork and why.
3. **PIR.** Cholesky task DAG; every `syrk`/`gemm` reduction gets `order = fixed_tree(spec)` — required by `D2` (gating table row 1); dynamic lookahead scheduling remains legal (row: execution order ≠ arithmetic order). `CriticalPath` certificate: `Θ(n/nb)` with `nb=256` pinned → depth = n/256, `Bound::Known`, gap computed later against the schedule.
4. **DIR.** `use process_grid=(4,8), block_size=256` → **verified, not searched**: coverage/disjointness proofs run on the pinned ownership functions (pass), knob constraint `256 mod mr == 0` (pass), memory feasibility (pass). Planner notes — diagnosis only, since the pin forbids action — that (6,6)... isn't available at P=32; that (4,8) is non-square costs +19% predicted `CommVolume` vs the (√32-ish) optimum; certificate gap for volume: **1.31** (vs 1.12 achievable). Printed in the artifact next to the pin. No override.
5. **CIR.** Panel broadcasts pipelined, lookahead 2 proposed. Under `D2`, each reducing collective's algorithm+tree is pinned as a config-pure function and recorded; SHARP offload for residual-norm allreduces (fp32 branch's IR loop) is **rejected by the D2 gate** — the artifact prices it: "in-network allreduce would save 0.8 µs × 2/iter; requires D4."
6. **LIR.** Blocking feedback: `ResourceViolation{Memory(hbm), required 131 GiB, available 128, at gpu[*]}` — lookahead-2 double buffers + fp32 shadow copies. Implicates: SIR `lookahead=2` (`auto`), DIR `nb` (`use` — skipped by routing). SIR revises → `lookahead=1`, alternative `2` moved to `excluded` with the capacity proof as justification. Re-lower: fits. One round, monotone, done.
7. **SIR/HIR.** Tuning (D3-would-allow-DB... note: `D2` means the *reduction trees* can't come from the DB, but tile sizes/stream counts that don't change arithmetic order can and do). Rank map embeds the 4×8 grid rows into racks; progress thread reserved (model: overlap prediction 0.88 requires async progress).
8. **Artifact.** Certificate vector per phase (`CommVolume` gap 1.31 ⟵ pinned grid; `CriticalPath` gap 1.06 for the factorization; trsm phase `MemIo`-bound, gap 1.4, note attached: "batch RHS to change assumptions"), the D2-pinned trees, the prefer-satisfaction line, the conditional-plan fork, evidence DAG for every claim.

**Runtime.**

- Condition estimator returns 2.1e4 → `PropertyDiscovered{C2, RuntimeChecked}` on the bus → the *pre-compiled* fp32+IR branch is taken (plans are immutable; the discovery selects among compiled branches). IR converges in 3 iterations; iteration count is input-pure → D2 intact.
- Suppose instead panel 7's `potrf` hits a non-positive pivot: guard fires → `Counterexample{C1, instance}`. `C1` demoted. Fallback edge: because `algorithm` was **prefer**, the planner had compiled a pivoted-LDLᵀ fallback branch; it fires, the override report ("preference cholesky overridden at runtime: spd falsified, instance attached") lands in the run log. Had the user written `require algorithm = cholesky`, §5.3 applies: no fallback was compiled — the run aborts carrying the counterexample, exactly as the compile-time note warned.
- `PerformanceMeasurement`: achieved overlap 0.61 vs predicted 0.88 → stored; the calibration residual model (AI Level 3) updates; next compilation of this shape re-plans HIR (second progress thread? different rail striping?) with corrected inputs. Feedback affected the future, not the running job.

### 8.2 End-to-end: imported loop → GEMM (semantic recognition)

Imported legacy kernel (Fortran 77 style, as found in the wild):

```fortran
      DO 30 J = 1, N
        DO 20 L = 1, K
          T = BETA_B(L,J)
          DO 10 I = 1, M
            CC(I,J) = CC(I,J) + AA(I,L) * T
   10     CONTINUE
   20   CONTINUE
   30 CONTINUE
```

1. **Ingest.** Lowered to affine loop IR. AI **Level 1** proposes `H1: Equivalence(region, math.matmul(AA, BETA_B) accumulated into CC)` with preconditions `P1: noalias(CC, AA)`, `P2: noalias(CC, BETA_B)`, `P3: leading dims ≥ M/K` — evidence `AiInferred{score .98}`. It also proposes `H2: Property(AA, symmetric)` (`AiInferred{.71}`, from observed call-site construction).
2. **Verification pipeline for H1 (deterministic).**
   - Polyhedral raising: access functions are exact affine; the recovered contraction pattern matches the `matmul+accumulate` template; dependence analysis confirms the `T` scalar is a loop-invariant load per (L,J) — equivalence of the *affine part* gets `StaticallyDerived{rule: raise-contraction}`.
   - `P1, P2`: interprocedural alias analysis proves `P2` (`StaticallyDerived`), cannot prove `P1` (arguments cross a foreign ABI). SMT on the caller's allocation facts — also inconclusive. Resolution: **runtime guard** — an O(1) extent-overlap check on (CC, AA) descriptors; `P1` gets `RuntimeChecked{guard}` and the code is *versioned*: guarded fast path = MIR `math.matmul`, fallback = the original loop nest, preserved verbatim as the semantic anchor.
   - `H1`'s effective status via the weakest-link rule: `RuntimeChecked` — **admissible for legality** (guard compiled in). The region is replaced by a MIR node. Everything above now applies: the full v1 §4 lowering menu — threaded BLAS, GPU, SUMMA, 2.5D — becomes reachable for a 50-year-old loop, with the contract defaulting to `⟨D3, A2⟩` *scoped to the guarded path* (the fallback loop keeps its original semantics bit-for-bit).
3. **H2 (symmetry) is different in kind.** Exploiting it (`gemm → symm/syrk`-class kernels that read only one triangle) is **legality-relevant**: if AA is not symmetric, the result is wrong, not slow. `AiInferred` is inadmissible; options, in the artifact: (a) prove it from the constructor (attempted: the construction site computes `AA = GᵀG` → property algebra fires → `StaticallyDerived` — if the constructor is visible); (b) full O(n²) runtime check — costed at 0.4% of the gemm at n=20k, offered as a guard; (c) an `assert` the user may add. In this run the constructor is in a foreign object file → the compiler takes (b) automatically *only because* the guard cost is below threshold; otherwise it ships the offer and uses plain gemm. `H2` either becomes `RuntimeChecked` or stays a suggestion — the score 0.71 never touched the gate.
4. **Feedback closes the loop.** Months later a caller passes overlapping CC/AA: the guard fires, the fallback loop runs (correct, slow), and `Counterexample{P1-universal-version}` is recorded — the claim was `RuntimeChecked` (per-execution), so nothing demotes; but the *measurement* that 12% of production calls take the fallback flows back as `PerformanceMeasurement`, and the diagnostics layer (AI Level 3) drafts the human-readable note: "consider `!DIR$ noalias` at these three call sites; predicted saving 14%." The suggestion carries `Predicted` evidence and waits for a human — which is exactly where authority for changing source code belongs.

---

## 9. Consequences for the v1 MVP plan

Small but structural additions, in priority order: (1) `Claim`/`Evidence`/`Decision` tables and the feedback bus exist from **Milestone 0** — they are cheap as data structures and impossible to retrofit as afterthoughts; (2) the contract lattice ships with only `⟨D2..D3⟩ × ⟨A2, A3⟩` populated and exactly two gates implemented (reduction trees, mixed precision) — the *table* is complete, the gate code grows; (3) control levels: `auto`/`use` first (the expert-pin path exercises verification-without-search, which flushes out planner/verifier entanglement early), `prefer`/`require` second; (4) placement domains: the trait with two kinds (Distributed, SharedMemory) at Milestone 2, Accelerator at Milestone 4, as already sequenced; (5) certificates: `Flops`, `CommVolume`, `CriticalPath` only; (6) AI levels: none in the MVP beyond the Level-3 hooks (the evidence system's `AiInferred` variant exists so the pipeline is AI-ready without containing a model).
