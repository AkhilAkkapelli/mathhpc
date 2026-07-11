# HPC Architecture for a Mathematical Programming Language and Compiler

**Design document — IR stack, hardware model, cost models, AI/deterministic division of labor, and MVP**

---

## 0. Working assumptions (project vision recap)

This design assumes the language has the following character (stated explicitly so every downstream decision is traceable to it):

1. **Programs express mathematics, not execution.** The user writes `C = A × B` or `x = solve(A, b)` against typed mathematical objects (matrices, tensors, operators) with declared *properties* (SPD, symmetric, banded, low-rank, sparse pattern class). There are no loops over indices in the surface language for linear-algebra-level operations; index-level math (sums, einsum-style contractions) is allowed but still value-semantic.

2. **The compiler owns the entire path from intent to machine**: algorithm selection, precision policy, parallelization, data distribution, communication schedule, memory layout, and hardware binding. The user may *constrain* any of these (e.g., "use ≤ 64 nodes", "residual ≤ 1e-10", "energy-preferred"), never *specify* them imperatively.

3. **Everything the compiler decides is inspectable and overridable** at the corresponding IR layer. Each IR has a stable textual form; an expert can pin a decision ("distribution: 2.5D, c=4") and the compiler re-plans everything below it.

4. **Correctness is deterministic; performance is model-guided and measured.** No AI or heuristic component may affect semantics. Heuristics choose among *proven-legal* alternatives.

The compilation pipeline is a strict tower of nine layers. Each lowering is a total function annotated with a cost estimate; each layer has its own verifier that runs in debug builds and in CI.

```
Mathematical IR   — what is true
Algorithm IR      — which algorithm computes it
Parallel IR       — what can run concurrently
Distribution IR   — who owns which data
Communication IR  — what messages move, when
Memory/Layout IR  — how bytes are arranged, in which memory
Schedule IR       — in what order, with what tiling/pipelining
Hardware Mapping  — on which physical resources
MLIR / LLVM       — codegen
```

A key architectural rule: **information only flows downward as refinement, and upward only as *cost feedback***. Lower layers never change the semantics decided above; they report costs (achieved bandwidth, measured kernel time) that the planner uses to revisit upper-layer choices in a bounded re-planning loop.

---

## 1. Hardware topology representation

Everything below Distribution IR consumes a **machine model**: a typed, attributed tree with cross-links (links are edges with latency/bandwidth attributes, so the "tree" is really a hierarchy plus a link graph). It is populated from `hwloc`, `nvidia-smi topo`, `slurm` topology plugins, and a one-time microbenchmark pass (STREAM, osu_latency/osu_bw, gemm peak).

### 1.1 Schema (textual form, s-expression style)

```lisp
(cluster :name "frontier-like"
  (network :type dragonfly :groups 8
           :inj-bw 25GB/s :link-bw 25GB/s
           :lat-intra-group 1.2us :lat-inter-group 2.0us
           :collectives (:allreduce sharp-offload :bcast tree))
  (group :id 0
    (rack :id 0
      (node :id 0 :count 128            ; homogeneous ranges compress
        (nic :count 4 :bw 25GB/s :lat 1.1us)
        (socket :id 0 :count 1          ; single-socket node example
          (numa :id 0..3 :count 4       ; 4 NUMA domains (NPS4)
            (mem :type ddr5 :cap 128GiB :bw 100GB/s :lat 95ns)
            (core :count 16 :freq 2.0..3.7GHz
              (simd :isa avx512 :width 512 :fma 2 :ports 2)
              (cache :l1 32KiB :l2 1MiB)
            )
            (cache :l3 32MiB :shared 16cores)))
        ;; --- accelerator subtree ---
        (gpu :id 0..7 :count 8 :arch cdna3-like
          (mem :type hbm3 :cap 128GiB :bw 5.3TB/s :lat 350ns)
          (sm  :count 228 :simd 64 :regs/sm 512KiB :lds/sm 160KiB
               :max-occupancy 2048threads)
          (peak :fp64 61TF :fp32 122TF :bf16 980TF))
        ;; --- link graph (edges, not tree) ---
        (link :kind cpu-gpu  :ends (numa 0) (gpu 0)   :bw 96GB/s  :lat 1.5us :name infinity-fabric)
        (link :kind gpu-gpu  :ends (gpu 0) (gpu 1)    :bw 200GB/s :lat 0.9us :name xgmi)
        (link :kind gpu-nic  :ends (gpu 0) (nic 0)    :bw 25GB/s  :gpudirect true)
        (affinity (gpu 0) (numa 0) (nic 0))           ; closest-resource triples
      ))))
```

### 1.2 Derived quantities (computed once, cached)

For every pair of resources the model precomputes:

- `bw(r1, r2)`, `lat(r1, r2)` — min-cut bandwidth and path latency between any two memories/devices (host↔device, device↔device, node↔node intra/inter group).
- **Roofline points** per compute resource: `(peak_flops, mem_bw)` → machine balance `B = peak_flops / mem_bw` in FLOP/byte. E.g., GPU above: 61 TF / 5.3 TB/s ≈ 11.5 FLOP/byte (fp64); CPU NUMA domain: ~2 TF / 100 GB/s ≈ 20 FLOP/byte.
- **α–β–γ parameters** per network level: latency α, inverse bandwidth β (s/byte), per-flop time γ. Collectives get per-algorithm models, e.g. ring allreduce of m bytes on p ranks: `2(p−1)α + 2m(p−1)/p·β`.
- **NUMA distance matrix**, **GPU peer matrix** (which pairs have direct links vs routed), **NIC affinity map**.
- **Energy coefficients** where the platform exposes them (RAPL / rocm-smi): nJ/flop, nJ/byte-moved per memory level, idle power. Used only as a tie-breaker objective unless the user selects `optimize: energy`.

### 1.3 Why a tree + link graph and not a flat table

Distribution IR needs the *hierarchy* (to build process grids congruent with racks→nodes→NUMA), Communication IR needs the *link graph* (to cost collectives and choose GPU-aware vs staged paths), and Hardware Mapping IR needs the *affinity triples* (core masks, closest NIC, closest NUMA for each GPU). One structure serves all three, and it is versioned: a plan compiled against topology hash `H` is re-validated if the job lands on a different partition.

---

## 2. Cross-cutting cost model

Every IR node can be asked for a **cost vector**, propagated bottom-up:

```
cost = { flops, words_moved: {l1,l2,l3,ddr,hbm,pcie/xgmi,network},
         messages, sync_points, critical_path_time, memory_footprint,
         energy_est }
```

Three analytical models are used, in increasing fidelity, and are decisive at different layers:

1. **Roofline / arithmetic intensity** (Memory/Layout, Schedule): `AI = flops / bytes_from_level`. GEMM with tiling: AI ≈ b/ (words per flop) — for an `m×k · k×n` tile resident in cache with tile size b, AI ≈ b/2 FLOP/word ≈ b/16 FLOP/byte fp64. The compiler *solves for b* such that `AI ≥ machine balance` and 3 tiles fit in the target cache level: `3b² · 8 ≤ cache_bytes`.
2. **α–β–γ / LogGP** (Distribution, Communication): closed-form per-collective, per-message costs; used to compare SUMMA vs Cannon vs 2.5D and to choose grid shapes.
3. **Communication lower bounds** (Distribution): for classical matmul on P processors with memory M per proc, words moved ≥ `Ω(n³ / (P·√M))`; with replication factor c (2.5D), bandwidth cost drops by √c and latency by c^(3/2), at c× memory. The planner treats the lower bound as an *optimality certificate*: if a candidate plan is within a small factor of the bound, search stops.

Strong/weak scaling are *predicted* from these models (and later validated): the planner emits, with every executable, a scaling card — predicted efficiency at 1×, 4×, 16× the requested resources, and the crossover point where latency (α·messages) dominates and strong scaling dies.

---
## 3. The IR stack

Notation: all textual examples use an MLIR-flavored syntax with per-layer dialects (`math.`, `algo.`, `par.`, `dist.`, `comm.`, `mem.`, `sched.`, `hw.`). Symbolic dimensions are SSA values of type `!sym`.

Running examples throughout:
- **E1:** `C = A × B`, A: n×k, B: k×n, dense fp64.
- **E2:** `x = solve(A, b)`, A: n×n SPD, dense fp64.

---

### 3.1 Mathematical IR (MIR)

**Purpose.** Capture *what is true*, with maximal algebraic structure and zero execution commitment. This is the layer where mathematics-level optimization happens; every layer below can only lose information, so MIR rewrites are the highest-leverage optimizations in the whole compiler.

**Core operations.**
- Tensor algebra: `math.matmul`, `math.add`, `math.scale`, `math.transpose`, `math.contract` (einsum), `math.kron`, `math.hadamard`.
- Equation/relation forms: `math.solve %A, %b` (declared as "the x with Ax=b", *not* an algorithm), `math.lstsq`, `math.eig`, `math.svd` (as mathematical objects with existence conditions).
- Reductions/functionals: `math.norm {p}`, `math.trace`, `math.det` (flagged: almost always a smell; rewritten away when possible).
- Structure constructors/assertions: `math.assert_prop %A {spd}`, `math.as_triangular`.

**Type system.**
```
!math.tensor<f64, dims=[%n,%k], props={dense}>
!math.tensor<f64, dims=[%n,%n], props={spd, symmetric}>
!math.tensor<f64, dims=[%n,%n], props={sparse<csr-class>, bandwidth=%bw}>
!math.op<...>            ; linear operator (matrix-free capable)
!sym                     ; symbolic nonneg integer with affine relations
```
Properties form a lattice (e.g., `diagonal ⊑ triangular ⊑ general`, `spd ⊑ symmetric ⊑ general`) with a propagation algebra: `spd × spd` is not generally SPD, but `Aᵀ A` with full-rank A is; `triangular × triangular` (same side) is triangular; sum of SPD is SPD. Precision requirements are types too: `!math.accuracy<rel=1e-12>` attached to results.

**Metadata.**
- Symbolic dimension relations (`%k == %n`, `%n ≥ 1`), value ranges/scaling if known.
- Conditioning hints (`cond ≤ 1e6` from user or from provenance, e.g., "A = M + σI, σ known").
- Accuracy contract per result; reproducibility contract (`bitwise`, `run-to-run`, `best-effort`) — this later constrains reduction reordering.
- Provenance: source location, user pins.

**Invariants.**
- Pure, value-semantic, no aliasing, no order. The IR is a DAG of mathematical facts.
- Dimension consistency proven by the affine solver over `!sym`.
- Every property annotation is either user-asserted (trusted, but generates an optional runtime check) or derived (must carry a derivation certificate).

**Legal transformations.**
- Associativity reorder of chained products with the classic matrix-chain DP *on symbolic dims* (specializing at runtime if dims unknown): `(A·B)·v → A·(B·v)` — O(n³)→O(n²) when v is a vector.
- `inverse(A)·b → solve(A,b)`; `det`/`inverse` elimination; `trace(A·B) → Σ hadamard` (O(n³)→O(n²)).
- Common-subexpression at math level: `AᵀA` shared across expressions.
- Structure exploitation: `solve(A,b)` with A triangular → mark as substitution-class; with A = I + UVᵀ → Woodbury rewrite.
- Distributivity chosen by cost: `A(B+C)` vs `AB + AC` (the former halves flops; the latter may expose better parallel structure — the decision is *deferred* by keeping both forms as an `math.alt` node until Algorithm IR costs them).
- Precision splitting: mark subexpressions eligible for lower precision given the accuracy contract (decision made at Algorithm IR).

**Optimization opportunities.** This is where 10×–1000× wins live: complexity-class changes, structure exploitation, elimination of explicit inverses, matrix-free operator recognition. No lower layer can recover a missed `trace(AB)` rewrite.

**Verification obligations.**
- Type/dimension checking (affine feasibility).
- Property-derivation soundness: each derived property must be produced by a rule in the (small, audited) property algebra; certificates checked structurally.
- Accuracy contracts must be satisfiable: MIR verifier rejects `det` of a 10⁵×10⁵ matrix with `rel=1e-15` fp64, with a diagnosis, instead of silently proceeding.

**Textual example (E1 + E2).**
```mlir
func @step(%A: !math.tensor<f64,[%n,%k],{dense}>,
           %B: !math.tensor<f64,[%k,%n],{dense}>,
           %S: !math.tensor<f64,[%n,%n],{spd}>,
           %b: !math.tensor<f64,[%n],{dense}>)
    -> (!math.tensor<f64,[%n,%n]>, !math.tensor<f64,[%n]>) {
  %C = math.matmul %A, %B  : ... {accuracy = rel<1e-13>, repro = run-to-run}
  %x = math.solve  %S, %b  : ... {accuracy = residual<1e-10>}
  return %C, %x
}
```

---

### 3.2 Algorithm IR (AIR)

**Purpose.** Bind each mathematical fact to a *named algorithm* with a parameter space, a flop/stability model, and a recursive decomposition structure — still machine-independent.

**Core operations.**
- `algo.gemm{variant}` — variants: `classical`, `strassen{cutoff}`, `structure<triangular|symmetric>`.
- `algo.chol`, `algo.lu{pivot=partial|tournament|none}`, `algo.qr{variant=hh|tsqr|caqr}`.
- `algo.trsm`, `algo.syrk`, `algo.cg{precond}`, `algo.gmres{restart}`, `algo.ir` (iterative refinement wrapper).
- Composition: `algo.pipeline`, `algo.alt` (cost-deferred alternatives, resolved before Parallel IR), `algo.recurse` — the blocked/recursive form that later layers tile and distribute (e.g., blocked right-looking Cholesky expressed as a recurrence over block columns).

**Type system.** MIR tensor types plus:
```
!algo.blocked<!math.tensor<...>, hier=[%b1,%b2]>   ; logical block hierarchy (sizes still symbolic)
!algo.precision_plan<{compute=f64, accum=f64, store=f64} | {compute=bf16, accum=f32, refine=f64}>
```

**Metadata.**
- Exact flop polynomial (`2nk n` for gemm; `n³/3` chol; Strassen `O(n^2.807)` with constant).
- Stability class + error bound template (`‖Ĉ−C‖ ≤ c·k·u·‖A‖‖B‖` classical; Strassen's weaker bound — checked against the MIR accuracy contract).
- Working-set formula per recursion level; parallelism *potential* (critical path under infinite processors, e.g. Cholesky's O(n) block critical path).
- Communication lower bound for the algorithm class (feeds Distribution IR's optimality check).

**Invariants.**
- Every AIR op refines exactly one MIR op (or a rewritten cluster) — link preserved for traceability.
- Chosen algorithm's error bound instantiated with known dims/cond must imply the MIR accuracy contract, or an `algo.ir` refinement wrapper must be inserted.
- `algo.alt` nodes must be resolved before lowering (no deferred choices leak downward).

**Legal transformations.**
- Algorithm selection (the big one): `solve{spd} → chol + trsm + trsm`; `gemm → strassen` only if the error-bound check passes and dims exceed cutoff.
- Mixed precision: `gemm{compute=bf16,accum=f32}` + `algo.ir` if contract demands fp64 result — legal only when the IR convergence model (cond·u_low < 1) certifies convergence.
- Fusion at algorithm granularity: `chol` + downstream `trsm` fused into one blocked recurrence (enables lookahead later).
- Communication-avoiding variant selection: `qr → tsqr/caqr`, `lu → tournament pivoting` (marked: changes pivot choice ⇒ needs MIR reproducibility contract ≠ `bitwise`).

**Optimization opportunities.** Complexity constants, precision (up to 8× on tensor-core-class hardware), enabling CA algorithms, refinement strategies. Interplay with Distribution IR via the lower-bound metadata.

**Verification obligations.**
- Error-bound entailment check (symbolic inequality over n, cond, u).
- Flop accounting: AIR total flops must equal MIR-derived flops for exact algorithms (Strassen exempted via its own certificate).
- Termination certificates for iterative algorithms (CG: SPD ⇒ convergence; GMRES(restart): only *progress* is guaranteed ⇒ requires user-visible max-iter contract).

**Textual example.**
```mlir
%C = algo.gemm classical %A, %B
       {flops = affine<2*n*k*n>, err = classical_bound,
        precision = {compute=f64, accum=f64},
        recurse = blocked<[%bm,%bn,%bk]>}          // sizes still free

%L = algo.chol right_looking %S
       {flops = affine<n^3/3>, recurse = block_col<%nb>, lookahead = eligible}
%y = algo.trsm lower %L, %b
%x = algo.trsm lower_t %L, %y
```

---

### 3.3 Parallel IR (PIR)

**Purpose.** Make *all* legal concurrency explicit as a typed task/loop structure with a dependence relation — before any decision about processes, threads, or devices. This is the layer where polyhedral machinery lives.

**Core operations.**
- `par.forall (%i,%j) in domain(...)` — independent iterations (proof attached).
- `par.reduce {op=+, order=tree|deterministic|any}` — reordering legality tied to the MIR reproducibility contract.
- `par.task %id deps(%a: read, %b: readwrite)` / `par.taskgraph` — dynamic DAG regions (needed for factorizations: Cholesky's DAG of `potrf/trsm/syrk/gemm` block tasks).
- `par.pipeline` — producer/consumer stages with explicit channel types.
- `par.tile`, `par.fuse`, `par.interchange` as *recorded transformations* (schedule tree edits), so the polyhedral schedule is inspectable.

**Type system.**
```
!par.domain<affine_set<(i,j,k): 0<=i<n, 0<=j<n, 0<=k<kk>>
!par.dep<flow|anti|output, distance_vector|relation>
!par.tile<!math.tensor<...>, [%bm,%bn]>       ; logical tile handle (no layout yet)
```

**Metadata.**
- Full dependence relation (exact where affine — dense LA is; summarized otherwise).
- Grain-size cost per task instance (from AIR flop formula ÷ iterations).
- Reduction reorder license (from reproducibility contract).
- Critical-path length and average parallelism of the task graph (computed; e.g., blocked Cholesky with n/nb = T block-columns: critical path Θ(T), width up to Θ(T²) — tells Distribution IR how many processors are even useful).

**Invariants.**
- Dependence-completeness: verifier recomputes dependences from access functions and checks the recorded relation is a superset. Missing dependence = hard error; spurious = warning (lost parallelism).
- `forall` bodies must be provably independent (polyhedral proof or task-dep proof attached).
- Deterministic-reduction domains must carry a fixed combination tree.

**Legal transformations.**
- Polyhedral tiling/fusion/interchange/skewing — legality = dependence preservation, checked by ISL-style machinery.
- forall ↔ taskgraph conversion (loops with carried deps become task DAGs — this is how Cholesky lookahead becomes expressible).
- Reduction reassociation/privatization (contract-gated).
- Granularity coarsening: fuse tasks below a grain threshold (threshold comes later from hardware; here it stays symbolic `%grain`).

**Optimization opportunities.** Exposing wavefront/DAG parallelism in factorizations (the difference between fork-join ScaLAPACK-style and DAG-runtime PLASMA/DPLASMA-style performance), fusion that later becomes kernel fusion, tiling structure that later maps 1:1 to distribution blocks and cache tiles.

**Verification obligations.** Dependence preservation for every recorded transform; race-freedom of forall; determinism audit (any `order=any` reduction must trace to a contract permitting it).

**Textual example (E1, and E2's task DAG).**
```mlir
// E1: gemm as a 3-level tiled reduction over logical tiles
par.forall (%it, %jt) in domain<(0..%n/%bm, 0..%n/%bn)> {
  %acc = par.reduce {op=+, order=any} (%kt in 0..%k/%bk) {
    yield algo.gemm_tile A[%it,%kt], B[%kt,%jt]     // grain = 2*bm*bn*bk flops
  }
  par.write C[%it,%jt], %acc
}

// E2: right-looking Cholesky as a task graph
par.taskgraph {
  for %j in 0..%T {
    %t0 = par.task potrf S[%j,%j]        deps(S[%j,%j]: rw)
    par.forall (%i in %j+1..%T) {
      par.task trsm  S[%i,%j], %t0       deps(S[%i,%j]: rw, S[%j,%j]: r)
    }
    par.forall (%i,%l in triangle(%j+1..%T)) {
      par.task syrk/gemm S[%i,%l] -= S[%i,%j]·S[%l,%j]ᵀ   deps(...)
    }
  }
}  // metadata: crit_path = Θ(T), width = Θ(T²)
```

---

### 3.4 Distribution IR (DIR)

**Purpose.** Decide data ownership and computation placement across the *distributed* machine: process grids, distributions, replication. Communication is still implicit — DIR states *who owns what*; the next layer derives *what must move*.

**Core operations.**
- `dist.grid %g = grid<%pr x %pc [x %c]>` — 2D or 2.5D (replication depth c) process grids; also `grid<hier: nodes x numa>` congruent with topology.
- `dist.tensor %Ad = distribute %A on %g by <blockcyclic(%mb,%nb)| block | cyclic | replicated | owner(fn)>` per dimension.
- `dist.redistribute %Ad from d1 to d2` — a semantic no-op with a (large) cost; realized by Communication IR.
- `dist.local_view %Ad` — the owned sub-tensor, with the ownership function as metadata.
- `dist.compute_on {owner_of=%Cd}` — placement policy for each PIR task class (owner-computes on output is the default; DIR may override, e.g., replicate small B).

**Type system.**
```
!dist.grid<2, [%pr,%pc], topo_map=hier<node,numa>>
!dist.tensor<!math.tensor<f64,[%n,%n]>,
             grid=%g, dist=[blockcyclic(%nb), blockcyclic(%nb)],
             repl=1, halo=0>
```
Ownership is a first-class affine function `own(i,j) = ((i/%nb) mod %pr, (j/%nb) mod %pc)` carried in the type — every later layer computes message sets *from this function*, never from ad-hoc code.

**Metadata.**
- Per-tensor memory footprint per rank; total replication factor; imbalance bound (block-cyclic gives ≤ one extra block row/col per rank — computed exactly).
- Predicted communication volume per PIR op class from ownership functions (e.g., SUMMA on √P×√P grid: each rank sends/receives `2n²/√P` words per full gemm — matches the lower bound to within constant).
- Congruence map: which grid axes align with which topology levels (grid rows ↔ nodes, grid cols ↔ NUMA domains), so Communication IR can pick hierarchy-aware collectives.

**Invariants.**
- Coverage & disjointness: for `repl=1`, ownership partitions the index space (affine check); for `repl=c`, each element owned by exactly c ranks with a designated primary.
- Every PIR task is placed on a rank that will own (or receive) all its operands — DIR itself must be *closed*: the set of implied transfers is computable and finite.
- Memory feasibility: Σ footprints + workspace ≤ per-rank memory (from topology model), with a reserve fraction.

**Legal transformations (the planner's search space).**
- **Grid shape:** `pr × pc` selection — square minimizes gemm volume; tall grids favor `trsm`-heavy phases; the planner may use *different grids per phase* with a costed `redistribute` between (classic ScaLAPACK lesson: one grid for factorization, another for the solve is rarely worth it; the cost model decides).
- **Block size %nb:** couples three layers — DIR load balance (small nb), CIR message aggregation (large nb), and Schedule IR cache tiling (nb multiple of the microkernel tile). Resolved by the coupled cost model + autotuning; typical fp64 CPU answer 192–384, GPU 512–2048.
- **Block ↔ block-cyclic:** block-cyclic is mandatory for factorizations (E2) — as elimination proceeds, block distribution would idle the ranks owning the finished top-left corner; cyclic wrapping keeps all ranks busy at every step. For a single gemm (E1), plain block is fine and has simpler halos.
- **2D → 2.5D / 3D (communication-avoiding):** add replication depth c when memory headroom exists: volume drops √c×. Legal iff memory invariant holds and reproducibility contract permits the extra reduction reordering (the c-fold partial-C reduce).
- **Replicate small operand:** if B is n×r with r ≪ n, `repl=all` for B turns E1 into embarrassingly parallel panel gemms — legality is just the memory check.

**Optimization opportunities.** This layer decides asymptotic communication behavior — the single biggest lever for strong scaling. The lower-bound metadata from AIR certifies when to stop searching.

**Verification obligations.** Coverage/disjointness proofs (affine); memory feasibility; redistribute correctness (`d2∘redistribute∘d1⁻¹ = id` on the index space); imbalance ≤ declared bound.

**Textual example (E1 → SUMMA-ready, E2 → block-cyclic).**
```mlir
%g  = dist.grid grid<%pr x %pc> {congruent = [node, numa], choose: pr*pc = P, pr≈pc}
%Ad = dist.distribute %A on %g by [blockcyclic(%nb), blockcyclic(%nb)]
%Bd = dist.distribute %B on %g by [blockcyclic(%nb), blockcyclic(%nb)]
%Cd = dist.distribute %C on %g by [blockcyclic(%nb), blockcyclic(%nb)]
dist.compute_on owner_of(%Cd) : par.forall(...)      // stationary-C ⇒ SUMMA family
  {pred_volume = 2*n*n/sqrt(P) words/rank, lower_bound_ratio = 1.15}

// 2.5D variant (chosen when mem_headroom ≥ c and P large):
%g3 = dist.grid grid<%pr x %pc x %c=4>   {volume_ratio = 1/2}   // √4 = 2× less traffic

// E2: same block-cyclic grid; task placement follows owner of the updated block
dist.compute_on owner_of(block) : par.taskgraph(chol)
  {imbalance ≤ 1 block-row, pipeline_ok = true}
```

---

### 3.5 Communication IR (CIR)

**Purpose.** Materialize the transfers implied by DIR as explicit, costed communication operations — collectives and point-to-point — with explicit *epochs* (start/complete pairs) so overlap with computation is a first-class, verifiable structure.

**Core operations.**
- Collectives over grid axes: `comm.bcast %buf on %g.row from %root`, `comm.reduce`, `comm.allreduce`, `comm.allgather`, `comm.reduce_scatter`, `comm.alltoall` — each with an `{alg = ring|tree|recursive_doubling|bruck|hier<node,numa>|offload}` attribute chosen by the α–β model against the topology.
- Point-to-point: `comm.send/recv`, `comm.isend/irecv` returning `!comm.req`, `comm.sendrecv` (shift patterns for Cannon-style variants), one-sided `comm.put/get + comm.fence` for DAG-driven factorization runtimes.
- Overlap structure: `comm.epoch %e = start(...)` … `comm.wait %e` — everything between is certified independent of the in-flight buffers.
- `comm.persistent` (fixed-pattern iterations: setup once, `comm.start/wait` per iteration — matches MPI persistent collectives).
- `comm.channel` types for pipelined producer/consumer (Cholesky panel broadcast lookahead).

**Type system.**
```
!comm.buf<f64, %len, space=host|device, pinned?>   ; space commitment appears HERE
!comm.req                                          ; nonblocking handle (affine: must be waited exactly once)
!comm.epoch                                        ; region token
!comm.comm<axis = %g.row | %g.col | %g.plane | world, hier = node|flat>
```
`!comm.req` and `!comm.epoch` are *linear/affine types*: the verifier statically guarantees every request is completed exactly once and buffers are not touched inside their epoch — this eliminates the classic MPI use-after-Isend bug class by construction.

**Metadata.**
- Per-op message size, predicted time from α–β model, chosen algorithm and why (`hier` bcast when grid axis spans nodes×NUMA).
- Aggregate per-phase: volume, message count, predicted overlap fraction (compute time available inside each epoch ÷ predicted comm time).
- Buffer provenance: which DIR redistribute / compute placement implied this op (traceability).
- Ordering/matching: static tag assignment; communicator = grid axis (never raw ranks — makes matching decidable).

**Invariants.**
- **Deadlock freedom, statically**: the CIR program restricted to blocking ops must be a DAG under the matches-before relation; cyclic shift patterns must use `sendrecv` or nonblocking pairs. Checked by building the match graph from the (affine) ownership functions.
- Matching totality: every send has a unique statically-identified receive (per epoch, per tag, per communicator).
- Buffer safety: linear typing of `!comm.req`; no write to a buffer between `start` and `wait`.
- Volume conservation: Σ CIR message bytes per phase == DIR's predicted implied-transfer set (bytes), ± aggregation. This is a powerful whole-layer check: CIR cannot silently move more or less data than DIR promised.

**Legal transformations.**
- **Collective algorithm selection** (α-dominated small msgs → tree/recursive-doubling; β-dominated large → ring/pipelined ring; multi-level → hierarchical: intra-node reduce on shared memory, inter-node allreduce among node leaders, intra-node bcast). Pure cost decision, always legal.
- **Coalescing/aggregation:** merge messages with same (src,dst,epoch) — legal by matching preservation; changes α·messages vs pipelining tradeoff.
- **Blocking → nonblocking + epoch motion:** hoist `start` as early as dependences allow, sink `wait` as late as possible; the enclosed computation is what gets overlapped. Legality = the epoch-independence proof (from PIR dependences).
- **Broadcast → pipelined broadcast** along k-panels (SUMMA's classic depth-pipelining: overlap panel p+1's bcast with panel p's gemm — *this transformation is where SUMMA gets its efficiency*).
- **Reduce+bcast → allreduce**, **allgather of partials → reduce_scatter+allgather** rewrites (cost-driven, semantics-preserving; reduction reorder gated by the reproducibility contract as always).
- **P2P ↔ collective**: a row-shift expressible as sendrecv ring, or the 2.5D final reduction over the c-axis as `reduce_scatter` on `%g.plane`.
- **GPU-awareness split** (deferred choice recorded here, bound in HW Mapping): `comm.bcast %buf{space=device}` may lower to GPUDirect (NIC reads HBM) or host-staged (D2H + bcast + H2D) — CIR keeps both with costs.

**Optimization opportunities.** Overlap is *the* game at scale: a SUMMA step with panel bcast time `t_c` and gemm time `t_g` runs at `max(t_c, t_g)` when pipelined vs `t_c + t_g` naive — up to 2× and, more importantly, it flattens the strong-scaling cliff where `t_c` grows relative to `t_g` (t_g/t_c ∝ nb·√P/n falling as P grows). Hierarchical collectives typically win 2–5× on fat-node machines. Persistent collectives shave α on iterative solvers.

**Verification obligations.** Deadlock-freedom proof; matching totality; linear-type discipline on requests/epochs; volume conservation vs DIR; tag/communicator uniqueness; for one-sided regions, fence discipline (put→fence→read ordering).

**Textual example (E1: pipelined SUMMA step; E2: panel bcast with lookahead).**
```mlir
// SUMMA over kt = 0..K-1 panels; grid axes as communicators
%rowc = comm.comm axis(%g.row)  {hier = <node,numa>}
%colc = comm.comm axis(%g.col)

scf.for %kt = 0 to %K {
  // prefetch NEXT panels while computing current (double-buffered)
  %eA = comm.epoch start(comm.ibcast %Abuf[(%kt+1)%2] on %rowc
                          from owner_col(%kt+1) {alg=hier, bytes=%mb*%nb*8})
  %eB = comm.epoch start(comm.ibcast %Bbuf[(%kt+1)%2] on %colc
                          from owner_row(%kt+1) {alg=hier})
  sched.compute gemm_local %Cloc += %Abuf[%kt%2] · %Bbuf[%kt%2]   // overlapped region
  comm.wait %eA  comm.wait %eB
}   // predicted: overlap_fraction = min(1, t_gemm/t_bcast) = 0.93 @ P=256, n=65536

// 2.5D closing reduction (only when c>1):
comm.reduce_scatter %Cpartial on %g.plane {op=+, alg=ring, order=tree}  // contract-gated

// E2 (Cholesky): lookahead = panel k+1 factored & bcast while trailing update of k runs
%e = comm.epoch start(comm.ibcast %panel[%k+1] on %colc from owner(%k+1))
sched.compute trailing_update(%k)          // the big syrk/gemm — hides the bcast
comm.wait %e
```

---

### 3.6 Memory / Layout IR (LIR)

**Purpose.** Bind every buffer to a *memory space* and a *layout*, make all allocation, packing, and host↔device movement explicit, and enforce capacity/alignment/NUMA constraints. After LIR there are no abstract tensors — only placed, laid-out buffers.

**Core operations.**
- `mem.alloc %buf : space(...) layout(...) {align, pinned, first_touch=..., pool=...}` / `mem.dealloc` / `mem.pool` (arena for tile buffers — factorizations allocate per-task workspace at high frequency; pooling is mandatory).
- `mem.pack %tile ← %view {layout=microkernel<mr x nr>}` / `mem.unpack` — the BLIS-style packing of A/B panels into contiguous, microkernel-ordered buffers (this *is* how real GEMMs reach peak; it must be an explicit, costed op, not a hidden detail).
- `mem.copy %dst ← %src {engine = dma|ce|cpu, stream = %s}` — the *only* op crossing spaces (host↔HBM, HBM↔HBM peer).
- `mem.view` — strided/tiled aliases with static extent arithmetic.
- `mem.touch %buf on numa(%d)` — first-touch initialization placement; `mem.advise` (read-mostly, preferred-location) for managed memory when used.

**Type system.**
```
!mem.space<ddr(numa=%d) | hbm(gpu=%dgpu) | gpu_shared | gpu_regs
           | host_pinned | scratch(l2)>
!mem.buf<f64, shape=[%m,%n],
         layout = colmajor(ld=%ld) | rowmajor | tiled<%tb, inner=colmajor>
                | packedA<mr=%mr> | packedB<nr=%nr>,
         space = ..., align = 64|128|256>
```
Layout is part of the type ⇒ a kernel expecting `packedA<8>` cannot receive a `colmajor` buffer; layout conversions are visible `mem.pack` ops with cost.

**Metadata.**
- Per-buffer: size, space, lifetime interval, NUMA/GPU home, whether it crosses an overlap epoch (⇒ must be double-buffered ⇒ 2× footprint accounted).
- Per-space occupancy timeline (peak footprint vs capacity from topology model, per NUMA domain and per GPU).
- Traffic accounting: bytes moved per `mem.copy`/`pack`, feeding the roofline check (`AI` per kernel recomputed *after* packing decisions — packing costs O(mk+kn) traffic but enables O(mkn) flops at cache speed; the model verifies the amortization `n ≥ threshold`).
- H2D/D2H schedule: which copies ride which copy engines/streams (bound in Schedule IR).

**Invariants.**
- Space correctness: every kernel's operands are in a space its executor can address (CPU kernel can't read HBM unless the platform says so; GPUDirect paths must be declared in topology).
- Capacity: per-space peak ≤ capacity − reserve, proven from lifetime intervals (interval-graph coloring gives the allocator and the proof simultaneously).
- Alignment/size legality for the target ISA (AVX-512 ⇒ 64B; HBM coalescing ⇒ 128B segments).
- Pinned-buffer discipline: any buffer used in an async `mem.copy` or GPUDirect op is `host_pinned` (or device-resident); pinning totals capped (pinning too much stalls the OS — model carries a budget).
- No cross-space implicit access; no aliasing between a packed buffer and its source during the pack's epoch.

**Legal transformations.**
- **Layout selection/propagation:** choose col-major + ld padding to avoid cache-set conflicts (ld ∤ critical stride); tiled layouts for GPU tensors (coalescing); propagate to avoid conversions, insert `pack` only where the amortization check passes.
- **NUMA placement:** allocate each rank's blocks on its own NUMA domain via first-touch by the bound thread team; replicate read-only panels per-NUMA when the α–β-like intra-node model says remote traffic exceeds copy cost.
- **Double buffering** (mandated by any CIR epoch or async copy crossing a compute region).
- **Pooling/recycling**, **in-place** updates when the dependence structure (from PIR) proves last-use.
- **Host staging vs GPUDirect binding** for the CIR deferred choice, now that pinnedness/space are concrete.
- **Compression on the wire** (fp64→fp32 for a bcast panel when the AIR error budget has slack) — an aggressive, contract-gated rewrite.

**Optimization opportunities.** On CPUs this layer decides whether GEMM runs at 15% or 90% of peak (packing + ld choice + NUMA-local operands). On GPUs, tiled/coalesced layouts and keeping panels resident in HBM across SUMMA steps (avoiding re-transfers) dominate. First-touch mistakes silently halve effective bandwidth on 4-NUMA nodes — this layer makes them impossible rather than discouraged.

**Verification obligations.** Capacity proofs per space; space-reachability of every access; alignment; lifetime/alias soundness (esp. across epochs); traffic-model consistency (LIR byte counts are what Schedule IR's roofline consumes — they must reconcile with CIR volumes at the boundaries).

**Textual example (one rank's SUMMA step, GPU node).**
```mlir
%Cloc  = mem.alloc : !mem.buf<f64,[%ml,%nl], tiled<128>, hbm(gpu=%d)>   {resident}
%Ab0/1 = mem.alloc x2 : !mem.buf<f64,[%ml,%nb], packedA<8>, hbm(gpu=%d)>  // double-buffered
%hA    = mem.alloc : !mem.buf<f64,[%ml,%nb], colmajor, host_pinned> {numa = home(%d)}
mem.touch %Cloc_host_mirror on numa(home(%d))     // first-touch by bound team if CPU path

// staged path (if no GPUDirect): bcast lands in %hA, then async H2D on copy stream
mem.copy %Ab1 ← %hA {engine = ce0, stream = %s_copy}    // overlaps %s_comp gemm
```

---

### 3.7 Schedule IR (SIR)

**Purpose.** Fix concrete execution order and all remaining numeric parameters: loop tilings bound to cache sizes, unroll/vector factors, GPU launch geometry, stream assignment, software pipelining, prefetch, and the task-graph execution policy. SIR is the **autotuning surface**: every knob is a named attribute with a declared legal range.

**Core operations.**
- `sched.loopnest` with bound tile sizes per memory level (`L3=%mc x %kc`, `L2/L1 = %nc/%mr x %nr`), `unroll`, `vectorize<width, isa>`, `prefetch<dist>`.
- `sched.kernel gemm_micro<%mr x %nr>` — the register-blocked microkernel contract (inputs `packedA/packedB`, FMA chain length ⇒ latency hiding requirement `mr·nr ≥ fma_lat·fma_ports·simd_width`).
- `sched.launch %kern grid(%gx,%gy) block(%bx,%by) {smem=%s, regs≤%r, occupancy_est}` — GPU geometry.
- `sched.stream %s`, `sched.event`, `sched.wait` — intra-device concurrency (compute streams vs copy streams), mirroring CIR's epochs at device scope.
- `sched.pipeline depth=%d` — software pipelining of {copy, pack, compute} stages.
- `sched.taskpolicy {order = priority(critical_path) | locality, steal = hierarchical}` — for PIR task graphs (Cholesky): priorities from the critical path (panel tasks first — classical lookahead falls out of CP-priority scheduling).
- `sched.tune %param in range {objective, budget}` — declared autotuning points.

**Type system.** Mostly attributes on LIR buffers/ops, plus `!sched.stream`, `!sched.event` (affine, like `!comm.req`) and `!sched.knob<int, range, constraint>` — knob constraints are part of the type (`%kc·%mc·8 ≤ L3/2`, `%nb mod %mr == 0` tying DIR's block size to the microkernel).

**Metadata.**
- Per-kernel roofline prediction: AI (from LIR byte counts), predicted % of peak, register pressure, occupancy estimate (regs/thread & smem/block vs SM limits).
- Pipeline fill/drain overhead; stream dependency graph.
- The **tuning record**: parameter values, whether from model, database (machine+shape keyed), or fresh autotuning, with measured numbers when available.

**Invariants.**
- Every schedule refines PIR's dependence relation (re-verified after reordering — the polyhedral legality check runs again here on the final schedule).
- Resource feasibility: regs/thread ≤ file, smem ≤ per-SM, occupancy ≥ declared floor for latency-bound kernels; CPU: live ranges of the microkernel ≤ architectural registers.
- Stream/event affine discipline; no cross-stream access without event ordering.
- All knobs bound (no symbolic parameters may reach HW Mapping).

**Legal transformations.** Any dependence-preserving reordering; tile-size/unroll/vector-width binding within knob constraints; pipelining; stream reassignment; kernel fusion of adjacent LIR ops with compatible geometry (e.g., fuse the SUMMA local scale-and-accumulate epilogue into the gemm kernel). **Autotuning is a transformation policy here, not a separate system**: the tuner proposes knob vectors, the verifier re-checks, the benchmark harness measures, results persist in the tuning database.

**Optimization opportunities.** Last 2–5× on kernels; occupancy vs per-thread-resources tradeoff on GPUs; pipelining that composes with CIR overlap (network-overlap outside, copy/compute overlap inside — three-deep pipelines are the norm on multi-GPU SUMMA).

**Verification obligations.** Dependence re-verification post-scheduling; resource proofs; occupancy/pressure sanity vs declared floors; knob-constraint satisfaction; reproducibility: tuned schedules are pinned by hash — the same program+machine+shape must reload the same schedule (performance reproducibility is a supported contract).

**Textual example (CPU microkernel nest + GPU launch).**
```mlir
sched.loopnest gemm_local {
  tile jc = %nc(L3B=4080), pc = %kc(=256), ic = %mc(=144)   // BLIS-style
  pack B→packedB<nr=8> at pc;  pack A→packedA<mr=8> at ic
  tile jr = %nr(8), ir = %mr(8)
  sched.kernel gemm_micro<8x8> {vectorize<avx512>, unroll_k=4, prefetch<dist=2·kc>}
} {pred_AI = 32 flop/B(L3), pred_peak = 0.92}

sched.launch gemm_tile_kern grid(%n/128, %n/128) block(16,16)
  {smem = 2·128·32·8 = 64KiB, regs = 120/thr, occupancy_est = 0.5, // fp64 gemm: occupancy 50% is fine — it's throughput-bound, not latency-bound
   streams: %s_comp; copies on %s_copy; pipeline depth = 2}
```

---

### 3.8 Hardware Mapping IR (HIR)

**Purpose.** Bind the logical execution structure to *named physical resources*: ranks→nodes/NUMA, threads→cores (affinity masks), device IDs, NIC selection, communicator construction with rank reordering, environment/launch configuration. HIR is what the job launcher consumes.

**Core operations.**
- `hw.rankmap %g → topo {order = topology_aware, objective = min inter-group traffic}` — embeds the DIR grid into the machine so that heavy grid axes (row bcasts) map to well-connected subsets (same rack/group); produces the reordered `MPI_Comm` construction (`MPI_Dist_graph_create_adjacent` / `MPI_Cart_create(reorder)` semantics, but computed by *our* model, not the MPI library's guess).
- `hw.bind rank(%r) → {node, numa=%d, cores=mask, nic=%k, gpu=%dgpu}` — the affinity triple from the topology model's closeness map (rank's GPU, its home NUMA, and its NIC must be mutually closest — the classic "wrong-NIC" 2× penalty is eliminated by construction).
- `hw.threads team(%t) → cores {pin = compact|scatter, smt = off}`; `hw.progress_thread on core(%c)` when async progress is needed for overlap (MPI progress is not free — a dedicated core is often the price of real overlap; the model decides if it pays).
- `hw.device %dgpu {sm_clock_policy, mem_pool_cfg}`; `hw.comm_backend {p2p: gpudirect|staged, coll: nccl/rccl | mpi | sharp-offload}` — binding CIR's deferred choices.
- `hw.launch {ranks/node, threads/rank, env = {OMP_PLACES, OMP_PROC_BIND, MPICH_*/UCX_* knobs}}`.

**Type system.** Resource handle types referencing topology nodes (`!hw.core_set`, `!hw.gpu`, `!hw.nic`, `!hw.comm`), all *linear at node scope* — a core is in at most one team's mask, a GPU serves one rank (or an explicitly declared MPS/MIG share).

**Metadata.** The embedding quality report: predicted inter-group vs intra-group volume after rank reordering; per-rank resource ledger (cores, memory per NUMA, pinned budget); the exact launch line (srun/mpirun flags) as an artifact.

**Invariants.**
- Resource exclusivity (linear types) and completeness (every SIR stream/team/comm bound).
- Consistency with topology hash; feasibility (ranks/node × threads/rank ≤ cores, honoring the progress-thread reservation).
- Grid congruence promised by DIR (`congruent=[node,numa]`) actually realized by the rank map — verified, since CIR's hierarchical-collective choice depended on it.

**Legal transformations.** Rank reordering (graph embedding — quadratic assignment heuristics guided by the CIR traffic matrix); ranks-per-node vs threads-per-rank rebalancing (one rank per GPU is the default on GPU nodes; one rank per NUMA on CPU nodes); NIC striping for multi-rail; backend selection (RCCL vs MPI per collective, decided by measured tables).

**Verification obligations.** Exclusivity/feasibility proofs; a **binding audit at startup** (runtime re-checks hwloc reality vs the plan and refuses or re-plans on mismatch — jobs land on heterogeneous partitions more often than anyone admits).

**Textual example.**
```mlir
hw.rankmap %g(16x16) → cluster {grid.row ⊂ group, node holds 2x2 subgrid}
  {pred: 78% of bcast bytes intra-group}
hw.bind rank(0) → {node 0, numa 0, cores 0-14, nic 0, gpu 0}
hw.progress_thread core(15)                       // 1 of 16 cores buys overlap
hw.comm_backend {coll.bcast = rccl, coll.allreduce = sharp, p2p = gpudirect}
hw.launch {ranks/node = 8, threads/rank = 15, OMP_PROC_BIND=close, OMP_PLACES=cores}
```

---

### 3.9 MLIR / LLVM lowering

Each of our layers has (or maps onto) an MLIR dialect; the top four are custom dialects, the bottom reuses upstream infrastructure heavily:

| Our layer | MLIR realization |
|---|---|
| Math IR | custom `math_lang` dialect (linalg-adjacent, but with the property lattice and symbolic-dim solver) |
| Algorithm IR | custom `algo` dialect; lowers into `linalg` named ops + `scf` recursion skeletons |
| Parallel IR | `scf.forall`, `scf.reduce`, custom `taskdag` ops; polyhedral via ISL side-structure (or `affine` where it fits) |
| Distribution IR | custom `dist` dialect (upstream `mesh`/sharding dialects are inspirations but our ownership-function types are richer) |
| Communication IR | custom `comm` dialect → lowers to MPI/RCCL runtime calls through `llvm.call`, plus `async` dialect for epochs |
| Memory/Layout IR | `memref` + custom space/layout attributes; `gpu.alloc`, pinned-host runtime calls |
| Schedule IR | `transform` dialect *is* the schedule representation for kernels (tile/fuse/vectorize as transform scripts); `gpu.launch_func`; `vector` dialect for microkernels |
| HW Mapping | runtime-init module: emits the launch artifact + startup binding code |
| Codegen | `vector→llvm`, `nvvm`/`rocdl` for device kernels; **or** a call into vendor BLAS — see below |

**Vendor-library escape hatch (important, pragmatic):** at Schedule IR, any kernel whose contract matches a vendor routine (`gemm_local` ↔ `dgemm`/`cublasDgemm`) may lower to a *call* instead of generated code, decided by the tuning database (vendor vs generated, measured). The IR stack above is identical either way — the library is just one more schedule. This is how the MVP ships fast and how the compiler stays honest (it must beat or match the library to justify generated code).

---

## 4. Running example E1: `C = A × B` — six lowerings of one MIR node

All six start from the identical MIR/AIR:
```
%C = math.matmul %A, %B → algo.gemm classical {flops = 2n²k, recurse = blocked}
```
The planner picks a branch based on problem size vs the machine model; the branch is *entirely* a Distribution/Memory/Schedule/HW-Mapping decision.

### 4.1 Sequential GEMM (n small, or single-core target)
- PIR: tiled reduction as in §3.3, all parallel constructs degenerate.
- DIR: trivial grid `1×1`. CIR: empty (verifier: DIR implied-transfer set is empty ⇒ CIR must be empty).
- LIR: col-major, single NUMA, BLIS packing if n ≥ ~200 (amortization check), else direct.
- SIR: the §3.7 loopnest; knobs from the tuning DB for this CPU.
- Decision rule: chosen when `2n²k / peak_1core < dispatch_overhead_threshold` or resources pinned to 1 core.

### 4.2 Multithreaded (BLAS-class) GEMM — multicore + NUMA
- PIR: `par.forall` over (ic, jc) macro-tiles; reduction over pc kept sequential per tile (avoids atomic/reduction traffic — the standard BLAS choice, legal since order within a tile is preserved).
- DIR: still 1 rank, but **grid<numa: 1×4>** as an intra-node "distribution": C macro-tiles homed per NUMA domain — DIR machinery reused below the MPI level (this uniformity is deliberate: NUMA is just the innermost distribution level).
- LIR: first-touch of each C tile by its owning team; A/B panels packed *per NUMA* (replicated read-only panels — the intra-node analog of a broadcast, decided by the intra-node α–β model).
- SIR: nested parallelism — teams over jc (one per NUMA), threads within team over ic/jr; microkernel per §3.7.
- HIR: `OMP_PLACES=cores, PROC_BIND=close`, one team per NUMA; SMT off for fp64 gemm (FMA ports saturate at 1 thread/core).
- Or: the whole thing is one `dgemm` call into a vendor BLAS with the same affinity setup — the tuning DB arbitrates.

### 4.3 Single-GPU GEMM
- LIR: decide **residency**: if A, B, C fit in HBM and are reused, home them there for the program region (the residency analysis is a lifetime/traffic optimization at LIR, not per-call); else stream tiles with double-buffered H2D on copy streams.
- The H2D amortization check is explicit: transfer `(n·k+k·n+n²)·8` bytes at link bw vs `2n²k` flops at GPU peak ⇒ profitable iff `k ≳ (peak/link_bw)·12` — for 61 TF vs 96 GB/s, k ≳ a few thousand, or the data must already be resident. The compiler *prints this reasoning* in the plan artifact.
- SIR: `sched.launch` tiled kernel (or cublas/rocblas call); epilogue fusion if C feeds an elementwise op.
- HIR: rank's home NUMA = GPU's closest; pinned staging buffers on that NUMA.

### 4.4 Multi-GPU (single node, 8 GPUs)
- DIR: grid `2×4` **over GPUs** — again the same DIR machinery, one level down; block distribution (no cyclic needed for one gemm).
- CIR: row/col broadcasts of panels over **xGMI links** — collectives lowered to RCCL/NCCL or direct peer `mem.copy` rings; the α–β model uses the GPU-GPU link table (200 GB/s direct pairs vs routed).
- LIR: panels resident in each GPU's HBM; double-buffered peer transfers; host untouched in steady state.
- SIR: per-GPU three-stage pipeline {peer-copy k+1 ∥ gemm k}; one CPU thread per GPU driving its streams.
- This is "SUMMA at node scope" — deliberately the *same plan shape* as 4.5, so 4.5 nests it.

### 4.5 Distributed SUMMA (the flagship)
Chosen when P·(mem/rank) is needed or time budget demands it. Full plan:
- DIR: grid `√P×√P` (topology-congruent: each node holds a 2×4 GPU subgrid ⇒ effective hierarchical grid), block-cyclic `%nb` (cyclic unnecessary for pure gemm but chosen if C later feeds a factorization — cross-op distribution planning).
- CIR: the pipelined ibcast structure of §3.5; hierarchical bcast (inter-node leaders, then intra-node xGMI); GPUDirect if available.
- LIR: HBM-resident C; two panel buffers per operand per GPU; pinned host bounce buffers only on the staged path.
- SIR: three-deep pipeline: network bcast (k+2) ∥ intra-node distribution (k+1) ∥ gemm (k).
- HIR: rank map puts grid rows within groups; progress thread if the MPI stack needs it; RCCL for intra-node legs, MPI for inter-node.
- Predicted cost per rank: `2n³/P·γ_gpu + 2(n²/√P)·β_net·(1−overlap)` + `K·α_hier` — the plan artifact shows the strong-scaling curve and the P beyond which α dominates.

### 4.6 Communication-avoiding variant (2.5D)
Triggered when the model shows 4.5 is bandwidth-bound (`t_comm/t_flop > θ` even with full overlap) **and** memory headroom ≥ c:
- DIR: grid `√(P/c)×√(P/c)×c`; A,B replicated across the c-axis; each plane computes a k-slab of the sum.
- CIR: initial `bcast` of A,B along the c-axis (one-time, pipelined with first slabs), final `reduce_scatter` of partial C over the c-axis. Volume: `O(n²/√(cP))` per rank — the √c win, with the ~1.15× lower-bound certificate closing the search.
- Gating: the final reduction reorders the k-sum across planes ⇒ requires `repro ≠ bitwise`. If the user demanded bitwise, the planner reports *why* 2.5D was rejected and what it would have saved — refusal with a price tag, a deliberate UX principle.

---

## 5. Running example E2: distributed SPD solve, end to end

```
MIR:  %x = math.solve %S{spd}, %b {residual ≤ 1e-10}
AIR:  chol(right-looking, blocked, lookahead-eligible) → trsm → trsmᵀ
      + alt: mixed-precision chol{fp32} + iterative refinement{fp64}   // resolved by cond estimate
PIR:  the potrf/trsm/syrk task DAG of §3.3   (crit path Θ(n/nb) — hard ceiling on strong scaling, reported)
DIR:  grid √P×√P, block-cyclic(nb) mandatory (load balance under shrinking active submatrix);
      panel column replicated transiently (owner factors, then bcast)
CIR:  per step k: column-comm bcast of L-panel (pipelined: lookahead depth 1–2),
      row-comm bcast of the trsm results; trailing syrk/gemm overlapped with panel k+1's
      factor+bcast (classic lookahead = CP-priority scheduling, §3.7)
LIR:  per-rank block storage tiled<nb>; workspace pool for panel copies; GPU: trailing update
      on device, panel factorization on CPU or GPU by size (small potrf is latency-bound —
      the model routes tasks by grain size, a genuinely heterogeneous schedule)
SIR:  task policy = priority(critical_path); trailing gemms batched per stream;
      nb tuned jointly: DIR balance ↔ panel bcast size ↔ gemm efficiency (typ. 256–512 GPU)
HIR:  as §3.8; solve phase reuses the factorization grid (redistribute for the triangular
      solves costed and — per the model — rejected: trsm's O(n²) flops don't amortize an
      O(n²) redistribution)
```
The two `trsm` solves are communication-bound (O(n²) flops, O(n²/√P) words moved — AI is flat); the planner reports that the solve phase will scale poorly and, if many right-hand sides exist, automatically blocks them (`trsm` on n×nrhs — restoring AI). If the AIR chose mixed precision, refinement iterations reuse the CIR persistent-collective setup (allreduce for residual norms), and the verifier carries the convergence certificate `cond(S)·u_fp32 < 1` — if cond is unknown, a cheap condition estimator is *inserted into the plan* with a runtime fallback edge to the fp64 branch. Runtime algorithm fallback is a first-class plan structure, not an exception path.

---

## 6. Where AI helps — and where it must not

The dividing line is a hard architectural rule: **AI proposes, deterministic machinery disposes.** No learned component ever asserts legality, and every AI-influenced decision is reproducible (pinned model version + recorded decision in the plan artifact).

**Deterministic analysis is strictly superior for:**
- **Legality of everything**: dependence analysis, polyhedral transformation legality, deadlock freedom, ownership coverage, capacity/alignment proofs, error-bound entailment. These are decidable (affine) or certifiable in our domain; a learned guess here is a correctness bug generator.
- **Closed-form performance regimes**: roofline tile sizing, α–β collective selection, 2D-vs-2.5D crossover, H2D amortization, block-size coupling constraints. Analytical models are exact enough to prune 99% of the search space instantly and — crucially — they *explain themselves* in the plan artifact.
- **Communication lower bounds** as stopping criteria: no learned model can certify "within 1.2× of optimal"; the bound can.
- **Polyhedral scheduling** for the affine loop nests that dominate dense linear algebra: complete, exact dependence information beats any learned scheduler in its domain.
- **Benchmarking as ground truth**: the tuning database's measured numbers override every model, learned or analytical. Autotuning search (bounded, cached, shape-keyed) is the final arbiter for kernel knobs.

**AI genuinely helps at:**
- **Search guidance** where spaces are large and non-convex: proposing autotuning starting points (learned cost model as a surrogate — cuts tuning time 5–20×), proposing rank-embedding solutions for the quadratic-assignment problem in `hw.rankmap`, proposing fusion groupings. Always followed by verification + measurement.
- **Cost-model calibration**: learning the residual between analytical predictions and measurements per machine (analytical model gives the shape; a small learned correction gives fidelity — far more sample-efficient than end-to-end learned costs).
- **Algorithm selection priors** in murky regimes (sparse patterns, preconditioner choice, GMRES restart values) where no closed form exists — as a prior over the `algo.alt` set, with runtime fallback edges as the safety net.
- **Frontend intelligence**: recognizing mathematical structure the user didn't annotate (this expression is a Sylvester equation; this matrix is Toeplitz-like), *proposed as an assertion the deterministic prover or a runtime check must confirm*.
- **Diagnostics**: translating a plan artifact and a performance trace into an explanation ("you are α-bound; strong scaling ended at 64 nodes; here is the nb change worth trying") — high value, zero correctness risk.

**Explicitly out of scope for AI:** generating communication code directly, choosing reduction orders under a bitwise contract, anything inside a verifier.

---

## 7. HPC MVP — buildable by one developer after the frontend works

Scope discipline: dense fp64, two ops (`gemm`, SPD `solve`), CPU + optional single-GPU per rank, MPI. Every IR layer exists *in miniature* (textual form + verifier), because retrofitting layers is the death of such projects — but each layer's transformation set is tiny.

**Milestone 0 (week 1–2): machine model + harness.**
hwloc-based topology ingestion → the §1 s-expression; microbenchmark pass (STREAM per NUMA, dgemm peak per core-count, osu latency/bw); benchmark harness with a results database (SQLite, keyed by machine-hash × op × shape × knobs). *Everything later is honest because this exists first.*

**Milestone 1 (week 3–5): MIR + AIR + sequential/threaded lowering.**
MIR with dims + {dense, spd} properties only; the `inverse→solve` and matrix-chain rewrites; AIR with `gemm classical` and `chol/trsm`; lower to vendor BLAS/LAPACK calls (`dgemm`, `dpotrf/dtrsm`) with correct OpenMP affinity (HIR emits `OMP_PLACES/PROC_BIND`). Verifiers: dims, property algebra, flop accounting. Deliverable: matches `numpy`/`MATLAB` semantics at vendor-BLAS speed, with a plan artifact that shows its reasoning.

**Milestone 2 (week 6–9): DIR + CIR + SUMMA.**
2D block-cyclic types with affine ownership; DIR verifier (coverage, memory). CIR with `ibcast/wait` epochs + the linear-type request checker + static match/deadlock check for this one pattern. Lower to MPI (Fortran 2018 `mpi_f08` runtime — see skeleton below) calling node-local BLAS. Grid/nb chosen by the α–β model; pipelined double-buffered SUMMA as the *only* CIR transform. Deliverable: ≥70% weak-scaling efficiency to 64 ranks vs single-node BLAS, plan artifact predicts the scaling curve, benchmark harness confirms.

**Milestone 3 (week 10–12): distributed Cholesky solve.**
Reuse the same grid/types; fork-join version first (ScaLAPACK-style per-step bcasts — no task runtime yet), lookahead depth-1 as the single scheduling optimization. Runtime residual check wired to the accuracy contract.

**Milestone 4 (week 13–15): single GPU per rank.**
LIR residency for the local blocks (HBM-homed C, double-buffered panels), cuBLAS/rocBLAS local gemm, staged host bcast path only (GPUDirect deferred). The H2D amortization check goes in the planner and the plan artifact.

**Milestone 5 (week 16+): one CA feature + autotuning loop.**
2.5D gemm behind the memory-headroom + reproducibility gate; `nb` and grid-shape autotuning through the harness into the database. Stop. (Task-DAG runtime, polyhedral kernels, codegen replacing vendor BLAS, sparse — all explicitly post-MVP.)

**MVP runtime skeleton — the SUMMA inner loop the compiler targets** (modern Fortran, `mpi_f08`, double-buffered, overlap-structured exactly as CIR emits it):

```fortran
! Compiler-emitted target: pipelined SUMMA step on a pr x pc grid.
! Buffers Abuf(:,:,0:1), Bbuf(:,:,0:1) are the LIR double buffers (64B-aligned,
! first-touched by the bound team). row_comm/col_comm are the CIR grid axes.
use mpi_f08
type(MPI_Request) :: reqA(0:1), reqB(0:1)
integer :: kt, cur, nxt

cur = 0
call panel_bcast(0, Abuf(:,:,cur), Bbuf(:,:,cur), reqA(cur), reqB(cur))
call MPI_Waitall(2, [reqA(cur), reqB(cur)], MPI_STATUSES_IGNORE)

do kt = 0, npanels - 1
   nxt = 1 - cur
   if (kt < npanels - 1) &                              ! CIR epoch: start(k+1)
      call panel_bcast(kt+1, Abuf(:,:,nxt), Bbuf(:,:,nxt), reqA(nxt), reqB(nxt))

   ! Overlapped compute region (SIR): local gemm on the CURRENT buffers.
   ! beta=1 accumulates into the resident local C block.
   call dgemm('N','N', mloc, nloc, nb, 1.0d0, Abuf(:,:,cur), mloc, &
              Bbuf(:,:,cur), nb, 1.0d0, Cloc, mloc)

   if (kt < npanels - 1) then                           ! CIR: wait(k+1)
      call MPI_Waitall(2, [reqA(nxt), reqB(nxt)], MPI_STATUSES_IGNORE)
      cur = nxt
   end if
end do

contains
  subroutine panel_bcast(kt, Ab, Bb, rA, rB)
    integer, intent(in) :: kt
    real(8), intent(inout) :: Ab(:,:), Bb(:,:)
    type(MPI_Request), intent(out) :: rA, rB
    ! Owner column/row of panel kt from the DIR ownership function.
    if (mycol == owner_col(kt)) call pack_local_panelA(kt, Ab)   ! LIR mem.pack
    if (myrow == owner_row(kt)) call pack_local_panelB(kt, Bb)
    call MPI_Ibcast(Ab, size(Ab), MPI_REAL8, owner_col(kt), row_comm, rA)
    call MPI_Ibcast(Bb, size(Bb), MPI_REAL8, owner_row(kt), col_comm, rB)
  end subroutine
```

Two implementation notes the compiler must encode (both come straight from the IR invariants): (1) the buffers under an in-flight `MPI_Ibcast` are never touched — guaranteed by CIR's linear `!comm.req` typing, realized here by the cur/nxt discipline; (2) real overlap may require an MPI progress thread or `MPICH_ASYNC_PROGRESS` — that is an HIR decision recorded in the launch artifact, and the harness *measures* the achieved overlap fraction against CIR's prediction, feeding the calibration loop.

**MVP success criteria.** (a) One MIR program produces §4.1–§4.5 variants (4.6 as stretch) with no user code changes; (b) every plan ships a human-readable artifact: flops, predicted AI per kernel, communication volume, predicted vs measured scaling; (c) all verifiers run in CI on every lowering; (d) SUMMA within 1.3× of a tuned ScaLAPACK `pdgemm` on the same machine — the honesty benchmark.
