# mathhpc

A math-first, AI-assisted, evidence-driven HPC programming language and compiler system:
mathematical intent — or semantics recovered from legacy Fortran — enters one
evidence-driven optimization pipeline and comes out as correct, explainable, optimized
implementations for CPUs, GPUs, and (eventually) distributed supercomputers.

**Status: Foundation v1.0 is frozen. Task 001 (immutability substrate) is implemented
and green.** Foundation changes require a formal Foundation Change Request backed by
concrete implementation evidence.

## What exists today (Task 001)

The minimal trustworthy immutable substrate that all frozen Foundation v1.0 records
will be built on:

- `FrozenDict` — immutable mapping; insertion-order iteration (presentation only),
  order-independent equality against any `Mapping`, lazy cached order-independent
  hashing (immutability ≠ hashability). Canonical serialization order is deliberately
  *not* defined here (Task 002 serializer).
- `assert_deeply_immutable` / `validate_frozen_instance` — recursive validator with
  path-precise `MutableFieldError` (e.g. `root.payload.metadata['shape'][2]`), cycle
  rejection, shared-DAG acceptance.
- `ClaimId` / `EvidenceId` — strongly distinct identity types, separated statically
  (pyright strict) and at runtime; no shared base class, no ordering.
- Frozen-type registry — every frozen foundation dataclass registers with a fixture
  factory; a discovery test fails CI if anything under `src/` is unregistered; a
  generic invariant suite (immutability, mutation rejection, hash–eq law) runs over
  every registered type.
- Negative static-type tests — intentionally ill-typed code asserted to produce the
  exact pyright diagnostics, matched by rule and line.

Zero runtime dependencies. Dev/test: pytest, hypothesis, pyright (strict), ruff.

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff format --check . && ruff check . && pyright && pytest
```

Expected: all gates pass, 62 tests green.

## Design documents

The full architecture — the IR stack, hardware topology model, Claim+Evidence+Decision
spine, contract lattice, Semantic Recovery Pipeline, MVP plan, and the freeze itself —
lives in [`docs/design/`](docs/design/), in reading order 01–07.

## Roadmap (near term)

Task 002+: semantic domains, source spans, claim/evidence records, the five-state
claim fold, the foundation store — then the math frontend, SpecPacks (BLAS/LAPACK),
codegen, and the first end-to-end vertical slice:
`C = matmul(A, B)` → typed HIR → decision → DGEMM → generated Fortran → differential
comparison → provenance report.
