# mathhpc — working rules for Claude Code

Foundation v1.0 is FROZEN. Before any task, read:
- docs/design/07-errata-a1.1-freeze-task-001.md  (errata, freeze, Task 001 spec — the current state)
- docs/design/05-mvp-implementation-spec.md      (MVP scope, IRs, core types, task list §18)
- docs/design/06-mvp-addendum-a1.md              (corrected runtime boundary, DPOTRF chain,
                                                  alias guard, reordered task sequence)

## Governing rule
Frozen architectural types may be changed only when concrete implementation evidence
demonstrates a blocking flaw that cannot be solved through an adapter, extension point,
implementation-local mechanism, or compatible refinement. If such a flaw appears, do not
silently redesign: produce a formal Foundation Change Request (scenario, responsible
type/invariant, why no adapter suffices, smallest change, compatibility, migration,
regression tests) and stop for review.

## Workflow
- One task = one branch = one PR, following the Task 002–030 sequence in
  docs/design/06 §5 (which reorders docs/design/05 §18; the vertical slice lands at Task 012).
- Every new frozen foundation dataclass must be registered in the frozen-type registry
  (with a fixture factory) in the same PR — the discovery test enforces this.
- Every new frozen type uses @dataclass(frozen=True, slots=True) and the canonical
  __post_init__ pattern: validate_frozen_instance(self), plus explicit exact-type checks
  for ID-typed fields.
- Equivalence claims always carry a semantic domain. Confidence is never proof:
  AiInferred/Predicted evidence never gates legality.
- No Task N+1 code inside Task N. No speculative abstractions.

## Gates (all must pass before every commit)
ruff format --check . && ruff check . && pyright && pytest

pyright is strict; tests/typecheck/invalid_id_assignment.py is intentionally ill-typed
and excluded from the main run (the harness checks it by rule+line).

## Environment
Python 3.12+, zero runtime dependencies. Dev setup:
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
