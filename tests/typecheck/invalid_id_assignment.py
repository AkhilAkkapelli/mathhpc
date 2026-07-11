"""Intentionally ill-typed source, excluded from the main pyright run and checked
only by the negative-diagnostics harness. Each expected diagnostic is annotated
with its pyright rule; the harness matches by (line, rule), not message text."""

from mathhpc.foundation import ClaimId, EvidenceId


def consume_claim(x: ClaimId) -> None:
    _ = x


consume_claim(EvidenceId(7))  # expect-error: reportArgumentType

bad: ClaimId = EvidenceId(8)  # expect-error: reportAssignmentType
consume_claim(bad)
