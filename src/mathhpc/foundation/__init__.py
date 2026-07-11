"""Public API of the Foundation v1.0 immutable substrate (Task 001)."""

from mathhpc.foundation.ids import ClaimId, EvidenceId
from mathhpc.foundation.immutable import (
    FrozenDict,
    MutableFieldError,
    assert_deeply_immutable,
    validate_frozen_instance,
)
from mathhpc.foundation.registry import register_frozen_type, registered_frozen_types

__all__ = [
    "ClaimId",
    "EvidenceId",
    "FrozenDict",
    "MutableFieldError",
    "assert_deeply_immutable",
    "register_frozen_type",
    "registered_frozen_types",
    "validate_frozen_instance",
]
