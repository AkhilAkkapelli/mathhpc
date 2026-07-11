"""Public API of the Foundation v1.0 substrate (Tasks 001–002)."""

from mathhpc.foundation.domains import (
    IEEE754,
    ContractNumeric,
    ExceptionProfile,
    FpFormat,
    IntegerExact,
    MathematicalComplex,
    MathematicalReal,
    OverflowBehavior,
    RoundingMode,
    SemanticDomain,
)
from mathhpc.foundation.ids import ClaimId, ContractId, DecisionId, EvidenceId, PlanId
from mathhpc.foundation.immutable import (
    FrozenDict,
    MutableFieldError,
    assert_deeply_immutable,
    validate_frozen_instance,
)
from mathhpc.foundation.registry import register_frozen_type, registered_frozen_types
from mathhpc.foundation.serialize import (
    SerializationError,
    canonical_json_bytes,
    from_json,
    register_schema_migration,
    to_json,
)
from mathhpc.foundation.span import SourceSpan

__all__ = [
    "IEEE754",
    "ClaimId",
    "ContractId",
    "ContractNumeric",
    "DecisionId",
    "EvidenceId",
    "ExceptionProfile",
    "FpFormat",
    "FrozenDict",
    "IntegerExact",
    "MathematicalComplex",
    "MathematicalReal",
    "MutableFieldError",
    "OverflowBehavior",
    "PlanId",
    "RoundingMode",
    "SemanticDomain",
    "SerializationError",
    "SourceSpan",
    "assert_deeply_immutable",
    "canonical_json_bytes",
    "from_json",
    "register_frozen_type",
    "register_schema_migration",
    "registered_frozen_types",
    "to_json",
    "validate_frozen_instance",
]
