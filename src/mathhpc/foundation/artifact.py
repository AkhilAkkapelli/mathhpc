"""``ArtifactRef``: content-addressed immutable payload reference (Addendum A1 §2).

Bulk payloads live *behind* an ``ArtifactRef`` — a sha256 content hash plus the
immutable ``bytes`` — instead of as structured fields on frozen records. The hash
is validated against the payload at construction, so a stored reference can never
silently disagree with its content. ``ArtifactRef.of(data)`` computes the hash;
``EMPTY_ARTIFACT`` is the canonical no-payload reference used by evidence whose
justification is fully carried by its status and provenance.
"""

import hashlib
from dataclasses import dataclass
from typing import Final

from mathhpc.foundation.immutable import validate_frozen_instance
from mathhpc.foundation.registry import register_frozen_type

__all__ = ["EMPTY_ARTIFACT", "ArtifactRef"]

_HEX_DIGITS: Final = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A sha256-addressed immutable byte payload."""

    sha256: str
    data: bytes

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.sha256) is not str:
            raise TypeError(f"ArtifactRef.sha256 must be str, got {type(self.sha256).__qualname__}")
        if type(self.data) is not bytes:
            raise TypeError(f"ArtifactRef.data must be bytes, got {type(self.data).__qualname__}")
        if len(self.sha256) != 64 or not set(self.sha256) <= _HEX_DIGITS:
            raise ValueError("ArtifactRef.sha256 must be 64 lowercase hex digits")
        actual = hashlib.sha256(self.data).hexdigest()
        if actual != self.sha256:
            raise ValueError(
                f"ArtifactRef content hash mismatch: declared {self.sha256}, actual {actual}"
            )

    @classmethod
    def of(cls, data: bytes) -> "ArtifactRef":
        """The reference for ``data``, with the hash computed here."""
        return cls(hashlib.sha256(data).hexdigest(), data)


EMPTY_ARTIFACT: Final = ArtifactRef.of(b"")


register_frozen_type(
    ArtifactRef,
    fixture=lambda: (
        EMPTY_ARTIFACT,
        ArtifactRef.of(b""),
        ArtifactRef.of(b"payload"),
    ),
    schema_version=1,
)
