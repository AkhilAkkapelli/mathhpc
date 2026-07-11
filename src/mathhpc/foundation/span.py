"""``SourceSpan``: the source range every AST/HIR node and provenance report carries.

Convention (fixed here, used verbatim by the Task 007 parser): ``file`` is a
non-empty source name (a path or a synthetic name like ``"<stdin>"``); lines are
1-based; columns are 0-based with an exclusive end column, so ``col0 == col1`` on
one line is the empty span at that position. Span arithmetic (join, contains) is
deliberately absent until a consumer exists.
"""

from dataclasses import dataclass

from mathhpc.foundation.immutable import validate_frozen_instance
from mathhpc.foundation.registry import register_frozen_type

__all__ = ["SourceSpan"]


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Half-open source range ``file:(line0,col0)–(line1,col1)``."""

    file: str
    line0: int
    col0: int
    line1: int
    col1: int

    def __post_init__(self) -> None:
        validate_frozen_instance(self)
        if type(self.file) is not str:
            raise TypeError(f"SourceSpan.file must be str, got {type(self.file).__qualname__}")
        if not self.file:
            raise ValueError("SourceSpan.file must be non-empty")
        for name in ("line0", "col0", "line1", "col1"):
            if type(getattr(self, name)) is not int:
                raise TypeError(
                    f"SourceSpan.{name} must be int (exactly), "
                    f"got {type(getattr(self, name)).__qualname__}"
                )
        if self.line0 < 1:
            raise ValueError(f"SourceSpan.line0 must be >= 1, got {self.line0}")
        if self.col0 < 0 or self.col1 < 0:
            raise ValueError(f"SourceSpan columns must be >= 0, got {self.col0}, {self.col1}")
        if self.line1 < self.line0:
            raise ValueError(f"SourceSpan end line {self.line1} before start line {self.line0}")
        if self.line1 == self.line0 and self.col1 < self.col0:
            raise ValueError(f"SourceSpan end col {self.col1} before start col {self.col0}")


register_frozen_type(
    SourceSpan,
    fixture=lambda: (
        SourceSpan("demo.mth", 1, 0, 1, 0),
        SourceSpan("demo.mth", 1, 0, 1, 0),
        SourceSpan("demo.mth", 14, 1, 14, 22),
        SourceSpan("<stdin>", 2, 4, 5, 0),
    ),
    schema_version=1,
)
