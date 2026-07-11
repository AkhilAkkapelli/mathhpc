"""Unit tests for ``SourceSpan`` construction validation and identity semantics."""

import pytest

from mathhpc.foundation import SourceSpan


def test_span_construction_and_fields() -> None:
    span = SourceSpan("demo.mth", 14, 1, 14, 22)
    assert (span.file, span.line0, span.col0, span.line1, span.col1) == ("demo.mth", 14, 1, 14, 22)


def test_span_equality_and_hash() -> None:
    a = SourceSpan("f", 1, 0, 2, 3)
    b = SourceSpan("f", 1, 0, 2, 3)
    c = SourceSpan("g", 1, 0, 2, 3)
    assert a == b and hash(a) == hash(b)
    assert a != c


def test_span_empty_span_at_position_is_legal() -> None:
    SourceSpan("f", 3, 5, 3, 5)


def test_span_multiline_end_col_may_precede_start_col() -> None:
    SourceSpan("f", 2, 40, 3, 0)


@pytest.mark.parametrize(
    "args",
    [
        ("", 1, 0, 1, 0),  # empty file name
        ("f", 0, 0, 1, 0),  # 0-based line
        ("f", 1, -1, 1, 0),  # negative start col
        ("f", 1, 0, 1, -1),  # negative end col
        ("f", 2, 0, 1, 0),  # end line before start line
        ("f", 1, 5, 1, 4),  # same-line end col before start col
    ],
)
def test_span_rejects_invalid_ranges(args: tuple[str, int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        SourceSpan(*args)


def test_span_rejects_bool_coordinates() -> None:
    with pytest.raises(TypeError):
        SourceSpan("f", True, 0, 1, 0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_span_rejects_non_str_file() -> None:
    with pytest.raises(TypeError):
        SourceSpan(3, 1, 0, 1, 0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
