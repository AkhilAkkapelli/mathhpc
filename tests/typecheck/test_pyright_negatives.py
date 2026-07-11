"""Negative static-type tests: assert that intentionally ill-typed code produces
exactly the expected pyright diagnostics, matched by rule and source line."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).parent
_TARGET = _DIR / "invalid_id_assignment.py"
_PATTERN = re.compile(r"#\s*expect-error:\s*([A-Za-z]+)")


def _expected_diagnostics() -> set[tuple[int, str]]:
    expected: set[tuple[int, str]] = set()
    for lineno, line in enumerate(_TARGET.read_text().splitlines(), start=1):
        match = _PATTERN.search(line)
        if match:
            expected.add((lineno, match.group(1)))
    return expected


def _pyright_command() -> list[str]:
    executable = shutil.which("pyright")
    if executable is not None:
        return [executable]
    return [sys.executable, "-m", "pyright"]


def test_pyright_negative_diagnostics() -> None:
    completed = subprocess.run(
        [*_pyright_command(), "--outputjson", "--project", str(_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout)
    actual: set[tuple[int, str]] = set()
    for diagnostic in payload["generalDiagnostics"]:
        if diagnostic.get("severity") != "error":
            continue
        if not str(diagnostic.get("file", "")).endswith("invalid_id_assignment.py"):
            continue
        line = int(diagnostic["range"]["start"]["line"]) + 1
        actual.add((line, str(diagnostic.get("rule", "<no-rule>"))))
    expected = _expected_diagnostics()
    assert actual == expected, (
        f"expected {sorted(expected)}, got {sorted(actual)}\nstderr: {completed.stderr}"
    )
