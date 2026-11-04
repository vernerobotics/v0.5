# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    "# Copyright (c) 2026 Verne Labs, Inc.\n"
    "# Licensed under the Verne V0.5 Model License. See LICENSE.\n"
)


def python_files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.py") if ".venv" not in p.parts)


def test_every_python_file_has_the_header():
    files = python_files()
    assert files, "found no Python files"
    missing = [str(p.relative_to(ROOT)) for p in files if not p.read_text().startswith(HEADER)]
    assert not missing, f"files without the Verne header: {missing}"
