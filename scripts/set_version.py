#!/usr/bin/env python
"""Stamp the release version into ``pyproject.toml`` (release-pipeline helper).

Usage:
    python scripts/set_version.py 0.1.0a1

The version must be a PEP 440 version (final ``0.1.0`` or pre-release
``0.1.0a1``/``0.1.0b2``/``0.1.0rc1``); the script rewrites the single static
``version = "..."`` assignment in pyproject.toml in place.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Acceptable release versions: X.Y.Z optionally followed by a/b/rc pre-release.
_PEP440 = re.compile(r"^\d+\.\d+\.\d+((?:a|b|rc)\d+)?$")

#: The one static version assignment hatchling reads (``version = "0.1.0"``).
_VERSION = re.compile(r'^(version\s*=\s*")[^"]+(")$', re.MULTILINE)


def set_version(project_root: Path, version: str) -> None:
    """Validate ``version`` and rewrite the pyproject.toml assignment."""
    if not _PEP440.fullmatch(version):
        raise SystemExit(f"error: version {version!r} is not PEP 440 (e.g. 0.1.0a1)")
    path = project_root / "pyproject.toml"
    content = path.read_text(encoding="utf-8")
    updated, count = _VERSION.subn(lambda m: f"{m[1]}{version}{m[2]}", content)
    if count != 1:
        raise SystemExit(
            f"error: expected exactly one static version assignment, found {count}"
        )
    path.write_text(updated, encoding="utf-8")
    print(f"stamped version {version}")


def main(argv: list[str]) -> None:
    """CLI wrapper; expects exactly one argument."""
    if len(argv) != 1:
        raise SystemExit("usage: set_version.py <pep440-version>")
    set_version(Path.cwd(), argv[0])


if __name__ == "__main__":
    main(sys.argv[1:])
