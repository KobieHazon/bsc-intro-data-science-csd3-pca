#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = "".join(("208", "234", "161"))


def main() -> None:
    checked = 0
    for path in sorted(ROOT.rglob("*")):
        if ".git" in path.parts or path.is_dir():
            continue
        if path.suffix == ".ipynb":
            json.loads(path.read_text(encoding="utf-8"))
            checked += 1
        if path.suffix in {".csv", ".ipynb", ".md", ".txt", ".py"}:
            if FORBIDDEN in path.read_text(encoding="utf-8", errors="ignore"):
                raise SystemExit(f"forbidden student identifier remains in {path.relative_to(ROOT)}")
    print(f"validated {checked} notebook(s)")


if __name__ == "__main__":
    main()
