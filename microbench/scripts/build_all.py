#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    families = sorted({(ROOT / item["source"]).parent for item in manifest["benchmarks"]})
    for family in families:
        command = [sys.executable, str(family / "scripts" / "build.py")]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
