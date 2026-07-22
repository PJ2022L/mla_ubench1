#!/usr/bin/env python3
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from family_runner import build_family

raise SystemExit(build_family(HERE.parents[1], sys.argv[1:]))
