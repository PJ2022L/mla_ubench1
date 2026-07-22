#!/usr/bin/env python3
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from family_runner import sweep_family

raise SystemExit(sweep_family(HERE.parents[1], sys.argv[1:]))
