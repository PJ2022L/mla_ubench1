PYTHON ?= python3

.PHONY: check dry-build build quick full clean

check:
	$(PYTHON) scripts/static_check.py

dry-build:
	$(PYTHON) scripts/build.py --dry-run

build:
	$(PYTHON) scripts/build.py

quick:
	$(PYTHON) scripts/run.py --preset quick

full:
	$(PYTHON) scripts/run.py --preset full

clean:
	rm -rf build/bin build/ptx build/cubin build/sass build/resources build/logs build/raw
