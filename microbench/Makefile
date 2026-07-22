.PHONY: validate static dry-build dry-run build quick run

validate:
	python3 scripts/validate_manifest.py

static:
	python3 scripts/static_check.py --compile

dry-build:
	python3 scripts/build_all.py --dry-run

dry-run:
	python3 scripts/run_all.py --mode quick --dry-run

build:
	python3 scripts/build_all.py

quick:
	python3 scripts/run_all.py --mode quick

run:
	python3 scripts/run_all.py --mode full
