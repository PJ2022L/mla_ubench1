NVCC ?= nvcc
PYTHON ?= python3

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BIN_DIR := $(ROOT)/build/bin
MANIFEST := $(ROOT)/manifest.json
MANIFEST_TOOL := $(ROOT)/scripts/manifest_tool.py

ATOM_BENCHES := $(shell $(PYTHON) $(MANIFEST_TOOL) --manifest $(MANIFEST) --kind atom list)
CALIBRATIONS := $(shell $(PYTHON) $(MANIFEST_TOOL) --manifest $(MANIFEST) --kind calibration list)
ALL_BENCHES := $(ATOM_BENCHES) $(CALIBRATIONS)
ATOM_BINARIES := $(addprefix $(BIN_DIR)/,$(ATOM_BENCHES))
CALIBRATION_BINARIES := $(addprefix $(BIN_DIR)/,$(CALIBRATIONS))
COMMON_HEADERS := $(wildcard $(ROOT)/common/*.h $(ROOT)/common/*.hpp $(ROOT)/common/*.cuh)
BENCH_SOURCES := $(shell find $(ROOT)/compute $(ROOT)/memory $(ROOT)/model/calibration -name benchmark.cu -type f 2>/dev/null)

# Emit only SM90a device code. A compute_90 fallback cannot assemble WGMMA.
NVCCFLAGS := -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a
CPPFLAGS := -I$(ROOT) -I$(ROOT)/common
LDLIBS := -lcuda

.DEFAULT_GOAL := all

.PHONY: all atoms calibration everything clean list list-calibration static static-calibration validate-manifest $(ALL_BENCHES)

all: atoms
atoms: validate-manifest $(ATOM_BINARIES)
calibration: validate-manifest $(CALIBRATION_BINARIES)
everything: atoms calibration

$(ALL_BENCHES): %: $(BIN_DIR)/%

$(BIN_DIR)/%: $(MANIFEST) $(MANIFEST_TOOL) $(COMMON_HEADERS) $(BENCH_SOURCES) | $(BIN_DIR)
	@source_file="$$($(PYTHON) $(MANIFEST_TOOL) --manifest $(MANIFEST) --kind all source $*)"; \
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(CPPFLAGS) "$(ROOT)/$$source_file" -o $@ $(LDLIBS)

$(BIN_DIR):
	mkdir -p $@

validate-manifest:
	$(PYTHON) $(MANIFEST_TOOL) --manifest $(MANIFEST) --kind all validate

static: atoms
	bash $(ROOT)/scripts/static_check.sh --kind atom

static-calibration: calibration
	bash $(ROOT)/scripts/static_check.sh --kind calibration

list:
	@$(PYTHON) $(MANIFEST_TOOL) --manifest $(MANIFEST) --kind atom list | tr ' ' '\n'

list-calibration:
	@$(PYTHON) $(MANIFEST_TOOL) --manifest $(MANIFEST) --kind calibration list | tr ' ' '\n'

clean:
	rm -rf $(ROOT)/build
