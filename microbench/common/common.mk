# Shared Makefile fragment for
# microbench/<memory|compute>/<instruction-family>/<configuration>/Makefile.
# Building is local-safe; execution is always an explicit `make run` action.

CUDA_HOME  ?= /usr/local/cuda
NVCC       ?= $(CUDA_HOME)/bin/nvcc
NVDISASM   ?= $(CUDA_HOME)/bin/nvdisasm
PYTHON     ?= python3
ARCH       ?= sm_90a
CXX_STD    ?= c++17

COMMON_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT  := $(abspath $(COMMON_DIR)/../..)
FLASHMLA   ?= $(REPO_ROOT)/operators/flash_mla/target

INCLUDES := -I$(COMMON_DIR) \
            -I$(FLASHMLA)/csrc \
            -I$(FLASHMLA)/csrc/kerutils/include \
            -I$(FLASHMLA)/csrc/cutlass/include

COMMON_NVCCFLAGS := -O3 -std=$(CXX_STD) -arch=$(ARCH) \
                    --expt-relaxed-constexpr --use_fast_math -lineinfo \
                    $(INCLUDES) $(DEFINES)
COMMON_LDFLAGS := -lcuda

SOURCE     ?= $(TARGET).cu
PTX_FILE   ?= $(TARGET).ptx
CUBIN_FILE ?= $(TARGET).cubin
SASS_FILE  ?= $(TARGET).sass
RUN_BINARY := $(if $(filter /%,$(TARGET)),$(TARGET),./$(TARGET))
RESULT_TOOL ?= $(REPO_ROOT)/tools/result_tool.py
RESULT_DIR  ?= result
RUN_ID_RAW  := $(value RUN_ID)
override export RUN_ID := $(RUN_ID_RAW)
COMMON_HEADERS := $(wildcard $(COMMON_DIR)/*.h) \
                  $(wildcard $(COMMON_DIR)/*.hpp) \
                  $(wildcard $(COMMON_DIR)/*.cuh)
BUILD_MAKEFILES := $(MAKEFILE_LIST)

compile: $(TARGET)

$(TARGET): $(SOURCE) $(COMMON_HEADERS) $(BUILD_MAKEFILES)
	$(NVCC) $(COMMON_NVCCFLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $< -o $@ \
		$(COMMON_LDFLAGS) $(LDFLAGS) $(EXTRA_LDFLAGS)

ptx: $(PTX_FILE)

$(PTX_FILE): $(SOURCE) $(COMMON_HEADERS) $(BUILD_MAKEFILES)
	$(NVCC) $(COMMON_NVCCFLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) --ptx $< -o $@

cubin: $(CUBIN_FILE)

$(CUBIN_FILE): $(SOURCE) $(COMMON_HEADERS) $(BUILD_MAKEFILES)
	$(NVCC) $(COMMON_NVCCFLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) --cubin $< -o $@

sass: $(SASS_FILE)

$(SASS_FILE): $(CUBIN_FILE)
	$(NVDISASM) --print-line-info $< > $@

static: ptx sass

# Running is deliberately not part of compile/static/default targets.
run: compile
	@set -eu; \
	case "$${RUN_ID-}" in \
		"" ) ;; \
		"."|".."|[!A-Za-z0-9]*|*[!A-Za-z0-9._-]*) \
			echo "error: invalid RUN_ID" >&2; exit 2 ;; \
	esac; \
	if [ -n "$${RUN_ID-}" ]; then \
		set -- --run-id "$$RUN_ID"; \
	else \
		set --; \
	fi; \
	$(PYTHON) "$(RESULT_TOOL)" run \
		--result-dir "$(abspath $(RESULT_DIR))" --kind micro "$$@" -- \
		$(RUN_BINARY) $(ARGS)

clean:
	rm -f $(TARGET) $(PTX_FILE) $(CUBIN_FILE) $(SASS_FILE) log

.DEFAULT_GOAL := compile
.PHONY: compile ptx cubin sass static run clean
