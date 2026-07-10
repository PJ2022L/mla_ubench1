# Shared Makefile fragment for
# microbench/<memory|compute>/<instruction-family>/<configuration>/Makefile.
# SM90a/H800 only. Most benchmark bodies are still scaffolds.

NVCC      ?= nvcc
ARCH      ?= sm_90a
CUDA_HOME ?= /usr/local/cuda

COMMON_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT  := $(abspath $(COMMON_DIR)/../..)
FLASHMLA   ?= $(REPO_ROOT)/operators/flash_mla/target

INCLUDES := -I$(COMMON_DIR) \
            -I$(FLASHMLA)/csrc \
            -I$(FLASHMLA)/csrc/kerutils/include \
            -I$(FLASHMLA)/csrc/cutlass/include

NVCCFLAGS := -O3 -arch=$(ARCH) --expt-relaxed-constexpr --use_fast_math -lineinfo $(INCLUDES) $(DEFINES)
LDFLAGS   := -lcuda -lnvidia-ml
SOURCE    ?= $(TARGET).cu

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) | tee log

clean:
	rm -f $(TARGET) log

.PHONY: run clean
