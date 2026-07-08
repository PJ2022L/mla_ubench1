# 公共 Makefile 片段 —— 每个 atoms/<x>/Makefile `include ../../common/common.mk`。SCAFFOLD。
# 范式源自 ref_ubench 每 bench 一个 Makefile。H800 专用（sm_90a）。

NVCC      ?= nvcc
ARCH      ?= sm_90a
CUDA_HOME ?= /usr/local/cuda

# FlashMLA 头文件（复用指令级封装：cvt_fp8x8_bf16x8 / GMMA::MMA_* / st_async_128b）
FLASHMLA  ?= ../../../target_op/FlashMLA
INCLUDES  := -I../../common \
             -I$(FLASHMLA)/csrc \
             -I$(FLASHMLA)/csrc/kerutils/include
# TODO(impl): 补 CUTLASS/CuTe include（FlashMLA submodule）

NVCCFLAGS := -O3 -arch=$(ARCH) --expt-relaxed-constexpr --use_fast_math -lineinfo $(INCLUDES)
LDFLAGS   := -lcuda -lnvidia-ml    # -lcuda: TMA driver API; -lnvidia-ml: getGPUClock()

# 每个 atom Makefile 定义 TARGET := aX_name，然后 include 本文件。
$(TARGET): $(TARGET).cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) | tee log

clean:
	rm -f $(TARGET) log

.PHONY: run clean
