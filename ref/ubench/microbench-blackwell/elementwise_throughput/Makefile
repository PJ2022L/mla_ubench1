NVCCFLAGS := -std=c++17 -O3 -lineinfo -gencode arch=compute_100,code=sm_100

OP ?= exp
THREADS_PER_BLOCK ?= 256
CTAS_PER_SM ?= 4
INNER_REPEATS ?= 1

ifeq ($(OP),exp)
OP_KIND := 0
else ifeq ($(OP),tanh)
OP_KIND := 1
else ifeq ($(OP),rsqrt)
OP_KIND := 2
else ifeq ($(OP),add)
OP_KIND := 3
else ifeq ($(OP),multiply)
OP_KIND := 4
else ifeq ($(OP),relu)
OP_KIND := 5
else
$(error Unsupported OP=$(OP). Use exp, tanh, rsqrt, add, multiply, or relu)
endif

DEFINES := -DOP_KIND=$(OP_KIND) \
	-DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) \
	-DCTAS_PER_SM=$(CTAS_PER_SM) \
	-DINNER_REPEATS=$(INNER_REPEATS)

all: elementwise_tput.out elementwise_tput.ptx

elementwise_tput.out: elementwise_tput.cu
	nvcc $(NVCCFLAGS) $(DEFINES) -o $@ $<

elementwise_tput.ptx: elementwise_tput.cu
	nvcc $(NVCCFLAGS) $(DEFINES) -ptx -o $@ $<

bench: elementwise_tput.out
	sudo $$(which ncu) --clock-control none \
		--launch-skip 5 --launch-count 1 \
		--metrics dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__cycles_elapsed.avg,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_special_pred_on.sum \
		./elementwise_tput.out

clean:
	rm -f *.ptx *.out
