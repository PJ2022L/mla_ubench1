#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

constexpr int32_t NUM_SMS = 148;  // B200 has 148 SMs

#ifndef OP_KIND
#define OP_KIND 0
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#ifndef CTAS_PER_SM
#define CTAS_PER_SM 4
#endif

#ifndef INNER_REPEATS
#define INNER_REPEATS 1
#endif

constexpr int OP_EXP = 0;
constexpr int OP_TANH = 1;
constexpr int OP_RSQRT = 2;
constexpr int OP_ADD = 3;
constexpr int OP_MULTIPLY = 4;
constexpr int OP_RELU = 5;

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t status__ = (expr);                                       \
        if (status__ != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(status__));                           \
            return 1;                                                        \
        }                                                                    \
    } while (0)

__device__ __forceinline__ float apply_op(float value) {
    if constexpr (OP_KIND == OP_EXP) {
        return __expf(value);
    } else if constexpr (OP_KIND == OP_TANH) {
        return tanhf(value);
    } else if constexpr (OP_KIND == OP_RSQRT) {
        return rsqrtf(value + 1.0f);
    } else if constexpr (OP_KIND == OP_ADD) {
        return value + 1.0009765625f;
    } else if constexpr (OP_KIND == OP_MULTIPLY) {
        return value * 1.0009765625f;
    } else if constexpr (OP_KIND == OP_RELU) {
        return fmaxf(value, 0.0f);
    } else {
        return value;
    }
}

__device__ __forceinline__ float normalized_input(size_t index) {
    int32_t bucket = static_cast<int32_t>(index & 1023);
    return static_cast<float>(bucket - 512) * (1.0f / 1024.0f);
}

__global__ void init_kernel(float* input, size_t elements) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < elements; i += stride) {
        input[i] = normalized_input(i);
    }
}

__global__ void elementwise_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t elements) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < elements; i += stride) {
        float value = input[i];
#pragma unroll
        for (int repeat = 0; repeat < INNER_REPEATS; ++repeat) {
            value = apply_op(value);
        }
        output[i] = value;
    }
}

const char* op_name() {
    if constexpr (OP_KIND == OP_EXP) {
        return "exp";
    } else if constexpr (OP_KIND == OP_TANH) {
        return "tanh";
    } else if constexpr (OP_KIND == OP_RSQRT) {
        return "rsqrt";
    } else if constexpr (OP_KIND == OP_ADD) {
        return "add";
    } else if constexpr (OP_KIND == OP_MULTIPLY) {
        return "multiply";
    } else if constexpr (OP_KIND == OP_RELU) {
        return "relu";
    } else {
        return "unknown";
    }
}

size_t parse_size_arg(const char* text) {
    char* end = nullptr;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        fprintf(stderr, "Invalid integer argument: %s\n", text);
        std::exit(2);
    }
    return static_cast<size_t>(value);
}

int parse_int_arg(const char* text) {
    char* end = nullptr;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        fprintf(stderr, "Invalid integer argument: %s\n", text);
        std::exit(2);
    }
    return static_cast<int>(value);
}

int main(int argc, char** argv) {
    size_t elements = size_t{1} << 28;
    int warmup_iters = 5;
    int timed_iters = 20;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--elements") == 0 && i + 1 < argc) {
            elements = parse_size_arg(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_iters = parse_int_arg(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            timed_iters = parse_int_arg(argv[++i]);
        } else {
            fprintf(stderr, "Usage: %s [--elements N] [--warmup N] [--iters N]\n", argv[0]);
            return 2;
        }
    }

    if (elements == 0 || warmup_iters < 0 || timed_iters <= 0) {
        fprintf(stderr, "Invalid configuration: elements > 0, warmup >= 0, iters > 0 required\n");
        return 2;
    }

    float* input = nullptr;
    float* output = nullptr;
    size_t bytes = elements * sizeof(float);
    CUDA_CHECK(cudaMalloc(&input, bytes));
    CUDA_CHECK(cudaMalloc(&output, bytes));

    int blocks = NUM_SMS * CTAS_PER_SM;
    init_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(output, 0, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < warmup_iters; ++i) {
        elementwise_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, output, elements);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_iters; ++i) {
        elementwise_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, output, elements);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    double seconds = static_cast<double>(elapsed_ms) / 1000.0;
    double bytes_read = static_cast<double>(bytes) * timed_iters;
    double bytes_written = static_cast<double>(bytes) * timed_iters;
    double op_count = static_cast<double>(elements) * timed_iters * INNER_REPEATS;
    double effective_gbps = (bytes_read + bytes_written) / seconds / 1e9;
    double effective_gops = op_count / seconds / 1e9;

    printf("RESULT op=%s elements=%zu bytes_read=%.0f bytes_written=%.0f "
           "event_ms=%.6f effective_GBps=%.6f effective_GOps=%.6f\n",
           op_name(), elements, bytes_read, bytes_written, elapsed_ms,
           effective_gbps, effective_gops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    return 0;
}
