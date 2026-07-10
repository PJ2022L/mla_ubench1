#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include "common.h"

// Required compile-time configs (pass via: make CLUSTER_SIZE=8 THREADS_PER_CTA=1024 LOAD_T=float4 ACCESS_MODE=local)
#ifndef CLUSTER_SIZE
#error "CLUSTER_SIZE must be defined"
#endif
#ifndef THREADS_PER_CTA
#error "THREADS_PER_CTA must be defined"
#endif
#ifndef LOAD_T
#error "LOAD_T must be defined"
#endif
#ifndef ACCESS_MODE
#error "ACCESS_MODE must be defined (0=local, 1=bcast)"
#endif

#define N_ITERS 10000
#define SMEM_SIZE 232448  // Greater than 24 KiB to force 1 CTA per SM

constexpr uint32_t LOAD_SIZE = sizeof(LOAD_T);
constexpr auto ACCESS_MODE_ENUM = static_cast<DsmemAccessMode>(ACCESS_MODE);

template <size_t LOAD_SIZE, DsmemAccessMode mode>
__device__ __forceinline__ int32_t swizzle_idx(int32_t idx) {
    //if constexpr (mode == DsmemAccessMode::LOCAL) {
    if constexpr (true) {
      // 32 banks * 4 bytes / (load width)
      constexpr int32_t threads_per_cycle = (32*4)/ LOAD_SIZE;
      constexpr int32_t xor_mask = threads_per_cycle - 1;
      
      constexpr int32_t shift = (LOAD_SIZE == 4)  ? 0 :
                                (LOAD_SIZE == 8)  ? 1 :
                                (LOAD_SIZE == 16) ? 2 : 0;
      
      return idx ^ ((idx >> shift) & xor_mask);
    } else {
      return idx;
    }
}

template <typename T>
__device__ __forceinline__ int32_t swizzle_idx(int32_t idx) {
    return swizzle_idx<sizeof(T), ACCESS_MODE_ENUM>(idx);
}

__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void dsmemThroughputKernel(float *data) {
    extern __shared__ LOAD_T buffer[];
    const size_t N_buffer = (SMEM_SIZE - (4*blockDim.x*LOAD_SIZE)) / LOAD_SIZE;
    cg::cluster_group cluster = cg::this_cluster();

    for (int32_t i = threadIdx.x; i < N_buffer; i += blockDim.x) {
        buffer[swizzle_idx<LOAD_T>(i)] = reinterpret_cast<LOAD_T*>(data)[N_buffer * blockIdx.x + i];
    }
    __syncthreads();
    cluster.sync();

    int32_t remote_cluster_rank = get_cluster_rank<ACCESS_MODE_ENUM>(cluster);
    LOAD_T *remote_buffer = cluster.map_shared_rank(buffer, remote_cluster_rank);
    float acc = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f;
    float acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    if (ACCESS_MODE_ENUM==DsmemAccessMode::LOCAL || cluster.block_rank() != 0) {
    for (int32_t j = 0; j < N_ITERS; j++) {
        #pragma unroll
        for (int32_t i = threadIdx.x; i < N_buffer; i += blockDim.x*4) {
            float v0 = dsmem_load<LOAD_T, ACCESS_MODE_ENUM>(remote_buffer + swizzle_idx<LOAD_T>(i));
            //float v1 = dsmem_load<LOAD_T, ACCESS_MODE_ENUM>(remote_buffer + swizzle_idx<LOAD_T>(i + blockDim.x));
            //float v2 = dsmem_load<LOAD_T, ACCESS_MODE_ENUM>(remote_buffer + swizzle_idx<LOAD_T>(i + blockDim.x * 2));
            //float v3 = dsmem_load<LOAD_T, ACCESS_MODE_ENUM>(remote_buffer + swizzle_idx<LOAD_T>(i + blockDim.x * 3));
        
            acc0 += v0;
            //acc1 += v1;
            //acc2 += v2;
            //acc3 += v3;
        }
        //acc += acc0;// + acc1 + acc2 + acc3;
    }
    }
    cluster.sync();

    data[cg::this_grid().thread_rank()] = acc0;
}

void benchDsmemThroughput() {
    int32_t num_ctas = CLUSTER_SIZE;
    size_t data_size = num_ctas * SMEM_SIZE;
    size_t N = data_size / sizeof(float);
    float *data, *d_data;

    data = (float*) malloc(data_size);
    srand((uint32_t) CLUSTER_SIZE + THREADS_PER_CTA + LOAD_SIZE);
    for (size_t i = 0; i < N; i++) {
        data[i] = rand();
    }
    cudaMalloc(&d_data, data_size);
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = CLUSTER_SIZE;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config = {0};
    config.gridDim = num_ctas;
    config.blockDim = THREADS_PER_CTA;
    config.dynamicSmemBytes = (SMEM_SIZE / 128) * 128;
    config.attrs = attribute;
    config.numAttrs = 1;

    cudaFuncSetAttribute(dsmemThroughputKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
    CUDA_CHECK(cudaLaunchKernelEx(&config, dsmemThroughputKernel, d_data));
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_data);
    free(data);
}

int main(int argc, char **argv) {
    benchDsmemThroughput();
    return 0;
}
