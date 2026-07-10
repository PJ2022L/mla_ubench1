// st.async DSM 128-bit cluster2 micro-benchmark. SM90 scaffold.
// 纯操作：st.async.weak.shared::cluster（st_async_128b）把 bf16 分发到 peer CTA smem。
// **不保留** 原 producer 的 gather/dequant，只测 cluster 内 CTA↔CTA 的 DSM 分发。
// 需 cluster_size=2（__cluster_dims__ / cudaLaunchKernelEx cluster）。
// 范式：ref_ubench NewFeatures/DSM/Throughput（Pair）—— 测 DSM 分发带宽/延迟。
//
// 真实指令来源：splitkv_mla.cuh WG2 的 st_async_128b(sK_peer_base, ..., peer_bar)。
// Use it to price peer-CTA distribution independently from load/convert/local store.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "attention_shapes.h"

using namespace microbench;

#define THREADS_PER_BLOCK 128
#define CLUSTER_SIZE 2
#define REPEAT 512

// TODO(impl): __cluster_dims__(CLUSTER_SIZE,1,1) 或 launch 时设 cluster。
__global__ void a3_dsm_crossover(uint32_t* startClk, uint32_t* stopClk, uint32_t* dsink) {
    __shared__ uint16_t local_bf16[shape::TOPK_BLOCK/2 * shape::D_NOPE];
    // TODO(impl): 取 peer CTA 的 smem 地址（cluster.map_shared_rank / mapa.shared::cluster）。
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0, acc = 0;

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl):
        //   st.async.weak.shared::cluster.mbarrier::complete_tx::bytes [peer_smem], {data}, [peer_bar];
        //   （即 st_async_128b）纯 DSM 分发；用 cluster barrier 收尾。
        //   acc ^= local_bf16[...];   // 防优化
    }
    CLK_STOP(stop);
    startClk[uid] = start; stopClk[uid] = stop; dsink[uid] = acc;
}

int main() {
    // TODO(impl): cudaLaunchKernelEx 设 cluster=2；cycles → DSM byte/clk。
    printf("[TODO] dsm_store/128b_cluster2_sm90 scaffold\n");
    return 0;
}
