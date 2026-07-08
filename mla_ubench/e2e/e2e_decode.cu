// e2e_decode — 端到端基线（用于标定组合模型）。SCAFFOLD。
// 不切割，调 FlashMLA 真实 launcher，wall-clock(cudaEvent) 计时；给出 T_measured 与 ~410 TFLOPS 对齐。
//
// 真实入口：sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<V32,128>(params)
//   （内部含 combine，PDL 重叠）。
// 组合模型标定：η = T_model / T_measured，见 ../model/compose.py。

#include <cstdio>
#include <cuda_runtime.h>
#include "gpu_check.h"
#include "mla_shapes.h"
// #include "sm90/decode/sparse_fp8/splitkv_mla.h"   // TODO: FlashMLA launcher
// #include "params.h"                                // SparseAttnDecodeParams

using namespace mla_ubench;

int main() {
    // TODO(impl):
    //   1) 构造 SparseAttnDecodeParams（gen_q_bf16 / gen_kv_fp8 / gen_indices + sched meta）。
    //   2) warmup + cudaEvent 计时 run_flash_splitkv_mla_fp8_sparse_kernel<V32,128>(params)。
    //   3) FLOPS = s_q*batch*topk*h_q*(576+512)*2；bytes = batch*topk*656；打印 TFLOPS/GB·s + T_measured(ms)。
    //   4) 落 log 供 compose.py 读取。
    printf("[TODO] e2e_decode scaffold — 复现 ~410 TFLOPS on H800\n");
    return 0;
}
