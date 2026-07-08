#pragma once
// TMA tensormap 封装（A1 kv_gather / A7 tma_store 用）。SCAFFOLD。
// 源自 ref/ubench/NVIDIA-Hopper-Benchmark/NewFeatures/TMA/util.h（cuTensorMapEncodeTiled 封装）。
//
// 核心思想：TMA 需要一个 CUtensorMap 描述符（rank/size/stride/box_size/swizzle）。
//   host 侧用 driver API cuTensorMapEncodeTiled 构造，__grid_constant__ 传进 kernel，
//   kernel 内 cp.async.bulk.tensor.Nd.shared::cluster.global + mbarrier 完成异步拷贝。

#include <cuda.h>
#include <cstdint>

namespace mla_ubench {

// 拿 cuTensorMapEncodeTiled 函数指针（driver API）。源自 util.h。TODO(impl)。
// PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled();

// 为 2D gather 构造 tensormap（MLA KV：box=[TOPK_BLOCK, d_qk tile]，swizzle=SW128）。TODO(impl)。
// void create_kv_tensor_map(CUtensorMap& desc, void* kv_global, int rows, int cols, int box_r, int box_c);

// 设备侧 TMA load 内联（cp.async.bulk.tensor.2d + mbarrier.expect_tx）。
// 参考 tma_bw_2d.cu 的 asm 片段。放进 A1。

}  // namespace mla_ubench
