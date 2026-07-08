#pragma once
// MLA (DeepSeek-V3.2) 形状常量 + 合成数据生成。SCAFFOLD。
// 所有原子共用这些常量，保证「形状真实」（哪怕数据是随机的）。

#include <cstdint>

namespace mla_ubench {

// ---- V3.2 (MODEL_TYPE::V32) 形状 ----
namespace shape {
constexpr int D_QK       = 576;   // head_dim_k = 512 NoPE + 64 RoPE
constexpr int D_V        = 512;   // head_dim_v
constexpr int D_NOPE     = 512;
constexpr int D_ROPE     = 64;
constexpr int QUANT_TILE = 128;   // 每 128 fp8 共享 1 个 fp32 scale
constexpr int NUM_SCALES = D_NOPE / QUANT_TILE;   // = 4
constexpr int BYTES_PER_TOKEN = D_NOPE /*fp8*/ + NUM_SCALES * 4 /*fp32 scale*/ + D_ROPE * 2 /*bf16 rope*/;  // 656
constexpr int BLOCK_M    = 64;    // query heads per tile
constexpr int TOPK_BLOCK = 64;    // topk tokens per block
constexpr int PAGE_BLOCK = 64;
}  // namespace shape

// ---- 可调参数（原子从命令行/宏读）----
struct AtomParams {
    int topk   = 2048;
    int repeat = 1024;     // REPEAT_TIMES，稳态循环次数
    int num_heads = 128;   // 128→cluster2→有 DSM crossover；64→无
    // TODO(impl): cache_hint 等
};

// ---- 合成数据（host malloc + cudaMemcpy），形状真实、值随机 ----
// TODO(impl):
//   gen_q_bf16(...)         : [BLOCK_M, D_QK] bf16
//   gen_kv_fp8(...)         : [topk, 656B]  —— 512 fp8_e4m3 | 4 fp32 scale | 64 bf16 rope
//   gen_indices(...)        : [topk] int32，分布可控（sequential/random/local）
//   gen_logits_f32(...)     : [BLOCK_M, TOPK_BLOCK] f32（A6 softmax 用）
//   gen_o_accum(...)        : [BLOCK_M, num_splits, D_V] f32（A8 combine 用）

}  // namespace mla_ubench
