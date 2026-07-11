# m64n256_b16_x4_sm90

- **Target**：exact CUTLASS `SM90_U32x4_STSM_N`，保存真实 WGMMA C-fragment 映射的 `[64,256]` B16 output half。
- **Geometry**：128-thread warpgroup、4 warps、K-major SW128 destination；kernel body 有 16 条静态 STMatrix opcodes，每个 warp 都执行，因此每 tile 共 64 条动态 warp instructions。
- **Timed body**：register fragment → shared tile；`--fence=true|false` 控制每 tile 后的 `fence_view_async_shared` 是否计时。global sink 和 validation fence 不计时。
- **Validation**：计时后统一建立 async-proxy visibility，把完整 logical shared tile 写到 global，并逐元素核对 BF16 `1.0`。
- **CLI**：`--fence=true|false --repeat N --warmup N --samples N --device N`。
- **Metrics**：cycle/tile、cycle/warp-instruction、warp-instruction/clk/SM、byte/clk/SM 和原始 cycle 分位数。
- **Consumers**：dense/sparse decode output staging。
- **Static evidence**：当前 `sm_90a` SASS 的 fence-on/off 两个 kernel 都各有 16 条 `STSM.16.M88.4`。两者各有一条 `FENCE.VIEW.ASYNC.S`：fence-on 位于 timed loop，fence-off 位于计时后的 validation path。ptxas 使用 39 registers，0 spill。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 `M=64,N=256,x4`。

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 fence mode；失败或 parse-error run 不得登记。
