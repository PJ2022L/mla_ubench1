# online_m64n64_exp2_shfl_sm90

- **Target**：`row max + shfl reduction + exp2 + BF16 score conversion + row sum + full O-fragment rescale` 紧耦合链。
- **Geometry**：`[64,64]` score tile 使用真实 128-thread fragment-to-row mapping；每 thread 保留 2 行 score 和 `[2,64]` FP32 O fragment。
- **Modes**：`--mode local` 测单 WG online step；`--mode dense-pair` 用 256 threads 测 WG0 publish max、WG1 merge、WG0 final score/O/L rescale，并保留 source-like NamedBarrier handoff。
- **Timed body**：完整 vector/shared-state 链；不含 QK/PV WGMMA。O fragment 和转换结果通过计时区外 sink 保持可观察，避免编译器删掉大部分 rescale。
- **Metrics**：cycle/iteration、cycle/page、exp2 element/clk/SM。
- **Consumers**：sparse decode 使用 local-state；dense decode 与 sparse prefill 使用 shared-state pair 数据流。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 `M=64,N=64`。

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写对应 variant；失败或 parse-error run 不得登记。
