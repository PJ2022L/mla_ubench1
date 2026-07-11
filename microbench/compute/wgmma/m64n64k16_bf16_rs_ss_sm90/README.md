# m64n64k16_bf16_rs_ss_sm90

- **Target**：exact `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`，A/B major 均为 K；SS/RS 分开报告。
- **Geometry**：每个 warpgroup 128 threads；A/B 使用 `GMMA::Layout_K_SW128_Atom`。`--resident-wg 1|2` 控制同 CTA resident WG 数。
- **Timed body**：latency 为每 instruction 完整 fence/commit/wait；throughput 为每 group issue `--instructions-per-group` 条 WGMMA，累计 `--issue-depth` 个 committed groups 后 wait0。global accumulator sink 不计时。
- **CLI**：`--operand rs|ss|both --measurement latency|throughput|both --issue-depth 1..8 --instructions-per-group 1..36 --resident-wg 1|2 --repeat N --warmup N --samples N`。
- **Metrics**：cycle/instruction/WG、instruction/clk/SM、flop/clk/SM，外加原始 cycle 分位数。
- **Consumers**：FlashMLA sparse decode QK（36 SS/page）；dense decode steady QK（32 SS+4 RS/page）。
- **Batch mapping**：dense 的 64-wide QK tile 是 4 instructions/group；sparse prefill 可用 16/20/36 检查大 group。该 benchmark 不复现 mixed group 与 TMA/barrier 交错。
- **Static evidence**：当前 `sm_90a` PTX 为 exact BF16 `m64n64k16`；SS 使用两个 descriptors，RS 使用 4 个 A registers + B descriptor。PTX 保留 `commit_group/wait_group<0>`；SASS 为 `HGMMA.64x64x16.F32.BF16` 和最终 `WARPGROUP.DEPBAR.LE`。depth=`D` 的 throughput kernel 有 `D` 个静态 HGMMA loops，每个 loop 动态执行 `instructions-per-group` 次。ptxas 为 RS 62 / SS 58 registers，0 spill。
- **Source/config**：共享 `../benchmark.cu`；本目录 Makefile 固定 `M=64,N=64,K=16`。

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 operand、measurement、issue depth 和 resident WG；失败或 parse-error run 不得登记。
