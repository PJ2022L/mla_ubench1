# m64n256k16_bf16_rs_ss_sm90

- **Target**：exact `wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16`；A major=K、B major=MN，RS/SS 分开报告。
- **Geometry**：每个 warpgroup 128 threads；A 为 K-major SW128，B 为 `GMMA::Layout_MN_SW128_Atom`。`--resident-wg 1|2` 控制同 CTA resident WG 数。
- **Timed body**：latency 为每 instruction 完整 fence/commit/wait；throughput 为每 group issue `--instructions-per-group` 条 WGMMA，累计 `--issue-depth` 个 committed groups 后 wait0。global accumulator sink 不计时。
- **CLI**：`--operand rs|ss|both --measurement latency|throughput|both --issue-depth 1..8 --instructions-per-group 1..36 --resident-wg 1|2 --repeat N --warmup N --samples N`。
- **Metrics**：cycle/instruction/WG、instruction/clk/SM、flop/clk/SM，外加原始 cycle 分位数。
- **Consumers**：FlashMLA sparse decode local/remote PV（每支每 block 4 次）。
- **Batch mapping**：PV 可用 `--instructions-per-group 4` 匹配一次源码 helper 的 issue/commit 数；不包含 RS/SS 跨 WG 交错与 shared handoff。
- **Static evidence**：当前 `sm_90a` PTX 为 exact BF16 `m64n256k16`；SS 使用 descriptors，RS 使用 4 个 A registers + B descriptor，B 的 MN-major 在 SASS 显示为 `.tnspB`。PTX 保留 `commit_group/wait_group<0>`；SASS 为 `HGMMA.64x256x16.F32.BF16` 和最终 `WARPGROUP.DEPBAR.LE`。ptxas 为 RS 158 / SS 154 registers，0 spill。
- **Source/config**：共享 `../benchmark.cu`；本目录 Makefile 固定 `M=64,N=256,K=16`。

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 operand、measurement、issue depth 和 resident WG；失败或 parse-error run 不得登记。
