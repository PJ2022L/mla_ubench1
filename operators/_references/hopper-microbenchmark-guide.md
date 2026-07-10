# Hopper micro-benchmark 指导

本文件提炼 `NVIDIA-Hopper-Benchmark` 的测试思路，作为本仓库 SM90 benchmark 的最小设计检查表。外部代码用于参考，不等于已经满足本仓库的统计与真实性要求。

## 1. 目录和实验矩阵

采用 `大类 / 指令族 / 参数配置`：例如 `compute/wgmma/m64n64k16_bf16_rs_ss_sm90/`。同一实验只改变一个主要自变量：

- 计算：MNK、dtype、SS/RS、稀疏/稠密、issue depth、resident warpgroup 数；
- TMA/访存：tile/rank、访问模式、working set/cache hint、CTA 数；
- DSM：cluster topology、local/peer、block/thread 数、stride。

因变量保留原始 cycles/ms，并归一为 `cycle/op`、`op/clk/SM`、`byte/clk/SM`、`flop/clk/SM` 或 `GB/s`。配置、原始值与归一值必须同时记录。

## 2. 延时与吞吐不是同一个实验

**Latency**：构造真依赖链，使第 `i+1` 次操作必须等待第 `i` 次结果；用单 CTA/单 SM 降低调度干扰。扣除或摊薄固定开销后报告 `cycles / repeat`。

```cuda
asm volatile("bar.sync 0;");
asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
for (int i = 0; i < repeat; ++i) {
    state = target_op(state);           // 真依赖，结果最后写入 sink
}
complete_target_op();                   // 如 wgmma.wait_group / mbarrier wait
asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) :: "memory");
```

**Throughput**：使用多个独立 accumulator/地址流打断依赖链，并用足够 warp/CTA 覆盖流水线；报告总操作或总字节除以统一时间窗。

```cuda
for (int i = 0; i < repeat; ++i) {
    s0 = target_op(s0);  s1 = target_op(s1);
    s2 = target_op(s2);  s3 = target_op(s3);  // 相互独立
}
sink = combine(s0, s1, s2, s3);
```

单 SM 指令吞吐宜用 `min(start[all threads])` 到 `max(stop[all threads])`；全 GPU/HBM/TMA 带宽宜 warmup 后用 CUDA event 包围多个稳态 kernel。换算式必须写清：

```text
latency_cycle_per_op = (stop - start) / repeat
throughput_byte_per_clk_per_SM = useful_bytes / elapsed_cycles / active_SM
throughput_GB_per_s = useful_bytes / elapsed_seconds / 1e9
throughput_flop_per_clk_per_SM = useful_flops / elapsed_cycles / active_SM
```

## 3. Hopper 指令的必要边界

- WGMMA：operand 在计时前预置；RS/SS 分开；说明 fence、`commit_group`、`wait_group` 是否计入，结束时必须 wait 后再读时钟。
- TMA：tensor-map 创建、host allocation、输入生成不计入；issue、`mbarrier.expect_tx` 与完成等待形成完整 transaction。
- DSM：用 `cudaLaunchKernelEx` 固定 cluster dimension；local/peer 分开，cluster 初始化和 steady-state 访问分开。
- 普通 ALU/访存：延时使用 pointer/dependency chasing；吞吐使用独立状态和足够占用率。

## 4. 最小真实性检查

1. 先 warmup，再收集多次样本并报告 median 与离散程度；锁频/功耗策略随日志记录。
2. 目标结果写入不可预测的 global sink，防止编译器删除或合并。
3. 保存 PTX/SASS，确认 source intrinsic 实际生成目标指令、重复次数和 cache hint。
4. 计时区不能混入初始化；异步指令不能漏掉 completion wait。
5. 区分 requested bytes、transferred bytes 与 unique useful bytes，吞吐分母说明 active SM/cluster。
6. scaffold、SM89 结果或单次运行都不能登记为 SM90 `measured`。

## 参考实现锚点

- WGMMA latency/throughput：`TensorCores/wgmma/latency/test_wgmma_fp16.cu`、`TensorCores/wgmma/throughput/test_wgmma_fp16.cu`、`TensorCores/wgmma/mma_sm90_gmma.hpp`；
- TMA latency/throughput：`NewFeatures/TMA/Latency/tma_lat_uniform/tma_lat.cu`、`NewFeatures/TMA/Throughput/tma_bw_2d/tma_bw_2d.cu`；
- DSM latency/throughput：`NewFeatures/DSM/Latency/OneToAll/dsm_latency.cu`、`NewFeatures/DSM/Throughput/Pair/dsm_throughput.cu`；
- 通用吞吐归一化：`RegularUnits/MaxFlops/MaxFlops.cu`、`RegularUnits/shared_bw/shared_bw.cu`。

以上路径均相对 `operators/_references/ubench/NVIDIA-Hopper-Benchmark/`。
