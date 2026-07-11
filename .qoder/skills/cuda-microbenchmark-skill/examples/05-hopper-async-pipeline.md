# 示例 5：Hopper 异步操作的完成周期与吞吐

本示例用伪代码说明TMA/WGMMA两种不同测量，不提供可脱离descriptor/layout直接复制的完整kernel。

## A. 串行完成周期

```cpp
CLK_START(start);
for (int i = 0; i < repeat; ++i) {
    issue_async(stage0, barrier0);
    arrive_expect_tx(barrier0, bytes);
    wait_completion(barrier0);
    consume_or_depend_on_result(stage0);
}
CLK_STOP(stop);
```

特点：

- `depth=1`。
- 下一轮复用同一stage/barrier/result，必须等上一轮完成。
- TMA结果在wait后从shared消费；WGMMA复用同一accumulator并commit/wait。
- 输出应叫 `full_completion_cycle/tile` 或 `full_protocol_cycle/group`。

## B. pipeline throughput

```cpp
CLK_START(start);
for (int base = 0; base < repeat; base += depth) {
    for (int stage = 0; stage < depth; ++stage) {
        issue_async(independent_stage[stage], independent_barrier[stage]);
    }
    for (int stage = 0; stage < depth; ++stage) {
        wait_completion(independent_barrier[stage]);
        consume(independent_stage[stage]);
    }
}
CLK_STOP(stop);
```

特点：

- 每个stage有独立shared区域、barrier/phase和目标地址。
- 扫描 `depth=1..合法上限`、resident CTA/WG/cluster。
- TMA store避免outstanding WAW alias；WGMMA遵守最大outstanding group和accumulator生命周期。
- 输出 `transaction/clk/SM`、`byte/clk/SM` 或CUDA Event GB/s/TFLOPS。

## 为什么两者不能混用

若连续issue `N` 次后只wait一次：

```text
elapsed / N = 聚合摊销完成周期
```

它可能很好地表示throughput，但不能证明“一条异步指令从issue到完成”的latency。

## 参考

- `operators/_references/ubench/NVIDIA-Hopper-Benchmark/NewFeatures/TMA/Latency/`
- `operators/_references/ubench/NVIDIA-Hopper-Benchmark/NewFeatures/TMA/Throughput/`
- `operators/_references/ubench/NVIDIA-Hopper-Benchmark/TensorCores/wgmma/latency/`
- `operators/_references/ubench/NVIDIA-Hopper-Benchmark/TensorCores/wgmma/throughput/`
- 本仓库 `microbench/memory/tma_load/benchmark.cu` 与 `microbench/compute/wgmma/benchmark.cu`
