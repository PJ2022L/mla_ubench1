# 计时与 CUDA 特性

## 1. 先选择计时作用域

| 目标 | 推荐计时 | 原因 |
|---|---|---|
| 单线程/warp/WG/CTA 的短依赖链 | `%clock64` | 直接报告 SM cycle，开销低 |
| 一个 CTA/cluster 的完成周期 | `%clock64`，每个参与者记录时间 | 能观察线程偏斜和 completion wait |
| 多 CTA、多 SM 饱和吞吐 | CUDA Event | 同一 stream 上覆盖完整 grid，避免跨 SM 时钟拼接 |
| public API/e2e | CUDA Event 或 profiler trace | 必须覆盖正常路径上的全部 kernel |

`%globaltimer` 可作为设备时间诊断，但不要未经目标架构验证就假定它恒定为纳秒。整 grid 默认使用 CUDA Event。

## 2. `%clock64` 与 inline PTX

使用 64 位 per-SM 周期计数器：

```cpp
__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%clock64;"
                 : "=l"(value) :: "memory");
    return value;
}
```

- `asm`：嵌入 PTX。
- `volatile`：要求保留每次 asm，不因结果看似冗余而删除/合并。
- `"=l"`：写入 64 位寄存器 operand。
- `"memory"` clobber：约束编译器跨该点重排内存访问；它本身不是 GPU memory fence。
- `%%clock64`：`%` 在 asm 模板中需要转义。

对整个 CTA 对齐边界：

```cpp
asm volatile("bar.sync 0;" ::: "memory");
uint64_t start = read_clock64();
// target work
asm volatile("bar.sync 0;" ::: "memory");
uint64_t stop = read_clock64();
```

所有 CTA 线程必须一致执行 `bar.sync 0`，否则可能死锁。只测单线程依赖链时不必为了形式引入全 CTA barrier，但必须保证编译器和目标指令顺序正确。

## 3. CTA/WG 的 cycle 归约

让每个参与线程写出 `start[i]` 和 `stop[i]`。一个同步 CTA/WG 的保守 envelope 为：

```text
cycles = max(stop[i]) - min(start[i])
```

同时报告每线程 `stop-start` 的 min/max/mean 以暴露 skew。不同 SM 的 `%clock64` 不保证同步，不能对整 grid 使用这个 envelope。

## 4. CUDA Event

在同一 stream 中记录：

```cpp
cudaEventRecord(start, stream);
kernel<<<grid, block, smem, stream>>>(...);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
```

要求：

- 先 warmup，避免首次 module load、page fault 和 cache 初始化污染。
- 放大循环，使单次 event 区间至少约 1 ms，降低 event 分辨率和 launch overhead 占比。
- 多次独立采样，不只在一次 event 内重复到无法观察波动。
- e2e 必须在相同 stream 里覆盖 main、combine 等正常路径 kernel。

## 5. 异步指令的完成边界

只发射异步指令后读取时钟，测到的通常是 issue 成本，不是完成延迟。stop 前加入语义必需的完成机制：

| 机制 | 完成边界 |
|---|---|
| WGMMA | `wgmma.commit_group` + 合法 `wgmma.wait_group`，必要时 operand fence |
| TMA load | mbarrier `expect_tx/arrive` + phase wait，随后消费 shared 数据 |
| TMA/bulk store | commit/arrive + completion wait，随后回读 global 校验 |
| `cp.async` | commit group + wait group / mbarrier，随后消费 shared 数据 |
| DSM async store | cluster 生命周期、peer ack/mbarrier 或可证明的消费闭环 |

区分字段名：`issue_cycle`、`completion_cycle`、`full_protocol_cycle`，不要统一叫 latency。

## 6. baseline 与计时开销

延迟 benchmark 同时提供：

```text
raw_cycle_per_op = measured_cycles / operation_count
net_cycle_per_op = (measured_cycles - baseline_cycles) / operation_count
```

baseline 保留相同循环、索引计算、barrier、sink 和分支，仅移除目标指令或替换成等价寄存器操作。若无法构造等价 baseline，明确只报告 full-chain cycle。

## 7. DVFS 与时钟

`%clock64` 报告周期，不是绝对时间：

```text
time = sum(each cycle / instantaneous SM frequency)
```

计算 pipeline latency 的 cycle 数通常较稳定；HBM/L2 等其他时钟域的固定 ns 延迟会随 SM 频率变化而表现为不同 cycle 数。正式实验应：

1. 尽量锁定 SM/memory clock 和 power policy。
2. 测量前后记录实际 SM/memory clock、P-state、功耗和温度。
3. 同时保存 cycle 指标与 event 的 ns/GB/s/TFLOPS。
4. 发现热降频或时钟档变化时重测，不用名义频率修补结果。

## 8. 采样与统计

- 至少 5 次不计结果的 warmup、20 个正式 sample。
- 自动放大 repeat，使 device interval 至少约 `10^4` cycles。
- 报告 raw samples、median、p10、p90、sample count；`min` 仅用于诊断。
- 样本分布多峰时先检查 clock、cache、occupancy 和系统干扰，不只给 mean。
