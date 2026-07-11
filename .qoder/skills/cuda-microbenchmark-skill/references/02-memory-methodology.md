# 访存类 micro-benchmark

访存必须分开回答“单次请求多久返回”和“系统每秒能搬多少数据”。两者的 kernel 结构相反。

## 1. 延迟：构造地址真依赖

使用 pointer chase：

```text
index_0 = seed
index_1 = load(array[index_0])
index_2 = load(array[index_1])
...
```

下一次地址依赖上一次 load 结果，硬件无法靠 memory-level parallelism 隐藏 latency。报告：

```text
cycle_per_load = (cycles - baseline) / dependent_load_count
```

要求：

- 单线程或受控 warp；不要让其他独立请求隐藏延迟。
- 用 `#pragma unroll 1` 和必要的 inline PTX 固定动态 load 数与 cache modifier。
- 最终 index 写入 global sink。
- 先检查 SASS，确认没有被 constant-fold、预取或改成另一种 load。

## 2. cache 层级控制

每个结果必须标注 cache 状态：

| 目标 | 工作集和准备 |
|---|---|
| shared latency | shared pointer chase；初始化后 `__syncthreads()`；分别测无冲突/固定 bank conflict |
| L1-hot | working set 小于有效 L1；预热在计时外；核对 `.ca`/目标 cache hint |
| L2-hot | working set 大于 L1、小于 L2；预热 L2；核对 `.cg` 等 modifier |
| HBM/DRAM | working set 明显超过实际 L2；随机/大 stride；必要时使用 cache eviction control 和 profiler 验证 |
| TLB/page walk | 单独扫描 page 数、page stride；不要混入普通 DRAM latency |

不要照搬参考代码中 Volta/A100 的固定 L1/L2 容量、block 数或 SM 数。运行时查询设备，并用 profiler counter 验证命中层级。

## 3. 吞吐：制造独立请求并饱和资源

使用多线程、多 warp、多 CTA、vector load/store 和多独立 accumulator。扫描：

```text
vector width
independent streams per thread
threads/CTA
resident CTA/SM
grid blocks / cluster count
working set
access pattern
```

整卡带宽使用 CUDA Event：

```text
GB/s = total useful or requested bytes / event seconds / 1e9
```

单 SM 诊断可报告：

```text
byte_per_clk_sm = bytes completed by one SM-scoped workload / CTA envelope cycles
```

二者不能混用。

## 4. 字节分母

至少区分：

- requested bytes：源码 load/store 请求的字节总数。
- unique/useful bytes：算法真正不同的数据量，去除线程间重复读取。
- cache-line/transaction bytes：实际 fabric 流量，只能由已验证的 transaction 规则或 profiler counter给出。
- read + write bytes：带 copy/reduction 时分别计数，不能只数主输入。

coalescing、重放、cache line overfetch 和 ECC 会让 requested bytes 与物理 HBM bytes 不同。

## 5. load、store 与 atomic 的不同闭环

### Load

读取结果天然提供完成信号，但必须让结果参与地址依赖、算术或 sink。纯 `volatile` 指针不足以证明目标 SASS 保留。

### Store

store 没有返回值。纯 store loop通常只能解释为 issue throughput。若测 completion latency，构造：

```text
store -> required fence/sync -> load/ack -> dependency/sink
```

并用只保留 load/fence 的 baseline 扣除闭环成本。

### Atomic

- latency：同一地址或返回值依赖链，明确 contention 模式。
- throughput：独立地址、固定热点地址和分片热点分别测；返回值是否使用会改变语义和指令。

## 6. TMA、cp.async 与 DSM

### 串行完成周期

`depth=1`：issue 一个 tile/group，立即执行合法 completion wait，再进入下一轮。它测 tile/group 的完整协议周期，不一定是单条 opcode latency。

### pipeline throughput

为每个 outstanding stage 准备独立 shared 区域和 mbarrier：

```text
for burst:
    issue stage 0..depth-1
    wait  stage 0..depth-1
```

扫描 depth、resident CTA/cluster 和 working set。禁止多个 outstanding store 写同一目标地址造成 WAW alias。

Hopper 特有要求：

- TMA tensor map、box/stride/swizzle/alignment 必须合法。
- async-proxy fence、mbarrier phase、expect bytes 和 consumer wait 必须保留。
- DSM 要固定 cluster launch、peer mapping、cluster sync 和退出前生命周期。
- logical tile 可能拆成多条 TMA，分母按 SASS/源码动态 transaction 数计算。

## 7. 访存验收清单

- cache 状态和 working-set bytes 已写入输出。
- latency 使用真 pointer chase，throughput 使用独立请求。
- requested/unique/physical bytes 没有混淆。
- load/store 结果可观察，异步结果被消费并校验。
- grid 扫描到平台而不是只跑一个固定 launch。
- SASS 中 load/store width、address space、cache hint 和动态数量正确。
