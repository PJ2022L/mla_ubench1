# SM90 micro-benchmark 设计要点

本文总结 `NVIDIA-Hopper-Benchmark` 与 `gpgpu-sim` 中可复用的测量方法，并给出本仓库 `microbench/` 的最低正确性要求。参考代码是方法来源，不是可直接照搬的实现；最终只在远端 H800/SM90 上运行。

## 先区分两个问题

- **延迟**：构造严格的循环依赖，下一次操作必须等上一次结果或完成信号。单 CTA/warp/warpgroup 用 `clock64`，同时测空循环基线。
- **吞吐**：构造多条独立链，扫描 issue depth、resident CTA/warpgroup/cluster，直到资源饱和。整网格用同一 stream 上的 CUDA event 计时。

不能用“很多条操作后只 wait 一次”的结果冒充单条完成延迟，也不能用一条依赖链的速度冒充峰值吞吐。

## 可以直接复用的经验

1. **指针追逐制造真依赖**：shared/L1/L2/DRAM、DSM 和 TMA latency 样例都用前一次 load 的结果生成下一次地址。这是隔离 load latency 最可靠的基本结构。
2. **多独立链隐藏延迟**：shared/L1 bandwidth、MaxFlops 和 DSM throughput 通过大量线程或多个寄存器 accumulator 提高 ILP/TLP。吞吐测试也应显式扫描并行度，而不是只测一个固定 launch。
3. **预热和 cache 状态在计时区外定义**：L1/L2 bandwidth 样例先填充 cache；DRAM 测试使用大 working set。每个结果必须标为 cold、L2-hot 或循环工作集，不能混报。
4. **设备内周期与整网格时间分开**：短依赖链用设备计数器；多 SM 饱和吞吐用 CUDA event。两类数据互相校验，但不混用单位。
5. **结果必须可观察**：寄存器结果写入 global sink 并拷回检查；inline PTX 使用显式输入/输出。源码循环存在不代表目标指令一定存在。
6. **Hopper 特有同步必须保留**：TMA 的 tensor map、async-proxy fence、mbarrier，DSM 的 cluster launch/map/sync，以及 WGMMA 的 fence/commit/wait 都属于操作语义，不是可随意删掉的噪声。
7. **最终以 PTX/SASS 为证**：保存反汇编，确认目标 opcode、operand mode、vector width、cache hint 和动态指令数，再计算 cycle/inst、byte/clk 或 FLOP/clk。

## 不能照搬的旧做法

- `gpgpu-sim/GPU_Microbenchmark` 大量代码来自 Volta；固定的 L1/L2 容量、block 数、resident thread 数和 `clock_freq_MHZ` 在 H800 上无效。必须运行时查询设备，并用实际 working set 验证 cache 层级。
- Volta/Turing 的 `wmma::mma_sync` 是 32-thread warp 指令；SM90 WGMMA 是 128-thread warpgroup 异步协议。fragment 布局、依赖、fence、commit/wait 和 `sm_90a` 编译目标都不能继承旧代码。
- `clock()`/`%clock` 是 32 bit，长循环会回绕；不同 SM 的 `%clock64` 也不应用 min/max 拼成整网格时间。短单 CTA 测量用 `%clock64`，整网格用 event。
- 不要用 `cudaDeviceProp::clockRate` 或一次 NVML 采样把 host 秒数硬换成 cycles；DVFS 下误差不可控。cycles 来自设备计数器，GB/s 来自 event 时间，并记录实测时钟策略。
- `random_latency` 的“一次预热、7 次取最小值”会系统性偏低；其 `MeasurementSeries::median()` 偶数样本索引也有错误。不要复用该统计实现。
- Hopper `RegularUnits/shared_lat` 在并行初始化 shared 后缺少 barrier；这个结构有数据竞争。初始化、预热与计时之间必须显式同步。
- `RegularUnits/mem_lat` 的注释、cache modifier 和 working set 并不一致；例如 timed load 使用 `ld.global.cg`，旧的 8 MiB 数组也小于 H800 L2。cache 命中层级必须由地址规模、hint 和 profiler 一起确认。
- Hopper TMA bandwidth 样例每 tile 立即 wait，适合“串行完成速率”，不能代表多 stage TMA 峰值吞吐；吞吐模式必须使用独立 smem stage/mbarrier 并扫描 pipeline depth。
- Hopper WGMMA latency 样例在很长 issue 循环后只 commit/wait 一次，不能单独证明 completion latency。依赖延迟模式应让每个被测 group 的下一次迭代依赖其完成。
- Hopper DSM `Throughput/Pair` 的核心是 `dst_smem[temp]`，测的是 peer DSM **load**。本仓库目标是 `st.async.weak.shared::cluster` **store**；只能复用 cluster 配置、地址映射和生命周期管理，不能复用指令或流量公式。
- 不要依赖永远不会执行的分支、未读取的 dummy buffer 或单纯的 C++ `volatile` 防优化。它们可能改变指令，也可能仍被消除。

## 所有 benchmark 的公共契约

### 环境与构建

- 运行时拒绝非 compute capability 9.0；WGMMA、TMA、DSM 相关目标使用 `sm_90a`，不是仅 `sm_90`。
- 记录 GPU 型号、CUDA/driver/nvcc 版本、SM 数、时钟/功耗策略和 ECC 状态。所有 CUDA API、kernel launch 与同步都检查错误。
- 本地只做代码生成、解析和静态检查；性能数字只能来自远端 H800。

### 计时与统计

- descriptor 创建、malloc/memcpy、输入生成、首次 page fault、普通初始化都在计时区外；只保留操作语义必需的 fence/wait/barrier。
- 至少 5 次不计样本的预热、20 个正式样本；报告 median、p10、p90 和样本数，原始样本落盘；公共 API 额外提供 p05/p95。`min` 只作为诊断值。
- 自动放大 repeat，使 device-cycle 区间至少约 `10^4` cycles，event 区间至少约 1 ms；使用 64-bit 计数器。
- 延迟结果同时给出 `raw` 与 `baseline-subtracted`。baseline 保留相同循环、地址计算、同步和 sink，仅移除目标指令或换成等价无操作。
- 测量前后记录实际 SM clock；出现热降频或时钟档变化时重测，不用名义时钟修补结果。

### 防优化、校验与计数

- 输入来自运行时数据；目标 inline PTX 使用 `asm volatile`、正确的约束和必要的 `memory` clobber。最终 accumulator/checksum 写到 global memory 并在 host 校验。
- 至少检查一个小规模数值结果；异步 store/load 还要检查目标内存确实更新，不能只看 elapsed time。
- 保存 cubin/SASS，核对每次循环实际执行的目标指令数。逻辑 tile 可能拆成多条 TMA/STMatrix/WGMMA 指令，分母必须按 SASS 动态数计算。
- 指标写清作用域：`/warpgroup`、`/SM` 或 `/cluster`。byte 指标同时区分 requested bytes、unique/useful bytes；不要凭理论 transaction 大小虚增流量。
- 同一个原子只计一次。若另有 `stmatrix` benchmark，默认 `tma_store` 不应再次把 register-to-shared staging 算进 TMA 原子；组合 epilogue 可以另报，但必须明确标为 composite。

## 本仓库各 family 的最小正确测量契约

| Family | 最小正确契约 |
|---|---|
| `wgmma` | RS/SS 分开；一个完整 128-thread warpgroup；operand 与 descriptor 在计时前准备。latency 使用 accumulator/完成依赖并按 group commit+wait；throughput 使用多套独立 accumulator、合法的 outstanding group 深度并扫描 resident warpgroup。结果 sink，SASS 核对 shape/dtype/mode 和指令数。 |
| `convert` | 复现 FlashMLA 的完整 FP8x8 + scale -> BF16x8 指令序列；输入预装寄存器，不计 global/shared 访存。latency 用输出反馈形成依赖，throughput 用独立 register tuple；有限非零输入覆盖正常值/饱和值，校验 8 个输出并保存 SASS。 |
| `softmax` | 使用真实 128-thread fragment-to-row mapping，完整执行约定的 max、shuffle、exp2、sum、rescale；local-state 与 cross-WG shared-state 分开。每次迭代从受控 logits/state 开始，状态重置成本应单列或用 baseline 扣除；检查 row max/sum/output 为有限值并与 CPU 对拍。 |
| `global_load` | SASS 必须是目标 128-bit `ld.global.nc` 及指定 L2 hint。分别跑 direct-address control 与真实 indexed gather，明确 sequential/local/random、working-set 大小和 cold/L2-hot 状态。latency 用 load 结果决定下一地址；throughput 用多独立 stream/warp，并报告 16 B/request 与 unique bytes。 |
| `cp_async_g2s` | SASS 必须是目标 16-byte `cp.async.cg.shared.global` 和指定 L2 hint；一个 128-thread producer warpgroup，真实 index mapping 与 `4/5/5/4` segment schedule 分开报告。shared 目标与 completion group/mbarrier 必须有效，wait 后消费并校验完整 tile。latency 串行完成，throughput 扫合法 outstanding group；73,728 B/block 按实际 4,608 copies 计数。 |
| `shared_store` | 值与地址在计时前准备，显式扫描无冲突、固定 bank conflict、真实 swizzle。纯 store 没有返回依赖，因此默认结果是 issue throughput；若报 completion latency，必须用 store -> 必需 fence/sync -> load/ack 的闭环，并扣除 load/sync baseline。目标 shared 内容必须被读取校验。 |
| `dsm_store` | cluster 固定 `(2,1,1)`，peer mapping 在计时前完成；cluster 首次使用前同步，退出前保证所有 peer 访问结束。peer CTA 的 mbarrier/consumer 必须等待并校验 16 B payload。分别报告 async-store issue throughput 与带完成确认的 latency，并同时跑 local-store control；指标作用域为 cluster。 |
| `tma_load` | tensor map 在 host 侧预建且编码成功；global/smem 地址、stride、box 和 swizzle 满足约束。一个 elected thread issue，mbarrier 的 init/expect/fence 顺序正确，consumer wait 后读取并校验 shared。latency 每 tile 等完成；throughput 使用独立 stage/mbarrier 扫 depth 与 CTA 数。逻辑 `64x576` 若被拆分，按实际 TMA 指令与字节计数。 |
| `tma_store` | shared 输入在计时前填好并校验，执行 required async-proxy fence、TMA store commit/等待，完成后回读 global。2D/3D/4D/5D descriptor 分开；latency 每 tile 确认完成，throughput 扫 outstanding group 与 CTA 数。默认 TMA-only；`STMatrix + TMA` 只能作为另行命名的 epilogue composite。 |
| `bulk_store` | shared FP32 source 在计时前按真实 stride-520 row layout 填好；64 个 active row threads 各 issue 一条 2,048-byte `cp.async.bulk.global.shared::cta.bulk_group`，随后执行 commit 与所选 completion wait。回读并校验全部 64x512 output；staging 不计时，扫描 completion depth/CTA 数，并按 64 instructions、131,072 useful B/tile 报告。 |
| `stmatrix` | 使用真实 WGMMA accumulator fragment mapping 和完整 128-thread warpgroup，SASS 确认目标 `stmatrix` form/x4 数量。寄存器值预置，shared swizzle/conflict 模式固定，store 后同步并校验 tile。纯 instruction 与 `stmatrix + async-proxy fence` 两个边界分开报告。 |
| `splitkv_reduce` | 复现真实 partial O/LSE layout、`num_splits`、exp2 rescale 和 FP32/float4 accumulation；整行 512 维输出与 CPU reference 对拍。L2-resident 与 HBM working set 分开，吞吐用 event 和 CTA 扫描；字节数包含 O partial、LSE 和最终写回，不能只数主数组。 |

## 代表性参考路径

- WGMMA latency/throughput：`NVIDIA-Hopper-Benchmark/TensorCores/wgmma/latency/test_wgmma_fp16.cu`、`TensorCores/wgmma/throughput/test_wgmma_fp16.cu`
- WGMMA SM90A PTX wrapper：`NVIDIA-Hopper-Benchmark/TensorCores/wgmma/mma_sm90_gmma.hpp`
- TMA dependent latency：`NVIDIA-Hopper-Benchmark/NewFeatures/TMA/Latency/tma_lat_uniform/tma_lat.cu`
- TMA random latency/statistics：`NVIDIA-Hopper-Benchmark/NewFeatures/TMA/Latency/tma_lat_random/main.cu`、`MeasurementSeries.hpp`
- TMA tensor map/bandwidth：`NVIDIA-Hopper-Benchmark/NewFeatures/TMA/Throughput/tma_bw_1d/tma_bw_1d.cu`、`tma_bw_2d/tma_bw_2d.cu`
- DSM cluster latency/throughput：`NVIDIA-Hopper-Benchmark/NewFeatures/DSM/Latency/OneToAll/dsm_latency.cu`、`NewFeatures/DSM/Throughput/Pair/dsm_throughput.cu`
- shared/L1/L2/DRAM：`NVIDIA-Hopper-Benchmark/RegularUnits/shared_lat/shared_lat.cu`、`shared_bw/shared_bw.cu`、`l1_bw_32f/l1_bw_32f.cu`、`l2_bw_32f/l2_bw_32f.cu`、`mem_lat/mem_lat.cu`、`mem_bw/mem_bw.cu`
- 旧架构依赖链/独立链：`gpgpu-sim/GPU_Microbenchmark/shared_lat/shared_lat.cu`、`l1_bw_32f_unroll/l1_bw_32f_unroll.cu`、`Atomic_ubench/Atomic_add/Atomic_add_lat/atomic_add_lat.cu`、`Atomic_add_bw/atomic_add_bw.cu`
- 旧 WMMA 单指令计时与 SASS 分解思路：`gpgpu-sim/tensorcore-microbenchmarks/Turing/ClockProfiling/16x16x16HHF_RR/clk_16x16x16HHF_RR.cu`、`Turing/SassProfiling/`
