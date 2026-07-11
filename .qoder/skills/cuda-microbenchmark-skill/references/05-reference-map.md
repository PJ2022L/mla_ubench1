# `operators/_references/ubench` 参考代码路由

参考代码用于学习测量结构和CUDA机制，不直接复制其固定常量或结论。

## 访存延迟

- shared pointer chase：`gpgpu-sim/GPU_Microbenchmark/shared_lat/shared_lat.cu`
- L1/L2/DRAM：`NVIDIA-Hopper-Benchmark/RegularUnits/l1_lat/`、`l2_lat/`、`mem_lat/`
- random latency和统计：`NVIDIA-Hopper-Benchmark/RegularUnits/random_latency/`
- TMA dependent completion：`NVIDIA-Hopper-Benchmark/NewFeatures/TMA/Latency/tma_lat_uniform/tma_lat.cu`
- TMA random latency：`NVIDIA-Hopper-Benchmark/NewFeatures/TMA/Latency/tma_lat_random/`
- DSM latency：`NVIDIA-Hopper-Benchmark/NewFeatures/DSM/Latency/`

复用要点：pointer chase、每次完成等待、cache/working-set控制。需要修正：旧代码的32位clock、固定cache容量、缺失barrier和注释/modifier不一致。

## 访存吞吐

- shared/L1/L2：`NVIDIA-Hopper-Benchmark/RegularUnits/shared_bw/`、`l1_bw_32f/`、`l2_bw_32f/`
- HBM：`NVIDIA-Hopper-Benchmark/RegularUnits/mem_bw/`
- 多独立load链：`gpgpu-sim/GPU_Microbenchmark/l1_bw_32f_unroll/`
- TMA 1D/2D/3D：`NVIDIA-Hopper-Benchmark/NewFeatures/TMA/Throughput/tma_bw_1d/`、`tma_bw_2d/`、`tma_bw_3d/`
- DSM throughput：`NVIDIA-Hopper-Benchmark/NewFeatures/DSM/Throughput/`

复用要点：多CTA/warp、vector access、Event计时、并行度扫描。需要修正：按实际设备查询SM/cache，区分 requested与物理流量，并验证sink。

## 标量计算延迟与吞吐

- ALU dependent chain：`NVIDIA-Hopper-Benchmark/RegularUnits/alu_lat_float/`及其他dtype目录
- MaxFlops：`NVIDIA-Hopper-Benchmark/RegularUnits/MaxFlops/`
- 旧架构独立 accumulator：`gpgpu-sim/GPU_Microbenchmark/MaxFlops/`

复用要点：单accumulator测latency，多accumulator+足量grid测throughput。最终以SASS确认编译器没有改变指令或引入额外依赖。

## Tensor Core/WGMMA

- SM90 wrapper：`NVIDIA-Hopper-Benchmark/TensorCores/wgmma/mma_sm90_gmma.hpp`
- WGMMA latency结构：`NVIDIA-Hopper-Benchmark/TensorCores/wgmma/latency/test_wgmma_fp16.cu`
- WGMMA throughput结构：`NVIDIA-Hopper-Benchmark/TensorCores/wgmma/throughput/test_wgmma_fp16.cu`
- 旧WMMA clock/SASS拆解：`gpgpu-sim/tensorcore-microbenchmarks/Turing/ClockProfiling/`、`SassProfiling/`

注意：参考WGMMA latency代码在长issue loop后只commit/wait一次，不能单独证明每条completion latency。正确的latency模式应每个group建立完成依赖；throughput模式才连续保留outstanding work。

## 本仓库改进实现

优先参考当前仓库经过静态验证的公共设施和family：

- `microbench/common/clock.cuh`：64位clock64、CTA barrier和cycle归约。
- `microbench/common/measure.hpp`、`benchmark_utils.hpp`：采样和分位数。
- `microbench/compute/wgmma/benchmark.cu`：latency/throughput分模式。
- `microbench/memory/tma_load/benchmark.cu`：独立stage/mbarrier和depth扫描。
- `microbench/memory/global_load/benchmark.cu`：source-shaped聚合load；它不是pointer-chase latency模板。
- `operators/_references/ubench/microbenchmark-design-notes.md`：本仓库已识别的参考实现陷阱。
