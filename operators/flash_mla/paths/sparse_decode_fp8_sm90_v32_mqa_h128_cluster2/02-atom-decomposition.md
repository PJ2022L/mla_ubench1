# 02 — Instruction-level atom decomposition

目标不是把源码每一行都变成 benchmark，而是提取**动态工作量大、可能决定关键路径、能跨 operator 复用**的指令或紧耦合指令簇。地址计算、一次性谓词、轻量 arrive 等暂不单测；若 profiler 显示占比显著再升级为原子。

旧 sparse benchmark registry 已移除；当前 `microbench/manifest.json` 只交付
dense decode。下表中的 legacy 名称保留为分析需求，不表示当前已有可运行 leaf。

## Counting unit

统一以 `cluster2` 下**一个 CTA 处理一个 64-token block**为建模粒度。WG2 有 128 threads，每个 thread 处理一个 token 的一个 128-bit dimension slice；32 个逻辑 token 由 4 个 dimension lanes/token 协作。

以下计数是源码级动态调用数，不等价于最终 SASS issue 数；microbench 必须复现相同的 128-thread geometry，PTX/SASS 后再确认编译器是否合并或改变指令。

## Memory atoms

| ID | Significant operation | Per block / request work | Reusable benchmark | Why keep it |
|---|---|---:|---|---|
| M0 | Q `SM90_TMA_LOAD` | request: `64×576×2 = 73,728 B` / CTA | `tile64x576_bf16_sm90` (legacy; not in current dense-only suite) | prologue 与 producer 重叠，TMA 管线独立 |
| M1 | indexed 128-bit global load | thread: `1 scale + 8 NoPE + 2 RoPE`; block: `128×11` source calls | `128b_nc_l2_sm90` (legacy; not in current dense-only suite) | sparse gather、L2 hint、地址分布可能主导 |
| M2 | 128-bit local shared store | thread: `16 NoPE + 2 RoPE`; block: `128×18` | `128b_sm90` (legacy; not in current dense-only suite) | convert 结果必须落入 WGMMA layout |
| M3 | 128-bit peer DSM async store | cluster2: 与 M2 同 geometry | `128b_cluster2_sm90` (legacy; not in current dense-only suite) | crossover 的额外代价 |
| M4a | non-split output STSM + 5D TMA store | request: `64×512×2 = 65,536 B` / CTA | `m64n256_b16_x4_sm90` (legacy; not in current dense-only suite) + `tile64x512_bf16_5d_sm90` (legacy; not in current dense-only suite) | 只用于 direct BF16 O path |
| M4b | split FP32 partial staging + bulk S2G | request: `64×512×4 = 131,072 B` / CTA | `tile64x512_f32_sm90` (legacy; not in current dense-only suite) | split path；不能用 M4a 的 TMA-store cycle 替代 |
| M5 | split-KV partial read/reduce | per output row 约 `num_splits×512×4 B` read + BF16 write | `dv512_f32_sm90` (legacy; not in current dense-only suite) | combine grid；必须按其独立 grid/waves 转换成 kernel latency |

M1 不把 index arithmetic 单独拆开；benchmark 用 sequential/local/random index 分布和工作集大小扫描 L2 命中率。它可服务使用同类 `ld.global.nc` register load 的 gather kernel，但不能替代 sparse prefill 的 `cp.async` GMEM→SMEM 路径。

## Compute atoms

| ID | Significant operation | Dynamic count | Reusable benchmark | Why keep it |
|---|---|---:|---|---|
| C0 | `cvt_fp8x8_bf16x8` | thread: 16; block: `128×16` source calls | `fp8x8_to_bf16x8_sm90` (legacy; not in current dense-only suite) | Hopper FP8→BF16 可能成为 producer 瓶颈 |
| C1 | QK WGMMA SS `m64n64k16` | `576/16 = 36` / block | `m64n64k16_bf16_rs_ss_sm90` (legacy; not in current dense-only suite) SS mode | WG0 Tensor Core 主工作 |
| C2 | online softmax chain | 1 × `[64,64]` / block | `online_m64n64_exp2_shfl_sm90` (legacy; not in current dense-only suite) | max/exp2/sum/rescale 有强依赖，继续拆会失真 |
| C3a | local PV WGMMA RS `m64n256k16` | `64/16 = 4` / block | `m64n256k16_bf16_rs_ss_sm90` (legacy; not in current dense-only suite) | WG0 输出左半 |
| C3b | shared-score PV WGMMA SS `m64n256k16` | `64/16 = 4` / block | 同上，SS mode | WG1 输出右半；A operand 来自 WG0 发布的 CTA shared S |

WG0/WG1 的两支 PV 虽在不同 warpgroup 发射，仍共享一个 SM 的 Tensor Core
执行资源。WGMMA benchmark 必须提供 `resident_wg=2` aggregate stage cycles；模型
只有在该数已包含争用时才可对两支取 `max`。单 WG `cycle/inst` 的两个结果直接
`max` 会系统性低估 PV 阶段。

## Explicit exclusions

- token/page 的整数除法和模：先留在 M1 的真实地址生成路径，不单建 ALU benchmark。
- `is_kv_valid`、tail mask、LSE 单次写：相对主循环工作量轻。
- NamedBarrier/mbarrier 的 arrive bookkeeping：同步等待时间由 e2e/model calibration 体现；只有出现显著 barrier throughput 问题时再单测。
- WG0/WG1 的少量 scale multiply：保留在 C2/C3 或 shared-score handoff correction 中。

## Measurement contract

每个原子同时报告原始周期和归一化指标；禁止只报 GB/s/TFLOPS 而丢失 cycle：

```text
latency       = cycles / repeat
instruction   = dynamic_ops × repeat / cycles
memory        = unique_bytes × repeat / cycles
tensor        = flops × repeat / cycles
```

计算类输入预置在 register/shared memory；访存类保留真实地址分布、cache hint 和布局。所有 benchmark 去掉 operator 的跨-WG barrier，但保留完成该指令所必需的 commit/wait 和防优化依赖。
