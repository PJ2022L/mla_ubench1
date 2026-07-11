# SM90 static validation

本表记录 2026-07-11 使用 CUDA 13.1、`-arch=sm_90a` 生成的 PTX/cubin/SASS 证据。静态验收要求：目标指令存在，源码动态计数与 JSON 分母一致，PTX 使用 `%clock64`，SASS 使用 `CS2R SR_CLOCKLO`，并且 `cuobjdump --dump-resource-usage` 显示 `STACK=0, LOCAL=0`；全量 SASS 不得出现 `LDL/STL` spill 指令。

`validated` 只表示通过上述静态契约，不表示已有性能数据。本地没有执行任何 benchmark binary；延迟、吞吐、cache/occupancy 行为和正确性仍须在远端 H800 上运行后，才能把 registry 状态改为 `measured`。

| Configuration | PTX/SASS evidence | Dynamic work per timed iteration | Register / spill boundary |
|---|---|---|---|
| `wgmma/m64n64k16_bf16_rs_ss_sm90` | exact `HGMMA.64x64x16.F32.BF16`，RS/SS 和 `WARPGROUP.DEPBAR.LE` | latency：1 instruction/WG；throughput：`issue_depth × instructions_per_group`/WG；总量再乘 `repeat` 和 `resident_wg` | max 62 REG；STACK/LOCAL 0 |
| `wgmma/m64n256k16_bf16_rs_ss_sm90` | exact `HGMMA.64x256x16.F32.BF16`，RS/SS 和 `WARPGROUP.DEPBAR.LE` | 同上；PV source-like batch 使用 `instructions_per_group=4` | max 158 REG；STACK/LOCAL 0 |
| `convert/fp8x8_to_bf16x8_sm90` | `F2FP.F16.E4M3.UNPACK_B` + `F2FP.BF16.F32.PACK_AB` | 16 helper calls/thread，128 threads，即 2,048 calls/CTA；总量乘 `repeat` | max 78 REG；STACK/LOCAL 0 |
| `softmax/online_m64n64_exp2_shfl_sm90` | `MUFU.EX2` + `SHFL.BFLY`，local/dense-pair 两个 kernel | local：1 page/iteration、4,096 exp2 elements；dense-pair：2 pages、8,192 elements；总量乘 `repeat` | max 254 REG；STACK/LOCAL 0；高寄存器占用需 H800 验证 occupancy |
| `global_load/128b_nc_l2_sm90` | 128-bit `LDG.E...CONSTANT`，含 B128/B256 L2 hint | 11 loads/thread，128 threads，即 1,408 loads/CTA；总量乘 `repeat` | max 32 REG；STACK/LOCAL 0 |
| `cp_async_g2s/gather64x576_bf16_sm90` | `LDGSTS.E.BYPASS.LTC256B.128` | block：4,608 copies；pair `4/5/5/4` schedule：9,216 copies；总量乘 `repeat` | max 40 REG；STACK/LOCAL 0 |
| `shared_store/128b_sm90` | `STS.128` | 18 stores/thread，128 threads，即 2,304 stores/CTA；总量乘 `repeat` | max 55 REG；STACK/LOCAL 0 |
| `dsm_store/128b_cluster2_sm90` | exact PTX `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64` | 18 stores/thread × 128 threads × 2 CTAs = 4,608 stores/cluster；73,728 B/block；总量乘 `repeat` | max 32 REG；STACK/LOCAL 0 |
| `tma_load/tile64x64_bf16_sm90` | timed kernel 为 `UTMALDG.2D/3D/4D` 与对应 `cp.async.bulk.tensor.{2,3,4}d` | 1 TMA transaction/tile；总量 `repeat`，rank 不改变字节数 | timed rank 2/3/4 均为 22 REG；STACK/LOCAL 0 |
| `tma_load/tile64x576_bf16_sm90` | timed kernel 为 `UTMALDG.2D/3D/4D` 与对应 `cp.async.bulk.tensor.{2,3,4}d` | 9 TMA transactions/tile；总量 `9×repeat`；H800 depth 预计最多 3 | timed rank 2/3/4 均为 22 REG；STACK/LOCAL 0 |
| `tma_store/tile64x512_bf16_2d_sm90` | timed kernel 为 `UTMASTG.2D` / `cp.async.bulk.tensor.2d` | 8 `[64,64]` transactions/tile；总量 `8×repeat` | timed 20 REG；STACK/LOCAL 0 |
| `tma_store/tile64x64_bf16_3d_sm90` | timed kernel 为 `UTMASTG.3D` / `cp.async.bulk.tensor.3d` | 1 transaction/tile；sparse prefill CTA output 由 8 tiles 组成 | timed 12 REG；STACK/LOCAL 0 |
| `tma_store/tile64x512_bf16_4d_sm90` | timed kernel 为 `UTMASTG.4D` / `cp.async.bulk.tensor.4d` | 8 `[64,64]` transactions/tile；总量 `8×repeat` | timed 23 REG；STACK/LOCAL 0 |
| `tma_store/tile64x512_bf16_5d_sm90` | timed kernel 为 `UTMASTG.5D` / `cp.async.bulk.tensor.5d` | 1 transaction/tile；总量 `repeat` | timed 12 REG；STACK/LOCAL 0 |
| `bulk_store/tile64x512_f32_sm90` | exact PTX `cp.async.bulk.global.shared::cta.bulk_group` + commit/wait | 8 warp leaders × 8 rows = 64 bulk copies/tile，131,072 B；总量乘 `repeat` | max 32 REG；STACK/LOCAL 0 |
| `stmatrix/m64n64_b16_x4_sm90` | `STSM.16.M88.4` + `FENCE.VIEW.ASYNC.S` | 4 static instructions/warp，4 warps，即 16 dynamic warp instructions/tile；总量乘 `repeat` | max 22 REG；STACK/LOCAL 0 |
| `stmatrix/m64n256_b16_x4_sm90` | `STSM.16.M88.4` + `FENCE.VIEW.ASYNC.S` | 16 static instructions/warp，4 warps，即 64 dynamic warp instructions/tile；总量乘 `repeat` | max 39 REG；STACK/LOCAL 0 |
| `splitkv_reduce/dv512_f32_sm90` | vector global loads、`MUFU.EX2`、warp shuffle、FP32 FMA 与 BF16 output store | 每 CTA iteration：`num_splits×1,024` 个 `float4` partial loads，`num_splits×8×512` FMAs，输出 8×512 BF16 + 8 LSE | max 64 REG；STACK/LOCAL 0 |

TMA-load correctness 使用独立 validation kernel 和非均匀确定性输入，消费首/末 working-set tile 的全部元素，并用 per-transaction 权重 checksum 检查漏传、错 tile 和跨 transaction 错位；它不区分同一 transaction 内的排列，不能单独证明 SW128 物理顺序。TMA-store 同样在 timed region 外写首/末 tile：rank 2/4 按 transaction 加权，rank 3 覆盖首末 64-column slice，rank 5 校验完整 tile multiset，并要求未触达区域保持为零。rank 5 的整 tile checksum 不能发现 8 个 64-column subtiles 在 tile 内互换；rank 2/4/5 还会拒绝 `working_set_tiles < min(depth, repeat)`，避免 outstanding WAW alias 污染 pipeline 结果。

这些 validation 均不进入 timed region。静态产物仍只能确认代码形状，不能替代 H800 测量。远端每个配置仍需保存 GPU 名称/时钟策略、完整 CLI、warmup/sample、原始 JSON、PTX/SASS hash，并检查输出 sink/CPU validation 后再登记 `measured`。
