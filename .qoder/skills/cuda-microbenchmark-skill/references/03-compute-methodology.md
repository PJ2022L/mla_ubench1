# 计算类 micro-benchmark

计算类覆盖 ALU、FMA、整数、SFU、格式转换、Tensor Core/WGMMA 和组合数值阶段。核心仍是区分依赖延迟与流水吞吐。

## 1. 标量/向量指令延迟

构造单 accumulator 真依赖：

```text
x1 = op(x0, a)
x2 = op(x1, a)
...
```

使用 runtime 输入、inline PTX 或受控 intrinsic，避免 constant folding 和编译器重写。输出：

```text
cycle_per_instruction = (cycles - baseline) / SASS_dynamic_instruction_count
```

对 transcendental/SFU 指令，确认源代码函数实际生成目标 `MUFU`/转换序列；一个 `exp2f()` 可能包含范围处理或多条指令。

## 2. 计算吞吐

使用多套独立 accumulator：

```text
x0 = op(x0, a)
x1 = op(x1, a)
...
xN = op(xN, a)
```

独立链提供 ILP，多线程/warp/CTA 提供 TLP。扫描 accumulator 数、threads、resident CTA/WG 和 grid size，直到 event throughput 达到平台。

```text
instruction/s = total dynamic instructions / event seconds
FLOP/s = total semantic FLOPs / event seconds
FLOP/clk/SM = per-SM semantic FLOPs / SM-scoped cycles
```

FMA通常按 2 FLOPs 计，但必须说明计数约定；integer、compare、convert 不应伪装成 FLOPs。

## 3. WGMMA/Tensor Core

SM90 WGMMA 是 128-thread warpgroup 异步协议，不是旧的 32-thread WMMA。固定并验证：

- exact `M/N/K`、input/accumulator dtype；
- A/B major、SS/RS operand mode和 shared swizzle；
- 完整 128-thread WG；
- fence、commit group、wait group 和 accumulator sink；
- 每个源码 helper 实际生成的 HGMMA 条数。

### WGMMA completion latency

每轮只发一个目标 group，并让下一轮依赖它完成：

```text
issue group -> commit -> wait0 -> next iteration using same accumulator
```

报告 `full_protocol_cycle/group`；若不能隔离 fence/commit/wait，不称为裸 opcode latency。

### WGMMA throughput

准备合法的独立 accumulator/operand，连续提交多个 outstanding group，再 wait：

```text
for group in issue_depth:
    issue instructions_per_group
wait0
```

扫描 issue depth、instructions/group、resident WG 和 CTA。整卡 TFLOPS 用 CUDA Event；单 CTA 的 `flop/clk/SM` 只是 SM proxy。

## 4. 格式转换与 SFU

- latency：输出反馈为下一次输入，形成单链。
- throughput：每线程多套独立 register tuple，不包含 global/shared staging。
- 若一个 helper 包含 unpack、scale、convert、pack，多条 SASS 共同构成一个“转换原子”，报告 full sequence cycle，而非单 mnemonic latency。
- 用有限、非零、正常/边界输入校验每个输出元素，避免 NaN 掩盖错误。

## 5. reduction/softmax 等组合计算

当 max、shuffle、exp2、sum、rescale 紧密依赖时，可作为 composite micro-benchmark，但必须：

- 使用真实 thread-to-row/fragment mapping；
- 分开 local-state 与 cross-WG shared-state；
- 将状态初始化/重置成本移出 timed region或单列 baseline；
- 明确输出是 `cycle/tile` 或有效 element/clk，不称为单条 exp latency；
- 用 CPU reference 对拍 max、sum、LSE 和 output。

## 6. 计算验收清单

- latency 中存在可证明的 RAW/completion 依赖。
- throughput 中存在足量独立链，且扫描并行度到平台。
- global/shared staging 不意外进入纯计算 timed region。
- 精确 SASS opcode、operand mode和动态数量已核对。
- accumulator/sink 保持目标工作可观察，无 local-memory spill。
- FLOP、instruction 和 group 三种分母没有混用。
