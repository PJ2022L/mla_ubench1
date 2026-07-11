# WGMMA family

层级固定为 `wgmma/<mM nN kK>_<dtype>_<operand-mode>_<arch>/`。family 共享 `benchmark.cu`；叶子 Makefile 只固定 MNK，运行时用 `--operand rs|ss|both` 分开报告 operand mode。

两个测量边界：

- `--measurement latency`：每条 WGMMA 独立执行 `fence → issue → commit_group → wait_group<0>`，报告 full completion latency。
- `--measurement throughput --issue-depth D --instructions-per-group I`：每个 group 连续 issue `I` 条 WGMMA 后 commit，累计 `D` 个 outstanding groups 再 `wait_group<0>`。`D∈[1,8]`，`I∈[1,36]`。

`--resident-wg 1|2` 用同一 CTA 中 128/256 threads 控制同时 resident 的独立 warpgroup。操作数预置在真实 SW128 shared layout；RS 的 A 在计时前搬入 registers；计时后把完整 FP32 accumulator fragment 写入 global sink。所有结果使用公共单行 JSON schema。

`I=4` 可匹配 dense QK/PV 的每-group issue 数；`I=16/20/36` 可分别测 sparse prefill 的大 group。这个开关只复现 WGMMA/commit batching，不包含源码中的 TMA、barrier、mixed group size 或 RS/SS 交错，因此结果不能标为完整 source schedule。
