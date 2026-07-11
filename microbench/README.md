# Reusable SM90 micro-benchmarks

这里存放跨 operator/kernel path 复用的原子 benchmark。当前只接受 SM90/Hopper。

**开始拆分前先查 [`index.md`](index.md)**。精确配置已存在就直接复用；同一指令族的新参数只在既有 family 下加配置，不按 operator 复制 benchmark。

当前 18 个配置的 opcode、动态计数和寄存器/spill 静态边界汇总见 [`static-validation.md`](static-validation.md)。

- [`memory/`](memory/)：global/shared/DSM/TMA/cache 数据移动。
- [`compute/`](compute/)：WGMMA、格式转换、softmax/SFU。

固定层级是 `<memory|compute>/<instruction-family>/<configuration>/`。family 共享 `benchmark.cu`；配置叶子用 Makefile 固定 shape/dtype/mode，并用 README 定义真实 geometry、计时边界、排除项和指标。当前 18 个配置均为 `validated`：本地 SM90a 编译、PTX/SASS 和 register/spill 静态证据已归档，但没有执行 benchmark。只有完成远端 H800 运行与原始数据保存后才能标为 `measured` 并用于定量性能结论。

公共构建/测量文件统一位于 [`common/`](common/)；`run_all.sh` 只处理公共 benchmark。operator-specific e2e/model 必须留在各自 path，不能放进公共层。

从仓库根目录批量操作：

```bash
./microbench/run_all.sh          # 默认 compile，不运行 binary
./microbench/run_all.sh static   # 生成 PTX/cubin/SASS，供静态核验
./microbench/run_all.sh clean
./microbench/run_all.sh run      # 仅远端 H800/SM90，必须显式指定
```

本地不得使用 `run`；单配置的 `make` 默认同样等价于 `make compile`。

## H800 output contract

每个 binary 成功时向 stdout 输出一个或多个 JSON record；公共 runner 同时保留完整 stdout/stderr，并把可解析 record 规范化为 JSONL。公共字段为：

- `benchmark`、`gpu`：稳定配置名和运行设备。
- `samples`、`min`、`mean`、`stddev`、`p05`、`p10`、`median`、`p90`、`p95`：设备 `%clock64` 区间统计；CTA、warpgroup 或 cluster 作用域由 leaf README 定义。
- shape/mode 字段：例如 `rank`、`operand_mode`、`measurement`、`pattern`、`depth`、`repeat` 和动态指令计数。
- 归一化指标：字段名显式带 `cycle_per_*`、`*_per_clk_sm` 或 `*_per_clk_cluster`；这些不是 host wall time 或整卡 GB/s。
- correctness 字段：`checksum`，以及适用 family 的 `expected_checksum`、`cpu_validation` 或 `untouched_zero`。校验失败时 binary 返回非零，不输出可登记为 `measured` 的结果。

批量 `run` 只覆盖每个配置的默认 CLI。TMA-load rank 2/3 controls、WGMMA operand/measurement 扫描等非默认点，应按各 leaf README 直接调用对应 `benchmark` 并逐行保存 JSON。

## Result layout

结果就近保存在配置叶子的 `result/`，不写配置根目录 `log`，也不复制到仓库顶层的中央结果树：

```text
<configuration>/
├── README.md
└── result/
    ├── summary.csv
    └── runs/<run_id>/
        ├── metadata.json
        ├── result.jsonl
        └── run.log
```

`run_id` 默认格式为 `YYYYMMDD-HHMMSS_<hostname>_<8位ID>`，已有目录永不覆盖。`metadata.json` 保存命令、环境白名单、Git/GPU/时钟信息、退出码和可用的 binary/SASS hash；`run.log` 保存完整输出；`result.jsonl` 只保存规范化 JSON record。`summary.csv` 是可重建索引，不是原始数据源。

从仓库根目录运行单个配置：

```bash
make -C microbench/compute/wgmma/m64n64k16_bf16_rs_ss_sm90 run \
  ARGS='--operand ss --measurement latency' RUN_ID='<optional>'

python tools/result_tool.py summarize \
  --result-dir microbench/compute/wgmma/m64n64k16_bf16_rs_ss_sm90/result
```

`./microbench/run_all.sh run` 为整批配置生成同一个 `RUN_ID`，便于跨目录关联；也可通过环境变量显式指定。命令失败或 JSON 解析失败时仍保留 `metadata.json` 和 `run.log`，但该 run 不得写入 leaf README 的 accepted 结果。

每个 leaf README 的 `H800 Results` 只总结选定的 accepted run：variant、median、p10/p90、主指标、correctness 和 run 路径。没有合格数据时统一写“尚无 accepted H800 run”，不得填示例数字。`index.md` 只维护状态和结果入口，不重复抄录性能值。
