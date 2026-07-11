# Shared microbench infrastructure

- `common.mk`：SM90a 构建、仓库/FlashMLA include 路由，以及显式 `compile/ptx/cubin/sass/static/run/clean` 目标；裸 `make` 等价于 `make compile`，不会隐式运行。`make run ARGS='...' RUN_ID='<optional>'` 通过仓库级结果工具把本次执行写入配置叶子的 `result/runs/<run_id>/`。
- `clock.cuh`：设备 `%clock64` bracket 和 host 侧 timestamp 汇总。
- `measure.hpp`：不改变样本顺序的 median/min/mean/stddev/p05/p10/p90/p95 统计。
- `benchmark_utils.hpp`：CLI、RAII `DeviceBuffer`、SM90 runtime guard、warmup/sample helper 和单行 JSON 输出。
- `gpu_check.h`：CUDA 错误检查。
- `tma_util.cuh`：通用 2D/3D/4D/5D TMA tiled tensor-map encode helper。
- `attention_shapes.h`：当前 benchmark 的默认 attention shape；最终应由命令行/宏覆盖。

这里只放跨 memory/compute benchmark 共享的基础设施，不放 operator-specific modeling 或 e2e。性能 binary 只能在远端 H800 上显式运行；本地允许 `make compile/ptx/cubin/sass/static` 静态检查。批量入口 `../run_all.sh` 默认也是 `compile`，只有 `../run_all.sh run` 才会执行 binary。

运行结果不写配置根目录的可覆盖 `log`。统一由 `../../tools/result_tool.py` 保存：

```text
result/
├── summary.csv
└── runs/<run_id>/
    ├── metadata.json
    ├── result.jsonl
    └── run.log
```

`result_tool.py summarize --result-dir <result>` 可随时从不可变 run 目录重建 `summary.csv`；`make clean` 只删除构建产物，不删除实验结果。
