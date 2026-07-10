# Shared microbench infrastructure

- `common.mk`：SM90a 构建、仓库/FlashMLA include 路由。
- `clock.cuh`：设备 cycle bracket 和主机时钟接口。
- `measure.hpp`：多次测量统计。
- `gpu_check.h`：CUDA 错误检查。
- `tma_util.cuh`：TMA tensor-map helper scaffold。
- `attention_shapes.h`：当前 benchmark 的默认 attention shape；最终应由命令行/宏覆盖。

这里只放跨 memory/compute benchmark 共享的基础设施，不放 operator-specific modeling 或 e2e。
