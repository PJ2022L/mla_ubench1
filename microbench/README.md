# Reusable SM90 micro-benchmarks

这里存放跨 operator/kernel path 复用的原子 benchmark。当前只接受 SM90/Hopper。

**开始拆分前先查 [`index.md`](index.md)**。精确配置已存在就直接复用；同一指令族的新参数只在既有 family 下加配置，不按 operator 复制 benchmark。

- [`memory/`](memory/)：global/shared/DSM/TMA/cache 数据移动。
- [`compute/`](compute/)：WGMMA、格式转换、softmax/SFU。

固定层级是 `<memory|compute>/<instruction-family>/<configuration>/`。family 共享 `benchmark.cu`；配置叶子用 Makefile 固定 shape/dtype/mode，并用 README 定义真实 geometry、计时边界、排除项和指标。CUDA 文件目前多数仍是 scaffold，不能把 TODO 输出当成实测结果。

公共构建/测量文件统一位于 [`common/`](common/)；`run_all.sh` 只运行公共 benchmark。operator-specific e2e/model 必须留在各自 path，不能放进公共层。
