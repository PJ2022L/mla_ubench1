# online_m64n64_exp2_shfl_sm90

- **Target**：`row max + shfl reduction + exp2 + row sum + O rescale` 紧耦合链。
- **Geometry**：`[64,64]` score tile 在一个 128-thread warpgroup 的真实 fragment mapping；分 local-state 与 cross-WG shared-max/scale 两种模式。
- **Timed body**：完整 online-softmax step；不含 QK/PV、跨-WG shared handoff。
- **Metrics**：cycle/tile、cycle/row、exp2 element/clk/SM。
- **Consumers**：sparse decode 使用 local-state；dense decode 使用 shared-state 和 P/O rescale。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 `M=64,N=64`。
